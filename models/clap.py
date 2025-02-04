import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import Backbone
from models.clip import CLIPModel
from collections import OrderedDict
from einops import rearrange


class AdapterMethod(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.apply_constraint = args.clap_constraint
        self.distance = "l2"

        self.augmentations = True  # True
        self.epochs_aumentation = 20  # 20

        self.prototypes = None

        if self.apply_constraint != "none":
            print("Applying constraint to the logistic regression weights: " + str(self.distance))

        self.lagrangian_initialized = False

    def init_prompts(self, base_text_features):
        self.register_buffer("base_text_features", base_text_features)
        self.prototypes = nn.Parameter(base_text_features.clone())

    def zero_shot_constraint(self):
        device = self.prototypes.device
        # Compute constraint
        if "l2" in self.apply_constraint:
            disimilitude = (self.prototypes - self.base_text_features.clone().to(device)).pow(2).sum(-1)
        elif "cosine" in self.apply_constraint:
            disimilitude = 1 - torch.nn.functional.cosine_similarity(
                self.prototypes, self.base_text_features.clone().to(device)
            )
        else:
            print("Dissimilitude metric for constraint not implemented")
            assert False

        return torch.mean(self.alpha_constraint.to(device) * disimilitude)

    def init_lagrangian_multipliers(self, labels_ds, logits_ds):
        logits_ds = logits_ds.cpu().to(torch.float32)
        if self.lagrangian_initialized:
            return

        if "balanced" in self.apply_constraint:
            performance = torch.ones(logits_ds.shape[-1]).to(torch.float)
        else:
            with torch.no_grad():
                # Get one-hot encoding ground-truth
                labels_one_hot = torch.nn.functional.one_hot(labels_ds).cpu()

                # Get zero_shot performance
                performance = torch.diag(
                    torch.softmax(logits_ds, -1).t() @ labels_one_hot.to(torch.float32)
                ) / labels_one_hot.sum(0)

                if "corrected" in self.apply_constraint:
                    performance *= logits_ds.shape[-1] / torch.sum(performance).item()
                if "constant" in self.apply_constraint:
                    performance = torch.ones(logits_ds.shape[-1]).to(torch.float) * torch.mean(performance).item()

        # set new alphas
        self.register_buffer("alpha_constraint", torch.clone(performance))
        self.register_buffer("penalty_parameter", self.alpha_constraint)

        self.lagrangian_initialized = True

    def outer_step(self):
        def phr(h, lambd, rho):
            x = lambd + rho * h
            y_sup = 1 / (2 * rho) * (x**2 - lambd**2)
            y_inf = -1 / (2 * rho) * (lambd**2)

            grad_y_sup = x
            grad_y_inf = torch.zeros_like(h)

            sup = x >= 0
            return (torch.where(sup, y_sup, y_inf), torch.where(sup, grad_y_sup, grad_y_inf))

        device = self.prototypes

        print("Outer step on Augmented Lagrangian Multiplier")

        # Cmpute current constraints
        disimilitude = (self.prototypes - self.base_text_features.clone().to(device)).pow(2).sum(-1)

        # Compute phr
        phr_value, phr_grad = phr(disimilitude, self.alpha_constraint.to(device), self.penalty_parameter.to(device))

        # Update lagrangian multipliers
        self.alpha_constraint = phr_grad.detach().clone()

        # Update penalty parameters rho
        self.penalty_parameter = disimilitude.detach().clone()

        print("New lagrangian multipliers:")
        print(self.alpha_constraint[0:5].detach().cpu().numpy())

    def forward(self):
        return self.prototypes


class CLAPModel(CLIPModel):
    def __init__(self, backbone: Backbone, args) -> None:
        super().__init__(
            backbone,
        )
        self.num_labels = args.num_labels

        self.adapters = nn.ModuleList([AdapterMethod(args) for _ in range(args.num_labels)])

    def init_text_features(self, texts_multitasks):
        super().init_text_features(texts_multitasks)

        for i in range(self.num_labels):
            self.adapters[i].init_prompts(self.text_features[i])

    def forward(self, image, text_features=None, is_multi_label=False):
        output = self.encode_image(image)

        if is_multi_label:
            text_features = torch.stack([self.adapters[i]() for i in range(self.num_labels)], dim=0)
            text_features = text_features.to(image.device)

            logits = (
                self.backbone.logit_scale
                * F.normalize(output["image_feature"], dim=-1)
                @ F.normalize(text_features, dim=-1).transpose(1, 2)
            )
            output["logits"] = logits.transpose(0, 1)

        else:
            raise NotImplementedError

        return output

    def load_state_dict(self, state_dict, strict=True):
        """
        Override the load_state_dict method to skip keys containing 'base_text_features',
        even when strict=True.

        Args:
            state_dict (dict): The state dictionary to load.
            strict (bool): Whether to strictly enforce key matching.

        Returns:
            NamedTuple with missing_keys and unexpected_keys.
        """
        # Filter out keys containing 'base_text_features' from the state_dict
        filtered_state_dict = {k: v for k, v in state_dict.items() if "base_text_features" not in k}

        # Identify the skipped keys
        skipped_keys = [k for k in state_dict.keys() if "base_text_features" in k]
        for k in skipped_keys:
            assert torch.equal(state_dict[k], self.state_dict()[k].to(state_dict[k].device))
        if skipped_keys:
            print(f"Skipping keys: {skipped_keys}")

        # Call the parent class's load_state_dict with the filtered dictionary
        result = super().load_state_dict(filtered_state_dict, strict=False)  # Use strict=False here

        if "alpha_constraint" in filtered_state_dict.keys():
            for i in range(len(self.adapters)):
                self.adapters[i].lagrangian_initialized = True

        if strict:
            # Handle strict mode: Check for missing and unexpected keys
            missing_keys = [k for k in result.missing_keys if "base_text_features" not in k]
            unexpected_keys = result.unexpected_keys

            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise RuntimeError(
                    f"Error(s) in loading state_dict: Missing keys: {missing_keys}, "
                    f"Unexpected keys: {unexpected_keys}"
                )

        return result
