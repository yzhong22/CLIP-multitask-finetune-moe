import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import Backbone
from models.clip import CLIPModel
from collections import OrderedDict
from einops import rearrange


class TaskResLearner(nn.Module):
    def __init__(self, residual_scale=1.0):
        super().__init__()
        self.alpha = residual_scale

        self.text_feature_residuals = None

    def init_prompts(self, base_text_features):
        try:
            self.register_buffer("base_text_features", base_text_features)
        except:
            return
        self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

    def forward(self):
        return self.base_text_features + self.alpha * self.text_feature_residuals  # t + a * x


class TaskResModel(CLIPModel):
    def __init__(self, backbone: Backbone, args) -> None:
        super().__init__(
            backbone,
        )
        self.num_labels = args.num_labels
        self.classes = None

        self.prompt_learners = nn.ModuleList([TaskResLearner(args.taskres_alpha) for _ in range(args.num_labels)])

    def set_classes(self, classes):
        self.classes = classes

    def init_text_features(self, texts_multitasks):
        super().init_text_features(texts_multitasks)

        for i in range(self.num_labels):
            self.prompt_learners[i].init_prompts(self.text_features[i])

    def forward(self, image, text_features=None, is_multi_label=False):
        output = self.encode_image(image)

        if is_multi_label:
            text_features = torch.stack([self.prompt_learners[i]() for i in range(self.num_labels)], dim=0)
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
            assert torch.all(
                torch.lt(torch.abs(torch.add(state_dict[k], -self.state_dict()[k].to(state_dict[k].device))), 1e-5)
            ), f"{state_dict[k]} and {self.state_dict()[k].to(state_dict[k].device)}."
        if skipped_keys:
            print(f"Skipping keys: {skipped_keys}")

        # Call the parent class's load_state_dict with the filtered dictionary
        result = super().load_state_dict(filtered_state_dict, strict=False)  # Use strict=False here

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
