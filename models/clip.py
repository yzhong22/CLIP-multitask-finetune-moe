import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import Backbone
from models.adapter import CLIPAdapter


class CLIPModel(nn.Module):
    def __init__(self, backbone: Backbone, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.text_features = None

    def init_text_features(self, texts_multitasks):
        training = self.training
        self.eval()
        self.text_features = [self.backbone.encode_text(texts).clone().detach() for texts in texts_multitasks]
        self.train(training)

    def encode_image(self, image):
        image_feature_pretrained = self.backbone.encode_image(image)

        return {
            "image_feature": image_feature_pretrained,
            "image_feature_pretrained": image_feature_pretrained,
        }

    def compute_logits(self, image_feature, text_features):
        logits = self.backbone.logit_scale * F.normalize(image_feature, dim=-1) @ F.normalize(text_features, dim=-1).T

        return {"logits": logits}

    def forward(self, image, text_features=None, is_multi_label=False):
        output = self.encode_image(image)

        if is_multi_label and text_features is not None:
            text_features = torch.stack(text_features, dim=0)
            logits = (
                self.backbone.logit_scale
                * F.normalize(output["image_feature"], dim=-1)
                @ F.normalize(text_features, dim=-1).transpose(1, 2)
            )
            output["logits"] = logits.transpose(0, 1)
        elif text_features is not None:
            logits = self.compute_logits(output["image_feature"], text_features)
            output.update(logits)

        return output

    def init_wise_ft(self, zero_shot_state_dict, alpha=0.5):
        theta_0 = {k: v.clone() for k, v in zero_shot_state_dict.items()}
        theta_1 = {k: v.clone() for k, v in self.state_dict().items()}

        theta = _merge(alpha, theta_0, theta_1, None, 1e-8)

        self.load_state_dict(theta)

        print(f"WiSE-FT is activated with alpha = {alpha}")

    # def forward(self, image, text_features=None):
    #     output = self.encode_image(image)

    #     if text_features is not None:
    #         logits = self.compute_logits(output["image_feature"], text_features)
    #         output.update(logits)

    #     return output

    #     image_feature_pretrained = self.backbone.encode_image(image)
    #     image_feature = F.normalize(image_feature_pretrained, dim=-1)

    #     logits = self.backbone.logit_scale * image_feature @ self.text_features.transpose(1, 2)

    #     return {"logits": logits.transpose(0, 1), "feature_pretrained": image_feature_pretrained}


def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {key: (1 - alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()}

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta
