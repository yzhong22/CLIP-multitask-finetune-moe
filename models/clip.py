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
