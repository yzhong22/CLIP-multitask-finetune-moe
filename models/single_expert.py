import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import Backbone
from models.adapter import CLIPAdapter

# from models.clip import CLIPModel
from models.clip_adapter import AdaptationModel


class SingleExpertModel(AdaptationModel):
    def __init__(self, backbone: Backbone, residual_scale=0.1, *args, **kwargs) -> None:
        super().__init__(backbone, *args, **kwargs)

        self.residual_scale = residual_scale

    def encode_image(self, image):
        image_feature_pretrained = self.backbone.encode_image(image)
        adapt_med, adapt_out = self.adapter(image_feature_pretrained)

        residual_scaled = (
            self.residual_scale
            * adapt_out
            / torch.norm(adapt_out, dim=-1, keepdim=True)
            * torch.norm(image_feature_pretrained, dim=-1, keepdim=True)
        )

        image_feature = image_feature_pretrained + residual_scaled

        return {
            "image_feature": image_feature,
            "image_feature_pretrained": image_feature_pretrained,
            "image_feature_residual": residual_scaled,
        }
