import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import Backbone
from models.adapter import CLIPAdapter
from models.clip import CLIPModel


class AdaptationModel(CLIPModel):
    def __init__(self, backbone: Backbone, *args, **kwargs) -> None:
        super().__init__(backbone, *args, **kwargs)

        self.adapter = CLIPAdapter(self.backbone.dim, bottleneck=768)

    def load(self, state_dict):
        adapter_state_dict = {k.replace("adapter.", ""): v for k, v in state_dict.items() if k.startswith("adapter.")}
        self.adapter.load_state_dict(adapter_state_dict)

    def encode_image(self, image):
        image_feature_pretrained = self.backbone.encode_image(image)
        adapt_med, adapt_out = self.adapter(image_feature_pretrained)

        image_feature = image_feature_pretrained + adapt_out

        return {
            "image_feature": image_feature,
            "image_feature_pretrained": image_feature_pretrained,
            "image_feature_residual": adapt_out,
        }
