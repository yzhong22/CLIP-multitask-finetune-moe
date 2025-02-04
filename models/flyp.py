import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import Backbone
from models.clip import CLIPModel
from collections import OrderedDict
from einops import rearrange


class FLYPModel(CLIPModel):
    def __init__(self, backbone: Backbone) -> None:
        super().__init__(
            backbone,
        )

    def forward(self, image, text_features=None, is_multi_label=False):
        if self.training:
            return self.encode_image(image)

        else:
            return super().forward(image, text_features, is_multi_label)
