import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod


class Backbone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dim = None
        self.model = None
        self.image_processor = None
        self.tokenizer = None

        self.logit_scale = 100.0

    @abstractmethod
    def encode_image(self, image):
        pass

    @abstractmethod
    def encode_text(self, text):
        pass

    def forward(self, image, text):
        pass
