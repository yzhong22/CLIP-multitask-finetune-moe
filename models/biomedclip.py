import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import Backbone

from open_clip import create_model_from_pretrained, get_tokenizer


class BiomedCLIP(Backbone):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model, self.image_processor = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.context_length = 256
        self.logit_scale = float(self.model.logit_scale.exp())

        self.dim = 512

    def encode_image(self, image):
        return self.model.encode_image(image, normalize=False)

    def encode_text(self, texts):
        texts = self.tokenizer(texts, context_length=self.context_length).to(next(self.model.parameters()).device)
        text_feature = self.model.encode_text(texts, normalize=False)

        return text_feature

    def forward(self, image, text):
        image_feature = self.encode_image(image)
        text_feature = self.encode_text(text)

        logits = self.logit_scale * image_feature @ text_feature.t()

        return logits
