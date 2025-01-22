import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import Backbone
from models.adapter import CLIPAdapter


class MoEModel(nn.Module):
    def __init__(self, backbone: Backbone, num_experts, residual_scale=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.num_experts = num_experts
        self.residual_scale = residual_scale

        self.adapters = nn.ModuleList(
            [CLIPAdapter(self.backbone.dim, bottleneck=768) for _ in range(self.num_experts)]
        )

        self.router = nn.Linear(self, backbone.dim, self.num_experts)

        self.text_features = None

    def load_experts(self, state_dicts):
        for i, adapter in enumerate(self.adapters):
            state_dict = {
                k.replace("adapter.", ""): v for k, v in state_dicts[i].items() if str.startswith(k, "adapter.")
            }
            adapter.load_state_dict(state_dict)

    def init_text_features(self, texts_multitasks):
        self.text_features = (
            torch.stack([F.normalize(self.backbone.encode_text(texts), dim - 1) for texts in texts_multitasks], dim=0)
            .clone()
            .detach()
        )  # Shape: D, 2, C

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
        # image_feature = F.normalize(image_feature, dim=-1)

        return image_feature

    def forward(self, image):
        assert self.text_features is not None, "Text features are not initialized."

        image_feature_pretrained = self.backbone.encode_image(image)

        expert_residuals = []

        for i, adapter in enumerate(self.adapters):
            adapt_med, adapt_out = adapter(image_feature_pretrained)

            residual_scaled = (
                self.residual_scale
                * adapt_out
                / torch.norm(adapt_out, dim=-1, keepdim=True)
                * torch.norm(image_feature_pretrained, dim=-1, keepdim=True)
            )

            expert_residuals.append(residual_scaled)

        expert_residuals = torch.stack(expert_residuals, dim=1)  # B, D, C

        logits = self.router(image_feature_pretrained)
        weights = torch.softmax(logits, dim=-1)

        residual = torch.sum(expert_residuals * weights.unsqueeze(-1))

        image_feature = image_feature_pretrained + residual
        image_feature = F.normalize(image_feature, dim=-1)

        logits = self.backbone.logit_scale * image_feature @ self.text_features.transpose(1, 2)

        return {"logits": logits.transpose(0, 1)}


# class SingleExpertModel(nn.Module):
#     def __init__(self, backbone: Backbone, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         # self.backbone = backbone
#         # self.adapter = CLIPAdapter(self.backbone.dim, bottleneck=768)

#         self.backbone = nn.Sequential(nn.Conv2d(3, 512, 3, 1, 1), nn.AdaptiveAvgPool2d(1))
#         self.classifier = nn.Linear(512, 2 * 5)

#         self.text_features = None

#     def init_text_features(self, texts_multitasks):
#         pass

#     def forward(self, image):
#         B = image.shape[0]
#         x = self.backbone(image).view(B, 512)

#         x = self.classifier(x).view(B, 5, 2)

#         return x
