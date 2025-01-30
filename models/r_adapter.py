import copy
import torch
import torch.nn as nn

from models.backbone import Backbone
from models.clip import CLIPModel

from timm.models.layers import DropPath


class StochasticAdapter(nn.Module):
    def __init__(self, embed_dim, r=64, init_value=0.1, eval_scale=0.5, drop_path=0, scale_train=True, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.r = r
        self.bias = bias
        self.scale_train = scale_train
        self.drop_path = drop_path
        self.init_value = init_value
        self.s = (
            nn.Parameter(init_value * torch.ones(1), requires_grad=self.scale_train)
            if self.scale_train
            else init_value
        )
        self.loss = torch.zeros([1])
        self.drop_paths = DropPath(drop_path, scale_by_keep=True)
        self.eval_scale = eval_scale

        if embed_dim > r:
            self.d = nn.Linear(embed_dim, r, bias=bias)
            self.u = nn.Linear(r, embed_dim, bias=bias)
            nn.init.xavier_uniform_(self.d.weight)
            nn.init.zeros_(self.u.weight)
            if bias:
                nn.init.zeros_(self.d.bias)
                nn.init.zeros_(self.u.bias)
        else:
            self.f = nn.Linear(embed_dim, embed_dim, bias=bias)
            nn.init.zeros_(self.f.weight)
            if bias:
                nn.init.zeros_(self.f.bias)

    def forward(self, x):
        if self.embed_dim > self.r:
            z = self.u(self.d(x))
        else:
            z = self.f(x)
        z = self.drop_paths(z)

        scale = self.s if self.training else self.s * self.eval_scale
        x = x + z * scale

        return x


def forward_vit_block_adapter(self, x):
    if not self.training and (self.ema or self.bma):
        adapter_attn, adapter_mlp = self.adapter_attn_cache, self.adapter_mlp_cache
    else:
        adapter_attn, adapter_mlp = self.adapter_attn, self.adapter_mlp

    if self.merged:
        # x = x + self.attention(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    else:
        # x = x + adapter_attn(self.attention(self.ln_1(x)))
        # x = x + adapter_mlp(self.mlp(self.ln_2(x)))

        x = x + self.drop_path1(self.ls1(adapter_attn(self.attn(self.norm1(x)))))
        x = x + self.drop_path2(self.ls2(adapter_mlp(self.mlp(self.norm2(x)))))

    return x


def peft_init(args):
    peft_config = {}
    peft_config["lora"] = args.r_lora
    peft_config["adapter"] = args.r_adapter
    peft_config["drop_path"] = args.r_drop_path
    peft_config["ema"] = args.r_ema
    peft_config["bma"] = args.r_bma
    peft_config["eval_scale"] = args.r_eval_scale
    return peft_config


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "lora_" in n and "cache" not in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def mark_only_adapter_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "adapter" in n and "cache" not in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


class RAdapterModel(CLIPModel):
    def __init__(self, backbone: Backbone, args) -> None:
        super().__init__(backbone)

        self.args = args

        self.use_peft = args.peft and any(
            [
                args.r_lora > 0,
                args.r_adapter > 0,
            ]
        )
        if self.use_peft:
            peft_config = peft_init(args)
            self.set_Adapter(768, peft_config)

        for p in self.backbone.parameters():
            p.requires_grad = False

        if args.r_lora > 0:
            mark_only_lora_as_trainable(self.backbone.model.visual)
        if args.r_adapter > 0:
            mark_only_adapter_as_trainable(self.backbone.model.visual)

    def set_Adapter(self, dim, peft_config):
        for i, _ in enumerate(self.backbone.model.visual.trunk.blocks.children()):
            if peft_config["adapter"] > 0:
                _.adapter_attn = StochasticAdapter(
                    embed_dim=dim,
                    r=peft_config["adapter"],
                    drop_path=peft_config["drop_path"],
                    eval_scale=peft_config["eval_scale"],
                )
                _.adapter_mlp = StochasticAdapter(
                    embed_dim=dim,
                    r=peft_config["adapter"],
                    drop_path=peft_config["drop_path"],
                    eval_scale=peft_config["eval_scale"],
                )
            else:
                _.adapter_attn, _.adapter_mlp = None, None

            with torch.no_grad():
                _.org_proj_weight, _.org_proj_bias = _.attn.proj.weight.data, _.attn.proj.bias.data
                _.org_fc2_weight, _.org_fc2_bias = _.mlp.fc2.weight.data, _.mlp.fc2.bias.data

            _.ema, _.bma = peft_config["ema"], peft_config["bma"]
            _.merged = False

            if _.ema or _.bma:
                _.adapter_attn_cache = copy.deepcopy(_.adapter_attn)
                _.adapter_mlp_cache = copy.deepcopy(_.adapter_mlp)

            bound_method = forward_vit_block_adapter.__get__(_, _.__class__)
            setattr(_, "forward", bound_method)
