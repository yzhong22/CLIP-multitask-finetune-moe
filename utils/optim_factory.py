# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import optim as optim


import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or name.endswith(".gamma")
            or name.endswith(".beta")
        ):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
            parameter_group_vars[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    # if weight_decay and filter_bias_and_bn:
    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    if "fused" in opt_lower:
        assert has_apex and torch.cuda.is_available(), "APEX and CUDA required for fused optimizers"

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, "opt_eps") and args.opt_eps is not None:
        opt_args["eps"] = args.opt_eps
    if hasattr(args, "opt_betas") and args.opt_betas is not None:
        opt_args["betas"] = args.opt_betas

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "sgd" or opt_lower == "nesterov":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "momentum":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "nadam":
        from timm.optim.nadam import Nadam

        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == "radam":
        from timm.optim.radam import RAdam

        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == "adamp":
        from timm.optim.adamp import AdamP

        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == "sgdp":
        from timm.optim.sgdp import SGDP

        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "adafactor":
        from timm.optim.adafactor import Adafactor

        if not args.lr:
            opt_args["lr"] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == "adahessian":
        from timm.optim.adahessian import Adahessian

        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == "rmsprop":
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == "rmsproptf":
        from timm.optim.rmsprop_tf import RMSpropTF

        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == "novograd":
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == "nvnovograd":
        from timm.optim.nvnovograd import NvNovoGrad

        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == "fusedsgd":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "fusedmomentum":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == "fusedadam":
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == "fusedadamw":
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == "fusedlamb":
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == "fusednovograd":
        opt_args.setdefault("betas", (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == "lookahead":
            from timm.optim.lookahead import Lookahead

            optimizer = Lookahead(optimizer)

    return optimizer


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
        )
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
