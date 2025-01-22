from models.biomedclip import BiomedCLIP

from models.clip import CLIPModel
from models.clip_adapter import AdaptationModel
from models.single_expert import SingleExpertModel
from models.moe import MoEModel


backbone_map = {"biomedclip": BiomedCLIP}


def build_backbone(args):
    return backbone_map[args.backbone]()


def build_single_expert_model(args):
    backbone = build_backbone(args)

    model = SingleExpertModel(backbone=backbone, residual_scale=args.residual_scale)

    return model


def build_moe_model(args):
    backbone = build_backbone(args)

    model = MoEModel(backbone=backbone, residual_scale=args.residual_scale, num_experts=len(args.subsets))

    return model
