import datetime
import argparse
from pathlib import Path

import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from utils.constants import *
from utils.util import str2bool, setup_logger, cosine_scheduler, auto_load_model, save_model
from utils.metrics import (
    multitask_binary_classification_report,
    binary_classification_report,
    multitask_classification_report,
    find_threshold,
)

from datasets import MultiLabelDataset, SingleExpertDataset
from models import build_single_expert_model, build_backbone, CLIPModel, AdaptationModel, build_adaptation_model
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("stage 1 evaluation", add_help=False)
    parser.add_argument("--batch_size", default=64, type=int, help="Per GPU batch size")
    parser.add_argument("--method", default="lp", type=str, help="name of finetuning method")

    # Model parameters
    parser.add_argument("--backbone", default="biomedclip", type=str, help="name of model backbone")
    parser.add_argument("--input_size", default=224, type=int, help="image input size")
    parser.add_argument("--residual_scale", default=0.1, type=float)
    # Dataset parameters
    parser.add_argument("--ckpt_dir", default="")
    parser.add_argument("--data_root", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument("--metadata_path", default="")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--experts",
        default=None,
        type=str,
        help="subsets for training",
    )
    parser.add_argument(
        "--eval_subsets",
        default="chexpert",
        type=str,
        help="subsets for training",
    )
    parser.add_argument("--save_pred", type=str2bool, default=True)

    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument(
        "--pin_mem",
        type=str2bool,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument(
        "--use_amp", type=str2bool, default=False, help="Use apex AMP (Automatic Mixed Precision) or not"
    )
    parser.add_argument("--eval_expert", type=str2bool, default=False)
    parser.add_argument("--peft", type=str2bool, default=True)

    # R-Adapter parameters
    parser.add_argument("--r_lora", type=float, default=0)
    parser.add_argument("--r_adapter", type=float, default=8)
    parser.add_argument("--r_drop_path", type=float, default=0.2)
    parser.add_argument("--r_ema", type=float, default=0.999)
    parser.add_argument("--r_bma", type=float, default=0)
    parser.add_argument("--r_eval_scale", type=float, default=0.5)

    # CoCoOp parameters
    parser.add_argument("-cocoop_n_ctx", type=int, default=4)
    parser.add_argument("-cocoop_ctx_init", type=str, default="")

    # TaskRes parameters
    parser.add_argument("--taskres_alpha", type=float, default=0.5)

    # CLAP parameters
    parser.add_argument("--clap_constraint", type=str, default="l2")
    return parser


def check_args(args):
    assert args.backbone in args.ckpt_dir

    args.experts_str = args.experts
    args.eval_subsets_str = args.eval_subsets

    subsets = [x.strip() for x in args.eval_subsets.split(",")]
    args.eval_subsets = subsets

    # assert len(subsets) == 1, f"Stage 1 training expect single subset, while got {len(subsets)}"
    assert all(
        [x in DATASETS for x in subsets]
    ), f"Argument {args.eval_subsets} contains invalid subsets. Supported datasets: {DATASETS}."

    if args.experts is None:
        args.experts = []
    else:
        experts = [x.strip() for x in args.experts.split(",")]
        args.experts = experts

        if len(experts) > 0:
            assert all(
                [x in TRAIN_DATASETS + OOD_DATASETS for x in experts]
            ), f"Argument {args.experts} contains invalid subsets. Supported datasets: {TRAIN_DATASETS}."


def build_model(args, device):
    if len(args.experts) == 0:
        backbone = build_backbone(args)
        model = CLIPModel(backbone=backbone).to(device)
    elif len(args.experts) >= 1:
        if args.method == "taskres":
            subsets_train = ["rsna-pulmonary-embolism", "chexpert", "CC-CCII", "ssim-covid19", "lung-pet-ct-dx"]

            dataset_train = MultiLabelDataset(args, subsets=subsets_train, split="train")
            args.num_labels = len(dataset_train.classes)
        model = build_adaptation_model(args)

        if args.method == "taskres":
            model.set_classes(dataset_train.classes)
            model.init_text_features(dataset_train.class_texts)
        checkpoint = torch.load(args.ckpt_dir, map_location="cpu")

        model.load_state_dict(checkpoint["model"])
        model = model.to(device)

        if args.method == "r-adapter":
            backbone = build_backbone(args)
            model_org = CLIPModel(backbone=backbone).to(device)

            org_state_dict = model.state_dict()

            model.init_wise_ft(args, org_state_dict, alpha=0.5)

            model_org.load_state_dict(org_state_dict, strict=False)

            return model_org

        elif args.method == "wise-ft":
            backbone = build_backbone(args)
            model_org = CLIPModel(backbone=backbone).to(device)

            org_state_dict = model.state_dict()
            model.init_wise_ft(org_state_dict, alpha=0.5)

        # adapter_state_dict = {
        #     k.replace("adapter.", ""): v for k, v in checkpoint["model"].items() if k.startswith("adapter.")
        # }
        # model.adapter.load_state_dict(adapter_state_dict)

    return model


def split_datasets(eval_subsets):
    id_datasets, ood_datasets, novel_datasets = [], [], []

    for subset in eval_subsets:
        if subset in TRAIN_DATASETS:
            id_datasets.append(subset)
        elif subset in OOD_DATASETS:
            ood_datasets.append(subset)
        elif subset in ZERO_SHOT_DATASETS:
            novel_datasets.append(subset)
        else:
            raise NotImplementedError

    return id_datasets, ood_datasets, novel_datasets


def main(args):
    logger = setup_logger("history", args.output_dir, "history.log", screen=True, tofile=True)
    logger.info(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    model = build_model(args, device)
    model.eval()

    for i, subset in enumerate(args.eval_subsets):
        # Multi-label datasets such as CheXpert should follow different evaluation pipelines
        if subset in ["chexpert", "mimic-cxr", "luna16"] + ZERO_SHOT_DATASETS:
            is_multi_label = True
        else:
            is_multi_label = False
            # dataset = SingleExpertDataset(args, subsets=subset, split="test")
        dataset = MultiLabelDataset(args, subsets=subset, split="test")

        data_loader_val = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
        )

        if args.method == "taskres":
            class_indices = [model.classes[cls] for cls in dataset.classes.keys()]

            logger.info(f"Using following class indices for method {args.method}: {class_indices}")
        else:
            model.init_text_features(dataset.class_texts)
        text_features = model.text_features

        logits_all = []
        labels_all = []

        feature_pretrained_all = []
        feature_residual_all = []

        logger.info(f"Validating on {subset}")
        logger.info(dataset.classes)
        logger.info(dataset.class_texts)
        for batch in tqdm(data_loader_val):
            with torch.no_grad():
                images = batch["image"].to(args.device, non_blocking=True)
                labels = batch["label"].to(args.device, non_blocking=True)

                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(images, text_features, True)
                else:
                    output = model(images, text_features, True)
                logits = output["logits"]

                if args.method == "taskres":
                    logits = logits[:, class_indices, :]

                logits_all.append(logits.detach().cpu().numpy())

                feature_pretrained = output["image_feature_pretrained"]
                feature_residual = (
                    output["image_feature_residual"]
                    if "image_feature_residual" in output.keys()
                    else torch.zeros_like(feature_pretrained)
                )

            labels_all.append(labels.detach().cpu().numpy())

            feature_pretrained_all.append(feature_pretrained.detach().cpu().numpy())
            feature_residual_all.append(feature_residual.detach().cpu().numpy())

        labels_all = np.concatenate(labels_all, axis=0)

        feature_pretrained_all = np.concatenate(feature_pretrained_all, axis=0).astype(np.float32)
        feature_residual_all = np.concatenate(feature_residual_all, axis=0).astype(np.float32)

        report = {}

        classes = list(data_loader_val.dataset.classes.keys())
        logits_all = np.concatenate(logits_all, axis=0).astype(np.float32)

        # prob shape: B, NC
        prob_all = torch.softmax(torch.from_numpy(logits_all), dim=-1)[:, :, -1].numpy()

        report.update(multitask_binary_classification_report(prob_all, labels_all, classes, threshold_max_f1=False))
        report.update(multitask_binary_classification_report(prob_all, labels_all, classes, threshold_max_f1=True))

        if not is_multi_label:
            # evaluate in the multi-class manner

            prob_multi_class = prob_all
            labels_multi_class = np.argmax(labels_all, axis=-1)

            report.update(
                multitask_classification_report(
                    prob_multi_class,
                    labels_multi_class,
                    classes=classes,
                    suffix="-multiclass",
                )
            )

        report = {k: float(v) for k, v in report.items()}

        logger.info(report)

        output_dir = os.path.join(args.output_dir, subset)
        with open(os.path.join(output_dir, "result.json"), "w") as fp:
            json.dump(report, fp, indent=2)

        if args.save_pred:
            np.savez(
                os.path.join(output_dir, "pred.npz"),
                logits=logits_all,
                labels=labels_all,
                image_feature_pretrained=feature_pretrained_all,
                image_feature_residual=feature_residual_all,
                text_features=(
                    np.asarray([x.cpu().numpy() for x in text_features], dtype=object) if text_features else None
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage 1 training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()

    check_args(args)
    if args.metadata_path == "":
        args.metadata_path = os.path.join(args.data_root, "metadata.csv")
    if args.experts_str is not None:
        if args.eval_expert:
            args.output_dir = os.path.join(args.ckpt_dir, args.method, f"train_{','.join(args.experts)}", "eval")
            args.ckpt_dir = os.path.join(
                args.ckpt_dir, args.method, f"train_{','.join(args.experts)}", "checkpoint-final.pth"
            )
        else:
            args.output_dir = os.path.join(
                args.ckpt_dir, args.method, f"train_adaptation_{','.join(args.experts)}", "eval"
            )
            args.ckpt_dir = os.path.join(
                args.ckpt_dir, args.method, f"train_adaptation_{','.join(args.experts)}", "checkpoint-final.pth"
            )
    else:
        args.output_dir = os.path.join(args.ckpt_dir, "eval")

    for subset in args.eval_subsets:
        Path(os.path.join(args.output_dir, subset)).mkdir(parents=True, exist_ok=True)
    main(args)
