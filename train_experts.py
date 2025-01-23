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

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from utils.constants import TRAIN_DATASETS
from utils.mixup import MixupADDiag
from utils.util import str2bool, setup_logger, cosine_scheduler, auto_load_model, save_model
from utils.optim_factory import create_optimizer, NativeScalerWithGradNormCount

from datasets import SingleExpertDataset, MultiLabelDataset
from models import build_single_expert_model
from trainers import SingleExpertTrainer, AdaptationTrainer


def get_args_parser():
    parser = argparse.ArgumentParser("stage 1 training", add_help=False)
    parser.add_argument("--method", default="lp", type=str, help="name of finetuning method")
    parser.add_argument("--batch_size", default=64, type=int, help="Per GPU batch size")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument(
        "--balance_subset",
        type=str2bool,
        default=False,
    )

    # Model parameters
    parser.add_argument("--backbone", default="biomedclip", type=str, help="name of model backbone")
    parser.add_argument("--input_size", default=224, type=int, help="image input size")
    parser.add_argument("--residual_scale", default=0.1, type=float)
    parser.add_argument("--drop_path", type=float, default=0.0, metavar="PCT", help="Drop path rate (default: 0.1)")

    # Optimization parameters
    parser.add_argument(
        "--clip_grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument("--layer_decay", type=float, default=1.0)
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-6)",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=20, metavar="N", help="epochs to warmup LR, if scheduler supports"
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw"')
    parser.add_argument(
        "--opt_eps", default=1e-8, type=float, metavar="EPSILON", help="Optimizer Epsilon (default: 1e-8)"
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    )
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")

    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # * Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")

    # * Mixup params
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup alpha, mixup enabled if > 0.")
    parser.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha, cutmix enabled if > 0.")
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Dataset parameters
    parser.add_argument("--data_root", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--subsets",
        default="chexpert",
        type=str,
        help="subsets for training",
    )
    parser.add_argument("--auto_resume", type=str2bool, default=True)
    parser.add_argument("--save_ckpt", type=str2bool, default=True)
    parser.add_argument("--save_pred", type=str2bool, default=True)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_num", default=3, type=int)

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument(
        "--pin_mem",
        type=str2bool,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument(
        "--use_amp", type=str2bool, default=False, help="Use apex AMP (Automatic Mixed Precision) or not"
    )
    return parser


def check_args(args):
    subsets = [x.strip() for x in args.subsets.split(",")]
    args.subsets = subsets

    # assert len(subsets) == 1, f"Stage 1 training expect single subset, while got {len(subsets)}"
    assert all(
        [x in TRAIN_DATASETS for x in subsets]
    ), f"Argument {args.subsets} contains invalid subsets. Supported datasets: {TRAIN_DATASETS}."


def main(args):
    logger = setup_logger("history", args.output_dir, "history.log", screen=True, tofile=True)
    logger.info(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # dataset_train = SingleExpertDataset(args, subsets=args.subsets, split="train")
    # dataset_test = SingleExpertDataset(args, subsets=args.subsets, split="test")
    dataset_train = MultiLabelDataset(args, subsets=args.subsets, split="train")
    dataset_test = MultiLabelDataset(args, subsets=args.subsets, split="test")

    dataset_test.set_classes(dataset_train.classes, dataset_train.class_texts)

    logger.info(f"Classes for diagnose: {dataset_train.classes}")
    logger.info(f"Class prompts: {dataset_train.class_texts}")

    if args.balance_subset:
        sample_weights = dataset_train.get_balance_sample_weights()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            sampler=sampler,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            shuffle=True,
        )

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
        )
    else:
        data_loader_test = None

    model = build_single_expert_model(args).to(device)
    model.init_text_features(dataset_train.class_texts)

    for p in model.backbone.parameters():
        p.requires_grad = False

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_paramteters = [n for n, p in model.named_parameters() if p.requires_grad]

    logger.info("Model = %s" % str(model))
    logger.info(f"Trainable params: {trainable_paramteters}")
    logger.info(f"Number of params: {n_parameters}")

    eff_batch_size = args.batch_size * args.gradient_accumulation_steps
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size
    args.num_training_steps_per_epoch = num_training_steps_per_epoch

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    logger.info("Base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logger.info("Actual lr: %.2e" % args.lr)

    logger.info("Batch size = %d" % eff_batch_size)
    logger.info("Gradient accumulation steps = %d" % args.gradient_accumulation_steps)
    logger.info("Number of training examples = %d" % len(dataset_train))
    logger.info("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    trainer = AdaptationTrainer(
        args=args, logger=logger, model=model, data_loader_train=data_loader_train, data_loader_test=data_loader_test
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage 1 training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    check_args(args)
    args.metadata_path = os.path.join(args.data_root, "metadata.csv")
    if args.output_dir:
        args.output_dir = os.path.join(
            args.output_dir, args.backbone, f"seed{args.seed}", args.method, f"train_{','.join(args.subsets)}"
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
