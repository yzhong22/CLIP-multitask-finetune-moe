import os
import time, datetime
import math
import json
import numpy as np
import torch
import copy
from tqdm import tqdm

from torch.utils.data import random_split

from utils.metrics import MetricLogger, multitask_binary_classification_report

from trainers.adaptation import AdaptationTrainer


class CLAPTrainer(AdaptationTrainer):
    def cal_loss(self, data_batch):
        text_features = self.model.text_features

        images = data_batch["image"].to(self.args.device, non_blocking=True)
        labels = data_batch["label"].to(self.args.device, non_blocking=True)

        labels = torch.stack([1 - labels, labels], dim=-1)

        if self.mixup_fn is not None:
            images, labels = self.mixup_fn(images, labels)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                output = self.model(images, text_features, True)["logits"]
                loss = self.criterion(output, labels)

                if self.model.adapters[0].apply_constraint != "none":
                    loss_constraint = torch.stack(
                        [self.model.adapters[i].zero_shot_constraint() for i in range(len(self.model.adapters))]
                    ).mean()
                    loss += loss_constraint

        else:
            output = self.model(images, text_features, True)["logits"]
            loss = self.criterion(output, labels) + loss_constraint
            if self.model.adapters[0].apply_constraint != "none":
                loss_constraint = torch.stack(
                    [self.model.adapters[i].zero_shot_constraint() for i in range(len(self.model.adapters))]
                ).mean()
                loss += loss_constraint

        return loss

    def _hook_before_train(self):
        if not self.model.adapters[0].lagrangian_initialized:
            self.logger.info(f"Extracting features on training dataset")
            logits_ds, labels_ds = self.extract_features(reps=1, augmentation=True)

        if self.model.adapters[0].apply_constraint != "none":
            print("Getting initial lagrangian multipliers for constraint formulation", end="\n")

            self.logger.info("Lagrangian multipliers: ")
            for i in range(len(self.model.adapters)):
                self.model.adapters[i].init_lagrangian_multipliers(labels_ds[:, i], logits_ds[:, i])

                self.logger.info(
                    list(torch.round(self.model.adapters[i].alpha_constraint.detach(), decimals=3).cpu().numpy())
                )

    def _hook_after_epoch(self):
        if "adaptative" in self.model.adapters[0].apply_constraint:
            for i in range(len(self.model.adapters)):
                self.model.adapters[i].outer_step()

    def extract_features(self, reps=20, augmentation=True):
        training = self.model.training
        self.model.eval()

        dataset = copy.deepcopy(self.data_loader_train.dataset)

        if not augmentation:
            dataset.transform = self.data_loader_test.dataset.transform

        subset_size = 50000
        remaining_size = len(dataset) - subset_size
        dataset, _ = random_split(dataset, [subset_size, remaining_size])

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(1.5 * self.args.batch_size),
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=False,
            shuffle=False,
        )

        labels_ds, logits_ds = [], []

        for rep in range(reps):
            for i, batch in tqdm(enumerate(data_loader)):
                images = batch["image"].to(self.args.device, non_blocking=True)
                labels = batch["label"].to(self.args.device, non_blocking=True)

                with torch.no_grad():
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            output = self.model(images, None, True)["logits"]
                    else:
                        output = self.model(images, None, True)["logits"]

                logits_ds.append(output)
                labels_ds.append(labels)

        labels_ds = torch.cat(labels_ds, dim=0)
        logits_ds = torch.cat(logits_ds, dim=0)

        self.model.train(training)

        return logits_ds, labels_ds
