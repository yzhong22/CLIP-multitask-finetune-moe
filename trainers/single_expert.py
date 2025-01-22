import os
import time, datetime
import math
import json
import numpy as np
import torch

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from utils.metrics import MetricLogger, find_threshold, binary_classification_report, multitask_classification_report
from utils.mixup import MixupADDiag
from utils.util import str2bool, setup_logger, cosine_scheduler, auto_load_model, adjust_learning_rate, save_model
from utils.optim_factory import create_optimizer, NativeScalerWithGradNormCount


class SingleExpertTrainer(object):
    def __init__(
        self,
        args,
        logger,
        model,
        data_loader_train,
        data_loader_test=None,
    ) -> None:
        self.args = args
        self.logger = logger

        self.model = model
        self.criterion = None
        self.optimizer = None
        self.mixup_fn = None
        self.loss_scaler = None

        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test
        self.max_accuracy = 0

        self.num_diag_classes = len(self.data_loader_train.dataset.classes)

        self._initialize()

    def train(self):
        self.logger.info("Start training for %d epochs" % self.args.epochs)
        start_time = time.time()

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.max_accuracy = 0.0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            train_stats = self.train_one_epoch(epoch)

            if self.args.output_dir and self.args.save_ckpt:
                if (epoch + 1) % self.args.save_ckpt_freq == 0 or epoch + 1 == self.args.epochs:
                    save_model(
                        args=self.args,
                        model_without_ddp=self.model,
                        optimizer=self.optimizer,
                        loss_scaler=self.loss_scaler,
                        epoch=epoch,
                    )

            if self.data_loader_test is not None:
                test_stats, is_best = self.evaluate(self.data_loader_test)
                if is_best:
                    if self.args.output_dir and self.args.save_ckpt:
                        save_model(
                            args=self.args,
                            epoch="best",
                            model_without_ddp=self.model,
                            optimizer=self.optimizer,
                            loss_scaler=self.loss_scaler,
                        )
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

            else:
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

            self.logger.info(log_stats)

        save_model(
            args=self.args,
            epoch="latest",
            model_without_ddp=self.model,
            optimizer=self.optimizer,
            loss_scaler=self.loss_scaler,
        )
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Training time {}".format(total_time_str))

    def train_one_epoch(self, epoch):
        self.model.train(True)
        self.optimizer.zero_grad()

        metric_logger = MetricLogger(logger=self.logger, delimiter=" ")

        header = "Epoch: [{}]".format(epoch)
        print_freq = 100

        text_features = self.model.text_features

        for data_iter_step, data_batch in enumerate(
            metric_logger.log_every(self.data_loader_train, print_freq, header)
        ):
            if data_iter_step % self.args.gradient_accumulation_steps == 0:
                adjust_learning_rate(self.optimizer, epoch, self.args)

            images = data_batch["image"].to(self.args.device, non_blocking=True)
            labels = data_batch["label"].to(self.args.device, non_blocking=True)

            labels_ad, labels_diag = labels[:, 0], labels[:, -1]

            if self.mixup_fn is not None:
                images, labels_ad, labels_diag = self.mixup_fn(images, labels_ad, labels_diag)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(images)

                    image_feature = output["image_feature"]
                    logits_ad = self.model.compute_logits(image_feature, text_features[0])["logits"]
                    logits_diag = self.model.compute_logits(image_feature, text_features[1])["logits"]

                    loss = (self.criterion(logits_ad, labels_ad) + self.criterion(logits_diag, labels_diag)) / 2
            else:
                output = self.model(images)

                image_feature = output["image_feature"]
                logits_ad = self.model.compute_logits(image_feature, text_features[0])["logits"]
                logits_diag = self.model.compute_logits(image_feature, text_features[1])["logits"]

                loss = (self.criterion(logits_ad, labels_ad) + self.criterion(logits_diag, labels_diag)) / 2

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                assert math.isfinite(loss_value)

            if self.args.use_amp:
                is_second_order = hasattr(self.optimizer, "is_second_order") and self.optimizer.is_second_order
                loss /= self.args.gradient_accumulation_steps
                grad_norm = self.loss_scaler(
                    loss,
                    self.optimizer,
                    clip_grad=self.args.clip_grad,
                    parameters=self.model.parameters(),
                    create_graph=is_second_order,
                    update_grad=(data_iter_step + 1) % self.args.gradient_accumulation_steps == 0,
                )
                if (data_iter_step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
            else:
                loss /= self.args.gradient_accumulation_steps
                loss.backward()

                if (data_iter_step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            min_lr = 10.0
            max_lr = 0.0

            for group in self.optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)

            weight_decay_value = None
            for group in self.optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]

            metric_logger.update(weight_decay=weight_decay_value)

            if self.args.use_amp:
                metric_logger.update(grad_norm=grad_norm)

            # break  # for debug only

        metric_logger.synchronize_between_processes()

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate(self, data_loader_val):
        # switch to evaluation mode
        self.model.eval()

        logits_ad_all = []
        logits_diag_all = []
        prob_ad_all = []
        prob_diag_all = []
        labels_all = []

        text_features = self.model.text_features

        for batch in data_loader_val:
            images = batch["image"].to(self.args.device, non_blocking=True)
            labels = batch["label"].to(self.args.device, non_blocking=True)

            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(images)
                    image_feature = output["image_feature"]
                    logits_ad = self.model.compute_logits(image_feature, text_features[0])["logits"]
                    logits_diag = self.model.compute_logits(image_feature, text_features[1])["logits"]
            else:
                output = self.model(images)
                image_feature = output["image_feature"]
                logits_ad = self.model.compute_logits(image_feature, text_features[0])["logits"]
                logits_diag = self.model.compute_logits(image_feature, text_features[1])["logits"]

            logits_ad_all.append(logits_ad.detach().cpu().numpy())
            logits_diag_all.append(logits_diag.detach().cpu().numpy())
            prob_ad_all.append(torch.softmax(logits_ad, dim=-1).detach().cpu().numpy())
            prob_diag_all.append(torch.softmax(logits_diag, dim=-1).detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

            torch.cuda.synchronize()

        logits_ad_all = np.concatenate(logits_ad_all, axis=0)
        logits_diag_all = np.concatenate(logits_diag_all, axis=0)
        prob_ad_all = np.concatenate(prob_ad_all, axis=0)
        prob_diag_all = np.concatenate(prob_diag_all, axis=0)

        labels_all = np.concatenate(labels_all, axis=0)

        thres_ad_max_f1 = find_threshold(prob_ad_all[:, -1], labels_all[:, 0])

        report = {}

        report.update(
            binary_classification_report(
                prob_ad_all[:, -1], labels_all[:, 0], threshold=thres_ad_max_f1, suffix="-ad@max_f1"
            )
        )
        report.update({"thres-ad@max_f1": thres_ad_max_f1})
        report.update(
            binary_classification_report(prob_ad_all[:, -1], labels_all[:, 0], threshold=0.5, suffix="-ad@0.5")
        )
        report.update(
            multitask_classification_report(
                prob_diag_all,
                labels_all[:, -1],
                classes=list(self.data_loader_train.dataset.classes.keys()),
                suffix="-diag",
            )
        )

        report = {k: float(v) for k, v in report.items()}

        is_best = False
        if self.max_accuracy < report["auc-diag"]:
            self.max_accuracy = report["auc-diag"]
            is_best = True

        with open(os.path.join(self.args.output_dir, "result_latest.json"), "w") as fp:
            json.dump(report, fp, indent=2)
        if is_best:
            with open(os.path.join(self.args.output_dir, "result_best.json"), "w") as fp:
                json.dump(report, fp, indent=2)

        if self.args.save_pred:
            np.savez(
                os.path.join(self.args.output_dir, "pred_latest.npz"),
                logits_ad=logits_ad_all,
                logits_diag=logits_diag_all,
                labels=labels_all,
                text_features=np.asarray([x.cpu().numpy() for x in text_features], dtype=object),
            )
            if is_best:
                np.savez(
                    os.path.join(self.args.output_dir, "pred_best.npz"),
                    logits_ad=logits_ad_all,
                    logits_diag=logits_diag_all,
                    labels=labels_all,
                    text_features=np.asarray([x.cpu().numpy() for x in text_features], dtype=object),
                )

        return report, is_best

    def _initialize(self):
        self._build_optimizer()
        self._build_criterion()

        auto_load_model(
            args=self.args,
            model_without_ddp=self.model,
            optimizer=self.optimizer,
            loss_scaler=self.loss_scaler,
        )

    def _build_optimizer(self):
        self.optimizer = create_optimizer(self.args, self.model, skip_list=None)
        self.loss_scaler = NativeScalerWithGradNormCount()

        # lr_schedule_values = cosine_scheduler(
        #     self.args.lr,
        #     self.args.min_lr,
        #     self.args.epochs,
        #     self.args.num_training_steps_per_epoch,
        #     warmup_epochs=self.args.warmup_epochs,
        #     warmup_steps=self.args.warmup_steps,
        # )

        # if self.args.weight_decay_end is None:
        #     self.args.weight_decay_end = self.args.weight_decay
        # wd_schedule_values = cosine_scheduler(
        #     self.args.weight_decay,
        #     self.args.weight_decay_end,
        #     self.args.epochs,
        #     self.args.num_training_steps_per_epoch,
        # )
        # self.logger.info("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    def _build_criterion(self):
        mixup_active = self.args.mixup > 0 or self.args.cutmix > 0.0 or self.args.cutmix_minmax is not None
        if mixup_active:
            self.logger.info("Mixup is activated!")
            self.mixup_fn = MixupADDiag(
                mixup_alpha=self.args.mixup,
                cutmix_alpha=self.args.cutmix,
                cutmix_minmax=self.args.cutmix_minmax,
                prob=self.args.mixup_prob,
                switch_prob=self.args.mixup_switch_prob,
                mode=self.args.mixup_mode,
                label_smoothing=self.args.smoothing,
                num_classes=self.num_diag_classes,
            )
        if self.mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif self.args.smoothing > 0.0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=self.args.smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.logger.info("criterion = %s" % str(self.criterion))
