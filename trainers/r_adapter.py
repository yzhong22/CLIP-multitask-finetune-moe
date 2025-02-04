import os
import time, datetime
import math
import json
import numpy as np
import torch

from utils.metrics import MetricLogger, multitask_binary_classification_report

from trainers.adaptation import AdaptationTrainer


def ema_update(args, model, m):
    encoders = []
    if args.r_adapter > 0:
        encoders.append(model.backbone.model.visual.trunk)
    for encoder in encoders:
        for i, _ in enumerate(encoder.blocks.children()):
            avg_model_params = list(_.adapter_attn_cache.parameters()) + list(_.adapter_mlp_cache.parameters())
            model_params = list(_.adapter_attn.parameters()) + list(_.adapter_mlp.parameters())
            for moving_avg_param, param in zip(avg_model_params, model_params):
                moving_avg_param.data = m * moving_avg_param.data + (1 - m) * param.data.detach()


def reparameterize(Wa, Wb=None, Ba=None, Bb=None, scale=1, do_residual=False):
    bias = 0
    id_tensor = 0
    if Ba is not None:
        bias = Ba @ Wb
    if Bb is not None:
        bias = bias + Bb
    if do_residual:
        if Wb is not None:
            id_tensor = torch.eye(Wa.shape[0], Wb.shape[1]).to(Wa.device)
        else:
            id_tensor = torch.eye(Wa.shape[0], Wa.shape[1]).to(Wa.device)
    if Wb is not None:
        weight = Wa @ Wb * scale + id_tensor
    else:
        weight = Wa * scale + id_tensor
    return weight.T, bias * scale if isinstance(bias, torch.Tensor) else None


def Rep_AdaptWeight(model, args, eval_scale=0.5):
    encoders = []
    if args.r_adapter > 0:
        encoders.append(model.backbone.model.visual.trunk)

    for encoder in encoders:
        for i, _ in enumerate(encoder.blocks.children()):
            _.merged = True

            with torch.no_grad():
                _.org_attn_weight, _.org_attn_bias = (
                    _.attn.proj.weight.data.clone().detach(),
                    _.attn.proj.bias.data.clone().detach(),
                )
                _.org_mlp_weight, _.org_mlp_bias = (
                    _.mlp.fc2.weight.data.clone().detach(),
                    _.mlp.fc2.bias.data.clone().detach(),
                )

            if _.ema or _.bma:
                adapter_attn, adapter_mlp = _.adapter_attn_cache, _.adapter_mlp_cache
            else:
                adapter_attn, adapter_mlp = _.adapter_attn, _.adapter_mlp

            adapt_attn_scale = adapter_attn.s * eval_scale

            if adapter_attn.embed_dim > adapter_attn.r:
                merged_weight, m = reparameterize(adapter_attn.d.weight.squeeze().T, adapter_attn.u.weight.squeeze().T)
                adapt_attn_weight, adapt_attn_bias = reparameterize(
                    merged_weight.squeeze().T, scale=adapt_attn_scale, do_residual=True
                )
            else:
                adapt_attn_weight, adapt_attn_bias = reparameterize(
                    adapter_attn.f.weight.squeeze().T, scale=adapt_attn_scale, do_residual=True
                )

            rep_attn_weight, rep_attn_bias = reparameterize(
                _.attn.proj.weight.T, adapt_attn_weight.T, _.attn.proj.bias, adapt_attn_bias
            )

            adapt_mlp_scale = adapter_mlp.s * eval_scale

            if adapter_mlp.embed_dim > adapter_mlp.r:
                merged_weight, m = reparameterize(adapter_mlp.d.weight.squeeze().T, adapter_mlp.u.weight.squeeze().T)
                adapt_mlp_weight, adapt_mlp_bias = reparameterize(
                    merged_weight.squeeze().T, scale=adapt_mlp_scale, do_residual=True
                )
            else:
                adapt_mlp_weight, adapt_mlp_bias = reparameterize(
                    adapter_mlp.f.weight.squeeze().T, scale=adapt_mlp_scale, do_residual=True
                )

            rep_mlp_weight, rep_mlp_bias = reparameterize(
                _.mlp.fc2.weight.T, adapt_mlp_weight.T, _.mlp.fc2.bias, adapt_mlp_bias
            )

            with torch.no_grad():
                _.attn.proj.weight.copy_(rep_attn_weight)
                _.attn.proj.bias.copy_(rep_attn_bias)
                _.mlp.fc2.weight.copy_(rep_mlp_weight)
                _.mlp.fc2.bias.copy_(rep_mlp_bias)


def Repback_AdaptWeight(model, args):
    encoders = []
    if args.r_adapter > 0:
        encoders.append(model.backbone.model.visual.trunk)

    for encoder in encoders:
        for i, _ in enumerate(encoder.blocks.children()):
            _.merged = False
            with torch.no_grad():
                _.attn.proj.weight.copy_(_.org_attn_weight)
                _.attn.proj.bias.copy_(_.org_attn_bias)
                _.mlp.fc2.weight.copy_(_.org_mlp_weight)
                _.mlp.fc2.bias.copy_(_.org_mlp_bias)


class RAdapterTrainer(AdaptationTrainer):
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
        else:
            output = self.model(images, text_features, True)["logits"]
            loss = self.criterion(output, labels)

        return loss

    def _hook_after_update(self):
        if self.args.r_ema and self.args.peft:
            with torch.no_grad():
                ema_update(self.args, self.model, self.args.r_ema)

    @torch.no_grad()
    def evaluate(self, data_loader_val):
        # switch to evaluation mode
        self.model.eval()

        Rep_AdaptWeight(
            self.model,
            self.args,
        )

        logits_all = []
        prob_all = []
        labels_all = []

        text_features = self.model.text_features

        count = 200

        for i, batch in enumerate(data_loader_val):
            # if i >= count:
            #     break
            images = batch["image"].to(self.args.device, non_blocking=True)
            labels = batch["label"].to(self.args.device, non_blocking=True)

            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(images, text_features, True)["logits"]
            else:
                output = self.model(images, text_features, True)["logits"]

            logits_all.append(output.detach().cpu().numpy())
            prob_all.append(torch.softmax(output, dim=-1)[:, :, -1].detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

            torch.cuda.synchronize()

        Repback_AdaptWeight(self.model, self.args)

        logits_all = np.concatenate(logits_all, axis=0)
        prob_all = np.concatenate(prob_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)

        report = {}
        classes = list(data_loader_val.dataset.classes.keys())
        report.update(multitask_binary_classification_report(prob_all, labels_all, classes, threshold_max_f1=False))
        report.update(multitask_binary_classification_report(prob_all, labels_all, classes, threshold_max_f1=True))

        is_best = False
        if self.max_accuracy < report["auc_avg@max_f1"]:
            self.max_accuracy = report["auc_avg@max_f1"]
            is_best = True

        with open(os.path.join(self.args.output_dir, "result_latest.json"), "w") as fp:
            json.dump(report, fp, indent=2)
        if is_best:
            with open(os.path.join(self.args.output_dir, "result_best.json"), "w") as fp:
                json.dump(report, fp, indent=2)

        if self.args.save_pred:
            np.savez(
                os.path.join(self.args.output_dir, "pred_latest.npz"),
                logits=logits_all,
                labels=labels_all,
                text_features=np.asarray([x.cpu().numpy() for x in text_features], dtype=object),
            )
            if is_best:
                np.savez(
                    os.path.join(self.args.output_dir, "pred_best.npz"),
                    logits=logits_all,
                    labels=labels_all,
                    text_features=np.asarray([x.cpu().numpy() for x in text_features], dtype=object),
                )

        return report, is_best
