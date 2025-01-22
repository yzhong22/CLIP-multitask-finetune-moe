import os
import math
import time
from collections import defaultdict, deque
import datetime
import numpy as np

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, accuracy_score
import torch
import torch.distributed as dist
from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger(object):
    def __init__(self, logger, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.logger = logger
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            "{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable))
        )


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def find_threshold(tol_output, tol_target):
    # to find this thresold, first we get the precision and recall without this, from there we calculate f1 score,
    # using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation
    # are used to calculate our binary output.

    p, r, t = precision_recall_curve(tol_target, tol_output)
    # Choose the best threshold based on the highest F1 measure
    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p) + 1e-8))
    bestthr = t[np.where(f1 == max(f1))]

    return bestthr[0]


def binary_classification_report(pred, y, threshold=0.5, suffix=""):
    auc = roc_auc_score(y, pred)

    tn, fp, fn, tp = confusion_matrix(y, (pred > threshold).astype(int)).ravel()
    report = {
        f"auc{suffix}": auc,
        f"acc{suffix}": (tp + tn) / (tn + fp + fn + tp),
        f"tpr{suffix}": tp / (tp + fn),
        f"tnr{suffix}": tn / (tn + fp),
        f"tn{suffix}": int(tn),
        f"fp{suffix}": int(fp),
        f"fn{suffix}": int(fn),
        f"tp{suffix}": int(tp),
    }

    return report


def one_hot_np(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def multitask_binary_classification_report(pred, y, classes, threshold_max_f1=False):
    n_classes = len(classes)

    overall_report = {}

    auc_list = []
    acc_list = []

    if threshold_max_f1:
        suffix = "@max_f1"
    else:
        suffix = "@0.5"

    for i, cls in enumerate(classes):
        pred_binary = pred[:, i]
        y_binary = y[:, i]

        if threshold_max_f1:
            threshold = find_threshold(pred_binary, y_binary)
        else:
            threshold = 0.5

        report = binary_classification_report(pred_binary, y_binary, threshold=threshold, suffix=f"-{cls}{suffix}")

        overall_report.update(report)

        auc_list.append(report[f"auc-{cls}{suffix}"])
        acc_list.append(report[f"acc-{cls}{suffix}"])

    overall_report.update({f"auc_avg{suffix}": np.mean(auc_list), f"acc_avg{suffix}": np.mean(acc_list)})

    return overall_report


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def multitask_classification_report(pred, y, classes, suffix=""):
    num_classes = len(classes)
    overall_report = {}

    y_one_hot = one_hot_np(y, num_classes)

    pred_hard = np.argmax(pred, axis=-1)
    acc = accuracy_score(y, pred_hard)
    auc = roc_auc_score(y_one_hot, pred)

    overall_report.update({f"auc{suffix}": auc, f"acc{suffix}": acc})

    return overall_report
