#!/usr/bin/env python3
"""SimCLR transfer evaluation (Chen et al. 2020, Appendix B.8).

Two protocols, both on the ResNet representation h (the projector is discarded):

* linear: L2-regularized multinomial logistic regression fit by L-BFGS on a
  frozen encoder. No augmentation. Images are resized so the shorter side is
  224 (bicubic) and center-cropped to 224. The L2 penalty is added to the
  *sum* of per-example losses; the coefficient is chosen from 45 log-spaced
  values in [1e-6, 1e5], warm-started along that path.
* finetune / scratch: SGD with Nesterov momentum 0.9 for 20_000 steps
  (40_000 from random init) at batch 256. Random resized crop + horizontal
  flip only. Test resize is shorter-side 256 then a 224 center crop. Learning
  rate and weight decay are a log grid; the optimizer weight decay is the grid
  value divided by the base learning rate. Batch-norm momentum follows the
  paper's TensorFlow convention. The cosine schedule is the one Kornblith et
  al. 2019 use; B.8 says it follows that work except for preprocessing.

Hyperparameters are chosen on a validation set, then the model is retrained on
train+val and scored on test. Metrics: top-1; mean per-class (Aircraft, Pets,
Caltech-101, Flowers); 11-point VOC 2007 mAP.

  python transfer/simclr_eval.py --self-check
  python transfer/simclr_eval.py --mode linear --dataset cifar10 \\
      --data "$DATA/cifar10" --checkpoint ckpt.pth.tar --output cifar10.json

Hold-out sizes and the Caltech-101 30-per-class draw were not published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FINETUNE_STEPS = 20_000
SCRATCH_STEPS = 40_000
# ponytail: unpublished train-set hold-out. 20% class-balanced, seed below.
# Swap holdout_indices for their split if it is ever released.
HOLDOUT_FRACTION = 0.2
CALTECH_PER_CLASS = 30
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
DATASET_ORDER = (
    "food101", "cifar10", "cifar100", "birdsnap", "sun397", "cars",
    "aircraft", "voc2007", "dtd", "pets", "caltech101", "flowers",
)


class TransferError(RuntimeError):
    pass


def log_spaced(low: float, high: float, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("grid count must be positive")
    return np.logspace(math.log10(low), math.log10(high), count)


def l2_grid(count: int = 45) -> np.ndarray:
    """λ on sum(cross-entropy) + 0.5 λ ||W||^2, from 1e-6 to 1e5."""
    return log_spaced(1e-6, 1e5, count)


def finetune_grid(lr_count: int = 7, wd_count: int = 7) -> tuple[np.ndarray, list[float]]:
    lrs = log_spaced(1e-4, 1e-1, lr_count)
    wds = [float(v) for v in log_spaced(1e-6, 1e-3, wd_count)] + [0.0]
    return lrs, wds


def scratch_grid(lr_count: int = 7, wd_count: int = 8) -> tuple[np.ndarray, np.ndarray]:
    return log_spaced(1e-3, 1.0, lr_count), log_spaced(1e-5, 10 ** -1.5, wd_count)


def optimizer_weight_decay(grid_wd: float, base_lr: float) -> float:
    """B.8: divide the grid weight decay by the learning rate.

    PyTorch SGD applies ``p -= lr * (g + wd * p)``. Setting ``wd = grid/base_lr``
    makes the decay term ``-(scheduled_lr / base_lr) * grid * p``.
    """
    if grid_wd == 0.0:
        return 0.0
    if base_lr <= 0:
        raise ValueError("learning rate must be positive")
    return float(grid_wd) / float(base_lr)


def pytorch_bn_momentum(steps_per_epoch: int) -> float:
    """Map B.8's TensorFlow BN momentum onto PyTorch.

    TF: ``running = m * running + (1-m) * batch`` with ``m = max(1-10/s, 0.9)``.
    PyTorch uses the complementary coefficient.
    """
    s = max(int(steps_per_epoch), 1)
    momentum_tf = max(1.0 - 10.0 / s, 0.9)
    return 1.0 - momentum_tf


def set_bn_momentum(module: nn.Module, steps_per_epoch: int) -> float:
    momentum = pytorch_bn_momentum(steps_per_epoch)
    for layer in module.modules():
        if isinstance(layer, _BatchNorm):
            layer.momentum = momentum
    return momentum


def cosine_lr(step: int, steps: int, base_lr: float) -> float:
    if steps <= 1:
        return base_lr
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * step / (steps - 1)))


def steps_per_epoch(n_examples: int, batch_size: int) -> int:
    if n_examples <= 0 or batch_size <= 0:
        raise ValueError("n_examples and batch_size must be positive")
    return max(1, math.ceil(n_examples / batch_size))


# --- metrics -----------------------------------------------------------------


def top1_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"shape {pred.shape} != {target.shape}")
    if len(target) == 0:
        raise ValueError("empty predictions")
    return float((pred == target).mean())


def mean_per_class_accuracy(pred: np.ndarray, target: np.ndarray, n_classes: int) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    correct = np.zeros(n_classes, dtype=np.float64)
    count = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask = target == c
        count[c] = mask.sum()
        correct[c] = (pred[mask] == c).sum()
    present = count > 0
    if not present.any():
        raise ValueError("no class has a test example")
    return float((correct[present] / count[present]).mean())


def eleven_point_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """VOC 2007 11-point interpolated average precision."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    npos = int(labels.sum())
    if npos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    recall = tp / npos
    precision = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        hit = recall >= t - 1e-12
        ap += float(precision[hit].max()) if hit.any() else 0.0
    return ap / 11.0


def voc_map(scores: np.ndarray, target: np.ndarray) -> float:
    """11-point mAP. ``target`` uses 1 positive, 0 difficult (ignored), -1 negative."""
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target)
    if scores.shape != target.shape or scores.ndim != 2:
        raise ValueError("VOC scores and targets must be [N, C]")
    aps = []
    for c in range(scores.shape[1]):
        valid = target[:, c] != 0
        # Difficult (0) is dropped. Remaining labels are +1 or -1.
        aps.append(eleven_point_ap(target[valid, c] == 1, scores[valid, c]))
    return float(np.mean(aps))


def score_predictions(metric: str, pred: np.ndarray, target: np.ndarray, n_classes: int) -> float:
    if metric == "top1":
        return top1_accuracy(pred, target)
    if metric == "mean_per_class":
        return mean_per_class_accuracy(pred, target, n_classes)
    if metric == "voc_map":
        return voc_map(pred, target)
    raise ValueError(f"unknown metric {metric}")


def permutation_pvalue(
    metric: str,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    n_classes: int,
    n_perm: int = 100_000,
    seed: int = 0,
) -> float:
    """Two-sided permutation test from B.8 (exchange predictions per example).

    Top-1 uses the exact binomial null (McNemar). Mean per-class and VOC
    label-accuracy sample ``n_perm`` exchanges. VOC compares thresholded
    accuracy, not mAP, as specified.
    """
    if metric == "top1":
        correct_a = np.asarray(pred_a) == np.asarray(target)
        correct_b = np.asarray(pred_b) == np.asarray(target)
        return _binomial_pvalue(correct_a, correct_b, n_perm, seed)
    if metric == "mean_per_class":
        return _per_class_pvalue(pred_a, pred_b, target, n_classes, n_perm, seed)
    if metric == "voc_map":
        return _voc_accuracy_pvalue(pred_a, pred_b, target, n_perm, seed)
    raise ValueError(f"unknown metric {metric}")


def _binomial_pvalue(correct_a: np.ndarray, correct_b: np.ndarray, n_perm: int, seed: int) -> float:
    diff = float(correct_a.mean() - correct_b.mean())
    disagree = int((correct_a != correct_b).sum())
    n = len(correct_a)
    if disagree == 0 or n == 0:
        return 1.0
    draws = np.random.RandomState(seed).binomial(disagree, 0.5, size=n_perm)
    null = (2.0 * draws - disagree) / n
    return float(np.mean(np.abs(null) >= abs(diff) - 1e-15))


def _per_class_pvalue(pred_a, pred_b, target, n_classes, n_perm, seed) -> float:
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    target = np.asarray(target)
    obs = mean_per_class_accuracy(pred_a, target, n_classes) - mean_per_class_accuracy(
        pred_b, target, n_classes
    )
    ca = pred_a == target
    cb = pred_b == target
    disagree = np.flatnonzero(ca != cb)
    if len(disagree) == 0:
        return 1.0
    y_d = target[disagree]
    counts = np.bincount(target, minlength=n_classes).astype(np.float64)
    present = counts > 0
    n_present = int(present.sum())
    rng = np.random.RandomState(seed)
    extreme = 0
    done = 0
    chunk = 250
    while done < n_perm:
        m = min(chunk, n_perm - done)
        assign_a = rng.rand(m, len(disagree)) < 0.5
        diffs = np.zeros(m, dtype=np.float64)
        for c in np.flatnonzero(present):
            sel = y_d == c
            n_dc = int(sel.sum())
            if n_dc == 0:
                continue
            n_a = assign_a[:, sel].sum(axis=1)
            diffs += (2.0 * n_a - n_dc) / counts[c]
        diffs /= n_present
        extreme += int(np.count_nonzero(np.abs(diffs) >= abs(obs) - 1e-15))
        done += m
    return extreme / n_perm


def _voc_accuracy_pvalue(scores_a, scores_b, target, n_perm, seed) -> float:
    target = np.asarray(target)
    valid = target != 0
    truth = target == 1
    hard_a = np.asarray(scores_a) >= 0.5
    hard_b = np.asarray(scores_b) >= 0.5
    contrib = ((hard_a == truth) & valid).sum(axis=1) - ((hard_b == truth) & valid).sum(axis=1)
    ntot = int(valid.sum())
    if ntot == 0:
        raise ValueError("VOC targets have no non-difficult labels")
    obs = float(contrib.sum()) / ntot
    nz = contrib[contrib != 0].astype(np.float64)
    if len(nz) == 0:
        return 1.0
    rng = np.random.RandomState(seed)
    extreme = 0
    done = 0
    chunk = 250
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, len(nz)))
        null = signs @ nz / ntot
        extreme += int(np.count_nonzero(np.abs(null) >= abs(obs) - 1e-15))
        done += m
    return extreme / n_perm


# --- logistic regression -----------------------------------------------------


def _multinomial_loss_grad(theta: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float, n_classes: int):
    n, d = x.shape
    w = theta[: d * n_classes].reshape(d, n_classes)
    b = theta[d * n_classes :]
    logits = x @ w + b
    peak = np.max(logits, axis=1, keepdims=True)
    shifted = logits - peak
    log_z = peak.ravel() + np.log(np.exp(shifted).sum(axis=1))
    nll = -(logits[np.arange(n), y] - log_z).sum()
    loss = nll + 0.5 * l2 * np.dot(w.ravel(), w.ravel())
    probs = np.exp(shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True)))
    probs[np.arange(n), y] -= 1.0
    grad_w = x.T @ probs + l2 * w
    grad_b = probs.sum(axis=0)
    return float(loss), np.concatenate([grad_w.ravel(), grad_b])


def _binary_loss_grad(theta: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float):
    """Independent binary logistics. ``y`` is 1 / -1 / 0 (difficult, ignored)."""
    n, d = x.shape
    c = y.shape[1]
    w = theta[: d * c].reshape(d, c)
    b = theta[d * c :]
    z = x @ w + b
    valid = y != 0
    y01 = (y == 1).astype(np.float64)
    abs_z = np.abs(z)
    bce = np.maximum(z, 0.0) - z * y01 + np.log1p(np.exp(-abs_z))
    loss = float(bce[valid].sum() + 0.5 * l2 * np.dot(w.ravel(), w.ravel()))
    sig = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    dz = (sig - y01) * valid
    grad_w = x.T @ dz + l2 * w
    grad_b = dz.sum(axis=0)
    return loss, np.concatenate([grad_w.ravel(), grad_b])


def fit_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    l2: float,
    n_classes: int,
    multilabel: bool,
    theta0: np.ndarray | None = None,
    maxiter: int = 500,
) -> tuple[np.ndarray, bool]:
    x = np.ascontiguousarray(features, dtype=np.float64)
    y = np.asarray(targets)
    if multilabel:
        y = np.asarray(y, dtype=np.int64)
        n_param = x.shape[1] * y.shape[1] + y.shape[1]
        fun = lambda theta: _binary_loss_grad(theta, x, y, l2)
    else:
        y = np.asarray(y, dtype=np.int64)
        if y.min() < 0 or y.max() >= n_classes:
            raise TransferError(f"labels outside [0, {n_classes})")
        n_param = x.shape[1] * n_classes + n_classes
        fun = lambda theta: _multinomial_loss_grad(theta, x, y, l2, n_classes)
    if theta0 is None:
        theta0 = np.zeros(n_param, dtype=np.float64)
    else:
        theta0 = np.array(theta0, dtype=np.float64, copy=True)
    result = minimize(
        fun, theta0, method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "ftol": 1e-12}
    )
    return result.x, bool(result.success)


def predict_logistic(
    theta: np.ndarray, features: np.ndarray, n_classes: int, multilabel: bool
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    d = x.shape[1]
    c = n_classes
    w = theta[: d * c].reshape(d, c)
    b = theta[d * c :]
    logits = x @ w + b
    if multilabel:
        z = logits
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return logits.argmax(axis=1)


def select_l2_metric(
    train_x, train_y, val_x, val_y, n_classes, multilabel, metric, grid, maxiter
) -> dict[str, Any]:
    theta = None
    best: dict[str, Any] | None = None
    trials = []
    for raw in grid:
        l2 = float(raw)
        theta, ok = fit_logistic(
            train_x, train_y, l2, n_classes, multilabel, theta0=theta, maxiter=maxiter
        )
        pred = predict_logistic(theta, val_x, n_classes, multilabel)
        val = score_predictions(metric, pred, val_y, n_classes)
        trials.append({"l2": l2, "val_percent": 100.0 * val, "converged": ok})
        print(f"  l2={l2:.4g} val={100 * val:.2f} converged={ok}", flush=True)
        if best is None or val > best["val"]:
            best = {"l2": l2, "val": val, "theta": theta.copy(), "converged": ok}
    assert best is not None
    best["trials"] = trials
    return best


# --- index splits ------------------------------------------------------------


def holdout_indices(targets: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced hold-out. At least one val image when a class has two or more."""
    targets = np.asarray(targets)
    if targets.ndim != 1:
        raise ValueError("holdout is for single-label targets")
    rng = np.random.RandomState(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for c in np.unique(targets):
        idx = np.flatnonzero(targets == c)
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(round(fraction * n))
        if n > 1:
            n_val = min(max(n_val, 1), n - 1)
        else:
            n_val = 0
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def per_class_indices(targets: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets)
    rng = np.random.RandomState(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for c in np.unique(targets):
        idx = np.flatnonzero(targets == c)
        if len(idx) <= k:
            raise TransferError(
                f"class {int(c)} has {len(idx)} images; Caltech-101 needs more than {k} per class"
            )
        rng.shuffle(idx)
        train_parts.append(idx[:k])
        test_parts.append(idx[k:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(test_parts))


# --- datasets ----------------------------------------------------------------


def dataset_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent = dataset_targets(dataset.dataset)
        return parent[np.asarray(dataset.indices)]
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([dataset_targets(part) for part in dataset.datasets])
    for attr in ("targets", "_labels"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return np.asarray(value)
    samples = getattr(dataset, "_samples", None)
    if samples is not None:
        return np.asarray([target for _, target in samples])
    raise TransferError(f"{type(dataset).__name__} has no targets")


def class_count(dataset: Dataset, fallback: int) -> int:
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        return len(classes)
    labels = dataset_targets(dataset)
    if labels.ndim == 2:
        return int(labels.shape[1])
    return fallback


class ListDataset(Dataset):
    def __init__(self, paths: Sequence[Path], targets: np.ndarray, classes: Sequence[str], transform):
        self.paths = list(paths)
        self.targets = np.asarray(targets)
        self.classes = list(classes)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]
        if np.ndim(target) == 0:
            return img, int(target)
        return img, torch.tensor(np.asarray(target), dtype=torch.int64)


def _rgb_resize(shorter: int, train: bool) -> transforms.Compose:
    ops: list[Any] = []
    if train:
        ops.append(transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC))
        ops.append(transforms.RandomHorizontalFlip())
    else:
        ops.append(transforms.Resize(shorter, interpolation=InterpolationMode.BICUBIC))
        ops.append(transforms.CenterCrop(224))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def linear_transform() -> transforms.Compose:
    return _rgb_resize(224, train=False)


def finetune_train_transform() -> transforms.Compose:
    return _rgb_resize(224, train=True)


def finetune_test_transform() -> transforms.Compose:
    return _rgb_resize(256, train=False)


def _sun_root(root: Path) -> Path:
    for candidate in (root / "SUN397", root, root / "SUN397" / "SUN397"):
        if (candidate / "ClassName.txt").is_file() and (candidate / "Partitions").is_dir():
            return candidate
    raise TransferError(
        f"SUN397 not found under {root}. Expected ClassName.txt and "
        "Partitions/Training_01.txt (first split only)."
    )


def _load_sun(root: Path, split: str, transform) -> ListDataset:
    base = _sun_root(root)
    names = [ln.strip() for ln in (base / "ClassName.txt").read_text().splitlines() if ln.strip()]
    class_to_idx = {name: i for i, name in enumerate(names)}
    part = "Training_01.txt" if split == "train" else "Testing_01.txt"
    if split not in {"train", "test"}:
        raise TransferError(f"SUN397 split must be train or test, got {split}")
    paths: list[Path] = []
    targets: list[int] = []
    for line in (base / "Partitions" / part).read_text().splitlines():
        rel = line.strip().replace("\\", "/")
        if not rel:
            continue
        rel = rel[1:] if rel.startswith("/") else rel
        key = "/" + Path(rel).parent.as_posix()
        if key not in class_to_idx and key[1:] in class_to_idx:
            key = key[1:]
        if key not in class_to_idx:
            raise TransferError(f"SUN397 class {key} from {part} is not in ClassName.txt")
        image = base / rel
        if not image.is_file():
            raise TransferError(f"SUN397 image missing: {image}")
        paths.append(image)
        targets.append(class_to_idx[key])
    return ListDataset(paths, np.asarray(targets, dtype=np.int64), names, transform)


def _load_caltech(root: Path, transform) -> ListDataset:
    candidates = [root / "caltech101" / "101_ObjectCategories", root / "101_ObjectCategories"]
    folder = next((path for path in candidates if path.is_dir()), None)
    if folder is None:
        raise TransferError(
            f"Caltech-101 not found under {root}. Expected caltech101/101_ObjectCategories "
            "(BACKGROUND_Google is kept: Kornblith Table 1 is 102-way)."
        )
    classes = sorted(path.name for path in folder.iterdir() if path.is_dir())
    if "BACKGROUND_Google" not in classes:
        print("warning: BACKGROUND_Google is missing; the paper's Caltech-101 is 102-way", flush=True)
    paths: list[Path] = []
    targets: list[int] = []
    for class_index, name in enumerate(classes):
        images = sorted((folder / name).glob("image_*.jpg"))
        if not images:
            raise TransferError(f"Caltech-101 class {name} has no image_*.jpg files")
        paths.extend(images)
        targets.extend([class_index] * len(images))
    return ListDataset(paths, np.asarray(targets, dtype=np.int64), classes, transform)


def _voc_root(root: Path) -> Path:
    for candidate in (root / "VOCdevkit" / "VOC2007", root / "VOC2007", root):
        if (candidate / "ImageSets" / "Main" / "train.txt").is_file():
            return candidate
    raise TransferError(
        f"VOC2007 classification files not found under {root}. "
        "Expected VOCdevkit/VOC2007/ImageSets/Main/{{train,val,trainval,test}}.txt "
        "and per-class label files."
    )


def _read_voc_targets(main: Path, split: str, ids: list[str]) -> np.ndarray:
    index = {image_id: i for i, image_id in enumerate(ids)}
    targets = np.full((len(ids), len(VOC_CLASSES)), -1, dtype=np.int64)
    for class_index, name in enumerate(VOC_CLASSES):
        path = main / f"{name}_{split}.txt"
        if not path.is_file():
            raise TransferError(f"missing VOC label file {path}")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            image_id, label = line.split()
            if image_id not in index:
                continue
            targets[index[image_id], class_index] = int(label)
    return targets


def _load_voc(root: Path, split: str, transform) -> ListDataset:
    if split not in {"train", "val", "trainval", "test"}:
        raise TransferError(f"bad VOC split {split}")
    base = _voc_root(root)
    ids = [
        line.strip()
        for line in (base / "ImageSets" / "Main" / f"{split}.txt").read_text().splitlines()
        if line.strip()
    ]
    targets = _read_voc_targets(base / "ImageSets" / "Main", split, ids)
    paths = []
    for image_id in ids:
        image = base / "JPEGImages" / f"{image_id}.jpg"
        if not image.is_file():
            raise TransferError(f"VOC image missing: {image}")
        paths.append(image)
    dataset = ListDataset(paths, targets, list(VOC_CLASSES), transform)
    return dataset


def _align_imagefolder(test: datasets.ImageFolder, train: datasets.ImageFolder) -> datasets.ImageFolder:
    if test.class_to_idx == train.class_to_idx:
        return test
    missing = set(train.class_to_idx) - set(test.class_to_idx)
    extra = set(test.class_to_idx) - set(train.class_to_idx)
    if missing or extra:
        raise TransferError(
            f"birdsnap train/test classes differ (missing {sorted(missing)[:3]}, extra {sorted(extra)[:3]})"
        )
    remap = {test.class_to_idx[name]: train.class_to_idx[name] for name in test.classes}
    test.samples = [(path, remap[label]) for path, label in test.samples]
    test.targets = [label for _, label in test.samples]
    test.imgs = test.samples
    test.classes = list(train.classes)
    test.class_to_idx = dict(train.class_to_idx)
    return test


def _load_birdsnap(root: Path, split: str, transform) -> datasets.ImageFolder:
    path = root / split
    if not path.is_dir():
        raise TransferError(
            f"Birdsnap expects {path}/<class>/* . The original host is offline; stage ImageFolder splits."
        )
    dataset = datasets.ImageFolder(str(path), transform=transform)
    if split == "test":
        train_path = root / "train"
        if train_path.is_dir():
            dataset = _align_imagefolder(dataset, datasets.ImageFolder(str(train_path)))
    return dataset


def _tv(name: str, root: Path, split: str, transform, download: bool) -> Dataset:
    if name == "cifar10":
        if split not in {"train", "test"}:
            raise TransferError("cifar10 has train and test only")
        return datasets.CIFAR10(str(root), train=split == "train", download=download, transform=transform)
    if name == "cifar100":
        if split not in {"train", "test"}:
            raise TransferError("cifar100 has train and test only")
        return datasets.CIFAR100(str(root), train=split == "train", download=download, transform=transform)
    if name == "food101":
        return datasets.Food101(str(root), split=split, download=download, transform=transform)
    if name == "cars":
        return datasets.StanfordCars(str(root), split=split, download=download, transform=transform)
    if name == "aircraft":
        return datasets.FGVCAircraft(
            str(root), split=split, annotation_level="variant", download=download, transform=transform
        )
    if name == "dtd":
        return datasets.DTD(str(root), split=split, partition=1, download=download, transform=transform)
    if name == "pets":
        pet_split = "trainval" if split == "train" else split
        if pet_split not in {"trainval", "test"}:
            raise TransferError("pets official files are trainval and test; val is held out of trainval")
        return datasets.OxfordIIITPet(str(root), split=pet_split, download=download, transform=transform)
    if name == "flowers":
        return datasets.Flowers102(str(root), split=split, download=download, transform=transform)
    raise TransferError(f"no torchvision loader for {name}")


def load_split(name: str, root: Path, split: str, transform, download: bool = False) -> Dataset:
    try:
        if name == "sun397":
            return _load_sun(root, split, transform)
        if name == "caltech101":
            if split != "all":
                raise TransferError("caltech101 is loaded as one pool; splits are drawn in-process")
            return _load_caltech(root, transform)
        if name == "voc2007":
            return _load_voc(root, split, transform)
        if name == "birdsnap":
            return _load_birdsnap(root, split, transform)
        if name == "flowers" and split == "trainval":
            return ConcatDataset(
                [
                    _tv(name, root, "train", transform, download),
                    _tv(name, root, "val", transform, download),
                ]
            )
        if name == "dtd" and split == "trainval":
            return ConcatDataset(
                [
                    _tv(name, root, "train", transform, download),
                    _tv(name, root, "val", transform, download),
                ]
            )
        return _tv(name, root, split, transform, download)
    except TransferError:
        raise
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise TransferError(f"{name} split={split} under {root}: {exc}") from exc


SPEC = {
    "food101": ("top1", 101, "holdout"),
    "cifar10": ("top1", 10, "holdout"),
    "cifar100": ("top1", 100, "holdout"),
    "birdsnap": ("top1", 500, "holdout"),
    "sun397": ("top1", 397, "holdout"),
    "cars": ("top1", 196, "holdout"),
    "aircraft": ("mean_per_class", 100, "official"),
    "voc2007": ("voc_map", 20, "official"),
    "dtd": ("top1", 47, "official"),
    "pets": ("mean_per_class", 37, "holdout"),
    "caltech101": ("mean_per_class", 102, "caltech"),
    "flowers": ("mean_per_class", 102, "official"),
}


def spec_of(name: str) -> tuple[str, int, str]:
    if name not in SPEC:
        raise TransferError(f"unknown dataset {name}; choose from {DATASET_ORDER}")
    return SPEC[name]


# --- model / features --------------------------------------------------------


def _load_backbone_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    from extract_backbone import extract_backbone, get_state_dict, load_checkpoint

    state = extract_backbone(get_state_dict(load_checkpoint(checkpoint)))
    if "conv1.weight" not in state:
        raise TransferError(f"{checkpoint} has no ResNet conv1.weight after dropping the projector")
    return state


def _check_load(missing: Iterable[str], unexpected: Iterable[str]) -> None:
    bad = [key for key in missing if not (key.startswith("fc.") or key.endswith("num_batches_tracked"))]
    if bad or list(unexpected):
        raise TransferError(f"checkpoint mismatch missing={bad[:8]} unexpected={list(unexpected)[:8]}")


def load_encoder(checkpoint: Path) -> nn.Module:
    from cl.model import build_backbone

    encoder = build_backbone("resnet50")
    missing, unexpected = encoder.load_state_dict(_load_backbone_state(checkpoint), strict=False)
    _check_load(missing, unexpected)
    return encoder


def load_classifier(checkpoint: Path | None, n_classes: int) -> nn.Module:
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    if checkpoint is not None:
        missing, unexpected = model.load_state_dict(_load_backbone_state(checkpoint), strict=False)
        _check_load(missing, unexpected)
    return model


def _worker_init(_worker_id: int) -> None:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def make_loader(
    dataset: Dataset, batch_size: int, workers: int, shuffle: bool, seed: int, pin_memory: bool
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "generator": generator,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = _worker_init
    return DataLoader(dataset, **kwargs)


def _pad(batch: torch.Tensor, n: int) -> torch.Tensor:
    extra = n - batch.shape[0]
    if extra <= 0:
        return batch
    return torch.cat([batch, batch.new_zeros((extra, *batch.shape[1:]))], 0)


@torch.inference_mode()
def extract_features(model: nn.Module, dataset: Dataset, batch_size: int, workers: int, device: torch.device, seed: int):
    loader = make_loader(dataset, batch_size, workers, False, seed, device.type == "cuda")
    model.to(device).eval()
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    print(f"extracting n={len(dataset)} batch={batch_size} device={device}", flush=True)
    for images, labels in loader:
        n = images.shape[0]
        output = model(_pad(images, batch_size).to(device, non_blocking=True))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = output.flatten(1)
        features.append(output[:n].detach().float().cpu())
        targets.append(labels if torch.is_tensor(labels) else torch.as_tensor(labels))
    if not features:
        raise TransferError("dataset is empty")
    return torch.cat(features).numpy(), torch.cat(targets).cpu().numpy()


def cached_features(cache: Path | None, model, dataset, batch_size, workers, device, seed) -> tuple[np.ndarray, np.ndarray]:
    if cache is not None and cache.is_file():
        stored = np.load(cache)
        print(f"cache hit {cache}", flush=True)
        return stored["features"], stored["targets"]
    features, targets = extract_features(model, dataset, batch_size, workers, device, seed)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, features=features, targets=targets)
    return features, targets


def _index_key(dataset: Dataset) -> str:
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return f"all-{len(dataset)}"
    payload = np.asarray(indices, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def cache_file(cache_dir: Path | None, checkpoint: Path, dataset_name: str, split: str, dataset: Dataset) -> Path | None:
    if cache_dir is None:
        return None
    stat = checkpoint.stat()
    key = hashlib.sha256(
        f"{checkpoint.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{dataset_name}|{split}|"
        f"{_index_key(dataset)}|linear224".encode()
    ).hexdigest()[:20]
    return cache_dir / f"{dataset_name}-{split}-{key}.npz"


# --- fine-tune ---------------------------------------------------------------


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sgd(model: nn.Module, base_lr: float, grid_wd: float) -> torch.optim.SGD:
    return torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=optimizer_weight_decay(grid_wd, base_lr),
    )


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def finetune(
    model: nn.Module,
    dataset: Dataset,
    steps: int,
    base_lr: float,
    grid_wd: float,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
    multilabel: bool,
) -> None:
    epoch_steps = steps_per_epoch(len(dataset), batch_size)
    momentum = set_bn_momentum(model, epoch_steps)
    print(
        f"finetune steps={steps} lr={base_lr:g} wd={grid_wd:g} "
        f"opt_wd={optimizer_weight_decay(grid_wd, base_lr):g} bn_momentum={momentum:.4f} "
        f"steps/epoch={epoch_steps}",
        flush=True,
    )
    loader = make_loader(
        dataset, min(batch_size, len(dataset)), workers, True, seed, device.type == "cuda"
    )
    optimizer = make_sgd(model, base_lr, grid_wd)
    model.to(device).train()
    stream = _cycle(loader)
    log_every = max(steps // 10, 1)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, steps, base_lr)
        images, target = next(stream)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if multilabel:
            valid = target != 0
            if not bool(valid.any()):
                continue
            loss = F.binary_cross_entropy_with_logits(logits[valid], (target[valid] == 1).float())
        else:
            loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise TransferError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step + 1 == steps:
            print(f"  step {step}/{steps} loss={loss.item():.4f} lr={cosine_lr(step, steps, base_lr):.4g}", flush=True)


@torch.inference_mode()
def predict_network(model, dataset, batch_size, workers, device, seed, multilabel: bool):
    loader = make_loader(dataset, batch_size, workers, False, seed, device.type == "cuda")
    model.to(device).eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for images, target in loader:
        logits = model(images.to(device, non_blocking=True))
        if multilabel:
            preds.append(torch.sigmoid(logits).float().cpu())
        else:
            preds.append(logits.argmax(dim=1).cpu())
        targets.append(target if torch.is_tensor(target) else torch.as_tensor(target))
    return torch.cat(preds).numpy(), torch.cat(targets).cpu().numpy()


# --- runs --------------------------------------------------------------------


def _warn_classes(found: int, expected: int, name: str) -> None:
    if found != expected:
        print(f"warning: {name} has {found} classes; the paper table lists {expected}", flush=True)


def linear_splits(name: str, root: Path, download: bool, seed: int, fraction: float, transform):
    metric, expected, kind = spec_of(name)
    if kind == "caltech":
        # ponytail: the 30-per-class draw is unpublished. seed keeps background (102-way).
        pool = load_split(name, root, "all", transform, download)
        targets = dataset_targets(pool)
        train_pool, test_idx = per_class_indices(targets, CALTECH_PER_CLASS, seed)
        local_train, local_val = holdout_indices(targets[train_pool], fraction, seed)
        print(
            f"caltech101: {CALTECH_PER_CLASS}/class then {fraction:.0%} hold-out, seed={seed}",
            flush=True,
        )
        return {
            "tune_train": Subset(pool, train_pool[local_train]),
            "tune_val": Subset(pool, train_pool[local_val]),
            "final_train": Subset(pool, train_pool),
            "test": Subset(pool, test_idx),
            "metric": metric,
            "n_classes": class_count(pool, expected),
            "multilabel": False,
        }
    if kind == "official":
        tune_train = load_split(name, root, "train", transform, download)
        n_classes = class_count(tune_train, expected)
        _warn_classes(n_classes, expected, name)
        return {
            "tune_train": tune_train,
            "tune_val": load_split(name, root, "val", transform, download),
            "final_train": load_split(name, root, "trainval", transform, download),
            "test": load_split(name, root, "test", transform, download),
            "metric": metric,
            "n_classes": n_classes,
            "multilabel": metric == "voc_map",
        }
    print(
        f"hold-out: class-balanced {fraction:.0%} of the training pool, seed={seed} "
        "(split size was not published)",
        flush=True,
    )
    pool = load_split(name, root, "train", transform, download)
    train_idx, val_idx = holdout_indices(dataset_targets(pool), fraction, seed)
    n_classes = class_count(pool, expected)
    _warn_classes(n_classes, expected, name)
    return {
        "tune_train": Subset(pool, train_idx),
        "tune_val": Subset(pool, val_idx),
        "final_train": pool,
        "test": load_split(name, root, "test", transform, download),
        "metric": metric,
        "n_classes": n_classes,
        "multilabel": False,
    }


def finetune_splits(name, root, download, seed, fraction, train_tf, test_tf):
    """Same indices as linear_splits, but train and eval transforms differ."""
    metric, expected, kind = spec_of(name)
    if kind == "caltech":
        pool_train = load_split(name, root, "all", train_tf, download)
        pool_eval = load_split(name, root, "all", test_tf, download)
        targets = dataset_targets(pool_train)
        train_pool, test_idx = per_class_indices(targets, CALTECH_PER_CLASS, seed)
        local_train, local_val = holdout_indices(targets[train_pool], fraction, seed)
        return {
            "tune_train": Subset(pool_train, train_pool[local_train]),
            "tune_val": Subset(pool_eval, train_pool[local_val]),
            "final_train": Subset(pool_train, train_pool),
            "test": Subset(pool_eval, test_idx),
            "metric": metric,
            "n_classes": class_count(pool_train, expected),
            "multilabel": False,
        }
    if kind == "official":
        tune_train = load_split(name, root, "train", train_tf, download)
        return {
            "tune_train": tune_train,
            "tune_val": load_split(name, root, "val", test_tf, download),
            "final_train": load_split(name, root, "trainval", train_tf, download),
            "test": load_split(name, root, "test", test_tf, download),
            "metric": metric,
            "n_classes": class_count(tune_train, expected),
            "multilabel": metric == "voc_map",
        }
    pool_train = load_split(name, root, "train", train_tf, download)
    pool_eval = load_split(name, root, "train", test_tf, download)
    train_idx, val_idx = holdout_indices(dataset_targets(pool_train), fraction, seed)
    n_classes = class_count(pool_train, expected)
    _warn_classes(n_classes, expected, name)
    return {
        "tune_train": Subset(pool_train, train_idx),
        "tune_val": Subset(pool_eval, val_idx),
        "final_train": pool_train,
        "test": load_split(name, root, "test", test_tf, download),
        "metric": metric,
        "n_classes": n_classes,
        "multilabel": False,
    }


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; using cpu", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def run_linear(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint is None:
        raise TransferError("linear evaluation needs --checkpoint")
    device = _device(args.device)
    torch.backends.cudnn.benchmark = False
    parts = linear_splits(args.dataset, args.data, args.download, args.seed, args.holdout_fraction, linear_transform())
    encoder = load_encoder(args.checkpoint)
    cache_dir = args.cache_dir
    if cache_dir is None and args.output is not None:
        cache_dir = args.output.parent / "features"

    def take(split_name, dataset):
        return cached_features(
            cache_file(cache_dir, args.checkpoint, args.dataset, split_name, dataset),
            encoder, dataset, args.batch_size, args.workers, device, args.seed,
        )

    # Official final_train is a separate loading of trainval; extract train and
    # val instead and concatenate so the images are not decoded twice.
    train_x, train_y = take("tune_train", parts["tune_train"])
    val_x, val_y = take("tune_val", parts["tune_val"])
    test_x, test_y = take("test", parts["test"])
    if len(train_y) + len(val_y) != len(parts["final_train"]):
        raise TransferError(
            f"train+val ({len(train_y)}+{len(val_y)}) != final train ({len(parts['final_train'])})"
        )
    n_classes = parts["n_classes"]
    print(f"selecting λ on n_train={len(train_y)} n_val={len(val_y)}", flush=True)
    chosen = select_l2_metric(
        train_x, train_y, val_x, val_y, n_classes, parts["multilabel"], parts["metric"],
        l2_grid(args.l2_count), args.lbfgs_maxiter,
    )
    if args.phase == "tune":
        pred = target = None
        test_score = None
    else:
        print(f"refit λ={chosen['l2']:g} on train+val", flush=True)
        full_x = np.concatenate([train_x, val_x])
        full_y = np.concatenate([train_y, val_y])
        theta, ok = fit_logistic(
            full_x, full_y, chosen["l2"], n_classes, parts["multilabel"],
            theta0=chosen["theta"], maxiter=args.lbfgs_maxiter,
        )
        chosen["final_converged"] = ok
        pred = predict_logistic(theta, test_x, n_classes, parts["multilabel"])
        target = test_y
        test_score = score_predictions(parts["metric"], pred, target, n_classes)
    return _record(args, parts, "linear", chosen["val"], test_score, {"l2": chosen["l2"]}, chosen["trials"], pred, target)


def _trial_path(output: Path) -> Path:
    return Path(str(output) + ".trials.jsonl")


def _read_trials(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _already(rows: list[dict[str, Any]], lr: float, wd: float) -> dict[str, Any] | None:
    for row in rows:
        if math.isclose(row["lr"], lr, rel_tol=0, abs_tol=1e-12) and math.isclose(
            row["wd"], wd, rel_tol=0, abs_tol=1e-15
        ):
            return row
    return None


def _best_trial(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (row["val_percent"], -row["wd"], -row["lr"]))


def run_finetune(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "finetune" and args.checkpoint is None:
        raise TransferError("finetune needs --checkpoint (scratch does not)")
    device = _device(args.device)
    torch.backends.cudnn.benchmark = False
    parts = finetune_splits(
        args.dataset, args.data, args.download, args.seed, args.holdout_fraction,
        finetune_train_transform(), finetune_test_transform(),
    )
    steps = args.steps if args.steps is not None else (SCRATCH_STEPS if args.mode == "scratch" else FINETUNE_STEPS)
    checkpoint = None if args.mode == "scratch" else args.checkpoint
    if args.lr is not None or args.weight_decay is not None:
        if args.lr is None or args.weight_decay is None:
            raise TransferError("pass both --lr and --weight-decay for a single run")
        grid = [(args.lr, args.weight_decay)]
    elif args.mode == "scratch":
        lrs, wds = scratch_grid(args.lr_count, args.wd_count)
        grid = [(float(lr), float(wd)) for lr in lrs for wd in wds]
    else:
        lrs, wds = finetune_grid(args.lr_count, args.wd_count)
        grid = [(float(lr), float(wd)) for lr in lrs for wd in wds]
    if args.phase == "all" and len(grid) > 1:
        print(
            f"{len(grid)} trials × {steps} steps at batch {args.batch_size}. "
            "This is the paper grid, not a smoke test.",
            flush=True,
        )
    trials_path = _trial_path(args.output) if args.output is not None else None
    rows = _read_trials(trials_path) if trials_path is not None else []
    if args.phase != "final":
        for lr, wd in grid:
            if _already(rows, lr, wd) is not None:
                print(f"skip lr={lr:g} wd={wd:g}", flush=True)
                continue
            seed_all(args.seed)
            model = load_classifier(checkpoint, parts["n_classes"])
            finetune(
                model, parts["tune_train"], steps, lr, wd, args.batch_size, args.workers,
                device, args.seed, parts["multilabel"],
            )
            pred, target = predict_network(
                model, parts["tune_val"], args.batch_size, args.workers, device, args.seed, parts["multilabel"]
            )
            val = score_predictions(parts["metric"], pred, target, parts["n_classes"])
            row = {"lr": lr, "wd": wd, "val_percent": 100.0 * val}
            rows.append(row)
            print(f"trial lr={lr:g} wd={wd:g} val={100 * val:.2f}", flush=True)
            if trials_path is not None:
                trials_path.parent.mkdir(parents=True, exist_ok=True)
                with trials_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
    if args.phase == "tune":
        best = _best_trial(rows)
        return _record(
            args, parts, args.mode, best["val_percent"] / 100.0, None,
            {"lr": best["lr"], "weight_decay": best["wd"], "steps": steps}, rows, None, None,
        )
    if args.lr is not None and args.phase == "final":
        best_lr, best_wd = args.lr, args.weight_decay
    else:
        if not rows:
            raise TransferError("no trials to select from; run --phase tune first or pass --lr and --weight-decay")
        best = _best_trial(rows)
        best_lr, best_wd = best["lr"], best["wd"]
    print(f"retrain lr={best_lr:g} wd={best_wd:g} on train+val", flush=True)
    seed_all(args.seed)
    model = load_classifier(checkpoint, parts["n_classes"])
    finetune(
        model, parts["final_train"], steps, best_lr, best_wd, args.batch_size, args.workers,
        device, args.seed, parts["multilabel"],
    )
    pred, target = predict_network(
        model, parts["test"], args.batch_size, args.workers, device, args.seed, parts["multilabel"]
    )
    test_score = score_predictions(parts["metric"], pred, target, parts["n_classes"])
    val_score = None
    if rows:
        match = _already(rows, best_lr, best_wd)
        if match is not None:
            val_score = match["val_percent"] / 100.0
    return _record(
        args, parts, args.mode, val_score, test_score,
        {"lr": best_lr, "weight_decay": best_wd, "steps": steps}, rows, pred, target,
    )


def _record(args, parts, mode, val_score, test_score, hparams, trials, pred, target) -> dict[str, Any]:
    metric = parts["metric"]
    record: dict[str, Any] = {
        "protocol": "simclr-b.8",
        "reference": "https://arxiv.org/abs/2002.05709",
        "dataset": args.dataset,
        "mode": mode,
        "metric": metric,
        "val_percent": None if val_score is None else float(100.0 * val_score),
        "test_percent": None if test_score is None else float(100.0 * test_score),
        "hparams": hparams,
        "trials": [{k: v for k, v in row.items() if k != "theta"} for row in trials],
        "n_tune_train": len(parts["tune_train"]),
        "n_val": len(parts["tune_val"]),
        "n_final_train": len(parts["final_train"]),
        "n_test": len(parts["test"]),
        "n_classes": parts["n_classes"],
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "steps_override": args.steps,
    }
    if pred is not None and args.output is not None:
        pred_path = Path(str(args.output) + ".npz")
        np.savez(pred_path, pred=np.asarray(pred), target=np.asarray(target))
        record["predictions"] = str(pred_path)
        if args.compare is not None:
            other = np.load(args.compare)
            record["permutation_pvalue"] = permutation_pvalue(
                metric, pred, other["pred"], target, parts["n_classes"], n_perm=args.permutations, seed=args.seed
            )
            record["compared_to"] = str(args.compare)
    if test_score is not None:
        val_text = "n/a" if val_score is None else f"{100 * val_score:.2f}"
        print(
            f"{args.dataset} {mode} {metric} test={100 * test_score:.2f}% val={val_text}% hparams={hparams}",
            flush=True,
        )
    return record


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


# --- self-check --------------------------------------------------------------


def _finite_difference(fun, theta: np.ndarray, eps: float = 1e-6) -> None:
    loss, grad = fun(theta)
    numeric = np.zeros_like(theta)
    for i in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric[i] = (fun(plus)[0] - fun(minus)[0]) / (2 * eps)
    if not np.allclose(grad, numeric, atol=1e-5):
        raise AssertionError((grad - numeric).max())
    assert math.isfinite(loss)


def self_check() -> None:
    grid = l2_grid()
    assert len(grid) == 45 and math.isclose(grid[0], 1e-6) and math.isclose(grid[-1], 1e5)
    lrs, wds = finetune_grid()
    assert len(lrs) == 7 and math.isclose(lrs[0], 1e-4) and math.isclose(lrs[-1], 0.1)
    assert len(wds) == 8 and wds[-1] == 0.0 and math.isclose(wds[0], 1e-6) and math.isclose(wds[-2], 1e-3)
    s_lrs, s_wds = scratch_grid()
    assert len(s_lrs) == 7 and len(s_wds) == 8
    assert math.isclose(s_lrs[0], 1e-3) and math.isclose(s_lrs[-1], 1.0)
    assert math.isclose(s_wds[0], 1e-5) and math.isclose(s_wds[-1], 10 ** -1.5)
    assert optimizer_weight_decay(1e-4, 0.01) == 0.01
    assert optimizer_weight_decay(0.0, 0.01) == 0.0
    assert math.isclose(pytorch_bn_momentum(200), 10 / 200)
    assert math.isclose(pytorch_bn_momentum(8), 0.1)
    assert math.isclose(cosine_lr(0, 100, 0.2), 0.2)
    assert math.isclose(cosine_lr(99, 100, 0.2), 0.0, abs_tol=1e-12)
    assert steps_per_epoch(2560, 256) == 10
    assert steps_per_epoch(2561, 256) == 11

    labels = np.array([1, 0, 1, 0, 1], dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert math.isclose(eleven_point_ap(labels, scores), 8.4 / 11)
    # A high-scoring difficult example must not count as a false positive.
    voc_scores = np.array([[0.9, 0.1], [0.99, 0.1], [0.1, 0.2]])
    voc_target = np.array([[1, -1], [0, -1], [-1, 1]])
    assert math.isclose(voc_map(voc_scores, voc_target), 1.0)

    pred = np.array([0, 0, 1, 1])
    target = np.array([0, 1, 1, 1])
    assert math.isclose(top1_accuracy(pred, target), 0.75)
    assert math.isclose(mean_per_class_accuracy(pred, target, 2), (1.0 + 2.0 / 3.0) / 2.0)

    train_idx, val_idx = holdout_indices(np.repeat(np.arange(3), 5), 0.2, 0)
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(train_idx) + len(val_idx) == 15
    # 0.2*5 = 1 val image per class.
    assert len(val_idx) == 3
    tiny_train, tiny_val = holdout_indices(np.array([0, 0]), 0.2, 0)
    assert len(tiny_val) == 1 and len(tiny_train) == 1
    try:
        per_class_indices(np.arange(30), 30, 0)
    except TransferError:
        pass
    else:
        raise AssertionError("30 images cannot yield a 30-per-class test remainder")
    pool, test = per_class_indices(np.repeat(np.arange(4), 40), 30, 0)
    assert len(pool) == 120 and len(test) == 40

    rng = np.random.RandomState(0)
    theta = rng.randn(3 * 3 + 3) * 0.1
    x = rng.randn(6, 3)
    y = rng.randint(0, 3, size=6)
    _finite_difference(lambda th: _multinomial_loss_grad(th, x, y, 0.3, 3), theta)
    y_bin = rng.randint(-1, 2, size=(6, 2))
    _finite_difference(lambda th: _binary_loss_grad(th, x, y_bin, 0.2), rng.randn(3 * 2 + 2) * 0.1)

    blobs = np.vstack([rng.randn(30, 2) - 3, rng.randn(30, 2) + 3])
    blob_y = np.array([0] * 30 + [1] * 30)
    fitted, ok = fit_logistic(blobs, blob_y, 1e-4, 2, multilabel=False, maxiter=200)
    assert ok
    assert top1_accuracy(predict_logistic(fitted, blobs, 2, False), blob_y) == 1.0

    correct_a = np.ones(5, dtype=bool)
    correct_b = np.zeros(5, dtype=bool)
    # Every example disagrees and the accuracy gap is 1. Exact two-sided tail is 2/32.
    p = _binomial_pvalue(correct_a, correct_b, n_perm=100_000, seed=0)
    assert abs(p - (2 / 32)) < 0.005, p
    assert _binomial_pvalue(correct_a, correct_a, 1000, 0) == 1.0

    y_pc = np.array([0, 0, 1, 1, 2, 2])
    pa = np.array([0, 0, 1, 0, 2, 1])
    pb = np.array([0, 1, 1, 1, 0, 2])
    obs = mean_per_class_accuracy(pa, y_pc, 3) - mean_per_class_accuracy(pb, y_pc, 3)
    extreme = 0
    total = 1 << len(y_pc)
    for mask in range(total):
        a = pa.copy()
        b = pb.copy()
        for i in range(len(y_pc)):
            if mask >> i & 1:
                a[i], b[i] = b[i], a[i]
        diff = mean_per_class_accuracy(a, y_pc, 3) - mean_per_class_accuracy(b, y_pc, 3)
        if abs(diff) >= abs(obs) - 1e-15:
            extreme += 1
    sampled = _per_class_pvalue(pa, pb, y_pc, 3, n_perm=20_000, seed=0)
    assert abs(sampled - extreme / total) < 0.02, (sampled, extreme / total)

    module = nn.BatchNorm2d(2)
    assert math.isclose(set_bn_momentum(nn.Sequential(module), 200), 0.05)
    assert math.isclose(module.momentum, 0.05)
    sgd = make_sgd(nn.Linear(2, 2), 0.01, 1e-4)
    assert math.isclose(sgd.param_groups[0]["weight_decay"], 0.01)
    assert sgd.param_groups[0]["nesterov"] is True

    torch.manual_seed(0)
    images = torch.randn(8, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
    before = model[2].weight.detach().clone()
    finetune(
        model, torch.utils.data.TensorDataset(images, labels), steps=4, base_lr=0.2, grid_wd=0.0,
        batch_size=4, workers=0, device=torch.device("cpu"), seed=0, multilabel=False,
    )
    assert not torch.equal(before, model[2].weight)

    from tempfile import TemporaryDirectory
    from torchvision.models import resnet50

    with TemporaryDirectory() as tmp:
        source = resnet50(weights=None)
        state = {
            f"module.backbone.{key}": value
            for key, value in source.state_dict().items()
            if not key.startswith("fc.")
        }
        state["module.projector.0.weight"] = torch.zeros(4, 4)
        path = Path(tmp) / "ckpt.pth.tar"
        torch.save({"state_dict": state, "epoch": 3}, path)
        encoder = load_encoder(path)
        encoder_sd = encoder.state_dict()
        for key, value in source.state_dict().items():
            if key.startswith("fc."):
                continue
            assert torch.equal(encoder_sd[key], value), key
        assert not torch.equal(encoder.layer4[2].bn3.weight, torch.zeros_like(encoder.layer4[2].bn3.weight))
        classifier = load_classifier(path, 17)
        assert classifier.fc.out_features == 17
        for key, value in source.state_dict().items():
            if key.startswith("fc."):
                continue
            assert torch.equal(classifier.state_dict()[key], value), key

    main_dir = None
    from tempfile import TemporaryDirectory as _TD

    with _TD() as tmp:
        main = Path(tmp) / "Main"
        main.mkdir()
        (main / "train.txt").write_text("000001\n000002\n")
        for name in VOC_CLASSES:
            (main / f"{name}_train.txt").write_text("000001 1\n000002 -1\n")
        (main / "aeroplane_train.txt").write_text("000001 0\n000002 1\n")
        parsed = _read_voc_targets(main, "train", ["000001", "000002"])
        assert parsed.shape == (2, 20)
        assert parsed[0, 0] == 0 and parsed[1, 0] == 1
        assert parsed[0, 1] == 1 and parsed[1, 1] == -1
        main_dir = main
    assert main_dir is not None
    print("ok simclr transfer B.8", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--mode", choices=("linear", "finetune", "scratch"))
    parser.add_argument("--dataset", choices=DATASET_ORDER)
    parser.add_argument("--data", type=Path, help="dataset root (see the module docstring)")
    parser.add_argument("--checkpoint", type=Path, help="SimCLR checkpoint; projector is dropped")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("tune", "final", "all"), default="all")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float, dest="weight_decay")
    parser.add_argument("--steps", type=int, help="override 20000/40000; recorded in the json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--l2-count", type=int, default=45)
    parser.add_argument("--lr-count", type=int, default=7)
    parser.add_argument("--wd-count", type=int, default=None, help="default 7 (+ zero) for finetune, 8 for scratch")
    parser.add_argument("--lbfgs-maxiter", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--compare", type=Path, help="other run's predictions npz for the B.8 permutation test")
    parser.add_argument("--permutations", type=int, default=100_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_check:
        self_check()
        return
    if not args.mode or not args.dataset or args.data is None or args.output is None:
        raise TransferError("--mode, --dataset, --data, and --output are required")
    if args.wd_count is None:
        args.wd_count = 8 if args.mode == "scratch" else 7
    if args.mode == "linear":
        record = run_linear(args)
    else:
        record = run_finetune(args)
    write_record(args.output, record)


if __name__ == "__main__":
    main()
