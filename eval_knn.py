#!/usr/bin/env python3
"""Frozen-backbone k-NN evaluation (standard SSL protocol).

Protocol (MoCo v3 / DINO-style):
  - ImageNet mean/std resize→center-crop preprocessing (from model_manifest)
  - Freeze backbone; extract full train and test embeddings
  - L2-normalize features; cosine similarity retrieval
  - k nearest neighbours (default 20; ViT-Tiny comparison uses k=10) with
    temperature-weighted soft voting (T=0.07)
  - Memory bank = official train split (Flowers uses train+val, matching transfer;
    ImageNet uses train→memory / val→query)
  - Query set = official test split (ImageNet: val)

This is intentionally separate from few-shot ridge probing in ``evaluate_frozen.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torchvision import datasets

from evaluate_frozen import (
    RESULT_SCHEMA_VERSION,
    EvaluationError,
    build_dataset,
    cached_embeddings,
    dataset_num_classes,
    evaluation_transform,
    model_record,
    prepare_frozen_model,
    read_manifest,
)

DEFAULT_K = 20
DEFAULT_TEMPERATURE = 0.07
KNN_DATASETS = (
    "cifar10",
    "dtd",
    "flowers",
    "cifar100",
    "pets",
    "mini_imagenet",
    "imagenet",
    "aircraft",
    "cars",
    "food101",
    "stl10",
)
# Table 2 (`tab:tl`): Air. CF10 Cars DTD FLW Food Pets STL
TABLE2_KNN_DATASETS = (
    "aircraft",
    "cifar10",
    "cars",
    "dtd",
    "flowers",
    "food101",
    "pets",
    "stl10",
)
IMAGENET_NUM_CLASSES = 1000


def resolve_imagenet_root(data_root: Path) -> Path:
    """Resolve ImageNet-1k ImageFolder root (train/ + val/)."""
    candidates = [
        data_root,
        data_root / "IN1K",
        data_root / "imagenet",
        data_root / "ImageNet",
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    raise EvaluationError(
        f"ImageNet root must contain train/ and val/ under {data_root} "
        "(checked ., IN1K/, imagenet/, ImageNet/)."
    )


def load_knn_splits(args: argparse.Namespace, transform):
    """Return (train_ds, query_ds, num_classes, train_split_name, test_split_name)."""
    if args.dataset == "imagenet":
        root = resolve_imagenet_root(args.data_root)
        train = datasets.ImageFolder(root / "train", transform=transform)
        query = datasets.ImageFolder(root / "val", transform=transform)
        return train, query, IMAGENET_NUM_CLASSES, "train", "val"
    root = args.data_root / args.dataset
    train = build_dataset(
        args.dataset, root, transform, True, download=args.download
    )
    query = build_dataset(
        args.dataset, root, transform, False, download=args.download
    )
    if args.dataset == "flowers":
        train_split = "train+val"
    elif args.dataset == "aircraft":
        train_split = "trainval"
    else:
        train_split = "train"
    return train, query, dataset_num_classes(args.dataset), train_split, "test"


@torch.inference_mode()
def knn_accuracy(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
    num_classes: int,
    chunk_size: int = 1024,
    device: torch.device | None = None,
) -> float:
    """Temperature-weighted cosine k-NN accuracy (features are L2-normalized)."""
    if k <= 0:
        raise ValueError("k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(train_features) < k:
        raise EvaluationError(
            f"Train set has {len(train_features)} samples; cannot retrieve k={k}"
        )
    if device is None:
        device = torch.device("cpu")
    # Keep the memory bank on device for ImageNet-scale retrieval; features are
    # extracted to CPU caches first, then moved here for the matmul.
    train_features = F.normalize(train_features.float(), dim=1).to(device, non_blocking=True)
    train_targets = train_targets.long().to(device, non_blocking=True)
    test_features = F.normalize(test_features.float(), dim=1)
    test_targets = test_targets.long()

    correct = 0
    total = 0
    for start in range(0, len(test_features), chunk_size):
        query = test_features[start : start + chunk_size].to(device, non_blocking=True)
        targets = test_targets[start : start + chunk_size].to(device, non_blocking=True)
        # Cosine similarity via matmul of L2-normalized vectors.
        similarity = query @ train_features.T
        topk_sim, topk_idx = similarity.topk(k, dim=1, largest=True, sorted=True)
        topk_labels = train_targets[topk_idx]
        weights = (topk_sim / temperature).exp()
        votes = torch.zeros(len(query), num_classes, dtype=torch.float32, device=device)
        votes.scatter_add_(1, topk_labels, weights)
        predictions = votes.argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += len(targets)
    return float(correct / max(total, 1))


def run_knn(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    record = model_record(manifest, args.model, allow_excluded=args.allow_excluded)
    print(f"eval_knn loading checkpoint {args.model}", flush=True)
    model, checkpoint_meta = prepare_frozen_model(record, manifest_path, args.download)
    print(f"eval_knn building splits {args.dataset} under {args.data_root}", flush=True)
    transform = evaluation_transform(manifest)
    train, test, num_classes, train_split, test_split = load_knn_splits(args, transform)
    print(
        f"eval_knn splits ready train={len(train)} test={len(test)} classes={num_classes}",
        flush=True,
    )
    train_features, train_targets = cached_embeddings(
        model,
        record,
        f"{args.dataset}-train",
        train,
        args.cache_dir,
        manifest["preprocessing"],
        args.batch_size,
        args.device,
        args.workers,
    )
    test_features, test_targets = cached_embeddings(
        model,
        record,
        f"{args.dataset}-{'val' if args.dataset == 'imagenet' else 'test'}",
        test,
        args.cache_dir,
        manifest["preprocessing"],
        args.batch_size,
        args.device,
        args.workers,
    )
    num_classes = max(
        num_classes,
        int(train_targets.max().item()) + 1,
        int(test_targets.max().item()) + 1,
    )
    accuracy = knn_accuracy(
        train_features,
        train_targets,
        test_features,
        test_targets,
        k=args.k,
        temperature=args.temperature,
        num_classes=num_classes,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "manifest_id": manifest.get("manifest_id"),
        "manifest_revision": manifest.get("sources", {}).get("huggingface", {}).get("revision"),
        "task": "knn",
        "model_id": args.model,
        "architecture_id": record["architecture_id"],
        "method": record["method"],
        "dataset": args.dataset,
        "checkpoint": checkpoint_meta,
        "protocol": {
            "k": args.k,
            "temperature": args.temperature,
            "distance": "cosine",
            "feature_normalization": "l2",
            "voting": "temperature_weighted",
            "train_split": train_split,
            "test_split": test_split,
            "preprocessing": manifest["preprocessing"],
            "num_train": int(len(train_targets)),
            "num_test": int(len(test_targets)),
        },
        "accuracy": accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("model_manifest.json")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=KNN_DATASETS)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("../results/embeddings"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--allow-excluded", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"eval_knn start model={args.model} dataset={args.dataset} "
        f"device={args.device} data_root={args.data_root}",
        flush=True,
    )
    try:
        args.device = torch.device(args.device)
        result = run_knn(args)
        serialized = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
            print(
                f"knn accuracy={result['accuracy']:.4f} "
                f"model={result['model_id']} dataset={result['dataset']} "
                f"k={result['protocol']['k']} → {args.output}",
                flush=True,
            )
        else:
            print(serialized)
    except EvaluationError as exc:
        raise SystemExit(f"evaluation error: {exc}") from exc


if __name__ == "__main__":
    main()
