#!/usr/bin/env python3
"""Declarative frozen-feature transfer and robustness evaluation.

This entry point deliberately performs no training of the backbone.  It reads
``model_manifest.json``, loads one approved checkpoint, caches embeddings, and
fits closed-form regularized linear probes.  It is suitable for a single
model/dataset smoke run as well as the later batch launchers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # Optional: local Hydra evaluation does not need it.
    hf_hub_download = None

from extract_backbone import extract_backbone, get_state_dict, load_checkpoint


DATASET_INFO = {
    "cifar10": (10, "CIFAR10"),
    "cifar100": (100, "CIFAR100"),
    "pets": (37, "Pets"),
    "flowers": (102, "Flowers"),
    "mini_imagenet": (100, "MiniImageNet"),
    "dtd": (47, "DTD"),
}
# Table 2 k-NN only; keep out of DATASET_INFO so few-shot aggregation stays unchanged.
KNN_ONLY_DATASETS = {
    "aircraft": (100, "Aircraft"),
    "cars": (196, "Cars"),
    "food101": (101, "Food101"),
    "stl10": (10, "STL10"),
}
SUPPORTED_DATASETS = tuple(DATASET_INFO)
FEW_SHOT_COUNTS = (1, 5, 10, 25)
# Oxford Flowers train has 20 images/class; 25-shot is impossible.
FLOWERS_SHOT_COUNTS = (1, 5, 10, 20)
DEFAULT_SEEDS = (0, 1, 2)
RESULT_SCHEMA_VERSION = "1.0.0"


def dataset_num_classes(name: str) -> int:
    if name in DATASET_INFO:
        return DATASET_INFO[name][0]
    if name in KNN_ONLY_DATASETS:
        return KNN_ONLY_DATASETS[name][0]
    raise EvaluationError(
        f"Unknown dataset {name!r}; choose from {SUPPORTED_DATASETS + tuple(KNN_ONLY_DATASETS)}"
    )


def shots_for_dataset(dataset: str) -> tuple[int, ...]:
    """Return the full-protocol shot schedule for a transfer dataset."""
    if dataset == "flowers":
        return FLOWERS_SHOT_COUNTS
    return FEW_SHOT_COUNTS


class EvaluationError(RuntimeError):
    """An actionable configuration, dataset, or checkpoint error."""


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise EvaluationError(f"Manifest does not exist: {path}") from exc


def model_record(
    manifest: dict[str, Any], model_id: str, *, allow_excluded: bool = False
) -> dict[str, Any]:
    for record in manifest.get("models", []):
        if record.get("id") == model_id:
            if record.get("evaluation_status") == "excluded" and not allow_excluded:
                raise EvaluationError(
                    f"Model {model_id!r} is excluded: "
                    f"{record.get('exclusion_reason', 'no reason recorded')}. "
                    "Pass --allow-excluded for unmatched/single-model reports."
                )
            return record
    raise EvaluationError(f"Model {model_id!r} is not present in the manifest")


def resolve_checkpoint(record: dict[str, Any], manifest_path: Path, download: bool) -> Path:
    checkpoint = record["checkpoint"]
    if record.get("source") == "timm":
        raise EvaluationError(
            f"Model {record['id']} uses source=timm; call prepare_frozen_model instead of "
            "resolve_checkpoint."
        )
    # Check cluster-local twins first. `local_path` is often the other
    # cluster's scratch; stating a foreign automount hangs the process.
    local_twins = list(checkpoint.get("local_paths") or [])
    if checkpoint.get("local_path"):
        local_twins.append(checkpoint["local_path"])
    for raw in local_twins:
        twin = Path(raw)
        if twin.is_file():
            _verify_checkpoint_hash(twin, checkpoint.get("sha256"), record["id"])
            return twin
    if record.get("source") == "local":
        path = Path(checkpoint["path"])
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        if not path.is_file():
            raise EvaluationError(
                f"Checkpoint for {record['id']} is missing: {path}. "
                "Supply the file at the manifest path before evaluating."
            )
        _verify_checkpoint_hash(path, checkpoint.get("sha256"), record["id"])
        return path
    if not download:
        raise EvaluationError(
            f"Checkpoint {record['id']} is remote-only. Pass --download to fetch "
            f"{checkpoint['repository']}@{checkpoint['revision']}:{checkpoint['path']}."
        )
    if hf_hub_download is None:
        raise EvaluationError(
            "Remote checkpoints require the optional 'huggingface_hub' package."
        )
    try:
        path = Path(
            hf_hub_download(
                repo_id=checkpoint["repository"],
                filename=checkpoint["path"],
                revision=checkpoint["revision"],
            )
        )
        _verify_checkpoint_hash(path, checkpoint.get("lfs_sha256"), record["id"])
        return path
    except Exception as exc:
        raise EvaluationError(
            f"Could not download pinned checkpoint {record['id']} from "
            f"{checkpoint['repository']}@{checkpoint['revision']}: {exc}"
        ) from exc


def _verify_checkpoint_hash(path: Path, expected: str | None, model_id: str) -> None:
    if not expected:
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise EvaluationError(
            f"SHA-256 mismatch for {model_id}: expected {expected}, "
            f"got {digest.hexdigest()} ({path})"
        )


def build_backbone(record: dict[str, Any]) -> tuple[nn.Module, str]:
    architecture = record["architecture_id"]
    if architecture == "resnet50":
        return models.resnet50(weights=None), "fc"
    if architecture in {
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",
    }:
        # Import lazily so CPU/static utilities can still be used without timm.
        try:
            import vits
        except ImportError as exc:
            raise EvaluationError("ViT evaluation requires the repo's timm dependency") from exc
        # Optional override for architecture variants that share tensor shapes
        # but differ in attention layout (e.g. MAE-Lite ViT-Tiny num_heads=12).
        factory_name = record.get("vit_factory")
        if factory_name:
            if not hasattr(vits, factory_name):
                raise EvaluationError(
                    f"Unknown vit_factory {factory_name!r} for {record.get('id')}"
                )
            return getattr(vits, factory_name)(), "head"
        constructor = {
            "vit_tiny_patch16_224": vits.vit_tiny,
            "vit_small_patch16_224": vits.vit_small,
            "vit_base_patch16_224": vits.vit_base,
        }[architecture]
        return constructor(), "head"
    raise EvaluationError(f"Unsupported architecture_id: {architecture}")


def _timm_model_name(record: dict[str, Any]) -> str:
    checkpoint = record.get("checkpoint") or {}
    name = checkpoint.get("timm_model_name") or record.get("architecture_id")
    if not name:
        raise EvaluationError(f"timm model {record.get('id')!r} is missing timm_model_name")
    return str(name)


def load_timm_frozen_backbone(record: dict[str, Any]) -> nn.Module:
    """Load ImageNet supervised weights from timm (no local checkpoint file)."""
    try:
        import timm
    except ImportError as exc:
        raise EvaluationError(
            f"timm is required for source=timm model {record['id']}"
        ) from exc
    name = _timm_model_name(record)
    try:
        # num_classes=0 yields pooled features from the forward pass.
        model = timm.create_model(name, pretrained=True, num_classes=0)
    except Exception as exc:
        raise EvaluationError(
            f"Could not create timm model {name!r} for {record['id']}: {exc}"
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_frozen_backbone(record: dict[str, Any], checkpoint_path: Path) -> nn.Module:
    model, head_name = build_backbone(record)
    try:
        state = extract_backbone(get_state_dict(load_checkpoint(checkpoint_path)))
        incompatible = model.load_state_dict(state, strict=False)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise EvaluationError(
            f"Could not normalize/load {record['id']} ({checkpoint_path}): {exc}"
        ) from exc
    missing = [key for key in incompatible.missing_keys if not key.startswith(f"{head_name}.")]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith(f"{head_name}.")]
    if missing or unexpected:
        raise EvaluationError(
            f"Checkpoint layout for {record['id']} is not faithful: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}. "
            "No non-head parameters were silently discarded."
        )
    if hasattr(model, head_name):
        # Preserve the architecture's feature-producing forward path while
        # replacing its supervised head with an identity operation.
        setattr(model, head_name, nn.Identity())
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def prepare_frozen_model(
    record: dict[str, Any], manifest_path: Path, download: bool
) -> tuple[nn.Module, dict[str, Any]]:
    """Return a frozen backbone and checkpoint provenance metadata."""
    if record.get("source") == "timm":
        model = load_timm_frozen_backbone(record)
        name = _timm_model_name(record)
        return model, {
            "path": f"timm:{name}",
            "source": "timm",
            "revision": record.get("checkpoint", {}).get("timm_tag") or "pretrained=True",
            "sha256": record.get("checkpoint", {}).get("sha256"),
        }
    checkpoint = resolve_checkpoint(record, manifest_path, download)
    model = load_frozen_backbone(record, checkpoint)
    return model, {
        "path": str(checkpoint),
        "source": record.get("source"),
        "revision": record.get("checkpoint", {}).get("revision"),
        "sha256": record.get("checkpoint", {}).get("sha256")
        or record.get("checkpoint", {}).get("lfs_sha256"),
    }


def evaluation_transform(manifest: dict[str, Any]) -> transforms.Compose:
    preprocessing = manifest["preprocessing"]
    return transforms.Compose(
        [
            transforms.Resize(preprocessing["resize_shorter_side"]),
            transforms.CenterCrop(preprocessing["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(preprocessing["mean"], preprocessing["std"]),
        ]
    )


def _resolve_stl10_root(root: Path) -> Path:
    for candidate in (root, root.parent / "STL10", root.parent / "stl10"):
        if (candidate / "stl10_binary").is_dir() or (candidate / "stl10_binary.tar.gz").is_file():
            return candidate
    return root


def build_dataset(
    name: str,
    root: Path,
    transform: transforms.Compose,
    train: bool,
    download: bool = False,
) -> Dataset:
    if name not in DATASET_INFO and name not in KNN_ONLY_DATASETS:
        raise EvaluationError(
            f"Unknown dataset {name!r}; choose from {SUPPORTED_DATASETS + tuple(KNN_ONLY_DATASETS)}"
        )
    try:
        if name == "cifar10":
            return datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        if name == "cifar100":
            return datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
        if name == "dtd":
            # torchvision DTD resolves ``<tv_root>/dtd/dtd/{images,labels}``.
            # Our convention passes root=data_root/dtd, so tv_root is data_root
            # when data_root/dtd/dtd/images exists (official torchvision layout).
            if (root / "dtd" / "images").is_dir():
                tv_root = root.parent
            elif (root / "dtd" / "dtd" / "images").is_dir():
                tv_root = root
            else:
                tv_root = root.parent
            split = "train" if train else "test"
            return datasets.DTD(root=str(tv_root), split=split, download=download, transform=transform)
        if name == "stl10":
            split = "train" if train else "test"
            return datasets.STL10(
                root=str(_resolve_stl10_root(root)),
                split=split,
                download=download,
                transform=transform,
            )
        if name == "food101":
            split = "train" if train else "test"
            return datasets.Food101(
                root=str(root), split=split, download=download, transform=transform
            )
        if name == "aircraft":
            split = "trainval" if train else "test"
            return datasets.FGVCAircraft(
                root=str(root),
                split=split,
                annotation_level="variant",
                download=download,
                transform=transform,
            )
        if name == "cars":
            split = "train" if train else "test"
            return datasets.StanfordCars(
                root=str(root), split=split, download=download, transform=transform
            )
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise EvaluationError(
            f"{name} dataset is missing or incomplete under {root}; "
            "pass --download or extract it under DATA_ROOT first."
        ) from exc
    try:
        if name == "mini_imagenet":
            from transfer.mini_imagenet_dataset import MiniImageNet
            return MiniImageNet(root=str(root), train=train, download=False, transform=transform)
        from transfer.oxford_flowers_dataset import Flowers
        from transfer.oxford_pets_dataset import Pets
        dataset_class = Pets if name == "pets" else Flowers
        return dataset_class(root=str(root), train=train, download=False, transform=transform)
    except FileNotFoundError as exc:
        raise EvaluationError(
            f"{name} dataset is missing or incomplete under {root}; "
            "download/extract it using transfer/README.md."
        ) from exc
    except ImportError as exc:
        raise EvaluationError(
            f"{name} requires the existing transfer dataset dependencies: {exc}"
        ) from exc


def class_targets(dataset: Dataset) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [dataset[index][1] for index in range(len(dataset))]
    return [int(target) for target in targets]


def balanced_support_indices(dataset: Dataset, shots: int, seed: int) -> list[int]:
    targets = class_targets(dataset)
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(target, []).append(index)
    generator = random.Random(seed)
    selected: list[int] = []
    for target in sorted(by_class):
        if len(by_class[target]) < shots:
            raise EvaluationError(
                f"Class {target} has only {len(by_class[target])} samples; "
                f"cannot select {shots}-shot support data."
            )
        candidates = by_class[target].copy()
        generator.shuffle(candidates)
        selected.extend(sorted(candidates[:shots]))
    return selected


def _dataloader_worker_init(_worker_id: int) -> None:
    """Keep decode workers from oversubscribing BLAS/OpenMP threads."""
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _pad_to_n(tensor: torch.Tensor, n: int) -> torch.Tensor:
    extra = n - tensor.size(0)
    if extra <= 0:
        return tensor
    return torch.cat([tensor, tensor.new_zeros((extra,) + tensor.shape[1:])], 0)


def _selfcheck_split_and_trim() -> None:
    """Stride split covers [0, N) once; pad-to-max then trim reconstructs."""
    n, world = 10000, 16
    parts = [list(range(rank, n, world)) for rank in range(world)]
    assert sorted(i for p in parts for i in p) == list(range(n))
    chunks = [torch.tensor(p) for p in parts]
    counts = [c.numel() for c in chunks]
    max_n = max(counts)
    padded = [_pad_to_n(c, max_n) for c in chunks]
    out = torch.cat([p[:k] for p, k in zip(padded, counts)])
    assert sorted(out.tolist()) == list(range(n))
    leftover = torch.arange(6).reshape(3, 2)
    padded_b = _pad_to_n(leftover, 5)
    assert padded_b.shape == (5, 2) and torch.equal(padded_b[:3], leftover)
    assert torch.equal(padded_b[3:], padded_b.new_zeros(2, 2))
    assert _pad_to_n(leftover, 3) is leftover


def _all_gather_cat(tensor: torch.Tensor) -> torch.Tensor:
    """NCCL all_gather along dim0 when ranks hold different lengths."""
    world = dist.get_world_size()
    if world == 1:
        return tensor
    n = torch.tensor([tensor.size(0)], device=tensor.device, dtype=torch.long)
    counts_t = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(counts_t, n)
    counts = [int(x.item()) for x in counts_t]
    tensor = _pad_to_n(tensor, max(counts))
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat([chunk[:c] for chunk, c in zip(gathered, counts)], dim=0)


_selfcheck_split_and_trim()


@torch.inference_mode()
def extract_embeddings(
    model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device, workers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    use_dist = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if use_dist else 1
    rank = dist.get_rank() if use_dist else 0
    local_ds: Dataset = (
        Subset(dataset, range(rank, len(dataset), world)) if use_dist else dataset
    )
    # DDP eval while the train loader still holds persistent workers: extra
    # forks deadlock on LUMI /dev/shm (same tmpfs as staged ImageNet).
    if use_dist:
        workers = 0
    pin_memory = device.type == "cuda" and workers > 0
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["worker_init_fn"] = _dataloader_worker_init
    loader = DataLoader(local_ds, **loader_kwargs)
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.to(device)
    model.eval()
    total = len(dataset)
    local_n = len(local_ds)
    print(
        f"extracting embeddings: n={total} local={local_n} world={world} "
        f"batch_size={batch_size} workers={workers} pin_memory={pin_memory} device={device}",
        flush=True,
    )
    done = 0
    last_report = 0
    report_every = max(min(batch_size * 4, max(local_n, 1)), 1)
    for images, labels in loader:
        # Keep N == batch_size. A leftover 53/113-wide batch is a new MIOpen
        # Find key; with cudnn.benchmark that search has run ~1h on MI250.
        n = images.size(0)
        output = model(_pad_to_n(images, batch_size).to(device, non_blocking=pin_memory))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = torch.flatten(output, 1)
        features.append(output[:n].detach().float())
        targets.append(torch.as_tensor(labels, device=device).long())
        done += len(labels)
        if done - last_report >= report_every or done >= local_n:
            print(f"  embedding progress: {done}/{local_n} (global {total})", flush=True)
            last_report = done
    if features:
        local_f = torch.cat(features)
        local_y = torch.cat(targets)
        dim = local_f.size(1)
    elif total == 0:
        raise EvaluationError("Dataset is empty; cannot extract embeddings")
    else:
        local_f = local_y = None
        dim = 0
    if use_dist:
        dim_t = torch.tensor([dim], device=device, dtype=torch.long)
        dist.all_reduce(dim_t, op=dist.ReduceOp.MAX)
        dim = int(dim_t.item())
        if local_f is None:
            local_f = torch.zeros(0, dim, device=device)
            local_y = torch.zeros(0, dtype=torch.long, device=device)
        local_f = _all_gather_cat(local_f)
        local_y = _all_gather_cat(local_y)
        if local_f.size(0) != total:
            raise EvaluationError(f"gathered {local_f.size(0)} embeddings != {total}")
    elif local_f is None:
        raise EvaluationError("Dataset is empty; cannot extract embeddings")
    return local_f.cpu(), local_y.cpu()


def cache_key(record: dict[str, Any], dataset: str, split: str, transform: Any) -> str:
    payload = json.dumps(
        {"model": record["id"], "checkpoint": record["checkpoint"], "dataset": dataset,
         "split": split, "transform": transform},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def cached_embeddings(
    model: nn.Module, record: dict[str, Any], dataset_name: str, dataset: Dataset,
    cache_dir: Path, transform: Any, batch_size: int, device: torch.device, workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(record, dataset_name, "all", transform)
    path = cache_dir / f"{record['id']}-{dataset_name}-{key}.pt"
    if path.is_file():
        try:
            cached = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            cached = torch.load(path, map_location="cpu")
        if isinstance(cached, dict) and {"features", "targets"} <= cached.keys():
            return cached["features"], cached["targets"]
    features, targets = extract_embeddings(model, dataset, batch_size, device, workers)
    torch.save({"features": features, "targets": targets}, path)
    return features, targets


def ridge_probe(features: torch.Tensor, targets: torch.Tensor, classes: int, regularization: float) -> torch.Tensor:
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("Features must be [N,D] and targets must be [N]")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    features = torch.nn.functional.normalize(features.float(), dim=1)
    design = torch.cat([features, torch.ones(len(features), 1)], dim=1)
    labels = torch.zeros(len(features), classes)
    labels[torch.arange(len(features)), targets.long()] = 1.0
    gram = design.T @ design
    gram.diagonal()[:-1].add_(regularization)
    return torch.linalg.solve(gram, design.T @ labels)


def probe_accuracy(weights: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> float:
    features = torch.nn.functional.normalize(features.float(), dim=1)
    design = torch.cat([features, torch.ones(len(features), 1)], dim=1)
    predictions = (design @ weights).argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def few_shot_results(
    train_features: torch.Tensor, train_targets: torch.Tensor,
    test_features: torch.Tensor, test_targets: torch.Tensor, classes: int,
    shots: Iterable[int], seeds: Iterable[int], regularization: float,
) -> list[dict[str, Any]]:
    records = []
    class_dataset = _TensorDataset(train_features, train_targets)
    for shot in shots:
        for seed in seeds:
            indices = balanced_support_indices(class_dataset, shot, seed)
            index_tensor = torch.tensor(indices)
            weights = ridge_probe(
                train_features[index_tensor], train_targets[index_tensor],
                classes, regularization,
            )
            records.append({
                "shots_per_class": shot,
                "seed": seed,
                "support_size": len(indices),
                "accuracy": probe_accuracy(weights, test_features, test_targets),
            })
    return records


class _TensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features, self.targets = features, targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.features[index], int(self.targets[index])


def imagenet_c_results(
    clean_features: torch.Tensor, clean_targets: torch.Tensor,
    corruption_features: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    classes: int, regularization: float,
) -> dict[str, Any]:
    weights = ridge_probe(clean_features, clean_targets, classes, regularization)
    clean_accuracy = probe_accuracy(weights, clean_features, clean_targets)
    per_corruption: dict[str, float] = {}
    severity_accuracies: list[float] = []
    for corruption, severity_data in sorted(corruption_features.items()):
        accuracies = [
            probe_accuracy(weights, features, targets)
            for severity, (features, targets) in sorted(severity_data.items())
        ]
        per_corruption[corruption] = float(sum(accuracies) / len(accuracies))
        severity_accuracies.extend(accuracies)
    mean_accuracy = float(sum(severity_accuracies) / len(severity_accuracies))
    return {
        "clean_accuracy": clean_accuracy,
        "mean_corruption_accuracy": mean_accuracy,
        "relative_corruption_error": (1.0 - mean_accuracy) / max(1.0 - clean_accuracy, 1e-12),
        "per_corruption_accuracy": per_corruption,
    }


def run_transfer(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    record = model_record(manifest, args.model, allow_excluded=args.allow_excluded)
    model, checkpoint_meta = prepare_frozen_model(record, manifest_path, args.download)
    transform = evaluation_transform(manifest)
    root = args.data_root / args.dataset
    train = build_dataset(args.dataset, root, transform, True, download=args.download)
    test = build_dataset(args.dataset, root, transform, False, download=args.download)
    train_features, train_targets = cached_embeddings(
        model, record, f"{args.dataset}-train", train, args.cache_dir, manifest["preprocessing"],
        args.batch_size, args.device, args.workers,
    )
    test_features, test_targets = cached_embeddings(
        model, record, f"{args.dataset}-test", test, args.cache_dir, manifest["preprocessing"],
        args.batch_size, args.device, args.workers,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "manifest_id": manifest.get("manifest_id"),
        "manifest_revision": manifest.get("sources", {}).get("huggingface", {}).get("revision"),
        "task": "transfer",
        "model_id": args.model,
        "architecture_id": record["architecture_id"],
        "method": record["method"],
        "dataset": args.dataset,
        "checkpoint": checkpoint_meta,
        "protocol": {"shots_per_class": list(args.shots), "seeds": list(args.seeds),
                     "regularization": args.regularization,
                     "preprocessing": manifest["preprocessing"]},
        "few_shot": few_shot_results(
            train_features, train_targets, test_features, test_targets,
            DATASET_INFO[args.dataset][0], args.shots, args.seeds, args.regularization,
        ),
    }


def run_imagenet_c(args: argparse.Namespace) -> dict[str, Any]:
    """Fit on clean ImageNet train data and reuse the probe on ImageNet-C."""
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    record = model_record(manifest, args.model, allow_excluded=args.allow_excluded)
    model, checkpoint_meta = prepare_frozen_model(record, manifest_path, args.download)
    transform = evaluation_transform(manifest)
    clean_root = args.imagenet_root
    if not (clean_root / "train").is_dir() or not (clean_root / "val").is_dir():
        raise EvaluationError(
            f"ImageNet root must contain train/ and val/ ImageFolder directories: {clean_root}"
        )
    if not args.imagenet_c_root.is_dir():
        raise EvaluationError(f"ImageNet-C root does not exist: {args.imagenet_c_root}")
    clean_train = datasets.ImageFolder(clean_root / "train", transform=transform)
    clean_val = datasets.ImageFolder(clean_root / "val", transform=transform)
    clean_train_features, clean_train_targets = cached_embeddings(
        model, record, "imagenet-train", clean_train, args.cache_dir,
        manifest["preprocessing"], args.batch_size, args.device, args.workers,
    )
    clean_val_features, clean_val_targets = cached_embeddings(
        model, record, "imagenet-val", clean_val, args.cache_dir,
        manifest["preprocessing"], args.batch_size, args.device, args.workers,
    )
    corruption_features: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    corruption_root = args.imagenet_c_root
    for corruption_path in sorted(path for path in corruption_root.iterdir() if path.is_dir()):
        severity_data: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for severity_path in sorted(path for path in corruption_path.iterdir() if path.is_dir()):
            try:
                severity = int(severity_path.name)
            except ValueError:
                continue
            dataset = datasets.ImageFolder(severity_path, transform=transform)
            severity_data[severity] = cached_embeddings(
                model, record, f"imagenet-c-{corruption_path.name}-{severity}",
                dataset, args.cache_dir, manifest["preprocessing"],
                args.batch_size, args.device, args.workers,
            )
        if severity_data:
            corruption_features[corruption_path.name] = severity_data
    if len(corruption_features) != 15 or any(
        len(severities) != 5 for severities in corruption_features.values()
    ):
        raise EvaluationError(
            f"ImageNet-C must contain exactly 15 corruption directories and five "
            f"severity directories; found {len(corruption_features)} corruptions."
        )
    result = imagenet_c_results(
        clean_train_features, clean_train_targets, corruption_features,
        len(clean_train.classes), args.regularization,
    )
    result.update({
        "schema_version": RESULT_SCHEMA_VERSION,
        "manifest_id": manifest.get("manifest_id"),
        "manifest_revision": manifest.get("sources", {}).get("huggingface", {}).get("revision"),
        "task": "imagenet-c",
        "model_id": args.model,
        "architecture_id": record["architecture_id"],
        "method": record["method"],
        "checkpoint": checkpoint_meta,
        "protocol": {
            "regularization": args.regularization,
            "preprocessing": manifest["preprocessing"],
            "corruptions": 15,
            "severities": 5,
        },
        "clean_validation_accuracy": probe_accuracy(
            ridge_probe(clean_train_features, clean_train_targets,
                        len(clean_train.classes), args.regularization),
            clean_val_features, clean_val_targets,
        ),
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model_manifest.json"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("transfer", "imagenet-c"), default="transfer")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--imagenet-c-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("../results/embeddings"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=None,
        help="Shots per class (default: 1 5 10 25; flowers uses 1 5 10 20)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--allow-excluded",
        action="store_true",
        help="Evaluate models marked evaluation_status=excluded (unmatched/single-model reports).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        args.device = torch.device(args.device)
        if args.task == "transfer":
            if args.dataset is None or args.data_root is None:
                raise EvaluationError("--dataset and --data-root are required for transfer")
            if args.shots is None:
                args.shots = list(shots_for_dataset(args.dataset))
            result = run_transfer(args)
        else:
            if args.imagenet_root is None or args.imagenet_c_root is None:
                raise EvaluationError(
                    "--imagenet-root and --imagenet-c-root are required for imagenet-c"
                )
            result = run_imagenet_c(args)
        serialized = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
        else:
            print(serialized)
    except EvaluationError as exc:
        raise SystemExit(f"evaluation error: {exc}") from exc


if __name__ == "__main__":
    main()
