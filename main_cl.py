#!/usr/bin/env python
"""Train SimCLR / SupCon / SimLAP / X-CLR (no momentum encoder).

SimCLR defaults follow Chen et al. 2020 ImageNet (paper §2 / Table 6):
  RN50, 2-layer projector 2048→128 + BN, NT-Xent τ=0.1, global BN,
  LARS, base LR 0.3 × batch/256 (= 4.8 at 4096), wd 1e-6, warmup 10, cosine,
  100 ep (64.5% linear), 1000 ep is the 69.3% ResNet-50 row,
  crop (0.08,1), jitter 0.8/0.8/0.8/0.2 p=0.8, gray 0.2, blur p=0.5, flip.

  # official: 4 GPU × 1024 = 4096
  python -m torch.distributed.run --nproc_per_node=4 main_cl.py \\
      --method simclr --data "$IMAGENET_ROOT" --output_dir "$OUT"

  # paper 69.3% row (1000 epoch)
  python -m torch.distributed.run --nproc_per_node=4 main_cl.py \\
      --method simclr --epochs 1000 --data "$IMAGENET_ROOT" --output_dir "$OUT"

  # X-CLR (Sobal et al. ICLR 2025): 8 GPU × 128 = 1024, AutoAugment, τ_s=0.1
  python -m torch.distributed.run --nproc_per_node=8 main_cl.py \\
      --method xclr --data "$IMAGENET_ROOT" --output_dir "$OUT"

SupCon / SimLAP keep the paper no-momentum recipe (absolute LR 0.2, 3-layer 256-d).
"""

from __future__ import annotations

import argparse
import builtins
import datetime
import math
import os
import random
import time
import warnings
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import moco.loader
from cl.imagenet import build_imagenet_dataset
from cl.model import build_model, keep_norm_fp32
from moco.optimizer import LARS

try:
    import wandb
except ImportError:
    wandb = None

KNN_K = 10
KNN_T = 0.07
# Official SimCLR ImageNet (Chen et al. 2020 §2 / google-research/simclr flags).
SIMCLR_RECIPE = dict(
    batch_size=4096,
    epochs=100,
    lr=0.3,
    scale_lr=True,
    weight_decay=1e-6,
    warmup_epochs=10,
    dim=128,
    mlp_dim=2048,
    mlp_layers=2,
    t=0.1,
    crop_min=0.08,
    last_norm="bn",
    jitter=(0.8, 0.8, 0.8, 0.2),
)
# X-CLR ImageNet (Sobal et al. ICLR 2025 §4.1 / A.7): RN50, 2-layer 128-d,
# LARS, 100 ep, bs 1024, absolute LR 0.075, AutoAugment, τ=τ_s=0.1.
XCLR_RECIPE = dict(
    batch_size=1024,
    epochs=100,
    lr=0.075,
    scale_lr=False,
    weight_decay=1e-6,
    warmup_epochs=10,
    dim=128,
    mlp_dim=2048,
    mlp_layers=2,
    t=0.1,
    t_s=0.1,
    crop_min=0.08,
    last_norm="bn",
    jitter=(0.8, 0.8, 0.8, 0.2),
    autoaugment=True,
)
# SupCon ImageNet: base LR 0.4 × batch/256, 8×512=4096, 350 ep, warmup 12, wd 1e-3.
SUPCON_RECIPE = dict(
    batch_size=4096,
    epochs=350,
    lr=0.4,
    scale_lr=True,
    weight_decay=1e-3,
    warmup_epochs=12,
    dim=256,
    mlp_dim=2048,
    mlp_layers=3,
    t=0.1,
    crop_min=0.08,
    last_norm=None,
    jitter=(0.4, 0.4, 0.4, 0.1),
)
# Paper tab:recipe (no momentum): SimLAP default (SupCon uses SUPCON_RECIPE).
PAPER_RECIPE = dict(
    batch_size=1024,
    epochs=1000,
    lr=0.2,
    scale_lr=False,
    weight_decay=1e-4,
    warmup_epochs=40,
    dim=256,
    mlp_dim=2048,
    mlp_layers=3,
    t=0.1,
    crop_min=0.08,
    last_norm=None,
    jitter=(0.4, 0.4, 0.4, 0.1),
)
EVAL_TF = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SimCLR / SupCon / SimLAP / X-CLR pre-training")
    p.add_argument("--method", default="simclr", choices=["simclr", "supcon", "simlap", "xclr"])
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--weights", default=None, type=str)
    p.add_argument("--data_set", default="IN1K", choices=["IN1K", "STL"])
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--data", default=None, help="ImageNet root (contains train/) or STL10 root")
    p.add_argument("data_dir", nargs="?", default=None, metavar="DIR", help="positional alias of --data")
    p.add_argument("-a", "--arch", default="resnet50")
    p.add_argument("-j", "--workers", default=8, type=int)
    p.add_argument("--epochs", default=None, type=int)
    p.add_argument("--ckpt-freq", default=50, type=int)
    p.add_argument("--start-epoch", default=0, type=int)
    p.add_argument("-b", "--batch-size", default=None, type=int, help="global batch")
    p.add_argument("--lr", default=None, type=float, help="SimCLR: base LR × batch/256; others: absolute")
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--wd", "--weight-decay", default=None, type=float, dest="weight_decay")
    p.add_argument("-p", "--print-freq", default=10, type=int)
    p.add_argument("--resume", default="", type=str)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--gpu", default=None, type=int)
    p.add_argument("--optimizer", default="lars", choices=["lars", "adamw"])
    p.add_argument("--warmup-epochs", default=None, type=int)
    p.add_argument("--crop-min", default=None, type=float)
    p.add_argument("--dim", default=None, type=int)
    p.add_argument("--mlp-dim", default=None, type=int)
    p.add_argument("--mlp-layers", default=None, type=int)
    p.add_argument("--t", default=None, type=float, help="softmax temperature")
    p.add_argument("--t-s", default=None, type=float, dest="t_s", help="X-CLR graph softmax temperature")
    p.add_argument("--xclr-graph", default=None, type=str, help="C×C class similarity .pt")
    p.add_argument("--xclr-text-model", default=None, type=str, help="Sentence Transformer id for the graph")
    p.add_argument("--num-classes", default=1000, type=int)
    p.add_argument("--gate", default="basic", choices=["basic", "open"])
    p.add_argument("--last-norm", default=None, choices=["bn", "ln", "none"])
    p.add_argument("--eval-freq", default=100, type=int, help="CIFAR-10 KNN every N epochs; 0 disables")
    p.add_argument(
        "--eval-data",
        default=None,
        type=str,
        help="CIFAR-10 root (cifar-10-batches-py). Default: $DATA_ROOT/cifar10 or <data>/../cifar10",
    )
    p.add_argument("--eval-batch-size", default=256, type=int)
    p.add_argument("--eval-download", action="store_true")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fp16 AMP; --no-amp trains in fp32 (SimCLR on MI250)",
    )
    p.add_argument("--grad-ckpt", action="store_true", help="checkpoint ResNet stages (fp32 8×512)")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "hydra"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    args.data = args.data or args.data_dir
    if not args.data:
        p.error("--data is required")
    del args.data_dir
    return args


def apply_recipe(args: argparse.Namespace) -> None:
    if args.method == "simclr":
        recipe = SIMCLR_RECIPE
    elif args.method == "xclr":
        recipe = XCLR_RECIPE
    elif args.method == "supcon":
        recipe = SUPCON_RECIPE
    else:
        recipe = PAPER_RECIPE
    for key, value in recipe.items():
        if key in {"scale_lr", "jitter"}:
            continue
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.scale_lr = recipe["scale_lr"]
    args.jitter = recipe["jitter"]
    if "autoaugment" in recipe:
        args.autoaugment = recipe["autoaugment"]


def main() -> None:
    args = parse_args()
    apply_recipe(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("CUDNN deterministic is on")

    # LUMI /tmp and /dev/shm are the same tmpfs as the staged ImageNet.
    # Default fd shm races on unlink under 8×8 workers (rank7 DataLoader crash).
    if args.workers:
        torch.multiprocessing.set_sharing_strategy("file_system")
    args.distributed = "RANK" in os.environ
    if args.distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
            # Rank-0 CIFAR KNN holds the post-eval NCCL barrier past 5 min.
            timeout=datetime.timedelta(seconds=1800),
        )
        dist.barrier()
        setup_for_distributed(args.rank == 0)
    else:
        args.rank = 0
        args.world_size = 1
        if args.gpu is None and torch.cuda.is_available():
            args.gpu = 0
    main_worker(args)


def main_worker(args: argparse.Namespace) -> None:
    if args.debug:
        args.output_dir = args.output_dir or None
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    elif args.rank == 0:
        raise SystemExit("--output_dir is required unless --debug")

    if args.data_set == "STL":
        args.num_classes = 10
    extra = f" t_s={args.t_s}" if args.method == "xclr" else ""
    print(
        f"=> recipe {args.method} {args.arch} bs={args.batch_size} ep={args.epochs} "
        f"lr={args.lr}{'×bs/256' if args.scale_lr else ''} wd={args.weight_decay} "
        f"warmup={args.warmup_epochs} dim={args.dim} layers={args.mlp_layers} t={args.t} "
        f"amp={args.amp} grad_ckpt={args.grad_ckpt}{extra}"
    )
    class_sim = None
    if args.method == "xclr":
        from cl.xclr_graph import class_names_for, offdiag_mean, resolve_class_sim

        class_sim = resolve_class_sim(
            args.num_classes,
            names=class_names_for(args.data_set),
            graph_path=args.xclr_graph,
            text_model=args.xclr_text_model or "sentence-transformers/all-mpnet-base-v2",
        )
        print(f"=> xclr graph {tuple(class_sim.shape)} offdiag={offdiag_mean(class_sim):.3f}")
    model = build_model(
        args.arch,
        args.method,
        dim=args.dim,
        mlp_dim=args.mlp_dim,
        num_layers=args.mlp_layers,
        temperature=args.t,
        num_classes=args.num_classes,
        last_norm=args.last_norm,
        gate=args.gate,
        weights=args.weights,
        tau_s=args.t_s if args.t_s is not None else 0.1,
        class_sim=class_sim,
    )
    if args.scale_lr:
        args.lr = args.lr * args.batch_size / 256
    print("=> peak lr", args.lr)

    if args.grad_ckpt:
        model.grad_ckpt = True
        print("=> gradient checkpoint on ResNet stages")
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        args.batch_size = int(args.batch_size / args.world_size)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], static_graph=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    device = next(model.parameters()).device
    use_amp = bool(args.amp) and device.type == "cuda"
    if use_amp:
        print(f"=> norm layers in fp32 ({keep_norm_fp32(model)})")

    if args.optimizer == "lars":
        optimizer = LARS(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(args.output_dir) if args.rank == 0 and args.output_dir else None
    if args.rank == 0 and args.output_dir and not args.no_wandb and wandb is not None:
        try:
            wandb.init(
                dir=args.output_dir,
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                job_type="train",
                name=os.environ.get("WANDB_NAME"),
                resume="allow" if args.resume else None,
            )
        except Exception as exc:
            print("=> wandb disabled:", exc)

    if args.resume and os.path.isfile(args.resume):
        loc = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
        ckpt = torch.load(args.resume, map_location=loc, weights_only=False)
        args.start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        saved_scaler = ckpt.get("scaler") or {}
        if saved_scaler and scaler.is_enabled():
            scaler.load_state_dict(saved_scaler)
        elif saved_scaler or scaler.is_enabled():
            print("=> skip scaler resume (fp32 ckpt ↔ fp16 run)")
        print("=> resumed", args.resume, "epoch", args.start_epoch)

    cudnn.benchmark = True
    train_loader, train_sampler = build_loader(args)

    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train(train_loader, model, optimizer, scaler, writer, epoch, args)
        step = (epoch + 1) * max(len(train_loader), 1)
        if args.rank == 0 and args.output_dir:
            payload = {
                "epoch": epoch + 1,
                "arch": args.arch,
                "method": args.method,
                "model": (model.module if args.distributed else model).state_dict(),
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }
            name = f"checkpoint_{epoch:04d}.pth.tar" if (epoch + 1) % args.ckpt_freq == 0 else "checkpoint.pth"
            torch.save(payload, os.path.join(args.output_dir, name))
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            evaluate_and_log(model, writer, epoch, step, args)
        if args.debug:
            break
    if writer:
        writer.close()
    if wandb is not None and wandb.run:
        wandb.finish()


def build_loader(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ops: list = [transforms.RandomResizedCrop(args.img_size, scale=(args.crop_min, 1.0))]
    if getattr(args, "autoaugment", False):
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        ]
    else:
        ops += [
            transforms.RandomApply([transforms.ColorJitter(*args.jitter)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ]
    aug = transforms.Compose(ops + [transforms.ToTensor(), normalize])
    transform = moco.loader.MultiCropsTransform(aug, aug, num_crops=0)
    if args.data_set == "STL":
        dataset = datasets.STL10(args.data, split="train", download=True, transform=transform)
    else:
        dataset = build_imagenet_dataset(args.data, split="train", transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    return loader, sampler


def resolve_cifar10_root(args) -> Path:
    candidates: list[Path] = []
    if args.eval_data:
        candidates.append(Path(args.eval_data))
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        candidates.append(Path(data_root) / "cifar10")
        candidates.append(Path(data_root))
    parent = Path(args.data).resolve().parent
    candidates.append(parent / "cifar10")
    candidates.append(Path(args.data) / "cifar10")
    candidates.append(parent)
    seen: set[str] = set()
    for root in candidates:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if (root / "cifar-10-batches-py").is_dir():
            return root
        nested = root / "cifar10"
        if (nested / "cifar-10-batches-py").is_dir():
            return nested
    return candidates[0] if candidates else parent / "cifar10"


def evaluate_cifar10_knn(model, args) -> float:
    """Frozen-backbone CIFAR-10 KNN (k=10, cosine, T=0.07). Returns accuracy in percent."""
    from eval_knn import knn_accuracy
    from evaluate_frozen import build_dataset, extract_embeddings

    root = resolve_cifar10_root(args)
    device = next(model.parameters()).device
    enc = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    train_ds = build_dataset("cifar10", root, EVAL_TF, True, download=args.eval_download)
    test_ds = build_dataset("cifar10", root, EVAL_TF, False, download=args.eval_download)
    train_f, train_y = extract_embeddings(enc.backbone, train_ds, args.eval_batch_size, device, args.workers)
    test_f, test_y = extract_embeddings(enc.backbone, test_ds, args.eval_batch_size, device, args.workers)
    acc_t = torch.zeros(1, device=device)
    if args.rank == 0:
        acc_t[0] = knn_accuracy(
            train_f,
            train_y,
            test_f,
            test_y,
            k=KNN_K,
            temperature=KNN_T,
            num_classes=10,
            device=device,
        ) * 100.0
    if args.distributed:
        dist.broadcast(acc_t, 0)
    return float(acc_t)


def evaluate_and_log(model, writer, epoch, step, args) -> None:
    model.eval()
    from evaluate_frozen import EvaluationError

    print("=> CIFAR-10 KNN (k=10)")
    try:
        acc = evaluate_cifar10_knn(model, args)
        print(f"=> eval/CF10 {acc:.2f} (epoch {epoch + 1})")
        if writer:
            writer.add_scalar("eval/CF10", acc, step)
        if wandb is not None and wandb.run:
            wandb.log({"eval/CF10": acc, "epoch": epoch + 1}, step=step)
    except EvaluationError as exc:
        print("=> skip eval/CF10:", exc)
    if args.distributed:
        dist.barrier()
    model.train()


def train(loader, model, optimizer, scaler, writer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    lrs = AverageMeter("LR", ":.4e")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(len(loader), [batch_time, data_time, lrs, losses], prefix=f"Epoch: [{epoch}]")
    model.train()
    end = time.time()
    iters = max(len(loader), 1)
    for i, (images, targets) in enumerate(loader):
        if args.gpu is not None:
            images = [x.cuda(args.gpu, non_blocking=True) for x in images]
            targets = targets.cuda(args.gpu, non_blocking=True)
        data_time.update(time.time() - end)
        lr = adjust_learning_rate(optimizer, epoch + i / iters, args)
        lrs.update(lr)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            loss, log = model(images, targets)
        loss_val = loss.item()
        finite = torch.tensor(int(math.isfinite(loss_val)), device=loss.device)
        if args.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() == 0:
            print(f"=> abort non-finite loss at epoch {epoch} [{i}/{iters}]: {loss_val}")
            raise SystemExit(2)
        losses.update(loss_val, images[0].size(0))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0:
            step = epoch * iters + i
            payload = {"loss": loss_val, "lr": lr, **log}
            if writer:
                for k, v in payload.items():
                    writer.add_scalar(k, v, step)
            if wandb is not None and wandb.run:
                wandb.log(payload, step=step)
        if i % args.print_freq == 0:
            progress.display(i)
        if args.debug and i >= 1:
            break


class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return ("{name} {val" + self.fmt + "} ({avg" + self.fmt + "})").format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        digits = len(str(num_batches))
        self.fmt = f"[{{:{digits}d}}/{num_batches}]"
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        print("\t".join([self.prefix + self.fmt.format(batch)] + [str(m) for m in self.meters]))


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / max(args.warmup_epochs, 1e-8)
    else:
        lr = args.lr * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1))
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def setup_for_distributed(is_master: bool) -> None:
    builtin = builtins.print

    def print(*a, **kw):
        force = kw.pop("force", False)
        if is_master or force:
            builtin("[{}]".format(datetime.datetime.now().time()), *a, **kw)

    builtins.print = print


if __name__ == "__main__":
    main()
