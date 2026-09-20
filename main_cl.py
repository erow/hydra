#!/usr/bin/env python
"""Train SimCLR / SupCon / SimLAP (no momentum encoder).

  python -m torch.distributed.run --nproc_per_node=4 main_cl.py \\
      --method simclr --data /path/to/IN1K --output_dir /path/to/out

  Logs ``loss`` to wandb each step and ``eval/CF10`` (KNN k=10) every 100 epochs.

  # single process (local smoke)
  python main_cl.py --method simclr --data /path/to/IN1K --debug
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
from cl.model import build_model
from moco.optimizer import LARS

try:
    import wandb
except ImportError:
    wandb = None

KNN_K = 10
KNN_T = 0.07
EVAL_TF = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SimCLR / SupCon / SimLAP pre-training")
    p.add_argument("--method", default="simclr", choices=["simclr", "supcon", "simlap"])
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--weights", default=None, type=str)
    p.add_argument("--data_set", default="IN1K", choices=["IN1K", "STL"])
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("data", metavar="DIR", help="ImageNet root (train/) or STL10 root")
    p.add_argument("-a", "--arch", default="resnet50")
    p.add_argument("-j", "--workers", default=8, type=int)
    p.add_argument("--epochs", default=1000, type=int)
    p.add_argument("--ckpt-freq", default=100, type=int)
    p.add_argument("--start-epoch", default=0, type=int)
    p.add_argument("-b", "--batch-size", default=1024, type=int, help="global batch")
    p.add_argument("--lr", default=0.2, type=float, help="base LR; scaled by batch/256")
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay")
    p.add_argument("-p", "--print-freq", default=10, type=int)
    p.add_argument("--resume", default="", type=str)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--gpu", default=None, type=int)
    p.add_argument("--optimizer", default="lars", choices=["lars", "adamw"])
    p.add_argument("--warmup-epochs", default=40, type=int)
    p.add_argument("--crop-min", default=0.08, type=float)
    p.add_argument("--dim", default=256, type=int)
    p.add_argument("--mlp-dim", default=2048, type=int)
    p.add_argument("--t", default=0.1, type=float, help="softmax temperature")
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
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "hydra"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("CUDNN deterministic is on")

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
            timeout=datetime.timedelta(seconds=300),
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
    print("=> creating", args.method, args.arch)
    model = build_model(
        args.arch,
        args.method,
        dim=args.dim,
        mlp_dim=args.mlp_dim,
        temperature=args.t,
        num_classes=args.num_classes,
        last_norm=args.last_norm,
        gate=args.gate,
        weights=args.weights,
    )
    args.lr = args.lr * args.batch_size / 256

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

    if args.optimizer == "lars":
        optimizer = LARS(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(args.output_dir) if args.rank == 0 and args.output_dir else None
    if args.rank == 0 and args.output_dir and not args.no_wandb:
        if wandb is None:
            raise SystemExit("wandb is required to log loss; pip install wandb or pass --no-wandb")
        wandb.init(
            dir=args.output_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            job_type="train",
            name=os.environ.get("WANDB_NAME"),
            resume="allow" if args.resume else None,
        )

    if args.resume and os.path.isfile(args.resume):
        loc = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
        ckpt = torch.load(args.resume, map_location=loc, weights_only=False)
        args.start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
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
    aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img_size, scale=(args.crop_min, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform = moco.loader.MultiCropsTransform(aug, aug, num_crops=0)
    if args.data_set == "STL":
        dataset = datasets.STL10(args.data, split="train", download=True, transform=transform)
    else:
        dataset = datasets.ImageFolder(os.path.join(args.data, "train"), transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
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
    acc = knn_accuracy(
        train_f,
        train_y,
        test_f,
        test_y,
        k=KNN_K,
        temperature=KNN_T,
        num_classes=10,
        device=device,
    )
    return acc * 100.0


def evaluate_and_log(model, writer, epoch, step, args) -> None:
    model.eval()
    if args.rank == 0:
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
        with torch.cuda.amp.autocast(enabled=args.gpu is not None):
            loss, log = model(images, targets)
        losses.update(loss.item(), images[0].size(0))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0:
            step = epoch * iters + i
            payload = {"loss": loss.item(), "lr": lr, **log}
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
