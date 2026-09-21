#!/usr/bin/env python3
"""ImageNet end-to-end fine-tuning for Hydra/MoCo ViT checkpoints.

Recipe defaults follow MoCo-v3 DeiT fine-tuning (150 epochs, AdamW, mixup/cutmix)
and are compatible with MAE-Lite MoCo-v3 ViT-Tiny settings except layer-wise LR
decay (not used here; MAE-Lite uses layer_decay=0.75 for 300e).

Checkpoint formats accepted:
  * convert_to_deit.py output: {'model': backbone_state_dict}
  * MoCo/Hydra: {'state_dict': module.base_encoder.*} (head stripped)
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets

import timm
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import accuracy


def parse_args():
    p = argparse.ArgumentParser(description="ViT ImageNet fine-tune")
    p.add_argument("--data-path", required=True, type=str)
    p.add_argument("--finetune", required=True, type=str, help="pretrained checkpoint")
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--model", default="vit_tiny_patch16_224", type=str)
    p.add_argument("--epochs", default=150, type=int)
    p.add_argument("--batch-size", default=256, type=int, help="per-GPU batch size")
    p.add_argument("--lr", default=5e-4, type=float, help="base lr before batch scaling")
    p.add_argument("--weight-decay", default=0.05, type=float)
    p.add_argument("--warmup-epochs", default=5, type=int)
    p.add_argument("--min-lr", default=1e-5, type=float)
    p.add_argument("--drop-path", default=0.1, type=float)
    p.add_argument("--mixup", default=0.8, type=float)
    p.add_argument("--cutmix", default=1.0, type=float)
    p.add_argument("--smoothing", default=0.1, type=float)
    p.add_argument("--reprob", default=0.25, type=float)
    p.add_argument("--workers", default=8, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--print-freq", default=50, type=int)
    p.add_argument("--clip-grad", default=None, type=float)
    p.add_argument("--local_rank", default=-1, type=int)
    return p.parse_args()


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def load_pretrained(model: nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("module.base_encoder.") and not k.startswith(
                "module.base_encoder.head."
            ):
                state[k[len("module.base_encoder.") :]] = v
            elif k.startswith("base_encoder.") and not k.startswith("base_encoder.head."):
                state[k[len("base_encoder.") :]] = v
    else:
        state = ckpt
    # Drop classifier if shape mismatch.
    for k in ("head.weight", "head.bias"):
        if k in state and k in model.state_dict() and state[k].shape != model.state_dict()[k].shape:
            del state[k]
    msg = model.load_state_dict(state, strict=False)
    if is_main():
        print(f"=> loaded {path}")
        print(f"   missing={msg.missing_keys}")
        print(f"   unexpected={msg.unexpected_keys}")


def build_loaders(args, world_size, rank):
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    train_tf = create_transform(
        input_size=224,
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=args.reprob,
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    val_tf = create_transform(
        input_size=224,
        is_training=False,
        interpolation="bicubic",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=val_tf)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler, len(train_set.classes)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_meter = 0.0
    acc1_meter = 0.0
    acc5_meter = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            output = model(images)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        bs = images.size(0)
        loss_meter += loss.item() * bs
        acc1_meter += acc1.item() * bs
        acc5_meter += acc5.item() * bs
        n += bs
    totals = torch.tensor([loss_meter, acc1_meter, acc5_meter, float(n)], device=device)
    if dist.is_initialized():
        dist.all_reduce(totals)
    loss_meter, acc1_meter, acc5_meter, n = totals.tolist()
    return {
        "loss": loss_meter / max(n, 1),
        "acc1": acc1_meter / max(n, 1),
        "acc5": acc5_meter / max(n, 1),
    }


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    # DeiT/MoCo-v3 lr scaling: lr * total_batch / 512
    total_batch = args.batch_size * world_size
    scaled_lr = args.lr * total_batch / 512.0
    if is_main():
        print(json.dumps({
            "model": args.model,
            "epochs": args.epochs,
            "batch_per_gpu": args.batch_size,
            "world_size": world_size,
            "total_batch": total_batch,
            "base_lr": args.lr,
            "scaled_lr": scaled_lr,
            "finetune": args.finetune,
        }, indent=2))

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_path_rate=args.drop_path,
    )
    load_pretrained(model, args.finetune)
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    train_loader, val_loader, train_sampler, _ = build_loaders(args, world_size, rank)
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.smoothing,
            num_classes=1000,
        )
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(
        model,
        opt="adamw",
        lr=scaled_lr,
        weight_decay=args.weight_decay,
    )
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )
    scaler = GradScaler(enabled=not torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    best_acc1 = 0.0
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_seen = 0
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                images, targets = mixup_fn(images, targets)
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            loss_sum += loss.item() * images.size(0)
            n_seen += images.size(0)
            if is_main() and step % args.print_freq == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch[{epoch}][{step}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} lr={lr_now:.3e}",
                    flush=True,
                )
        scheduler.step(epoch + 1)
        metrics = evaluate(model, val_loader, device)
        train_loss = loss_sum / max(n_seen, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "acc1": metrics["acc1"],
            "acc5": metrics["acc5"],
            "epoch_time_sec": time.time() - t0,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if is_main():
            print(
                f"* Epoch {epoch}: Acc@1 {metrics['acc1']:.3f} Acc@5 {metrics['acc5']:.3f} "
                f"(train_loss={train_loss:.4f}, {row['epoch_time_sec']:.1f}s)",
                flush=True,
            )
            is_best = metrics["acc1"] > best_acc1
            best_acc1 = max(best_acc1, metrics["acc1"])
            state = {
                "epoch": epoch + 1,
                "model": (model.module if hasattr(model, "module") else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc1": best_acc1,
                "args": vars(args),
            }
            torch.save(state, out_dir / "checkpoint.pth")
            if is_best:
                torch.save(state, out_dir / "model_best.pth")
            metrics_path = out_dir / "metrics.json"
            payload = {
                "model_id": "hydra-moco-vitt-beta1-e300",
                "eval": "imagenet_finetune",
                "recipe": "moco_v3_deit_style_timm",
                "arch": args.model,
                "epochs": args.epochs,
                "batch_size_per_gpu": args.batch_size,
                "total_batch": total_batch,
                "base_lr": args.lr,
                "scaled_lr": scaled_lr,
                "best_acc1": best_acc1,
                "last_acc1": metrics["acc1"],
                "history_tail": history[-5:],
                "elapsed_sec": time.time() - start,
            }
            metrics_path.write_text(json.dumps(payload, indent=2) + "\n")

    if is_main():
        total = str(datetime.timedelta(seconds=int(time.time() - start)))
        print(f"Done. best Acc@1={best_acc1:.3f} total={total}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
