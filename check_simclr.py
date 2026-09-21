#!/usr/bin/env python3
"""Fails if official SimCLR CLI / recipe / projector drift from Chen et al. 2020."""

from __future__ import annotations

import sys
from unittest.mock import patch

import torch.nn as nn

import main_cl
from cl.model import build_model


def _parse(argv: list[str]):
    with patch.object(sys, "argv", ["main_cl.py", *argv]):
        args = main_cl.parse_args()
    main_cl.apply_recipe(args)
    return args


def check_cli() -> None:
    args = _parse(["--method", "simclr", "--data", "/tmp/IN1K", "--output_dir", "/tmp/out"])
    assert args.batch_size == 4096, args.batch_size
    assert args.epochs == 100, args.epochs
    assert args.lr == 0.3, args.lr
    assert args.scale_lr
    assert args.weight_decay == 1e-6, args.weight_decay
    assert args.warmup_epochs == 10, args.warmup_epochs
    assert args.dim == 128 and args.mlp_layers == 2 and args.mlp_dim == 2048
    assert args.t == 0.1 and args.crop_min == 0.08 and args.last_norm == "bn"
    assert args.jitter == (0.8, 0.8, 0.8, 0.2)
    assert abs(args.lr * args.batch_size / 256 - 4.8) < 1e-9

    args = _parse(
        ["--method", "simclr", "--epochs", "1000", "--data", "/tmp/IN1K", "--output_dir", "/tmp/out"]
    )
    assert args.epochs == 1000 and args.batch_size == 4096


def check_model() -> None:
    model = build_model("resnet50", "simclr", dim=128, mlp_dim=2048, num_layers=2, last_norm="bn")
    assert model.backbone.out_dim == 2048
    kinds = [type(m) for m in model.projector]
    assert kinds == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Linear, nn.BatchNorm1d], kinds
    assert model.projector[0].in_features == 2048 and model.projector[0].out_features == 2048
    assert model.projector[3].in_features == 2048 and model.projector[3].out_features == 128
    assert model.projector[4].affine is False


if __name__ == "__main__":
    check_cli()
    check_model()
    print("ok simclr official recipe + --data CLI + 2-layer 128-d projector")
