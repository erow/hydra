#!/usr/bin/env python3
"""Fails if X-CLR CLI / recipe / loss drift from Sobal et al. ICLR 2025."""

from __future__ import annotations

import sys
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

import main_cl
from cl.losses import nt_xent, supcon, xclr
from cl.model import build_model


def _parse(argv: list[str]):
    with patch.object(sys, "argv", ["main_cl.py", *argv]):
        args = main_cl.parse_args()
    main_cl.apply_recipe(args)
    return args


def check_cli() -> None:
    args = _parse(["--method", "xclr", "--data", "/tmp/IN1K", "--output_dir", "/tmp/out"])
    assert args.batch_size == 1024, args.batch_size
    assert args.epochs == 100, args.epochs
    assert args.lr == 0.075, args.lr
    assert not args.scale_lr
    assert args.weight_decay == 1e-6, args.weight_decay
    assert args.warmup_epochs == 10, args.warmup_epochs
    assert args.dim == 128 and args.mlp_layers == 2 and args.mlp_dim == 2048
    assert args.t == 0.1 and args.t_s == 0.1
    assert args.crop_min == 0.08 and args.last_norm == "bn"
    assert args.autoaugment


def check_model() -> None:
    model = build_model(
        "resnet50",
        "xclr",
        dim=128,
        mlp_dim=2048,
        num_layers=2,
        last_norm="bn",
        class_sim=torch.eye(1000),
        tau_s=0.1,
    )
    assert model.method == "xclr"
    kinds = [type(m) for m in model.projector]
    assert kinds == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Linear, nn.BatchNorm1d], kinds


def check_graph() -> None:
    from cl.xclr_graph import BUNDLED_IN1K, load_class_embeds, load_class_sim, offdiag_mean

    emb = load_class_embeds(BUNDLED_IN1K)
    assert emb.shape[0] == 1000 and emb.ndim == 2, emb.shape
    sim = load_class_sim(BUNDLED_IN1K)
    assert sim.shape == (1000, 1000)
    assert torch.allclose(sim.diag(), torch.ones(1000), atol=1e-5)
    off = offdiag_mean(sim)
    assert 0.30 < off < 0.40, off  # paper ImageNet ST histogram ~0.35
    # related classes closer than unrelated (Sobal et al. fig. 2)
    assert float(sim[207, 208]) > float(sim[207, 504])  # retrievers vs mug


def check_loss() -> None:
    torch.manual_seed(0)
    b, d, c = 8, 16, 4
    z1 = F.normalize(torch.randn(b, d), dim=1)
    z2 = F.normalize(z1 + 0.05 * torch.randn(b, d), dim=1)
    y = torch.arange(b) % c
    same = nt_xent(z1, z2, 0.1)
    sc = supcon(z1, z2, y, 0.1)
    assert abs(xclr(z1, z2, torch.arange(b), torch.eye(b), 0.1, 1e-5).item() - same.item()) < 1e-4
    assert abs(xclr(z1, z2, y, torch.eye(c), 0.1, 1e-5).item() - sc.item()) < 1e-4


if __name__ == "__main__":
    check_cli()
    check_model()
    check_graph()
    check_loss()
    print("ok xclr recipe + ST graph + 2-layer 128-d projector + τ_s→0 limits")
