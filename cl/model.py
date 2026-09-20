"""Backbone + projector contrastive model (SimCLR / SupCon / SimLAP)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as torchvision_models

from cl.losses import nt_xent, simlap_loss, supcon


def build_backbone(arch: str, weights=None) -> nn.Module:
    if arch.startswith("timm:"):
        import timm

        enc = timm.create_model(arch[5:], pretrained=weights is not None, num_classes=0)
        enc.out_dim = enc.num_features
        return enc
    if not hasattr(torchvision_models, arch):
        raise ValueError(f"unknown arch {arch!r}")
    enc = torchvision_models.__dict__[arch](zero_init_residual=True, weights=weights)
    enc.out_dim = enc.fc.in_features
    enc.fc = nn.Identity()
    return enc


def build_projector(in_dim: int, hidden: int, out_dim: int, last_norm: str) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden, bias=False),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden, bias=False),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim, bias=False),
    ]
    if last_norm == "bn":
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    elif last_norm == "ln":
        layers.append(nn.LayerNorm(out_dim))
    elif last_norm != "none":
        raise ValueError(f"last_norm={last_norm!r}")
    return nn.Sequential(*layers)


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        method: str,
        dim: int = 256,
        mlp_dim: int = 2048,
        temperature: float = 0.1,
        num_classes: int = 1000,
        last_norm: str | None = None,
        gate: str = "basic",
    ):
        super().__init__()
        if method not in {"simclr", "supcon", "simlap"}:
            raise ValueError(f"method={method!r}")
        self.method = method
        self.temperature = temperature
        self.backbone = backbone
        if last_norm is None:
            last_norm = "ln" if method == "simlap" else "bn"
        self.projector = build_projector(backbone.out_dim, mlp_dim, dim, last_norm)
        self.filter = None
        if method == "simlap":
            from moco.filter import BasicGate, Filter, OpenGate

            gate_fn = OpenGate if gate == "open" else BasicGate
            self.filter = Filter(num_classes, dim, gate_fn=gate_fn)

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))

    def forward(self, images, targets=None) -> tuple[torch.Tensor, dict[str, float]]:
        z1 = self.project(images[0])
        z2 = self.project(images[1])
        log: dict[str, float] = {}
        if self.method == "simclr":
            loss = nt_xent(z1, z2, self.temperature)
        elif self.method == "supcon":
            if targets is None:
                raise ValueError("SupCon needs labels")
            loss = supcon(z1, z2, targets, self.temperature)
        else:
            if targets is None:
                raise ValueError("SimLAP needs labels")
            loss = simlap_loss(z1, z2, targets, self.filter, self.temperature)
            gate = getattr(self.filter, "gate", None)
            if gate is not None and hasattr(gate, "statistics") and z1.is_cuda:
                with torch.no_grad():
                    activation, entropy = gate.statistics()
                log["activation"] = float(activation)
                log["entropy"] = float(entropy)
        log["z@sim"] = float(torch.nn.functional.cosine_similarity(z1, z2).mean())
        return loss, log


def build_model(arch: str, method: str, **kwargs: Any) -> ContrastiveModel:
    weights = kwargs.pop("weights", None)
    return ContrastiveModel(build_backbone(arch, weights=weights), method, **kwargs)
