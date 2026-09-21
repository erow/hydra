"""Backbone + projector contrastive model (SimCLR / SupCon / SimLAP / X-CLR)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as torchvision_models

from cl.losses import nt_xent, simlap_loss, supcon, xclr


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


def build_projector(
    in_dim: int, hidden: int, out_dim: int, last_norm: str, num_layers: int = 3
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError(f"num_layers={num_layers}")
    layers: list[nn.Module] = []
    for i in range(num_layers):
        dim1 = in_dim if i == 0 else hidden
        dim2 = out_dim if i == num_layers - 1 else hidden
        layers.append(nn.Linear(dim1, dim2, bias=False))
        if i < num_layers - 1:
            layers.append(nn.BatchNorm1d(dim2))
            layers.append(nn.ReLU(inplace=True))
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
        num_layers: int | None = None,
        temperature: float = 0.1,
        num_classes: int = 1000,
        last_norm: str | None = None,
        gate: str = "basic",
        tau_s: float = 0.1,
        class_sim: torch.Tensor | None = None,
    ):
        super().__init__()
        if method not in {"simclr", "supcon", "simlap", "xclr"}:
            raise ValueError(f"method={method!r}")
        self.method = method
        self.temperature = temperature
        self.tau_s = tau_s
        self.backbone = backbone
        if last_norm is None:
            last_norm = "ln" if method == "simlap" else "bn"
        if num_layers is None:
            num_layers = 2 if method in {"simclr", "xclr"} else 3
        self.projector = build_projector(backbone.out_dim, mlp_dim, dim, last_norm, num_layers)
        self.filter = None
        if method == "simlap":
            from moco.filter import BasicGate, Filter, OpenGate

            gate_fn = OpenGate if gate == "open" else BasicGate
            self.filter = Filter(num_classes, dim, gate_fn=gate_fn)
        if method == "xclr":
            if class_sim is None:
                raise ValueError("X-CLR needs class_sim")
            if class_sim.shape != (num_classes, num_classes):
                raise ValueError(f"class_sim {tuple(class_sim.shape)} != ({num_classes}, {num_classes})")
            self.register_buffer("class_sim", class_sim.detach().float().contiguous(), persistent=False)

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
        elif self.method == "xclr":
            if targets is None:
                raise ValueError("X-CLR needs labels")
            loss = xclr(z1, z2, targets, self.class_sim, self.temperature, self.tau_s)
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
        with torch.no_grad():
            log["z@sim"] = float(torch.nn.functional.cosine_similarity(z1, z2).mean())
        return loss, log


def build_model(arch: str, method: str, **kwargs: Any) -> ContrastiveModel:
    weights = kwargs.pop("weights", None)
    return ContrastiveModel(build_backbone(arch, weights=weights), method, **kwargs)
