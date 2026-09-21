#!/usr/bin/env python3
"""Extract backbone-only state dictionaries from SimLAP checkpoints.

The checkpoints in ``weights/`` use three formats:

* a raw model state dictionary;
* a dictionary containing ``model`` or ``state_dict``;
* a DDP checkpoint whose model keys start with ``module.``.

This script normalizes those formats and removes classifier/projector heads.
The output is a plain PyTorch state dictionary with backbone-compatible keys.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint while remaining compatible with older PyTorch."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} does not contain a dictionary checkpoint")
    return checkpoint


def get_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Find the model state dictionary in common checkpoint layouts."""
    for key in ("state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    # Several files in weights/ are already raw state dictionaries.
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Could not find a model state dictionary in the checkpoint")


def extract_backbone(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return backbone parameters with a normalized, model-local key prefix."""
    keys = list(state_dict)

    # Hydra-MoCo / MAE-Lite MoCo-v3 ViT: use the base encoder, not its projector
    # or predictor. The momentum encoder has the same architecture but is not
    # the encoder used by the standard linear-evaluation loading path.
    # Also accept non-DDP ``base_encoder.*`` (MAE-Lite sometimes saves without
    # the ``module.`` prefix).
    for prefix in ("module.base_encoder.", "base_encoder."):
        if any(key.startswith(prefix) for key in keys):
            return {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix) and not key[len(prefix):].startswith("head.")
            }

    # MAE-Lite MAE: DDP-wrapped encoder+decoder under ``module.model.*``.
    # Keep encoder tensors only (drop mask_token / decoder_*).
    for prefix in ("module.model.", "model."):
        if any(key.startswith(prefix) for key in keys) and any(
            key.startswith(prefix + "patch_embed.") for key in keys
        ):
            out = {}
            for key, value in state_dict.items():
                if not key.startswith(prefix):
                    continue
                local = key[len(prefix) :]
                if local == "mask_token" or local.startswith("decoder"):
                    continue
                if local.startswith("head."):
                    continue
                out[local] = value
            if out:
                return out

    # Common SSL comparator wrappers use ``encoder`` or ``backbone`` instead
    # of MoCo-v3's ``base_encoder``.  Normalize only a known wrapper; strict
    # model loading in the evaluator still rejects incompatible layouts.
    for prefix in (
        "module.encoder_q.",
        "module.encoder.",
        "encoder.",
        "module.backbone.",
        "backbone.",
    ):
        if any(key.startswith(prefix) for key in keys):
            return {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
                and not any(
                    key[len(prefix):].startswith(head)
                    for head in ("fc.", "head.", "projector.", "predictor.")
                )
            }

    # HydraV1/rebuttal ResNet and Hydra ViT-B wrap the backbone under ``visual``.
    # Drop classifier/projector heads (``fc.*`` / ``head.*``) and ignore sibling
    # modules such as ``projector``, ``filter``, and ``logit_scale``.
    if any(key.startswith("visual.") for key in keys):
        prefix = "visual."
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
            and not key[len(prefix):].startswith("fc.")
            and not key[len(prefix):].startswith("head.")
        }

    # Raw ViT checkpoints use ``head`` for the ImageNet classifier.
    if "patch_embed.proj.weight" in state_dict:
        return {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("head.")
        }

    # Raw ResNet checkpoints use ``fc`` either as a classifier or as the MoCo
    # projector. Keep the stem and residual stages, and remove all fc.* keys.
    if "conv1.weight" in state_dict:
        return {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("fc.")
        }

    raise ValueError("Unknown checkpoint layout; no supported backbone prefix found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Input .pth or .ckpt file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output state-dict path (default: results/backbones/<input-name>.pth)",
    )
    args = parser.parse_args()

    backbone = extract_backbone(get_state_dict(load_checkpoint(args.checkpoint)))
    output = args.output or Path("results/backbones") / f"{args.checkpoint.stem}_backbone.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone, output)
    print(f"Saved {len(backbone)} tensors to {output}")


if __name__ == "__main__":
    main()
