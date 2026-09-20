#!/usr/bin/env python3
"""One-shot check: FastSSL SimCLR/SupCon checkpoint layout → torchvision ResNet-50."""

from __future__ import annotations

import torch

from extract_backbone import extract_backbone, get_state_dict


def main() -> None:
    ckpt = {
        "model": {
            "backbone.conv1.weight": torch.zeros(1),
            "backbone.layer1.0.conv1.weight": torch.zeros(1),
            "projector.0.weight": torch.zeros(1),
            "scale_logit": torch.zeros(1),
        }
    }
    out = extract_backbone(get_state_dict(ckpt))
    assert set(out) == {"conv1.weight", "layer1.0.conv1.weight"}, out
    print("ok FastSSL backbone.* layout")


if __name__ == "__main__":
    main()
