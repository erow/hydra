"""Mini-ImageNet ImageFolder wrapper for frozen few-shot evaluation.

Expected layout (100-class full protocol by default)::

    root/
      train/<class_id>/*.jpg
      test/<class_id>/*.jpg

``train`` is the support pool; ``test`` is the query set. Images are typically
84x84 and are upsampled by the shared ImageNet eval transform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from torchvision.datasets import ImageFolder


class MiniImageNet(ImageFolder):
    """ImageFolder over ``train/`` or ``test/`` under ``root``."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        if download:
            raise RuntimeError(
                "Mini-ImageNet download is not automated; prepare ImageFolder "
                "splits under root/train and root/test (see transfer/README.md)."
            )
        split = "train" if train else "test"
        split_root = Path(root) / split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Mini-ImageNet split missing: {split_root}. "
                "Expect root/train/<class>/* and root/test/<class>/*."
            )
        super().__init__(
            str(split_root), transform=transform, target_transform=target_transform
        )
        # Expose list[int] targets for balanced_support_indices.
        self.targets = [int(t) for t in self.targets]
