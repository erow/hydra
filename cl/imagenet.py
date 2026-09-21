"""ImageNet-1k from ImageFolder or the HF parquet hub cache."""

from __future__ import annotations

import io
from pathlib import Path

SPLIT_GLOB = {
    "train": "train-*.parquet",
    "val": "validation-*.parquet",
    "validation": "validation-*.parquet",
}


def find_hf_snapshot(root: str | Path) -> Path | None:
    root = Path(root)
    data = root / "data"
    if data.is_dir() and any(data.glob("train-*.parquet")):
        return root
    snaps = root / "snapshots"
    if not snaps.is_dir():
        return None
    found: list[Path] = []
    for snap in snaps.iterdir():
        if any((snap / "data").glob("train-*.parquet")):
            found.append(snap)
    return found[-1] if found else None


def _image_bytes(cell) -> bytes:
    if hasattr(cell, "as_py"):
        cell = cell.as_py()
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data:
            return bytes(data)
        path = cell.get("path")
        if path:
            return Path(path).read_bytes()
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    raise TypeError(f"unexpected image cell {type(cell)}")


class ParquetImageNet:
    """Map-style Dataset over ILSVRC parquet shards.

    ponytail: one open shard (~0.5G) per worker; shuffle jumps shards.
    Upgrade: row-group cache, or ImageFolder inside the squashfs.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None):
        import pyarrow.parquet as pq

        snap = find_hf_snapshot(root)
        if snap is None:
            raise FileNotFoundError(f"no HF imagenet parquet under {root}")
        self.files = sorted((snap / "data").glob(SPLIT_GLOB[split]))
        if not self.files:
            raise FileNotFoundError(f"no {SPLIT_GLOB[split]} under {snap / 'data'}")
        self.transform = transform
        self._pq = pq
        n = [pq.read_metadata(f).num_rows for f in self.files]
        total = 0
        self._cum: list[int] = []
        for rows in n:
            total += rows
            self._cum.append(total)
        self._tables: dict[int, object] = {}

    def __len__(self) -> int:
        return self._cum[-1]

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self._cum[-1]:
            raise IndexError(idx)
        lo, hi = 0, len(self._cum)
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        prev = 0 if lo == 0 else self._cum[lo - 1]
        return lo, idx - prev

    def _table(self, shard: int):
        cached = self._tables.get(shard)
        if cached is not None:
            return cached
        table = self._pq.read_table(
            self.files[shard], columns=["image", "label"], use_threads=False
        )
        self._tables.clear()
        self._tables[shard] = table
        return table

    def __getitem__(self, idx: int):
        from PIL import Image

        shard, row = self._locate(idx)
        table = self._table(shard)
        img = Image.open(io.BytesIO(_image_bytes(table.column("image")[row]))).convert("RGB")
        label = int(table.column("label")[row].as_py())
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_imagenet_dataset(root: str | Path, split: str = "train", transform=None):
    from torchvision.datasets import ImageFolder

    root = Path(root)
    folder = root / ("train" if split == "train" else "val")
    if folder.is_dir():
        return ImageFolder(folder, transform=transform)
    return ParquetImageNet(root, split=split, transform=transform)
