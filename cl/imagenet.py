"""ImageNet-1k from LMDB, ImageFolder, or the HF parquet hub cache."""

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


def find_lmdb(root: str | Path, split: str = "train") -> Path | None:
    """LMDB env dir (contains data.mdb). LUMI-AI-Guide layout: train/ or train_images/."""
    root = Path(root)
    names = {
        "train": ("train", "train_images"),
        "val": ("val", "validation", "val_images"),
        "validation": ("val", "validation", "val_images"),
    }
    for name in names.get(split, (split,)):
        cand = root / name
        if (cand / "data.mdb").is_file():
            return cand
    if (root / "data.mdb").is_file():
        return root
    return None


class LmdbImageNet:
    """Random-access JPEG LMDB (LUMI-AI-Guide / ImageFolderLMDB pickle records).

    Values are pickle.dumps((jpeg_bytes, label)). Open lazily so DataLoader
    workers do not inherit a forked env. lock=False / readahead=False is the
    random-read recipe on Lustre.
    """

    def __init__(self, root: str | Path, transform=None):
        import pickle

        import lmdb

        self.root = str(root)
        self.transform = transform
        self._env = None
        env = lmdb.open(self.root, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            raw_len = txn.get(b"__len__")
            if raw_len is None:
                raise FileNotFoundError(f"no __len__ in {self.root}")
            self.length = int(pickle.loads(raw_len))
            raw_keys = txn.get(b"__keys__")
            self.keys = (
                pickle.loads(raw_keys)
                if raw_keys is not None
                else [f"{i}".encode("ascii") for i in range(self.length)]
            )
        env.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def _open(self):
        if self._env is None:
            import lmdb

            self._env = lmdb.open(
                self.root, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self._env

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        import pickle

        from PIL import Image

        with self._open().begin(write=False) as txn:
            raw = txn.get(self.keys[idx])
        if raw is None:
            raise IndexError(idx)
        jpeg, label = pickle.loads(raw)
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def build_imagenet_dataset(root: str | Path, split: str = "train", transform=None):
    from torchvision.datasets import ImageFolder

    root = Path(root)
    lmdb_root = find_lmdb(root, split)
    if lmdb_root is not None:
        return LmdbImageNet(lmdb_root, transform=transform)
    folder = root / ("train" if split == "train" else "val")
    if folder.is_dir():
        return ImageFolder(folder, transform=transform)
    return ParquetImageNet(root, split=split, transform=transform)
