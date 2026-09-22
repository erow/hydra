#!/usr/bin/env python3
"""Fails if HF-cache snapshot discovery drifts."""

from __future__ import annotations

import tempfile
from pathlib import Path

from cl.imagenet import LmdbImageNet, build_imagenet_dataset, find_hf_snapshot, find_lmdb


def check_resolve() -> None:
    root = Path(tempfile.mkdtemp())
    data = root / "snapshots" / "rev" / "data"
    data.mkdir(parents=True)
    (data / "train-00000-of-00001.parquet").write_bytes(b"x")
    assert find_hf_snapshot(root) == data.parent
    assert find_hf_snapshot(data.parent) == data.parent
    assert find_hf_snapshot(root / "missing") is None

    folder = Path(tempfile.mkdtemp())
    (folder / "train").mkdir()
    assert find_hf_snapshot(folder) is None


def check_lmdb() -> None:
    import io
    import pickle

    import lmdb
    from PIL import Image

    root = Path(tempfile.mkdtemp())
    env_dir = root / "train"
    env_dir.mkdir()
    env = lmdb.open(str(env_dir), subdir=True, map_size=1 << 20)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, format="JPEG")
    jpeg = buf.getvalue()
    with env.begin(write=True) as txn:
        txn.put(b"0", pickle.dumps((jpeg, 7), protocol=5))
        txn.put(b"__keys__", pickle.dumps([b"0"], protocol=5))
        txn.put(b"__len__", pickle.dumps(1, protocol=5))
    env.close()
    assert find_lmdb(root, "train") == env_dir
    ds = build_imagenet_dataset(root, "train")
    assert isinstance(ds, LmdbImageNet) and len(ds) == 1
    img, y = ds[0]
    assert y == 7 and img.size == (8, 8)


if __name__ == "__main__":
    check_resolve()
    check_lmdb()
    print("ok imagenet HF-cache resolve + LMDB")
