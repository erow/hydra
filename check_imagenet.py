#!/usr/bin/env python3
"""Fails if HF-cache snapshot discovery drifts."""

from __future__ import annotations

import tempfile
from pathlib import Path

from cl.imagenet import find_hf_snapshot


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


if __name__ == "__main__":
    check_resolve()
    print("ok imagenet HF-cache resolve")
