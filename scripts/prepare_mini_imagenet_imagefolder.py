#!/usr/bin/env python3
"""Export HF Mini-ImageNet parquet dump to ImageFolder for evaluate_frozen.

Expected by ``transfer/mini_imagenet_dataset.py``::

    <out_root>/{train,test}/<class_id>/*.jpg

``train`` = support pool; ``test`` = query set.

Source layout (HuggingFace ``timm/mini-imagenet`` dump)::

    <hf_root>/data/{train,validation,test}-*.parquet

Split mapping (documented; NOT Vinyales 2016 class-disjoint few-shot)::

    ImageFolder train  <- HF train + HF validation
        (both drawn from ImageNet-1k train; shared 100 classes)
    ImageFolder test   <- HF test
        (ImageNet-1k validation images; 50 per class)

This HF dump uses the same 100 classes in every split (see dataset README:
it does **not** match Vinyales et al. 2016 train/val/test class partitions).
Class folders are zero-padded HF label ids ``000``..``099`` so ImageFolder
alphabetical order matches label indices 0..99.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

# HF class_label names from timm/mini-imagenet README (label index -> wnid).
HF_LABEL_WNIDS: dict[int, str] = {
    0: "n01532829",
    1: "n01558993",
    2: "n01704323",
    3: "n01749939",
    4: "n01770081",
    5: "n01843383",
    6: "n01855672",
    7: "n01910747",
    8: "n01930112",
    9: "n01981276",
    10: "n02074367",
    11: "n02089867",
    12: "n02091244",
    13: "n02091831",
    14: "n02099601",
    15: "n02101006",
    16: "n02105505",
    17: "n02108089",
    18: "n02108551",
    19: "n02108915",
    20: "n02110063",
    21: "n02110341",
    22: "n02111277",
    23: "n02113712",
    24: "n02114548",
    25: "n02116738",
    26: "n02120079",
    27: "n02129165",
    28: "n02138441",
    29: "n02165456",
    30: "n02174001",
    31: "n02219486",
    32: "n02443484",
    33: "n02457408",
    34: "n02606052",
    35: "n02687172",
    36: "n02747177",
    37: "n02795169",
    38: "n02823428",
    39: "n02871525",
    40: "n02950826",
    41: "n02966193",
    42: "n02971356",
    43: "n02981792",
    44: "n03017168",
    45: "n03047690",
    46: "n03062245",
    47: "n03075370",
    48: "n03127925",
    49: "n03146219",
    50: "n03207743",
    51: "n03220513",
    52: "n03272010",
    53: "n03337140",
    54: "n03347037",
    55: "n03400231",
    56: "n03417042",
    57: "n03476684",
    58: "n03527444",
    59: "n03535780",
    60: "n03544143",
    61: "n03584254",
    62: "n03676483",
    63: "n03770439",
    64: "n03773504",
    65: "n03775546",
    66: "n03838899",
    67: "n03854065",
    68: "n03888605",
    69: "n03908618",
    70: "n03924679",
    71: "n03980874",
    72: "n03998194",
    73: "n04067472",
    74: "n04146614",
    75: "n04149813",
    76: "n04243546",
    77: "n04251144",
    78: "n04258138",
    79: "n04275548",
    80: "n04296562",
    81: "n04389033",
    82: "n04418357",
    83: "n04435653",
    84: "n04443257",
    85: "n04509417",
    86: "n04515003",
    87: "n04522168",
    88: "n04596742",
    89: "n04604644",
    90: "n04612504",
    91: "n06794110",
    92: "n07584110",
    93: "n07613480",
    94: "n07697537",
    95: "n07747607",
    96: "n09246464",
    97: "n09256479",
    98: "n13054560",
    99: "n13133613",
}

# HF split name -> ImageFolder split name (None = skip).
DEFAULT_SPLIT_MAP: dict[str, str] = {
    "train": "train",
    "validation": "train",  # merge into support pool
    "test": "test",
}


def class_dir_name(label: int) -> str:
    return f"{int(label):03d}"


def image_bytes_from_cell(cell) -> bytes:
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is None:
            raise ValueError(f"image cell missing bytes: keys={list(cell)}")
        return raw
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unsupported image cell type: {type(cell)}")


def export_parquet_file(
    path: Path,
    out_split_root: Path,
    *,
    counters: dict[str, Counter],
    dry_run: bool,
) -> int:
    # Dry-run only needs labels (avoids loading multi-GB image blobs).
    columns = ["label"] if dry_run else ["image", "label"]
    table = pq.read_table(path, columns=columns)
    n = table.num_rows
    labels = table.column("label")
    images = None if dry_run else table.column("image")
    written = 0
    for i in range(n):
        label = int(labels[i].as_py())
        if label not in HF_LABEL_WNIDS:
            raise ValueError(f"unexpected label {label} in {path}")
        class_name = class_dir_name(label)
        counters[out_split_root.name][class_name] += 1
        idx = counters[out_split_root.name][class_name]
        # Stable, unique filename: wnid + per-class index within this export.
        wnid = HF_LABEL_WNIDS[label]
        rel_name = f"{wnid}_{idx:05d}.jpg"
        dest_dir = out_split_root / class_name
        dest = dest_dir / rel_name
        if dry_run:
            written += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            written += 1
            continue
        assert images is not None
        raw = image_bytes_from_cell(images[i].as_py())
        with Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert("RGB")
            rgb.save(dest, format="JPEG", quality=95)
        written += 1
    return written


def discover_parquet(hf_data: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(hf_data.glob("*.parquet")):
        name = path.name
        for split in ("train", "validation", "test"):
            if name.startswith(f"{split}-"):
                found[split].append(path)
                break
        else:
            raise RuntimeError(f"unrecognized parquet name: {path}")
    return dict(found)


def verify_imagefolder(out_root: Path, expected_classes: int = 100) -> dict:
    report: dict = {}
    for split in ("train", "test"):
        split_root = out_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing split: {split_root}")
        classes = sorted(p.name for p in split_root.iterdir() if p.is_dir())
        if len(classes) != expected_classes:
            raise RuntimeError(
                f"{split}: expected {expected_classes} classes, got {len(classes)}"
            )
        per_class = {}
        empty = []
        for c in classes:
            n = sum(1 for p in (split_root / c).iterdir() if p.is_file())
            per_class[c] = n
            if n == 0:
                empty.append(c)
        if empty:
            raise RuntimeError(f"{split}: empty classes: {empty[:10]}")
        report[split] = {
            "num_classes": len(classes),
            "num_images": sum(per_class.values()),
            "min_per_class": min(per_class.values()),
            "max_per_class": max(per_class.values()),
            "class_dirs_sample": classes[:5],
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-root",
        type=Path,
        default=Path("/parallel_scratch/jw02425/data/ssl_eval/mini_imagenet/_hf"),
        help="HuggingFace dump root (contains data/*.parquet)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/parallel_scratch/jw02425/data/ssl_eval/mini_imagenet"),
        help="ImageFolder root (will create train/ and test/ alongside _hf/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows only; do not write images",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-export ImageFolder verification",
    )
    args = parser.parse_args(argv)

    hf_data = args.hf_root / "data"
    if not hf_data.is_dir():
        print(f"ERROR: missing {hf_data}", file=sys.stderr)
        return 1

    parquet_by_split = discover_parquet(hf_data)
    for split in ("train", "validation", "test"):
        if split not in parquet_by_split:
            print(f"ERROR: no parquet for split {split}", file=sys.stderr)
            return 1

    print("Split mapping (HF -> ImageFolder):")
    for hf_split, out_split in DEFAULT_SPLIT_MAP.items():
        nfiles = len(parquet_by_split[hf_split])
        print(f"  {hf_split} ({nfiles} parquet) -> {out_split}")
    print(
        "Note: shared 100-class protocol (timm/mini-imagenet); "
        "NOT Vinyales class-disjoint splits."
    )

    counters: dict[str, Counter] = defaultdict(Counter)
    total_written = 0
    for hf_split, out_split in DEFAULT_SPLIT_MAP.items():
        out_split_root = args.out_root / out_split
        for path in parquet_by_split[hf_split]:
            print(f"export {path.name} -> {out_split}/ ...", flush=True)
            n = export_parquet_file(
                path, out_split_root, counters=counters, dry_run=args.dry_run
            )
            total_written += n
            print(f"  rows={n}", flush=True)

    summary = {
        "hf_root": str(args.hf_root),
        "out_root": str(args.out_root),
        "dry_run": args.dry_run,
        "split_map": DEFAULT_SPLIT_MAP,
        "protocol_note": (
            "timm/mini-imagenet: same 100 classes in all splits; "
            "train+validation -> support; test -> query"
        ),
        "rows_processed": total_written,
        "counts": {
            split: {
                "num_classes": len(ctr),
                "num_images": int(sum(ctr.values())),
                "min_per_class": int(min(ctr.values())) if ctr else 0,
                "max_per_class": int(max(ctr.values())) if ctr else 0,
            }
            for split, ctr in counters.items()
        },
        "label_wnids": {str(k): v for k, v in HF_LABEL_WNIDS.items()},
    }

    if not args.dry_run and not args.skip_verify:
        summary["verify"] = verify_imagefolder(args.out_root)

    meta_path = args.out_root / "imagefolder_export_meta.json"
    if not args.dry_run:
        args.out_root.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote {meta_path}")

    print(json.dumps({k: summary[k] for k in summary if k != "label_wnids"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
