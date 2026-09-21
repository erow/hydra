#!/usr/bin/env python3
"""Validate and aggregate frozen-representation evaluation records.

The command intentionally reports missing evaluations as incomplete coverage.
It never fills absent metrics with estimates or values from another model.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from evaluate_frozen import DATASET_INFO, DEFAULT_SEEDS, read_manifest, shots_for_dataset
from validate_results import ValidationError, validate_record


def _result_paths(result_files: Iterable[Path], result_dir: Path | None) -> list[Path]:
    paths = list(result_files)
    if result_dir is not None:
        paths.extend(sorted(result_dir.rglob("*.json")))
    unique = {path.resolve() for path in paths}
    return sorted(unique)


def _expected_evaluations(manifest: dict[str, Any]) -> set[tuple[str, str, str]]:
    groups = {
        model_id: group
        for group in manifest.get("approved_comparison_groups", [])
        for model_id in group.get("hydra", []) + group.get("comparators", [])
    }
    expected = set()
    for model_id in groups:
        for dataset in DATASET_INFO:
            # DTD is used for frozen k-NN only in this campaign, not few-shot transfer.
            if dataset == "dtd":
                continue
            expected.add((model_id, "transfer", dataset))
        expected.add((model_id, "imagenet-c", "imagenet-c"))
    return expected


def _group_for(manifest: dict[str, Any], model_id: str) -> dict[str, Any]:
    for group in manifest.get("approved_comparison_groups", []):
        if model_id in group.get("hydra", []) + group.get("comparators", []):
            return group
    raise ValueError(f"validated model {model_id!r} is not in an approved group")


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        record["model_id"],
        record["task"],
        record.get("dataset", "imagenet-c"),
    )


def _transfer_rows(record: dict[str, Any], group_id: str) -> list[dict[str, Any]]:
    values: dict[int, list[float]] = {}
    for item in record["few_shot"]:
        values.setdefault(item["shots_per_class"], []).append(float(item["accuracy"]))
    rows = []
    for shots in sorted(values):
        scores = values[shots]
        rows.append(
            {
                "task": "transfer",
                "architecture_id": record["architecture_id"],
                "group_id": group_id,
                "model_id": record["model_id"],
                "method": record["method"],
                "dataset": record["dataset"],
                "shots_per_class": shots,
                "mean_accuracy": statistics.fmean(scores),
                "std_accuracy": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                "clean_accuracy": None,
                "mean_corruption_accuracy": None,
                "relative_corruption_error": None,
                "corruption": None,
                "corruption_accuracy": None,
            }
        )
    return rows


def _corruption_rows(record: dict[str, Any], group_id: str) -> list[dict[str, Any]]:
    common = {
        "task": "imagenet-c",
        "architecture_id": record["architecture_id"],
        "group_id": group_id,
        "model_id": record["model_id"],
        "method": record["method"],
        "dataset": "imagenet-c",
        "shots_per_class": None,
        "mean_accuracy": None,
        "std_accuracy": None,
        "clean_accuracy": record["clean_accuracy"],
        "mean_corruption_accuracy": record["mean_corruption_accuracy"],
        "relative_corruption_error": record["relative_corruption_error"],
    }
    rows = [{**common, "corruption": name, "corruption_accuracy": accuracy}
            for name, accuracy in sorted(record["per_corruption_accuracy"].items())]
    return rows or [{**common, "corruption": None, "corruption_accuracy": None}]


def _markdown(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# SSL representation comparison ({report['status']})",
        "",
        report["conclusion"],
        "",
        f"Validated result records: {report['coverage']['validated_records']}; "
        f"expected complete evaluations: {report['coverage']['expected_evaluations']}; "
        f"missing or incomplete: {len(report['coverage']['missing'])}.",
        "",
    ]
    architectures = sorted({row["architecture_id"] for row in rows})
    for architecture in architectures or ["resnet50", "vit_small_patch16_224"]:
        lines.extend([f"## {architecture}", "", "### Few-shot transfer", ""])
        lines.append("| Group | Model | Dataset | Shots | Accuracy mean | Accuracy std |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        transfer = [row for row in rows if row["task"] == "transfer"
                    and row["architecture_id"] == architecture]
        for row in transfer:
            lines.append(
                f"| {row['group_id']} | {row['model_id']} | {row['dataset']} | "
                f"{row['shots_per_class']} | {row['mean_accuracy']:.4f} | "
                f"{row['std_accuracy']:.4f} |"
            )
        if not transfer:
            lines.append("| — | — | — | — | unavailable | unavailable |")
        lines.extend(["", "### ImageNet-C", ""])
        lines.append("| Group | Model | Clean | Mean corruption | Relative corruption error | Corruption breakdown |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        corruption = [row for row in rows if row["task"] == "imagenet-c"
                      and row["architecture_id"] == architecture]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in corruption:
            grouped.setdefault((row["group_id"], row["model_id"]), []).append(row)
        for (group_id, model_id), model_rows in sorted(grouped.items()):
            first = model_rows[0]
            breakdown = ", ".join(
                f"{row['corruption']}={row['corruption_accuracy']:.4f}"
                for row in model_rows if row["corruption"] is not None
            ) or "unavailable"
            lines.append(
                f"| {group_id} | {model_id} | {first['clean_accuracy']:.4f} | "
                f"{first['mean_corruption_accuracy']:.4f} | "
                f"{first['relative_corruption_error']:.4f} | {breakdown} |"
            )
        if not corruption:
            lines.append("| — | — | unavailable | unavailable | unavailable | unavailable |")
        lines.append("")
    if report["coverage"]["missing"]:
        lines.extend(["## Incomplete runtime evaluation", ""])
        lines.append("No metric is inferred for these missing or incomplete evaluations:")
        lines.extend(f"- `{item}`" for item in report["coverage"]["missing"])
        lines.append("")
    return "\n".join(lines)


def aggregate(manifest: dict[str, Any], records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = _expected_evaluations(manifest)
    by_key = {_record_key(record): record for record in records}
    missing = []
    for model_id, task, dataset in sorted(expected):
        key = (model_id, task, dataset)
        record = by_key.get(key)
        if record is None:
            missing.append(f"{model_id}/{task}/{dataset}: no validated record")
        elif task == "transfer" and (
            set(record["protocol"]["shots_per_class"]) != set(shots_for_dataset(dataset))
            or set(record["protocol"]["seeds"]) != set(DEFAULT_SEEDS)
        ):
            missing.append(f"{model_id}/{task}/{dataset}: default shots/seeds incomplete")
    rows = []
    for record in records:
        group = _group_for(manifest, record["model_id"])
        rows.extend(
            _transfer_rows(record, group["id"])
            if record["task"] == "transfer"
            else _corruption_rows(record, group["id"])
        )
    coverage = {
        "expected_evaluations": len(expected),
        "validated_records": len(records),
        "complete_evaluations": len(expected) - len(missing),
        "missing": missing,
    }
    report = {
        "schema_version": "1.0.0",
        "manifest_id": manifest.get("manifest_id"),
        "status": "complete" if not missing else "incomplete",
        "conclusion": (
            "All planned evaluations are present and validated."
            if not missing
            else "Runtime evaluation is incomplete; no missing metric was inferred."
        ),
        "coverage": coverage,
        "records": [
            {"model_id": record["model_id"], "task": record["task"],
             "dataset": record.get("dataset", "imagenet-c")}
            for record in records
        ],
        "rows": rows,
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model_manifest.json"))
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("results", type=Path, nargs="*")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest.resolve())
    paths = _result_paths(args.results, args.results_dir)
    records = []
    failures = []
    seen = set()
    for path in paths:
        try:
            record = validate_record(path, manifest)
            key = _record_key(record)
            if key in seen:
                raise ValidationError(f"duplicate result identity {key}")
            seen.add(key)
            records.append(record)
        except ValidationError as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        raise SystemExit("INVALID RESULT RECORDS\n" + "\n".join(failures))
    report, rows = aggregate(manifest, records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "task", "architecture_id", "group_id", "model_id", "method", "dataset",
        "shots_per_class", "mean_accuracy", "std_accuracy", "clean_accuracy",
        "mean_corruption_accuracy", "relative_corruption_error", "corruption",
        "corruption_accuracy",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "report.md").write_text(_markdown(report, rows))
    print(
        f"{report['status']}: {report['coverage']['complete_evaluations']}/"
        f"{report['coverage']['expected_evaluations']} complete evaluations; "
        f"wrote {args.output_dir}/summary.json, summary.csv, report.md"
    )


if __name__ == "__main__":
    main()
