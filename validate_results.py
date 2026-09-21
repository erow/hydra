#!/usr/bin/env python3
"""Validate frozen-representation result records before aggregation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluate_frozen import DATASET_INFO, RESULT_SCHEMA_VERSION, read_manifest


class ValidationError(ValueError):
    """A result record does not satisfy the aggregation contract."""


def _require(record: dict[str, Any], key: str) -> Any:
    value = record.get(key)
    if value is None or value == "":
        raise ValidationError(f"missing required field {key!r}")
    return value


def _accuracy(value: Any, field: str) -> None:
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValidationError(f"{field} must be a number in [0, 1]")


def _nonnegative_number(value: Any, field: str) -> None:
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise ValidationError(f"{field} must be a non-negative number")


def _validate_manifest_identity(
    record: dict[str, Any],
    manifest: dict[str, Any],
    *,
    allow_excluded: bool = False,
) -> None:
    if record.get("manifest_id") != manifest.get("manifest_id"):
        raise ValidationError("manifest_id does not match the supplied manifest")
    expected_revision = manifest.get("sources", {}).get("huggingface", {}).get("revision")
    if record.get("manifest_revision") != expected_revision:
        raise ValidationError("manifest_revision does not match the supplied manifest")
    model_id = _require(record, "model_id")
    models = {model.get("id"): model for model in manifest.get("models", [])}
    model = models.get(model_id)
    if model is None:
        raise ValidationError(f"model_id {model_id!r} is absent from the manifest")
    if model.get("evaluation_status") == "excluded" and not allow_excluded:
        # Intentional MAE / unmatched evals: --allow-excluded or ALLOW_EXCLUDED=1.
        # Soft-warn otherwise so single-file checks can proceed after
        # evaluate_frozen --allow-excluded wrote metrics.
        if os.environ.get("ALLOW_EXCLUDED", "0") != "1":
            print(
                f"WARN: validating excluded model {model_id!r} "
                f"(pass --allow-excluded or ALLOW_EXCLUDED=1)",
                flush=True,
            )
    if record.get("architecture_id") != model.get("architecture_id"):
        raise ValidationError("architecture_id does not match the manifest")


def validate_record(
    path: Path,
    manifest: dict[str, Any],
    *,
    allow_excluded: bool = False,
) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise ValidationError("top-level JSON value must be an object")
    if record.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValidationError(
            f"schema_version must be {RESULT_SCHEMA_VERSION!r}, "
            f"got {record.get('schema_version')!r}"
        )
    task = _require(record, "task")
    if task not in {"transfer", "imagenet-c"}:
        raise ValidationError(f"unsupported task {task!r}")
    _require(record, "architecture_id")
    _require(record, "method")
    checkpoint = _require(record, "checkpoint")
    if not isinstance(checkpoint, dict) or not _require(checkpoint, "path"):
        raise ValidationError("checkpoint must contain a path")
    _validate_manifest_identity(record, manifest, allow_excluded=allow_excluded)

    if task == "transfer":
        dataset = _require(record, "dataset")
        if dataset not in DATASET_INFO:
            raise ValidationError(f"unsupported transfer dataset {dataset!r}")
        protocol = _require(record, "protocol")
        if not isinstance(protocol, dict):
            raise ValidationError("protocol must be an object")
        shots = protocol.get("shots_per_class")
        seeds = protocol.get("seeds")
        regularization = protocol.get("regularization")
        if not shots or not seeds or any(not isinstance(v, int) or v <= 0 for v in shots):
            raise ValidationError("protocol.shots_per_class must contain positive integers")
        if any(not isinstance(v, int) for v in seeds):
            raise ValidationError("protocol.seeds must contain integers")
        if not isinstance(regularization, (int, float)) or regularization <= 0:
            raise ValidationError("protocol.regularization must be positive")
        results = _require(record, "few_shot")
        if not isinstance(results, list):
            raise ValidationError("few_shot must be a list")
        if any(not isinstance(item, dict) for item in results):
            raise ValidationError("each few_shot record must be an object")
        expected = {(shot, seed) for shot in shots for seed in seeds}
        actual = {(item.get("shots_per_class"), item.get("seed")) for item in results}
        if actual != expected or len(results) != len(expected):
            raise ValidationError("few_shot records do not cover each shot/seed exactly once")
        classes = DATASET_INFO[dataset][0]
        for index, item in enumerate(results):
            _accuracy(item.get("accuracy"), f"few_shot[{index}].accuracy")
            if item.get("support_size") != item.get("shots_per_class") * classes:
                raise ValidationError(f"few_shot[{index}].support_size is inconsistent")
    else:
        for field in ("clean_accuracy", "clean_validation_accuracy", "mean_corruption_accuracy"):
            _accuracy(_require(record, field), field)
        _nonnegative_number(
            _require(record, "relative_corruption_error"), "relative_corruption_error"
        )
        per_corruption = _require(record, "per_corruption_accuracy")
        if not isinstance(per_corruption, dict) or len(per_corruption) != 15:
            raise ValidationError("per_corruption_accuracy must contain 15 corruptions")
        for name, accuracy in per_corruption.items():
            _accuracy(accuracy, f"per_corruption_accuracy[{name!r}]")
        protocol = _require(record, "protocol")
        if not isinstance(protocol, dict):
            raise ValidationError("protocol must be an object")
        if protocol.get("corruptions") != 15 or protocol.get("severities") != 5:
            raise ValidationError("ImageNet-C protocol must declare 15 corruptions and 5 severities")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model_manifest.json"))
    parser.add_argument(
        "--allow-excluded",
        action="store_true",
        help="Allow result records for models marked evaluation_status=excluded.",
    )
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    allow_excluded = bool(args.allow_excluded) or os.environ.get("ALLOW_EXCLUDED", "0") == "1"
    manifest = read_manifest(args.manifest.resolve())
    failures = []
    for path in args.results:
        try:
            validate_record(path, manifest, allow_excluded=allow_excluded)
            print(f"VALID {path}")
        except ValidationError as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        raise SystemExit("INVALID RESULT RECORDS\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
