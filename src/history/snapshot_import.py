from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Any

from .archive import append_records, load_records
from .schemas import (
    SCHEMA_VERSION,
    PredictionRecord,
    normalize_utc,
    parse_utc,
)


DEFAULT_APP_DIR = Path("data/app")
DEFAULT_HISTORY_PATH = (
    DEFAULT_APP_DIR
    / "history"
    / "predictions.jsonl"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _canonical_team(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", value)
        .strip()
        .casefold()
        .split()
    )


def _derive_fixture_id(
    home_team: str,
    away_team: str,
    kickoff_utc: str,
) -> str:
    canonical = "|".join(
        [
            normalize_utc(kickoff_utc),
            _canonical_team(home_team),
            _canonical_team(away_team),
        ]
    )

    return "fx_" + _sha256_text(canonical)[:20]


def _derive_prediction_id(
    fixture_id: str,
    source_run_id: str,
    model_version: str,
) -> str:
    canonical = "|".join(
        [
            fixture_id,
            source_run_id.strip(),
            model_version.strip(),
        ]
    )

    return "pred_" + _sha256_text(canonical)[:24]


def _optional_string(
    row: dict[str, str],
    key: str,
) -> str | None:
    value = row.get(key)

    if value is None:
        return None

    stripped = str(value).strip()

    return stripped if stripped else None


def _required_string(
    row: dict[str, str],
    key: str,
) -> str:
    value = _optional_string(row, key)

    if value is None:
        raise ValueError(
            f"Required prediction column is missing or blank: {key}"
        )

    return value


def _optional_float(
    row: dict[str, str],
    key: str,
) -> float | None:
    value = _optional_string(row, key)

    if value is None:
        return None

    return float(value)


def _required_float(
    row: dict[str, str],
    key: str,
) -> float:
    value = _optional_float(row, key)

    if value is None:
        raise ValueError(
            f"Required prediction column is missing or blank: {key}"
        )

    return value


def _optional_int(
    row: dict[str, str],
    key: str,
) -> int | None:
    value = _optional_string(row, key)

    if value is None:
        return None

    numeric = float(value)

    if not numeric.is_integer():
        raise ValueError(
            f"{key} must contain an integer-compatible value."
        )

    return int(numeric)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in {path}"
        )

    return payload


def _load_prediction_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"Prediction CSV has no header: {path}"
            )

        return list(reader)


def _snapshot_row_hash(
    row: dict[str, str],
    *,
    source_run_id: str,
    created_at_utc: str,
    model_version: str,
) -> str:
    payload = {
        "source_run_id": source_run_id,
        "created_at_utc": created_at_utc,
        "model_version": model_version,
        "row": row,
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return _sha256_text(canonical)


def import_public_snapshot(
    app_dir: str | Path = DEFAULT_APP_DIR,
    archive_path: str | Path | None = None,
):
    app_dir = Path(app_dir)

    predictions_path = (
        app_dir
        / "current_predictions.csv"
    )

    metadata_path = (
        app_dir
        / "snapshot_metadata.json"
    )

    status_path = (
        app_dir
        / "pipeline_status.json"
    )

    if archive_path is None:
        archive_path = (
            app_dir
            / "history"
            / "predictions.jsonl"
        )

    archive_path = Path(archive_path)

    for required_path in (
        predictions_path,
        metadata_path,
        status_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required public snapshot file is missing: "
                f"{required_path}"
            )

    metadata = _load_json(metadata_path)
    status = _load_json(status_path)
    rows = _load_prediction_rows(predictions_path)

    if status.get("status") != "SUCCESS":
        raise ValueError(
            "Public pipeline status is not SUCCESS."
        )

    if status.get("production_ready") is not True:
        raise ValueError(
            "Public pipeline status is not production-ready."
        )

    source_run_id = str(
        metadata.get("source_run_id", "")
    ).strip()

    status_run_id = str(
        status.get("run_id", "")
    ).strip()

    if not source_run_id:
        raise ValueError(
            "snapshot_metadata.json is missing source_run_id."
        )

    if (
        status_run_id
        and source_run_id != status_run_id
    ):
        raise ValueError(
            "Snapshot source_run_id does not match "
            "pipeline status run_id."
        )

    model_version = str(
        metadata.get("model_version", "")
    ).strip()

    if not model_version:
        raise ValueError(
            "snapshot_metadata.json is missing model_version."
        )

    created_at_raw = str(
        metadata.get("created_at_utc", "")
    ).strip()

    if not created_at_raw:
        raise ValueError(
            "snapshot_metadata.json is missing created_at_utc."
        )

    created_at_utc = normalize_utc(
        created_at_raw
    )

    expected_fixture_count = metadata.get(
        "fixture_count"
    )

    if (
        expected_fixture_count is not None
        and int(expected_fixture_count) != len(rows)
    ):
        raise ValueError(
            "Snapshot fixture_count does not match "
            "current_predictions.csv."
        )

    existing_records = load_records(
        archive_path
    )

    existing_by_prediction_id = {
        record.prediction_id: record
        for record in existing_records
    }

    max_revision_by_fixture: dict[str, int] = {}

    for record in existing_records:
        max_revision_by_fixture[record.fixture_id] = max(
            record.revision,
            max_revision_by_fixture.get(
                record.fixture_id,
                0,
            ),
        )

    data_cutoff_raw = metadata.get(
        "data_cutoff_utc"
    )

    data_cutoff_utc = (
        normalize_utc(str(data_cutoff_raw))
        if data_cutoff_raw
        else None
    )

    dataset_version_raw = metadata.get(
        "dataset_version"
    )

    dataset_version = (
        str(dataset_version_raw).strip()
        if dataset_version_raw
        else None
    )

    records: list[PredictionRecord] = []

    for row in rows:
        home_team = _required_string(
            row,
            "home_team",
        )

        away_team = _required_string(
            row,
            "away_team",
        )

        kickoff_utc = normalize_utc(
            _required_string(
                row,
                "kickoff_utc",
            )
        )

        if parse_utc(created_at_utc) >= parse_utc(
            kickoff_utc
        ):
            raise ValueError(
                "Cannot archive a public snapshot as a "
                "LIVE_FORWARD_TEST because it was generated "
                "at or after kickoff: "
                f"{home_team} vs {away_team}"
            )

        fixture_id = _derive_fixture_id(
            home_team=home_team,
            away_team=away_team,
            kickoff_utc=kickoff_utc,
        )

        prediction_id = _derive_prediction_id(
            fixture_id=fixture_id,
            source_run_id=source_run_id,
            model_version=model_version,
        )

        existing_prediction = (
            existing_by_prediction_id.get(
                prediction_id
            )
        )

        if existing_prediction is not None:
            revision = existing_prediction.revision
        else:
            revision = (
                max_revision_by_fixture.get(
                    fixture_id,
                    0,
                )
                + 1
            )

            max_revision_by_fixture[
                fixture_id
            ] = revision

        record = PredictionRecord(
            schema_version=SCHEMA_VERSION,

            prediction_id=prediction_id,
            fixture_id=fixture_id,
            fixture_id_source=(
                "derived:sha256("
                "kickoff_utc|home_team|away_team)"
            ),

            revision=revision,
            prediction_type="LIVE_FORWARD_TEST",
            prediction_status="PUBLISHED",
            record_origin="PUBLIC_SNAPSHOT_IMPORT",

            home_team=home_team,
            away_team=away_team,
            kickoff_utc=kickoff_utc,
            published_at_utc=created_at_utc,

            model_version=model_version,
            source_run_id=source_run_id,

            data_cutoff_utc=data_cutoff_utc,
            dataset_version=dataset_version,

            home_probability=_required_float(
                row,
                "home_probability",
            ),
            draw_probability=_required_float(
                row,
                "draw_probability",
            ),
            away_probability=_required_float(
                row,
                "away_probability",
            ),

            predicted_result=_required_string(
                row,
                "predicted_result",
            ),
            predicted_selection=_required_string(
                row,
                "predicted_selection",
            ),

            max_probability=_optional_float(
                row,
                "max_probability",
            ),

            evidence_score=_optional_float(
                row,
                "confidence",
            ),
            evidence_label=_optional_string(
                row,
                "confidence_label",
            ),
            evidence_caps=_optional_string(
                row,
                "confidence_caps",
            ),

            model_top_two_margin=_optional_float(
                row,
                "model_top_two_margin",
            ),
            model_decisiveness_profile=_optional_string(
                row,
                "model_decisiveness_profile",
            ),

            elo_features_available=_optional_int(
                row,
                "elo_features_available",
            ),
            ucl_form_features_available=_optional_int(
                row,
                "ucl_form_features_available",
            ),
            domestic_features_available=_optional_int(
                row,
                "domestic_features_available",
            ),
            squad_features_available=_optional_int(
                row,
                "squad_features_available",
            ),

            overall_features_available=_optional_int(
                row,
                "overall_features_available",
            ),
            overall_feature_availability=_optional_float(
                row,
                "overall_feature_availability",
            ),

            predicted_home_goals=_optional_int(
                row,
                "predicted_home_goals",
            ),
            predicted_away_goals=_optional_int(
                row,
                "predicted_away_goals",
            ),
            predicted_scoreline=_optional_string(
                row,
                "predicted_scoreline",
            ),
            predicted_scoreline_probability=_optional_float(
                row,
                "predicted_scoreline_probability",
            ),

            global_modal_scoreline=_optional_string(
                row,
                "global_modal_scoreline",
            ),
            global_modal_result=_optional_string(
                row,
                "global_modal_result",
            ),

            score_home_lambda=_optional_float(
                row,
                "score_home_lambda",
            ),
            score_away_lambda=_optional_float(
                row,
                "score_away_lambda",
            ),
            score_total_lambda=_optional_float(
                row,
                "score_total_lambda",
            ),

            score_bridge_max_error_pp=_optional_float(
                row,
                "score_bridge_max_error_pp",
            ),
            score_bridge_quality=_optional_string(
                row,
                "score_bridge_quality",
            ),

            source_snapshot_hash=_snapshot_row_hash(
                row,
                source_run_id=source_run_id,
                created_at_utc=created_at_utc,
                model_version=model_version,
            ),
        )

        record.validate()
        records.append(record)

    return append_records(
        archive_path,
        records,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import the sanitized public prediction snapshot "
            "into the immutable prediction-history ledger."
        )
    )

    parser.add_argument(
        "--app-dir",
        default=str(DEFAULT_APP_DIR),
        help="Public app snapshot directory.",
    )

    parser.add_argument(
        "--archive",
        default=None,
        help=(
            "Optional prediction-history JSONL path. "
            "Defaults to <app-dir>/history/predictions.jsonl."
        ),
    )

    args = parser.parse_args()

    result = import_public_snapshot(
        app_dir=args.app_dir,
        archive_path=args.archive,
    )

    print("Prediction history import complete.")
    print(f"Imported: {result.imported}")
    print(f"Skipped:  {result.skipped}")
    print(f"Total:    {result.total}")


if __name__ == "__main__":
    main()