from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .schemas import PredictionRecord


class ImmutableArchiveConflict(RuntimeError):
    """
    Raised when an existing immutable prediction ID or revision
    is encountered with different content.
    """


@dataclass(frozen=True)
class ArchiveWriteResult:
    imported: int
    skipped: int
    total: int


def _canonical_json(payload: dict) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load_records(path: str | Path) -> list[PredictionRecord]:
    archive_path = Path(path)

    if not archive_path.exists():
        return []

    records: list[PredictionRecord] = []

    with archive_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
                record = PredictionRecord.from_dict(payload)
            except Exception as exc:
                raise ValueError(
                    f"Invalid prediction-history record at "
                    f"{archive_path}:{line_number}"
                ) from exc

            records.append(record)

    return records


def append_records(
    path: str | Path,
    records: Iterable[PredictionRecord],
) -> ArchiveWriteResult:
    archive_path = Path(path)
    incoming = list(records)

    for record in incoming:
        record.validate()

    existing = load_records(archive_path)

    existing_by_prediction_id = {
        record.prediction_id: record
        for record in existing
    }

    existing_by_fixture_revision = {
        (record.fixture_id, record.revision): record
        for record in existing
    }

    staged_payloads: list[dict] = []
    skipped = 0

    pending_prediction_ids = set(existing_by_prediction_id)
    pending_fixture_revisions = set(existing_by_fixture_revision)

    for record in incoming:
        payload = record.to_dict()

        existing_prediction = existing_by_prediction_id.get(
            record.prediction_id
        )

        if existing_prediction is not None:
            existing_payload = existing_prediction.to_dict()

            if _canonical_json(existing_payload) != _canonical_json(payload):
                raise ImmutableArchiveConflict(
                    "Prediction ID already exists with different content: "
                    f"{record.prediction_id}"
                )

            skipped += 1
            continue

        fixture_revision_key = (
            record.fixture_id,
            record.revision,
        )

        existing_revision = existing_by_fixture_revision.get(
            fixture_revision_key
        )

        if existing_revision is not None:
            if (
                _canonical_json(existing_revision.to_dict())
                != _canonical_json(payload)
            ):
                raise ImmutableArchiveConflict(
                    "Fixture revision already exists with different content: "
                    f"{record.fixture_id} revision {record.revision}"
                )

            skipped += 1
            continue

        if record.prediction_id in pending_prediction_ids:
            raise ImmutableArchiveConflict(
                "Duplicate prediction ID inside incoming batch: "
                f"{record.prediction_id}"
            )

        if fixture_revision_key in pending_fixture_revisions:
            raise ImmutableArchiveConflict(
                "Duplicate fixture revision inside incoming batch: "
                f"{record.fixture_id} revision {record.revision}"
            )

        staged_payloads.append(payload)

        pending_prediction_ids.add(record.prediction_id)
        pending_fixture_revisions.add(fixture_revision_key)

    if staged_payloads:
        archive_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        needs_separator = False

        if archive_path.exists() and archive_path.stat().st_size > 0:
            with archive_path.open("rb") as existing_handle:
                existing_handle.seek(-1, 2)
                needs_separator = existing_handle.read(1) != b"\n"

        with archive_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            if needs_separator:
                handle.write("\n")

            for payload in staged_payloads:
                handle.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                handle.write("\n")

    return ArchiveWriteResult(
        imported=len(staged_payloads),
        skipped=skipped,
        total=len(existing) + len(staged_payloads),
    )