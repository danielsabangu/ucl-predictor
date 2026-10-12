from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .result_schemas import ResultRecord


class ImmutableResultConflict(RuntimeError):
    """
    Raised when an existing immutable result is encountered
    with different content.
    """


@dataclass(frozen=True)
class ResultArchiveWriteResult:
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


def load_results(
    path: str | Path,
) -> list[ResultRecord]:
    result_path = Path(path)

    if not result_path.exists():
        return []

    records: list[ResultRecord] = []

    with result_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
                record = ResultRecord.from_dict(
                    payload
                )
            except Exception as exc:
                raise ValueError(
                    "Invalid result-history record at "
                    f"{result_path}:{line_number}"
                ) from exc

            records.append(record)

    return records


def append_results(
    path: str | Path,
    records: Iterable[ResultRecord],
) -> ResultArchiveWriteResult:
    result_path = Path(path)

    incoming = list(records)

    for record in incoming:
        record.validate()

    existing = load_results(
        result_path
    )

    existing_by_result_id = {
        record.result_id: record
        for record in existing
    }

    existing_by_source_match_id = {
        record.source_match_id: record
        for record in existing
    }

    staged: list[dict] = []
    skipped = 0

    pending_result_ids = set(
        existing_by_result_id
    )

    pending_source_ids = set(
        existing_by_source_match_id
    )

    for record in incoming:
        payload = record.to_dict()

        existing_result = (
            existing_by_result_id.get(
                record.result_id
            )
        )

        if existing_result is not None:
            if (
                _canonical_json(
                    existing_result.to_dict()
                )
                != _canonical_json(payload)
            ):
                raise ImmutableResultConflict(
                    "Result ID already exists with "
                    "different content: "
                    f"{record.result_id}"
                )

            skipped += 1
            continue

        existing_source = (
            existing_by_source_match_id.get(
                record.source_match_id
            )
        )

        if existing_source is not None:
            if (
                _canonical_json(
                    existing_source.to_dict()
                )
                != _canonical_json(payload)
            ):
                raise ImmutableResultConflict(
                    "Provider match ID already exists with "
                    "different result content: "
                    f"{record.source_match_id}"
                )

            skipped += 1
            continue

        if record.result_id in pending_result_ids:
            raise ImmutableResultConflict(
                "Duplicate result ID inside incoming batch: "
                f"{record.result_id}"
            )

        if (
            record.source_match_id
            in pending_source_ids
        ):
            raise ImmutableResultConflict(
                "Duplicate provider match ID inside "
                "incoming batch: "
                f"{record.source_match_id}"
            )

        staged.append(payload)

        pending_result_ids.add(
            record.result_id
        )

        pending_source_ids.add(
            record.source_match_id
        )

    if staged:
        result_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        needs_separator = False

        if (
            result_path.exists()
            and result_path.stat().st_size > 0
        ):
            with result_path.open(
                "rb"
            ) as existing_handle:
                existing_handle.seek(-1, 2)

                needs_separator = (
                    existing_handle.read(1)
                    != b"\n"
                )

        with result_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            if needs_separator:
                handle.write("\n")

            for payload in staged:
                handle.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                handle.write("\n")

    return ResultArchiveWriteResult(
        imported=len(staged),
        skipped=skipped,
        total=len(existing) + len(staged),
    )