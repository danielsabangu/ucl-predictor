from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .schemas import normalize_utc


RESULT_SCHEMA_VERSION = 1

VALID_FINAL_STATUSES = {"FINISHED"}
VALID_RESULTS = {"H", "D", "A"}


def _validate_score(
    name: str,
    value: int,
) -> None:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


@dataclass(frozen=True)
class ResultRecord:
    schema_version: int

    result_id: str
    source_match_id: str
    fixture_id: str

    home_team: str
    away_team: str
    kickoff_utc: str

    competition: str | None
    stage: str | None
    matchday: int | None

    status: str

    home_goals: int
    away_goals: int
    actual_result: str

    venue: str | None

    provider_source: str
    provider_last_updated_utc: str | None

    source_row_hash: str

    def validate(self) -> "ResultRecord":
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported result schema version: "
                f"{self.schema_version}"
            )

        required_strings = {
            "result_id": self.result_id,
            "source_match_id": self.source_match_id,
            "fixture_id": self.fixture_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "kickoff_utc": self.kickoff_utc,
            "status": self.status,
            "provider_source": self.provider_source,
            "source_row_hash": self.source_row_hash,
        }

        for name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} must be a non-empty string."
                )

        if self.home_team == self.away_team:
            raise ValueError(
                "Result home and away teams cannot be identical."
            )

        normalize_utc(self.kickoff_utc)

        if self.provider_last_updated_utc is not None:
            normalize_utc(
                self.provider_last_updated_utc
            )

        if self.status not in VALID_FINAL_STATUSES:
            raise ValueError(
                "Only verified FINISHED fixtures may enter "
                "the result ledger."
            )

        _validate_score(
            "home_goals",
            self.home_goals,
        )

        _validate_score(
            "away_goals",
            self.away_goals,
        )

        if self.actual_result not in VALID_RESULTS:
            raise ValueError(
                "actual_result must be H, D or A."
            )

        expected_result = (
            "H"
            if self.home_goals > self.away_goals
            else "A"
            if self.home_goals < self.away_goals
            else "D"
        )

        if self.actual_result != expected_result:
            raise ValueError(
                "actual_result does not agree with "
                "the final score."
            )

        if self.matchday is not None:
            if (
                not isinstance(self.matchday, int)
                or self.matchday < 0
            ):
                raise ValueError(
                    "matchday must be a non-negative integer."
                )

        if len(self.source_row_hash) != 64:
            raise ValueError(
                "source_row_hash must be a SHA-256 digest."
            )

        if not all(
            character in "0123456789abcdef"
            for character in self.source_row_hash.lower()
        ):
            raise ValueError(
                "source_row_hash must contain hexadecimal characters."
            )

        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "ResultRecord":
        record = cls(**payload)
        record.validate()
        return record