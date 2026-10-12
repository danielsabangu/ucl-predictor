from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any


SCHEMA_VERSION = 1

VALID_PREDICTION_TYPES = {
    "LIVE_FORWARD_TEST",
    "HISTORICAL_BACKTEST",
    "SIMULATION",
    "UNKNOWN",
}

VALID_PREDICTION_STATUSES = {
    "PUBLISHED",
}

VALID_RESULTS = {"H", "D", "A"}


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("UTC timestamp must be a non-empty string.")

    candidate = value.strip()

    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO timestamp: {value}") from exc

    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must be timezone-aware: {value}")

    return parsed.astimezone(timezone.utc)


def normalize_utc(value: str) -> str:
    return parse_utc(value).isoformat().replace("+00:00", "Z")


def _validate_probability(name: str, value: float | None) -> None:
    if value is None:
        return

    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _validate_optional_non_negative_int(
    name: str,
    value: int | None,
) -> None:
    if value is None:
        return

    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


@dataclass(frozen=True)
class PredictionRecord:
    schema_version: int

    prediction_id: str
    fixture_id: str
    fixture_id_source: str

    revision: int
    prediction_type: str
    prediction_status: str
    record_origin: str

    home_team: str
    away_team: str
    kickoff_utc: str
    published_at_utc: str

    model_version: str
    source_run_id: str

    data_cutoff_utc: str | None
    dataset_version: str | None

    home_probability: float
    draw_probability: float
    away_probability: float

    predicted_result: str
    predicted_selection: str

    max_probability: float | None

    evidence_score: float | None
    evidence_label: str | None
    evidence_caps: str | None

    model_top_two_margin: float | None
    model_decisiveness_profile: str | None

    elo_features_available: int | None
    ucl_form_features_available: int | None
    domestic_features_available: int | None
    squad_features_available: int | None

    overall_features_available: int | None
    overall_feature_availability: float | None

    predicted_home_goals: int | None
    predicted_away_goals: int | None
    predicted_scoreline: str | None
    predicted_scoreline_probability: float | None

    global_modal_scoreline: str | None
    global_modal_result: str | None

    score_home_lambda: float | None
    score_away_lambda: float | None
    score_total_lambda: float | None

    score_bridge_max_error_pp: float | None
    score_bridge_quality: str | None

    source_snapshot_hash: str

    def validate(self) -> "PredictionRecord":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported prediction history schema version: "
                f"{self.schema_version}"
            )

        required_strings = {
            "prediction_id": self.prediction_id,
            "fixture_id": self.fixture_id,
            "fixture_id_source": self.fixture_id_source,
            "record_origin": self.record_origin,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "model_version": self.model_version,
            "source_run_id": self.source_run_id,
            "predicted_selection": self.predicted_selection,
            "source_snapshot_hash": self.source_snapshot_hash,
        }

        for name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

        if self.home_team == self.away_team:
            raise ValueError("Home and away team cannot be identical.")

        if self.revision < 1:
            raise ValueError("Prediction revision must be at least 1.")

        if self.prediction_type not in VALID_PREDICTION_TYPES:
            raise ValueError(
                f"Unsupported prediction_type: {self.prediction_type}"
            )

        if self.prediction_status not in VALID_PREDICTION_STATUSES:
            raise ValueError(
                f"Unsupported prediction_status: {self.prediction_status}"
            )

        if self.predicted_result not in VALID_RESULTS:
            raise ValueError(
                f"predicted_result must be one of {sorted(VALID_RESULTS)}"
            )

        kickoff = parse_utc(self.kickoff_utc)
        published = parse_utc(self.published_at_utc)

        if self.data_cutoff_utc is not None:
            parse_utc(self.data_cutoff_utc)

        if self.prediction_type == "LIVE_FORWARD_TEST":
            if published >= kickoff:
                raise ValueError(
                    "LIVE_FORWARD_TEST predictions must be published "
                    "strictly before kickoff."
                )

        probabilities = {
            "H": float(self.home_probability),
            "D": float(self.draw_probability),
            "A": float(self.away_probability),
        }

        for name, value in (
            ("home_probability", self.home_probability),
            ("draw_probability", self.draw_probability),
            ("away_probability", self.away_probability),
        ):
            _validate_probability(name, value)

        probability_sum = sum(probabilities.values())

        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Prediction probabilities must sum to 1. "
                f"Observed: {probability_sum:.12f}"
            )

        highest_probability = max(probabilities.values())

        if (
            probabilities[self.predicted_result]
            + 1e-12
            < highest_probability
        ):
            raise ValueError(
                "predicted_result does not correspond to the "
                "highest model probability."
            )

        if self.max_probability is not None:
            _validate_probability(
                "max_probability",
                self.max_probability,
            )

            if not math.isclose(
                float(self.max_probability),
                highest_probability,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "max_probability does not match the probability "
                    "distribution."
                )

        _validate_probability("evidence_score", self.evidence_score)
        _validate_probability(
            "overall_feature_availability",
            self.overall_feature_availability,
        )
        _validate_probability(
            "predicted_scoreline_probability",
            self.predicted_scoreline_probability,
        )

        for name, value in (
            ("elo_features_available", self.elo_features_available),
            (
                "ucl_form_features_available",
                self.ucl_form_features_available,
            ),
            (
                "domestic_features_available",
                self.domestic_features_available,
            ),
            (
                "squad_features_available",
                self.squad_features_available,
            ),
            (
                "overall_features_available",
                self.overall_features_available,
            ),
            ("predicted_home_goals", self.predicted_home_goals),
            ("predicted_away_goals", self.predicted_away_goals),
        ):
            _validate_optional_non_negative_int(name, value)

        if len(self.source_snapshot_hash) != 64:
            raise ValueError(
                "source_snapshot_hash must be a SHA-256 hexadecimal digest."
            )

        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "PredictionRecord":
        record = cls(**payload)
        record.validate()
        return record