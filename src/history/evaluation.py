from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .archive import load_records
from .results_archive import load_results


DEFAULT_PREDICTIONS = Path(
    "data/app/history/predictions.jsonl"
)

DEFAULT_RESULTS = Path(
    "data/app/history/results.jsonl"
)

DEFAULT_EVALUATIONS = Path(
    "data/app/history/evaluations.jsonl"
)

DEFAULT_SUMMARY = Path(
    "data/app/history/summary.json"
)


EVALUATION_SCHEMA_VERSION = 1

VALID_RESULTS = {"H", "D", "A"}


@dataclass(frozen=True)
class PredictionEvaluation:
    schema_version: int

    prediction_id: str
    result_id: str
    fixture_id: str

    revision: int

    home_team: str
    away_team: str
    kickoff_utc: str
    published_at_utc: str

    model_version: str
    prediction_type: str

    predicted_result: str
    actual_result: str

    home_probability: float
    draw_probability: float
    away_probability: float

    actual_result_probability: float

    result_pick_correct: bool

    predicted_scoreline: str | None
    actual_scoreline: str
    exact_score_correct: bool | None

    multiclass_brier: float
    log_loss: float

    evidence_score: float | None
    evidence_label: str | None
    overall_feature_availability: float | None

    def validate(
        self,
    ) -> "PredictionEvaluation":
        if (
            self.schema_version
            != EVALUATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported evaluation schema version."
            )

        if self.predicted_result not in VALID_RESULTS:
            raise ValueError(
                "predicted_result must be H, D or A."
            )

        if self.actual_result not in VALID_RESULTS:
            raise ValueError(
                "actual_result must be H, D or A."
            )

        if not 0.0 <= self.actual_result_probability <= 1.0:
            raise ValueError(
                "actual_result_probability must be "
                "between 0 and 1."
            )

        if not math.isfinite(
            self.multiclass_brier
        ):
            raise ValueError(
                "multiclass_brier must be finite."
            )

        if not 0.0 <= self.multiclass_brier <= 2.0:
            raise ValueError(
                "Multiclass Brier score must fall "
                "between 0 and 2."
            )

        if not math.isfinite(
            self.log_loss
        ):
            raise ValueError(
                "log_loss must be finite."
            )

        if self.log_loss < 0.0:
            raise ValueError(
                "log_loss cannot be negative."
            )

        return self

    def to_dict(
        self,
    ) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _actual_vector(
    actual_result: str,
) -> dict[str, float]:
    if actual_result not in VALID_RESULTS:
        raise ValueError(
            f"Invalid result label: {actual_result}"
        )

    return {
        label: 1.0 if label == actual_result else 0.0
        for label in ("H", "D", "A")
    }


def multiclass_brier_score(
    *,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
    actual_result: str,
) -> float:
    """
    Multiclass Brier score using the sum of squared
    probability errors across H/D/A.

    Range: 0.0 to 2.0.
    Lower is better.
    """

    probabilities = {
        "H": float(home_probability),
        "D": float(draw_probability),
        "A": float(away_probability),
    }

    actual = _actual_vector(
        actual_result
    )

    return sum(
        (
            probabilities[label]
            - actual[label]
        ) ** 2
        for label in ("H", "D", "A")
    )


def multiclass_log_loss(
    *,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
    actual_result: str,
) -> float:
    probabilities = {
        "H": float(home_probability),
        "D": float(draw_probability),
        "A": float(away_probability),
    }

    probability = probabilities[
        actual_result
    ]

    epsilon = 1e-15

    clipped = min(
        max(probability, epsilon),
        1.0 - epsilon,
    )

    return -math.log(
        clipped
    )


def _evaluate_prediction(
    prediction,
    result,
) -> PredictionEvaluation:
    probabilities = {
        "H": float(
            prediction.home_probability
        ),
        "D": float(
            prediction.draw_probability
        ),
        "A": float(
            prediction.away_probability
        ),
    }

    actual_probability = probabilities[
        result.actual_result
    ]

    actual_scoreline = (
        f"{result.home_goals}-"
        f"{result.away_goals}"
    )

    if prediction.predicted_scoreline:
        exact_score_correct = (
            prediction.predicted_scoreline
            == actual_scoreline
        )
    else:
        exact_score_correct = None

    evaluation = PredictionEvaluation(
        schema_version=(
            EVALUATION_SCHEMA_VERSION
        ),

        prediction_id=(
            prediction.prediction_id
        ),

        result_id=result.result_id,

        fixture_id=(
            prediction.fixture_id
        ),

        revision=prediction.revision,

        home_team=prediction.home_team,
        away_team=prediction.away_team,

        kickoff_utc=(
            prediction.kickoff_utc
        ),

        published_at_utc=(
            prediction.published_at_utc
        ),

        model_version=(
            prediction.model_version
        ),

        prediction_type=(
            prediction.prediction_type
        ),

        predicted_result=(
            prediction.predicted_result
        ),

        actual_result=(
            result.actual_result
        ),

        home_probability=(
            float(
                prediction.home_probability
            )
        ),

        draw_probability=(
            float(
                prediction.draw_probability
            )
        ),

        away_probability=(
            float(
                prediction.away_probability
            )
        ),

        actual_result_probability=(
            actual_probability
        ),

        result_pick_correct=(
            prediction.predicted_result
            == result.actual_result
        ),

        predicted_scoreline=(
            prediction.predicted_scoreline
        ),

        actual_scoreline=(
            actual_scoreline
        ),

        exact_score_correct=(
            exact_score_correct
        ),

        multiclass_brier=(
            multiclass_brier_score(
                home_probability=(
                    prediction.home_probability
                ),
                draw_probability=(
                    prediction.draw_probability
                ),
                away_probability=(
                    prediction.away_probability
                ),
                actual_result=(
                    result.actual_result
                ),
            )
        ),

        log_loss=(
            multiclass_log_loss(
                home_probability=(
                    prediction.home_probability
                ),
                draw_probability=(
                    prediction.draw_probability
                ),
                away_probability=(
                    prediction.away_probability
                ),
                actual_result=(
                    result.actual_result
                ),
            )
        ),

        evidence_score=(
            prediction.evidence_score
        ),

        evidence_label=(
            prediction.evidence_label
        ),

        overall_feature_availability=(
            prediction.overall_feature_availability
        ),
    )

    evaluation.validate()

    return evaluation


def build_evaluations(
    *,
    predictions_path: str | Path = (
        DEFAULT_PREDICTIONS
    ),
    results_path: str | Path = (
        DEFAULT_RESULTS
    ),
) -> list[PredictionEvaluation]:
    predictions = load_records(
        predictions_path
    )

    results = load_results(
        results_path
    )

    results_by_fixture = {
        result.fixture_id: result
        for result in results
    }

    evaluations: list[
        PredictionEvaluation
    ] = []

    for prediction in predictions:
        result = results_by_fixture.get(
            prediction.fixture_id
        )

        if result is None:
            continue

        evaluations.append(
            _evaluate_prediction(
                prediction,
                result,
            )
        )

    return evaluations


def latest_revision_per_fixture(
    evaluations: list[
        PredictionEvaluation
    ],
) -> list[PredictionEvaluation]:
    latest: dict[
        str,
        PredictionEvaluation,
    ] = {}

    for evaluation in evaluations:
        existing = latest.get(
            evaluation.fixture_id
        )

        if (
            existing is None
            or evaluation.revision
            > existing.revision
        ):
            latest[
                evaluation.fixture_id
            ] = evaluation

    return sorted(
        latest.values(),
        key=lambda item: (
            item.kickoff_utc,
            item.fixture_id,
        ),
    )


def _safe_mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(
        mean(values)
    )


def build_summary(
    evaluations: list[
        PredictionEvaluation
    ],
) -> dict[str, Any]:
    headline = (
        latest_revision_per_fixture(
            evaluations
        )
    )

    resolved = len(headline)

    correct = sum(
        evaluation.result_pick_correct
        for evaluation in headline
    )

    exact_eligible = [
        evaluation
        for evaluation in headline
        if (
            evaluation.exact_score_correct
            is not None
        )
    ]

    exact_correct = sum(
        evaluation.exact_score_correct
        for evaluation in exact_eligible
    )

    by_result = {}

    for result_label in (
        "H",
        "D",
        "A",
    ):
        subset = [
            evaluation
            for evaluation in headline
            if (
                evaluation.actual_result
                == result_label
            )
        ]

        by_result[result_label] = {
            "sample_size": len(
                subset
            ),
            "correct_picks": sum(
                evaluation.result_pick_correct
                for evaluation in subset
            ),
            "accuracy": (
                (
                    sum(
                        evaluation.result_pick_correct
                        for evaluation in subset
                    )
                    / len(subset)
                )
                if subset
                else None
            ),
        }

    model_versions = sorted(
        {
            evaluation.model_version
            for evaluation in headline
        }
    )

    return {
        "evaluation_schema_version": (
            EVALUATION_SCHEMA_VERSION
        ),

        "headline_population": (
            "LATEST_PREMATCH_REVISION_PER_FIXTURE"
        ),

        "all_published_revision_count": (
            len(evaluations)
        ),

        "resolved_fixture_count": (
            resolved
        ),

        "correct_result_picks": (
            correct
        ),

        "result_pick_accuracy": (
            correct / resolved
            if resolved
            else None
        ),

        "exact_score_eligible": (
            len(exact_eligible)
        ),

        "exact_scores_correct": (
            exact_correct
        ),

        "exact_score_accuracy": (
            (
                exact_correct
                / len(exact_eligible)
            )
            if exact_eligible
            else None
        ),

        "mean_multiclass_brier": (
            _safe_mean(
                [
                    evaluation.multiclass_brier
                    for evaluation in headline
                ]
            )
        ),

        "mean_log_loss": (
            _safe_mean(
                [
                    evaluation.log_loss
                    for evaluation in headline
                ]
            )
        ),

        "mean_actual_result_probability": (
            _safe_mean(
                [
                    evaluation.actual_result_probability
                    for evaluation in headline
                ]
            )
        ),

        "performance_by_actual_result": (
            by_result
        ),

        "model_versions": (
            model_versions
        ),

        "sample_warning": (
            "INSUFFICIENT_SAMPLE_FOR_RELIABLE_"
            "PERFORMANCE_CLAIMS"
            if resolved < 30
            else None
        ),

        "metric_definitions": {
            "result_pick_accuracy": (
                "Highest-probability H/D/A selection "
                "equals the verified final match result."
            ),

            "multiclass_brier": (
                "Sum of squared probability errors "
                "across H/D/A. Range 0 to 2. "
                "Lower is better."
            ),

            "log_loss": (
                "Negative natural logarithm of the "
                "probability assigned to the actual "
                "H/D/A result. Lower is better."
            ),

            "exact_score_accuracy": (
                "Predicted regulation/final scoreline "
                "exactly equals the stored final scoreline."
            ),
        },
    }


def write_evaluation_artifacts(
    *,
    evaluations_path: str | Path = (
        DEFAULT_EVALUATIONS
    ),
    summary_path: str | Path = (
        DEFAULT_SUMMARY
    ),
    predictions_path: str | Path = (
        DEFAULT_PREDICTIONS
    ),
    results_path: str | Path = (
        DEFAULT_RESULTS
    ),
) -> tuple[
    list[PredictionEvaluation],
    dict[str, Any],
]:
    evaluations = build_evaluations(
        predictions_path=(
            predictions_path
        ),
        results_path=(
            results_path
        ),
    )

    summary = build_summary(
        evaluations
    )

    evaluations_path = Path(
        evaluations_path
    )

    summary_path = Path(
        summary_path
    )

    evaluations_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with evaluations_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for evaluation in evaluations:
            handle.write(
                json.dumps(
                    evaluation.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )

            handle.write("\n")

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )

        handle.write("\n")

    return (
        evaluations,
        summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate immutable UCL predictions "
            "against verified result records."
        )
    )

    parser.add_argument(
        "--predictions",
        default=str(
            DEFAULT_PREDICTIONS
        ),
    )

    parser.add_argument(
        "--results",
        default=str(
            DEFAULT_RESULTS
        ),
    )

    parser.add_argument(
        "--evaluations",
        default=str(
            DEFAULT_EVALUATIONS
        ),
    )

    parser.add_argument(
        "--summary",
        default=str(
            DEFAULT_SUMMARY
        ),
    )

    args = parser.parse_args()

    evaluations, summary = (
        write_evaluation_artifacts(
            evaluations_path=(
                args.evaluations
            ),
            summary_path=(
                args.summary
            ),
            predictions_path=(
                args.predictions
            ),
            results_path=(
                args.results
            ),
        )
    )

    print(
        "Prediction evaluation complete."
    )

    print(
        f"Published revisions evaluated: "
        f"{len(evaluations)}"
    )

    print(
        f"Resolved fixtures: "
        f"{summary['resolved_fixture_count']}"
    )

    print(
        f"Correct result picks: "
        f"{summary['correct_result_picks']}"
    )

    accuracy = (
        summary[
            "result_pick_accuracy"
        ]
    )

    if accuracy is not None:
        print(
            f"Result-pick accuracy: "
            f"{accuracy:.2%}"
        )

    exact_accuracy = (
        summary[
            "exact_score_accuracy"
        ]
    )

    if exact_accuracy is not None:
        print(
            f"Exact-score accuracy: "
            f"{exact_accuracy:.2%}"
        )

    brier = summary[
        "mean_multiclass_brier"
    ]

    if brier is not None:
        print(
            f"Mean multiclass Brier: "
            f"{brier:.6f}"
        )

    log_loss = summary[
        "mean_log_loss"
    ]

    if log_loss is not None:
        print(
            f"Mean log loss: "
            f"{log_loss:.6f}"
        )

    if summary["sample_warning"]:
        print(
            "WARNING: Sample is too small "
            "for reliable performance claims."
        )


if __name__ == "__main__":
    main()