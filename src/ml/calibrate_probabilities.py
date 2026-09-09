from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src.config import (
    REPORTS_DIR,
    MODEL_METADATA_DIR,
)


# ============================================================
# PATHS
# ============================================================

MODEL_REPORT_DIR = (
    REPORTS_DIR
    / "model_reports"
)

METRICS_DIR = (
    REPORTS_DIR
    / "metrics"
)

BOOSTED_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "boosted_rolling_backtest_predictions.csv"
)

CALIBRATION_FOLD_FILE = (
    METRICS_DIR
    / "catboost_calibration_fold_metrics.csv"
)

CALIBRATION_SUMMARY_FILE = (
    METRICS_DIR
    / "catboost_calibration_summary.csv"
)

CALIBRATED_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "catboost_calibrated_oof_predictions.csv"
)

TEMPERATURE_SEARCH_FILE = (
    METRICS_DIR
    / "catboost_temperature_search.csv"
)

CALIBRATION_METADATA_FILE = (
    MODEL_METADATA_DIR
    / "catboost_calibration.json"
)


# ============================================================
# TARGET
# ============================================================

TARGET_MAP = {
    "H": 0,
    "D": 1,
    "A": 2,
}

TARGET_NAMES = {
    0: "H",
    1: "D",
    2: "A",
}

TARGET_LABELS = [
    0,
    1,
    2,
]


# ============================================================
# CONSTANTS
# ============================================================

EPSILON = 1e-12

MIN_TEMPERATURE = 0.25

MAX_TEMPERATURE = 4.0


# ============================================================
# FILE CHECK
# ============================================================


def require_file(
    path,
    description: str,
) -> None:
    """
    Fail immediately if a required OOF file is unavailable.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}"
        )

    if path.stat().st_size == 0:

        raise RuntimeError(
            f"{description} is empty:\n"
            f"{path}"
        )


# ============================================================
# NORMALIZE
# ============================================================


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Guarantee every probability row sums exactly to one.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0,
    )

    row_sums = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    if (
        row_sums
        <=
        0
    ).any():

        raise RuntimeError(
            "Invalid zero-sum probability row."
        )

    normalized = (
        probabilities
        /
        row_sums
    )

    if not np.allclose(
        normalized.sum(
            axis=1
        ),
        1.0,
        atol=1e-10,
    ):

        raise RuntimeError(
            "Probability normalization failed."
        )

    return normalized


# ============================================================
# LOAD CATBOOST OOF
# ============================================================


def load_catboost_oof() -> pd.DataFrame:
    """
    Load CatBoost rolling OOF predictions from 2018-2024.
    """

    require_file(
        BOOSTED_PREDICTIONS_FILE,
        "Boosted rolling predictions",
    )

    dataframe = pd.read_csv(
        BOOSTED_PREDICTIONS_FILE,
        low_memory=False,
    )

    required = {
        "validation_season",
        "model",
        "match_id",
        "date",
        "home_team",
        "away_team",
        "actual_result",
        "prob_H",
        "prob_D",
        "prob_A",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Boosted prediction file "
            f"missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe[
        dataframe[
            "model"
        ]
        ==
        "catboost"
    ].copy()

    if dataframe.empty:

        raise RuntimeError(
            "No CatBoost OOF predictions found."
        )

    dataframe[
        "target"
    ] = dataframe[
        "actual_result"
    ].map(
        TARGET_MAP
    )

    if dataframe[
        "target"
    ].isna().any():

        raise RuntimeError(
            "Invalid result labels found."
        )

    dataframe[
        "target"
    ] = dataframe[
        "target"
    ].astype(
        int
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    probabilities = normalize_probabilities(
        dataframe[
            [
                "prob_H",
                "prob_D",
                "prob_A",
            ]
        ].to_numpy(
            dtype=float
        )
    )

    dataframe[
        "prob_H"
    ] = probabilities[
        :,
        0
    ]

    dataframe[
        "prob_D"
    ] = probabilities[
        :,
        1
    ]

    dataframe[
        "prob_A"
    ] = probabilities[
        :,
        2
    ]

    dataframe = dataframe.sort_values(
        by=[
            "validation_season",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    return dataframe


# ============================================================
# TEMPERATURE SCALING
# ============================================================


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Multiclass temperature scaling applied to probabilities.

    We convert probabilities to log-space, divide by T,
    then apply softmax.

    T > 1:
        softer / less confident probabilities

    T < 1:
        sharper / more confident probabilities
    """

    if temperature <= 0:

        raise ValueError(
            "Temperature must be positive."
        )

    probabilities = normalize_probabilities(
        probabilities
    )

    logits = np.log(
        np.clip(
            probabilities,
            EPSILON,
            1.0,
        )
    )

    scaled_logits = (
        logits
        /
        temperature
    )

    scaled_logits = (
        scaled_logits
        -
        scaled_logits.max(
            axis=1,
            keepdims=True,
        )
    )

    exponentials = np.exp(
        scaled_logits
    )

    calibrated = (
        exponentials
        /
        exponentials.sum(
            axis=1,
            keepdims=True,
        )
    )

    return normalize_probabilities(
        calibrated
    )


# ============================================================
# FIT TEMPERATURE
# ============================================================


def fit_temperature(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Fit a single multiclass temperature by minimizing
    Log Loss.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = normalize_probabilities(
        probabilities
    )

    def objective(
        temperature: float,
    ) -> float:

        calibrated = apply_temperature(
            probabilities,
            temperature,
        )

        return float(
            log_loss(
                y_true,
                calibrated,
                labels=TARGET_LABELS,
            )
        )

    result = minimize_scalar(
        objective,
        bounds=(
            MIN_TEMPERATURE,
            MAX_TEMPERATURE,
        ),
        method="bounded",
        options={
            "xatol": 1e-5,
        },
    )

    if not result.success:

        raise RuntimeError(
            "Temperature optimization failed."
        )

    return float(
        result.x
    )


# ============================================================
# BRIER SCORE
# ============================================================


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Multiclass Brier score.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    one_hot = np.zeros_like(
        probabilities,
        dtype=float,
    )

    one_hot[
        np.arange(
            len(
                y_true
            )
        ),
        y_true,
    ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    -
                    one_hot
                )
                ** 2,
                axis=1,
            )
        )
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_probabilities(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Evaluate probability quality and class predictions.
    """

    probabilities = normalize_probabilities(
        probabilities
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    (
        precision,
        recall,
        class_f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=TARGET_LABELS,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=TARGET_LABELS,
            )
        ),

        "brier_score": (
            multiclass_brier_score(
                y_true,
                probabilities,
            )
        ),

        "home_recall": float(
            recall[
                0
            ]
        ),

        "draw_recall": float(
            recall[
                1
            ]
        ),

        "away_recall": float(
            recall[
                2
            ]
        ),

        "home_f1": float(
            class_f1[
                0
            ]
        ),

        "draw_f1": float(
            class_f1[
                1
            ]
        ),

        "away_f1": float(
            class_f1[
                2
            ]
        ),

        "predicted_home": int(
            (
                predictions
                ==
                0
            ).sum()
        ),

        "predicted_draw": int(
            (
                predictions
                ==
                1
            ).sum()
        ),

        "predicted_away": int(
            (
                predictions
                ==
                2
            ).sum()
        ),

        "actual_home": int(
            support[
                0
            ]
        ),

        "actual_draw": int(
            support[
                1
            ]
        ),

        "actual_away": int(
            support[
                2
            ]
        ),
    }


# ============================================================
# WEIGHTED MEAN
# ============================================================


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """
    Validation-row weighted average.
    """

    return float(
        np.average(
            values.astype(
                float
            ),
            weights=weights.astype(
                float
            ),
        )
    )


# ============================================================
# LEAVE-ONE-SEASON-OUT CALIBRATION
# ============================================================


def run_meta_calibration(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    For each season:

        fit temperature on the OTHER six OOF seasons
        apply to held-out season

    This avoids evaluating calibration on the same rows used
    to estimate the temperature.
    """

    seasons = sorted(
        dataframe[
            "validation_season"
        ].unique()
    )

    metric_rows = []

    prediction_frames = []

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LEAVE-ONE-SEASON-OUT TEMPERATURE CALIBRATION"
    )

    print(
        "=" * 75
    )

    for held_out_season in seasons:

        calibration_train = dataframe[
            dataframe[
                "validation_season"
            ]
            !=
            held_out_season
        ].copy()

        held_out = dataframe[
            dataframe[
                "validation_season"
            ]
            ==
            held_out_season
        ].copy()

        train_probabilities = calibration_train[
            [
                "prob_H",
                "prob_D",
                "prob_A",
            ]
        ].to_numpy(
            dtype=float
        )

        train_targets = calibration_train[
            "target"
        ].to_numpy(
            dtype=int
        )

        temperature = fit_temperature(
            y_true=train_targets,
            probabilities=train_probabilities,
        )

        held_out_probabilities = held_out[
            [
                "prob_H",
                "prob_D",
                "prob_A",
            ]
        ].to_numpy(
            dtype=float
        )

        raw_metrics = evaluate_probabilities(
            y_true=held_out[
                "target"
            ].to_numpy(
                dtype=int
            ),
            probabilities=held_out_probabilities,
        )

        calibrated_probabilities = apply_temperature(
            held_out_probabilities,
            temperature,
        )

        calibrated_metrics = evaluate_probabilities(
            y_true=held_out[
                "target"
            ].to_numpy(
                dtype=int
            ),
            probabilities=calibrated_probabilities,
        )

        metric_rows.append(
            {
                "held_out_season": int(
                    held_out_season
                ),

                "rows": len(
                    held_out
                ),

                "temperature": (
                    temperature
                ),

                "raw_log_loss": (
                    raw_metrics[
                        "log_loss"
                    ]
                ),

                "calibrated_log_loss": (
                    calibrated_metrics[
                        "log_loss"
                    ]
                ),

                "log_loss_improvement": (
                    raw_metrics[
                        "log_loss"
                    ]
                    -
                    calibrated_metrics[
                        "log_loss"
                    ]
                ),

                "raw_brier": (
                    raw_metrics[
                        "brier_score"
                    ]
                ),

                "calibrated_brier": (
                    calibrated_metrics[
                        "brier_score"
                    ]
                ),

                "brier_improvement": (
                    raw_metrics[
                        "brier_score"
                    ]
                    -
                    calibrated_metrics[
                        "brier_score"
                    ]
                ),

                "raw_accuracy": (
                    raw_metrics[
                        "accuracy"
                    ]
                ),

                "calibrated_accuracy": (
                    calibrated_metrics[
                        "accuracy"
                    ]
                ),

                "raw_macro_f1": (
                    raw_metrics[
                        "macro_f1"
                    ]
                ),

                "calibrated_macro_f1": (
                    calibrated_metrics[
                        "macro_f1"
                    ]
                ),

                "raw_draw_recall": (
                    raw_metrics[
                        "draw_recall"
                    ]
                ),

                "calibrated_draw_recall": (
                    calibrated_metrics[
                        "draw_recall"
                    ]
                ),
            }
        )

        held_output = held_out[
            [
                "validation_season",
                "match_id",
                "date",
                "home_team",
                "away_team",
                "actual_result",
                "target",
            ]
        ].copy()

        held_output[
            "temperature"
        ] = temperature

        held_output[
            "raw_prob_H"
        ] = held_out_probabilities[
            :,
            0
        ]

        held_output[
            "raw_prob_D"
        ] = held_out_probabilities[
            :,
            1
        ]

        held_output[
            "raw_prob_A"
        ] = held_out_probabilities[
            :,
            2
        ]

        held_output[
            "calibrated_prob_H"
        ] = calibrated_probabilities[
            :,
            0
        ]

        held_output[
            "calibrated_prob_D"
        ] = calibrated_probabilities[
            :,
            1
        ]

        held_output[
            "calibrated_prob_A"
        ] = calibrated_probabilities[
            :,
            2
        ]

        raw_predictions = np.argmax(
            held_out_probabilities,
            axis=1,
        )

        calibrated_predictions = np.argmax(
            calibrated_probabilities,
            axis=1,
        )

        held_output[
            "raw_prediction"
        ] = [
            TARGET_NAMES[
                int(
                    value
                )
            ]
            for value in raw_predictions
        ]

        held_output[
            "calibrated_prediction"
        ] = [
            TARGET_NAMES[
                int(
                    value
                )
            ]
            for value in calibrated_predictions
        ]

        prediction_frames.append(
            held_output
        )

        print(
            f"\nSeason {held_out_season}:"
        )

        print(
            f"  Temperature: "
            f"{temperature:.4f}"
        )

        print(
            f"  Raw Log Loss:        "
            f"{raw_metrics['log_loss']:.4f}"
        )

        print(
            f"  Calibrated Log Loss: "
            f"{calibrated_metrics['log_loss']:.4f}"
        )

        print(
            f"  Improvement:         "
            f"{raw_metrics['log_loss'] - calibrated_metrics['log_loss']:+.4f}"
        )

        print(
            f"  Raw Brier:           "
            f"{raw_metrics['brier_score']:.4f}"
        )

        print(
            f"  Calibrated Brier:    "
            f"{calibrated_metrics['brier_score']:.4f}"
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    return (
        metrics,
        predictions,
    )


# ============================================================
# FULL OOF TEMPERATURE
# ============================================================


def fit_final_temperature(
    dataframe: pd.DataFrame,
) -> tuple[
    float,
    dict,
    dict,
]:
    """
    Fit deployment temperature using every historical OOF
    prediction after the meta-calibration test is complete.

    This final temperature is NOT evaluated as an unbiased
    result. Its purpose is future deployment/training.
    """

    probabilities = dataframe[
        [
            "prob_H",
            "prob_D",
            "prob_A",
        ]
    ].to_numpy(
        dtype=float
    )

    y_true = dataframe[
        "target"
    ].to_numpy(
        dtype=int
    )

    temperature = fit_temperature(
        y_true=y_true,
        probabilities=probabilities,
    )

    raw_metrics = evaluate_probabilities(
        y_true=y_true,
        probabilities=probabilities,
    )

    calibrated = apply_temperature(
        probabilities,
        temperature,
    )

    calibrated_metrics = evaluate_probabilities(
        y_true=y_true,
        probabilities=calibrated,
    )

    return (
        temperature,
        raw_metrics,
        calibrated_metrics,
    )


# ============================================================
# TEMPERATURE CURVE
# ============================================================


def build_temperature_curve(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an interpretable temperature-vs-Log-Loss curve.
    """

    probabilities = dataframe[
        [
            "prob_H",
            "prob_D",
            "prob_A",
        ]
    ].to_numpy(
        dtype=float
    )

    y_true = dataframe[
        "target"
    ].to_numpy(
        dtype=int
    )

    temperatures = np.linspace(
        0.50,
        2.00,
        151,
    )

    rows = []

    for temperature in temperatures:

        calibrated = apply_temperature(
            probabilities,
            float(
                temperature
            ),
        )

        metrics = evaluate_probabilities(
            y_true=y_true,
            probabilities=calibrated,
        )

        rows.append(
            {
                "temperature": float(
                    temperature
                ),

                "log_loss": (
                    metrics[
                        "log_loss"
                    ]
                ),

                "brier_score": (
                    metrics[
                        "brier_score"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================


def calibrate_probabilities() -> None:
    """
    Evaluate whether temperature scaling improves CatBoost's
    rolling out-of-sample probability estimates.

    2025/26 remains untouched.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - CATBOOST PROBABILITY CALIBRATION"
    )

    print(
        "=" * 75
    )

    dataframe = load_catboost_oof()

    print(
        f"\nCatBoost OOF matches: "
        f"{len(dataframe):,}"
    )

    print(
        f"Seasons: "
        f"{dataframe['validation_season'].min()} "
        f"to "
        f"{dataframe['validation_season'].max()}"
    )

    if len(
        dataframe
    ) != 924:

        raise RuntimeError(
            f"Expected 924 CatBoost OOF rows, "
            f"found {len(dataframe):,}."
        )

    # --------------------------------------------------------
    # META CALIBRATION
    # --------------------------------------------------------

    (
        fold_metrics,
        calibrated_predictions,
    ) = run_meta_calibration(
        dataframe
    )

    weights = fold_metrics[
        "rows"
    ]

    raw_weighted_log_loss = weighted_mean(
        fold_metrics[
            "raw_log_loss"
        ],
        weights,
    )

    calibrated_weighted_log_loss = weighted_mean(
        fold_metrics[
            "calibrated_log_loss"
        ],
        weights,
    )

    raw_weighted_brier = weighted_mean(
        fold_metrics[
            "raw_brier"
        ],
        weights,
    )

    calibrated_weighted_brier = weighted_mean(
        fold_metrics[
            "calibrated_brier"
        ],
        weights,
    )

    raw_weighted_accuracy = weighted_mean(
        fold_metrics[
            "raw_accuracy"
        ],
        weights,
    )

    calibrated_weighted_accuracy = weighted_mean(
        fold_metrics[
            "calibrated_accuracy"
        ],
        weights,
    )

    raw_weighted_draw_recall = weighted_mean(
        fold_metrics[
            "raw_draw_recall"
        ],
        weights,
    )

    calibrated_weighted_draw_recall = weighted_mean(
        fold_metrics[
            "calibrated_draw_recall"
        ],
        weights,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CALIBRATION META-BACKTEST SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nRaw weighted Log Loss:        "
        f"{raw_weighted_log_loss:.4f}"
    )

    print(
        f"Calibrated weighted Log Loss: "
        f"{calibrated_weighted_log_loss:.4f}"
    )

    print(
        f"Log Loss improvement:         "
        f"{raw_weighted_log_loss - calibrated_weighted_log_loss:+.4f}"
    )

    print(
        f"\nRaw weighted Brier:           "
        f"{raw_weighted_brier:.4f}"
    )

    print(
        f"Calibrated weighted Brier:    "
        f"{calibrated_weighted_brier:.4f}"
    )

    print(
        f"Brier improvement:            "
        f"{raw_weighted_brier - calibrated_weighted_brier:+.4f}"
    )

    print(
        f"\nRaw weighted Accuracy:        "
        f"{raw_weighted_accuracy:.4f}"
    )

    print(
        f"Calibrated weighted Accuracy: "
        f"{calibrated_weighted_accuracy:.4f}"
    )

    print(
        f"\nRaw weighted Draw Recall:     "
        f"{raw_weighted_draw_recall:.4f}"
    )

    print(
        f"Calibrated weighted Draw Recall: "
        f"{calibrated_weighted_draw_recall:.4f}"
    )

    # --------------------------------------------------------
    # TEMPERATURE STABILITY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEMPERATURE STABILITY"
    )

    print(
        "=" * 75
    )

    print(
        "\n"
        + fold_metrics[
            [
                "held_out_season",
                "temperature",
                "raw_log_loss",
                "calibrated_log_loss",
                "log_loss_improvement",
                "raw_brier",
                "calibrated_brier",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    mean_temperature = float(
        fold_metrics[
            "temperature"
        ].mean()
    )

    std_temperature = float(
        fold_metrics[
            "temperature"
        ].std(
            ddof=0
        )
    )

    print(
        f"\nMean held-out temperature: "
        f"{mean_temperature:.4f}"
    )

    print(
        f"Temperature std: "
        f"{std_temperature:.4f}"
    )

    # --------------------------------------------------------
    # FINAL DEPLOYMENT TEMPERATURE
    # --------------------------------------------------------

    (
        final_temperature,
        full_raw_metrics,
        full_calibrated_metrics,
    ) = fit_final_temperature(
        dataframe
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL DEPLOYMENT TEMPERATURE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nTemperature: "
        f"{final_temperature:.6f}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "This temperature is fitted on all historical OOF "
        "rows for future deployment."
    )

    print(
        "Its same-data improvement is not treated as an "
        "unbiased performance estimate."
    )

    print(
        f"\nFull OOF raw Log Loss:        "
        f"{full_raw_metrics['log_loss']:.4f}"
    )

    print(
        f"Full OOF calibrated Log Loss: "
        f"{full_calibrated_metrics['log_loss']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        CALIBRATION_FOLD_FILE,
        index=False,
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        [
            {
                "model": "catboost",

                "oof_matches": len(
                    dataframe
                ),

                "folds": len(
                    fold_metrics
                ),

                "raw_weighted_log_loss": (
                    raw_weighted_log_loss
                ),

                "calibrated_weighted_log_loss": (
                    calibrated_weighted_log_loss
                ),

                "log_loss_improvement": (
                    raw_weighted_log_loss
                    -
                    calibrated_weighted_log_loss
                ),

                "raw_weighted_brier": (
                    raw_weighted_brier
                ),

                "calibrated_weighted_brier": (
                    calibrated_weighted_brier
                ),

                "brier_improvement": (
                    raw_weighted_brier
                    -
                    calibrated_weighted_brier
                ),

                "raw_weighted_accuracy": (
                    raw_weighted_accuracy
                ),

                "calibrated_weighted_accuracy": (
                    calibrated_weighted_accuracy
                ),

                "raw_weighted_draw_recall": (
                    raw_weighted_draw_recall
                ),

                "calibrated_weighted_draw_recall": (
                    calibrated_weighted_draw_recall
                ),

                "mean_meta_temperature": (
                    mean_temperature
                ),

                "meta_temperature_std": (
                    std_temperature
                ),

                "deployment_temperature": (
                    final_temperature
                ),
            }
        ]
    )

    summary.to_csv(
        CALIBRATION_SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    output_predictions = (
        calibrated_predictions.copy()
    )

    output_predictions[
        "date"
    ] = pd.to_datetime(
        output_predictions[
            "date"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    output_predictions.to_csv(
        CALIBRATED_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    temperature_curve = build_temperature_curve(
        dataframe
    )

    temperature_curve.to_csv(
        TEMPERATURE_SEARCH_FILE,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "model": "catboost",

        "calibration_method": (
            "multiclass_temperature_scaling"
        ),

        "meta_validation_method": (
            "leave_one_oof_season_out"
        ),

        "oof_matches": len(
            dataframe
        ),

        "oof_seasons": sorted(
            int(
                season
            )
            for season in dataframe[
                "validation_season"
            ].unique()
        ),

        "raw_weighted_log_loss": (
            raw_weighted_log_loss
        ),

        "calibrated_weighted_log_loss": (
            calibrated_weighted_log_loss
        ),

        "raw_weighted_brier": (
            raw_weighted_brier
        ),

        "calibrated_weighted_brier": (
            calibrated_weighted_brier
        ),

        "mean_meta_temperature": (
            mean_temperature
        ),

        "meta_temperature_std": (
            std_temperature
        ),

        "deployment_temperature": (
            final_temperature
        ),

        "final_test_season": 2025,

        "final_test_evaluated": False,

        "important_note": (
            "The deployment temperature is fitted using all "
            "historical rolling OOF predictions only. "
            "The 2025/26 final test remains untouched."
        ),
    }

    with CALIBRATION_METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CATBOOST CALIBRATION COMPLETED"
    )

    print(
        "=" * 75
    )

    if (
        calibrated_weighted_log_loss
        <
        raw_weighted_log_loss
    ):

        print(
            "\nTemperature calibration improved "
            "out-of-sample Log Loss ✓"
        )

    else:

        print(
            "\nTemperature calibration did NOT improve "
            "out-of-sample Log Loss."
        )

    print(
        "\n2025/26 remains untouched ✓"
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    calibrate_probabilities()