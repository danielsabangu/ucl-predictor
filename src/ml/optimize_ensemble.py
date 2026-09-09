from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product

import numpy as np
import pandas as pd

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

RF_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "rolling_backtest_predictions.csv"
)

BOOSTED_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "boosted_rolling_backtest_predictions.csv"
)

META_BACKTEST_FILE = (
    METRICS_DIR
    / "ensemble_meta_backtest_metrics.csv"
)

FINAL_WEIGHT_SEARCH_FILE = (
    METRICS_DIR
    / "ensemble_weight_search.csv"
)

ENSEMBLE_PREDICTIONS_FILE = (
    MODEL_REPORT_DIR
    / "ensemble_oof_predictions.csv"
)

ENSEMBLE_METADATA_FILE = (
    MODEL_METADATA_DIR
    / "ensemble_optimization.json"
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
# CANDIDATE MODELS
# ============================================================

MODEL_NAMES = [
    "catboost",
    "random_forest",
    "elo_plus_ucl_form",
    "xgboost",
]


# ============================================================
# WEIGHT SEARCH
# ============================================================

WEIGHT_STEP = 0.05

EPSILON = 1e-12


# ============================================================
# FILE CHECK
# ============================================================


def require_file(
    path,
    description: str,
) -> None:
    """
    Fail immediately if an expected OOF prediction file
    is unavailable.
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
# NORMALIZE PROBABILITIES
# ============================================================


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Explicitly normalize each probability row.

    This also corrects the small XGBoost numerical issue
    observed during rolling backtesting.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        None,
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

    final_sums = normalized.sum(
        axis=1
    )

    if not np.allclose(
        final_sums,
        1.0,
        atol=1e-10,
    ):

        raise RuntimeError(
            "Probability normalization failed."
        )

    return normalized


# ============================================================
# LOAD RANDOM FOREST OOF
# ============================================================


def load_random_forest_predictions() -> dict[str, pd.DataFrame]:
    """
    Load Random Forest rolling predictions for:

        without_transfer
        elo_plus_ucl_form
    """

    require_file(
        RF_PREDICTIONS_FILE,
        "Random Forest rolling predictions",
    )

    dataframe = pd.read_csv(
        RF_PREDICTIONS_FILE,
        low_memory=False,
    )

    required = {
        "validation_season",
        "scenario",
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
            f"Random Forest prediction file "
            f"missing columns: {sorted(missing)}"
        )

    output = {}

    scenario_map = {
        "random_forest": (
            "without_transfer"
        ),

        "elo_plus_ucl_form": (
            "elo_plus_ucl_form"
        ),
    }

    for (
        model_name,
        scenario,
    ) in scenario_map.items():

        selected = dataframe[
            dataframe[
                "scenario"
            ]
            ==
            scenario
        ].copy()

        if selected.empty:

            raise RuntimeError(
                f"Random Forest scenario "
                f"{scenario} not found."
            )

        selected = selected.sort_values(
            by=[
                "validation_season",
                "match_id",
            ]
        ).reset_index(
            drop=True
        )

        probabilities = normalize_probabilities(
            selected[
                [
                    "prob_H",
                    "prob_D",
                    "prob_A",
                ]
            ].to_numpy(
                dtype=float
            )
        )

        selected[
            "prob_H"
        ] = probabilities[
            :,
            0
        ]

        selected[
            "prob_D"
        ] = probabilities[
            :,
            1
        ]

        selected[
            "prob_A"
        ] = probabilities[
            :,
            2
        ]

        output[
            model_name
        ] = selected

    return output


# ============================================================
# LOAD BOOSTED OOF
# ============================================================


def load_boosted_predictions() -> dict[str, pd.DataFrame]:
    """
    Load CatBoost and XGBoost rolling predictions.
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
            f"missing columns: {sorted(missing)}"
        )

    output = {}

    for model_name in [
        "catboost",
        "xgboost",
    ]:

        selected = dataframe[
            dataframe[
                "model"
            ]
            ==
            model_name
        ].copy()

        if selected.empty:

            raise RuntimeError(
                f"{model_name} predictions "
                f"not found."
            )

        selected = selected.sort_values(
            by=[
                "validation_season",
                "match_id",
            ]
        ).reset_index(
            drop=True
        )

        probabilities = normalize_probabilities(
            selected[
                [
                    "prob_H",
                    "prob_D",
                    "prob_A",
                ]
            ].to_numpy(
                dtype=float
            )
        )

        selected[
            "prob_H"
        ] = probabilities[
            :,
            0
        ]

        selected[
            "prob_D"
        ] = probabilities[
            :,
            1
        ]

        selected[
            "prob_A"
        ] = probabilities[
            :,
            2
        ]

        output[
            model_name
        ] = selected

    return output


# ============================================================
# MERGE OOF PREDICTIONS
# ============================================================


def merge_predictions(
    model_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all candidate-model OOF predictions on the
    exact same historical validation matches.
    """

    base_columns = [
        "validation_season",
        "match_id",
        "date",
        "home_team",
        "away_team",
        "actual_result",
    ]

    combined = None

    for model_name in MODEL_NAMES:

        if model_name not in model_frames:

            raise RuntimeError(
                f"Missing model predictions: "
                f"{model_name}"
            )

        frame = model_frames[
            model_name
        ].copy()

        renamed = frame[
            base_columns
            +
            [
                "prob_H",
                "prob_D",
                "prob_A",
            ]
        ].rename(
            columns={
                "prob_H": (
                    f"{model_name}_prob_H"
                ),

                "prob_D": (
                    f"{model_name}_prob_D"
                ),

                "prob_A": (
                    f"{model_name}_prob_A"
                ),
            }
        )

        if combined is None:

            combined = renamed

        else:

            combined = combined.merge(
                renamed,
                on=base_columns,
                how="inner",
                validate="one_to_one",
            )

    if combined is None:

        raise RuntimeError(
            "No model predictions merged."
        )

    combined[
        "target"
    ] = combined[
        "actual_result"
    ].map(
        TARGET_MAP
    )

    if combined[
        "target"
    ].isna().any():

        raise RuntimeError(
            "Invalid actual result label."
        )

    combined[
        "target"
    ] = combined[
        "target"
    ].astype(
        int
    )

    combined[
        "date"
    ] = pd.to_datetime(
        combined[
            "date"
        ],
        errors="raise",
    )

    combined = combined.sort_values(
        by=[
            "validation_season",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    return combined


# ============================================================
# WEIGHT COMBINATIONS
# ============================================================


def generate_weight_grid() -> list[dict[str, float]]:
    """
    Generate non-negative weights that sum to exactly 1.

    At 0.05 increments there are only 1,771 combinations
    for four models.
    """

    units = int(
        round(
            1.0
            /
            WEIGHT_STEP
        )
    )

    combinations = []

    for catboost_units in range(
        units
        +
        1
    ):

        for rf_units in range(
            units
            -
            catboost_units
            +
            1
        ):

            for elo_units in range(
                units
                -
                catboost_units
                -
                rf_units
                +
                1
            ):

                xgb_units = (
                    units
                    -
                    catboost_units
                    -
                    rf_units
                    -
                    elo_units
                )

                weights = {
                    "catboost": (
                        catboost_units
                        *
                        WEIGHT_STEP
                    ),

                    "random_forest": (
                        rf_units
                        *
                        WEIGHT_STEP
                    ),

                    "elo_plus_ucl_form": (
                        elo_units
                        *
                        WEIGHT_STEP
                    ),

                    "xgboost": (
                        xgb_units
                        *
                        WEIGHT_STEP
                    ),
                }

                total = sum(
                    weights.values()
                )

                if not np.isclose(
                    total,
                    1.0,
                    atol=1e-10,
                ):

                    raise RuntimeError(
                        "Invalid blend weight sum."
                    )

                combinations.append(
                    weights
                )

    return combinations


# ============================================================
# BLEND
# ============================================================


def blend_probabilities(
    *,
    dataframe: pd.DataFrame,
    weights: dict[str, float],
) -> np.ndarray:
    """
    Weighted average of model probabilities.
    """

    blended = np.zeros(
        (
            len(
                dataframe
            ),
            3,
        ),
        dtype=float,
    )

    for model_name in MODEL_NAMES:

        weight = weights[
            model_name
        ]

        model_probs = dataframe[
            [
                f"{model_name}_prob_H",
                f"{model_name}_prob_D",
                f"{model_name}_prob_A",
            ]
        ].to_numpy(
            dtype=float
        )

        blended += (
            weight
            *
            model_probs
        )

    return normalize_probabilities(
        blended
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
# EVALUATE
# ============================================================


def evaluate_probabilities(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Probability and classification metrics.
    """

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    model_log_loss = log_loss(
        y_true,
        probabilities,
        labels=TARGET_LABELS,
    )

    brier = multiclass_brier_score(
        y_true,
        probabilities,
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
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "log_loss": float(
            model_log_loss
        ),

        "brier_score": float(
            brier
        ),

        "home_precision": float(
            precision[
                0
            ]
        ),

        "home_recall": float(
            recall[
                0
            ]
        ),

        "home_f1": float(
            class_f1[
                0
            ]
        ),

        "draw_precision": float(
            precision[
                1
            ]
        ),

        "draw_recall": float(
            recall[
                1
            ]
        ),

        "draw_f1": float(
            class_f1[
                1
            ]
        ),

        "away_precision": float(
            precision[
                2
            ]
        ),

        "away_recall": float(
            recall[
                2
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
# SEARCH BEST WEIGHTS
# ============================================================


def search_best_weights(
    dataframe: pd.DataFrame,
    weight_grid: list[dict[str, float]],
) -> tuple[
    dict[str, float],
    pd.DataFrame,
]:
    """
    Find weights minimizing multiclass Log Loss.
    """

    y_true = dataframe[
        "target"
    ].to_numpy(
        dtype=int
    )

    rows = []

    best_weights = None
    best_log_loss = np.inf

    for weights in weight_grid:

        probabilities = blend_probabilities(
            dataframe=dataframe,
            weights=weights,
        )

        metrics = evaluate_probabilities(
            y_true=y_true,
            probabilities=probabilities,
        )

        row = {
            "catboost_weight": (
                weights[
                    "catboost"
                ]
            ),

            "random_forest_weight": (
                weights[
                    "random_forest"
                ]
            ),

            "elo_plus_ucl_form_weight": (
                weights[
                    "elo_plus_ucl_form"
                ]
            ),

            "xgboost_weight": (
                weights[
                    "xgboost"
                ]
            ),
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

        if (
            metrics[
                "log_loss"
            ]
            <
            best_log_loss
        ):

            best_log_loss = metrics[
                "log_loss"
            ]

            best_weights = weights.copy()

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        by=[
            "log_loss",
            "brier_score",
            "accuracy",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    results[
        "rank"
    ] = range(
        1,
        len(
            results
        )
        +
        1,
    )

    if best_weights is None:

        raise RuntimeError(
            "Weight optimization failed."
        )

    return (
        best_weights,
        results,
    )


# ============================================================
# META BACKTEST
# ============================================================


def run_meta_backtest(
    *,
    dataframe: pd.DataFrame,
    weight_grid: list[dict[str, float]],
) -> pd.DataFrame:
    """
    Leave-one-season-out meta-level ensemble backtest.

    For each held-out season:

        1. choose ensemble weights using every OTHER
           rolling OOF season;
        2. evaluate those weights on the held-out season.

    This gives us a much less optimistic estimate than
    choosing weights and evaluating on the exact same rows.
    """

    seasons = sorted(
        dataframe[
            "validation_season"
        ].unique()
    )

    rows = []

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LEAVE-ONE-SEASON-OUT ENSEMBLE META-BACKTEST"
    )

    print(
        "=" * 75
    )

    for held_out_season in seasons:

        training_meta = dataframe[
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

        (
            best_weights,
            _,
        ) = search_best_weights(
            training_meta,
            weight_grid,
        )

        probabilities = blend_probabilities(
            dataframe=held_out,
            weights=best_weights,
        )

        metrics = evaluate_probabilities(
            y_true=held_out[
                "target"
            ].to_numpy(
                dtype=int
            ),
            probabilities=probabilities,
        )

        row = {
            "held_out_season": int(
                held_out_season
            ),

            "rows": int(
                len(
                    held_out
                )
            ),

            "catboost_weight": (
                best_weights[
                    "catboost"
                ]
            ),

            "random_forest_weight": (
                best_weights[
                    "random_forest"
                ]
            ),

            "elo_plus_ucl_form_weight": (
                best_weights[
                    "elo_plus_ucl_form"
                ]
            ),

            "xgboost_weight": (
                best_weights[
                    "xgboost"
                ]
            ),
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

        print(
            f"\nSeason {held_out_season}:"
        )

        print(
            "  Weights: "
            f"Cat={best_weights['catboost']:.2f}, "
            f"RF={best_weights['random_forest']:.2f}, "
            f"EloForm={best_weights['elo_plus_ucl_form']:.2f}, "
            f"XGB={best_weights['xgboost']:.2f}"
        )

        print(
            f"  Log Loss:   "
            f"{metrics['log_loss']:.4f}"
        )

        print(
            f"  Brier:      "
            f"{metrics['brier_score']:.4f}"
        )

        print(
            f"  Accuracy:   "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Draw Recall:"
            f" {metrics['draw_recall']:.4f}"
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WEIGHTED META SUMMARY
# ============================================================


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """
    Row-count weighted average.
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
# FINAL OOF ENSEMBLE
# ============================================================


def build_final_oof_predictions(
    *,
    dataframe: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Apply final optimized weights to all historical OOF rows.
    """

    probabilities = blend_probabilities(
        dataframe=dataframe,
        weights=weights,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    output = dataframe[
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

    output[
        "prob_H"
    ] = probabilities[
        :,
        0
    ]

    output[
        "prob_D"
    ] = probabilities[
        :,
        1
    ]

    output[
        "prob_A"
    ] = probabilities[
        :,
        2
    ]

    output[
        "prediction"
    ] = [
        TARGET_NAMES[
            int(
                value
            )
        ]
        for value in predictions
    ]

    return output


# ============================================================
# MAIN
# ============================================================


def optimize_ensemble() -> None:
    """
    Optimize a probability blend using rolling OOF data only.

    No 2025/26 predictions or labels are used.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - ENSEMBLE WEIGHT OPTIMIZATION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    random_forest_frames = (
        load_random_forest_predictions()
    )

    boosted_frames = (
        load_boosted_predictions()
    )

    model_frames = {
        **random_forest_frames,
        **boosted_frames,
    }

    combined = merge_predictions(
        model_frames
    )

    print(
        f"\nMerged OOF matches: "
        f"{len(combined):,}"
    )

    print(
        f"Seasons: "
        f"{combined['validation_season'].min()} "
        f"to "
        f"{combined['validation_season'].max()}"
    )

    if len(
        combined
    ) != 924:

        raise RuntimeError(
            f"Expected 924 merged OOF matches, "
            f"found {len(combined):,}."
        )

    print(
        "OOF alignment: valid ✓"
    )

    # --------------------------------------------------------
    # INDIVIDUAL NORMALIZED METRICS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "NORMALIZED STANDALONE OOF METRICS"
    )

    print(
        "=" * 75
    )

    individual_rows = []

    for model_name in MODEL_NAMES:

        probabilities = combined[
            [
                f"{model_name}_prob_H",
                f"{model_name}_prob_D",
                f"{model_name}_prob_A",
            ]
        ].to_numpy(
            dtype=float
        )

        probabilities = normalize_probabilities(
            probabilities
        )

        metrics = evaluate_probabilities(
            y_true=combined[
                "target"
            ].to_numpy(
                dtype=int
            ),
            probabilities=probabilities,
        )

        row = {
            "model": model_name,
        }

        row.update(
            metrics
        )

        individual_rows.append(
            row
        )

    individual = pd.DataFrame(
        individual_rows
    ).sort_values(
        by="log_loss"
    )

    print(
        "\n"
        + individual[
            [
                "model",
                "accuracy",
                "macro_f1",
                "log_loss",
                "brier_score",
                "draw_recall",
                "predicted_draw",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # WEIGHT GRID
    # --------------------------------------------------------

    weight_grid = generate_weight_grid()

    print(
        f"\nBlend combinations: "
        f"{len(weight_grid):,}"
    )

    # --------------------------------------------------------
    # META BACKTEST
    # --------------------------------------------------------

    meta_backtest = run_meta_backtest(
        dataframe=combined,
        weight_grid=weight_grid,
    )

    meta_weights = meta_backtest[
        "rows"
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "META-BACKTEST SUMMARY"
    )

    print(
        "=" * 75
    )

    meta_summary = {
        "weighted_log_loss": weighted_mean(
            meta_backtest[
                "log_loss"
            ],
            meta_weights,
        ),

        "weighted_brier": weighted_mean(
            meta_backtest[
                "brier_score"
            ],
            meta_weights,
        ),

        "weighted_accuracy": weighted_mean(
            meta_backtest[
                "accuracy"
            ],
            meta_weights,
        ),

        "weighted_macro_f1": weighted_mean(
            meta_backtest[
                "macro_f1"
            ],
            meta_weights,
        ),

        "weighted_draw_recall": weighted_mean(
            meta_backtest[
                "draw_recall"
            ],
            meta_weights,
        ),
    }

    print(
        f"\nWeighted Log Loss: "
        f"{meta_summary['weighted_log_loss']:.4f}"
    )

    print(
        f"Weighted Brier: "
        f"{meta_summary['weighted_brier']:.4f}"
    )

    print(
        f"Weighted Accuracy: "
        f"{meta_summary['weighted_accuracy']:.4f}"
    )

    print(
        f"Weighted Macro F1: "
        f"{meta_summary['weighted_macro_f1']:.4f}"
    )

    print(
        f"Weighted Draw Recall: "
        f"{meta_summary['weighted_draw_recall']:.4f}"
    )

    # --------------------------------------------------------
    # FINAL WEIGHT SEARCH
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL OOF WEIGHT SEARCH"
    )

    print(
        "=" * 75
    )

    (
        final_weights,
        search_results,
    ) = search_best_weights(
        combined,
        weight_grid,
    )

    best_search = search_results.iloc[
        0
    ]

    print(
        "\nBest weights:"
    )

    print(
        f"  CatBoost:          "
        f"{final_weights['catboost']:.2f}"
    )

    print(
        f"  Random Forest:     "
        f"{final_weights['random_forest']:.2f}"
    )

    print(
        f"  Elo + UCL Form RF: "
        f"{final_weights['elo_plus_ucl_form']:.2f}"
    )

    print(
        f"  XGBoost:           "
        f"{final_weights['xgboost']:.2f}"
    )

    print(
        f"\nOOF Log Loss: "
        f"{best_search['log_loss']:.4f}"
    )

    print(
        f"OOF Brier: "
        f"{best_search['brier_score']:.4f}"
    )

    print(
        f"OOF Accuracy: "
        f"{best_search['accuracy']:.4f}"
    )

    print(
        f"OOF Macro F1: "
        f"{best_search['macro_f1']:.4f}"
    )

    print(
        f"OOF Draw Recall: "
        f"{best_search['draw_recall']:.4f}"
    )

    # --------------------------------------------------------
    # TOP WEIGHTS
    # --------------------------------------------------------

    print(
        "\nTop 15 weight combinations:"
    )

    print(
        search_results[
            [
                "rank",
                "catboost_weight",
                "random_forest_weight",
                "elo_plus_ucl_form_weight",
                "xgboost_weight",
                "log_loss",
                "brier_score",
                "accuracy",
                "macro_f1",
                "draw_recall",
            ]
        ]
        .head(
            15
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # FINAL OOF PREDICTIONS
    # --------------------------------------------------------

    final_oof = build_final_oof_predictions(
        dataframe=combined,
        weights=final_weights,
    )

    final_metrics = evaluate_probabilities(
        y_true=final_oof[
            "target"
        ].to_numpy(
            dtype=int
        ),
        probabilities=final_oof[
            [
                "prob_H",
                "prob_D",
                "prob_A",
            ]
        ].to_numpy(
            dtype=float
        ),
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

    meta_backtest.to_csv(
        META_BACKTEST_FILE,
        index=False,
        encoding="utf-8",
    )

    search_results.to_csv(
        FINAL_WEIGHT_SEARCH_FILE,
        index=False,
        encoding="utf-8",
    )

    output_predictions = final_oof.copy()

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
        ENSEMBLE_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "candidate_models": (
            MODEL_NAMES
        ),

        "weight_step": (
            WEIGHT_STEP
        ),

        "weight_combinations": len(
            weight_grid
        ),

        "oof_matches": len(
            combined
        ),

        "oof_seasons": sorted(
            int(
                season
            )
            for season in combined[
                "validation_season"
            ].unique()
        ),

        "xgboost_probability_normalization": True,

        "meta_backtest": (
            meta_summary
        ),

        "final_weights": (
            final_weights
        ),

        "final_oof_metrics": (
            final_metrics
        ),

        "final_test_season": 2025,

        "final_test_evaluated": False,

        "selection_metric": (
            "multiclass log loss"
        ),
    }

    with ENSEMBLE_METADATA_FILE.open(
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
        "ENSEMBLE OPTIMIZATION COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nStandalone benchmark:"
    )

    print(
        "  CatBoost Log Loss: 0.9166"
    )

    print(
        "  CatBoost Brier:    0.5390"
    )

    print(
        "\nRobust meta-backtest ensemble:"
    )

    print(
        f"  Log Loss: "
        f"{meta_summary['weighted_log_loss']:.4f}"
    )

    print(
        f"  Brier:    "
        f"{meta_summary['weighted_brier']:.4f}"
    )

    print(
        "\nFinal optimized weights:"
    )

    for model_name in MODEL_NAMES:

        print(
            f"  {model_name:<20} "
            f"{final_weights[model_name]:.2f}"
        )

    print(
        "\n2025/26 remains untouched ✓"
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    optimize_ensemble()