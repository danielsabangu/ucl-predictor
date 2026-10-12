from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import evaluate_final_test as v1


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

UEFA_FEATURES_PATH = (
    ROOT
    / "data"
    / "research"
    / "ucl_uefa_strength_features.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

COMPARISON_PATH = (
    REPORT_DIR
    / "uefa_strength_candidate_comparison.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "uefa_strength_candidate_test_predictions.csv"
)

CALIBRATION_PATH = (
    REPORT_DIR
    / "uefa_strength_candidate_calibration_bins.csv"
)

CONTROL_CONFUSION_PATH = (
    REPORT_DIR
    / "uefa_strength_control_confusion.csv"
)

CANDIDATE_CONFUSION_PATH = (
    REPORT_DIR
    / "uefa_strength_candidate_confusion.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "uefa_strength_candidate_summary.json"
)


# ============================================================
# EXPERIMENT CONTRACT
# ============================================================


CONTROL_FEATURE_COUNT = 290

CANDIDATE_FEATURE_COUNT = 296

EXPECTED_MATCH_ROWS = 2089


CANDIDATE_UEFA_FEATURES = [
    "home_uefa_coefficient",
    "away_uefa_coefficient",
    "uefa_coefficient_diff",
    "home_uefa_strength_percentile",
    "away_uefa_strength_percentile",
    "uefa_strength_percentile_diff",
]


LABEL_TO_RESULT = {
    0: "H",
    1: "D",
    2: "A",
}


# ============================================================
# HELPERS
# ============================================================


def canonical_match_id(
    value,
) -> str:
    """
    Build a deterministic join key.

    This is not fuzzy matching.

    It only protects against pandas representing an integer
    match ID as e.g. 123 instead of 123.0.
    """

    if pd.isna(
        value
    ):
        raise RuntimeError(
            "Encountered missing match_id."
        )

    text = str(
        value
    ).strip()

    numeric_float_match = re.fullmatch(
        r"([0-9]+)\.0+",
        text,
    )

    if numeric_float_match:
        return (
            numeric_float_match.group(
                1
            )
        )

    return text


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Reuse the V1 probability normalization exactly.
    """

    return v1.normalize_probabilities(
        probabilities
    )


# ============================================================
# LOAD V1 DATA
# ============================================================


def load_v1_dataframe() -> tuple[
    pd.DataFrame,
    list[str],
]:
    print(
        "Loading frozen V1 feature manifest..."
    )

    manifest = (
        v1.load_manifest()
    )

    all_features = (
        manifest[
            "feature_columns"
        ]
    )

    print(
        "Applying frozen V1 feature policy..."
    )

    control_features = (
        v1.build_final_feature_set(
            all_features
        )
    )

    if len(
        control_features
    ) != CONTROL_FEATURE_COUNT:

        raise RuntimeError(
            "Candidate experiment expected "
            f"{CONTROL_FEATURE_COUNT} frozen V1 "
            "features, but the V1 policy returned "
            f"{len(control_features)}."
        )

    print(
        "Loading frozen V1 historical matrix..."
    )

    dataframe = (
        v1.load_data(
            all_features
        )
    )

    if len(
        dataframe
    ) != EXPECTED_MATCH_ROWS:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_MATCH_ROWS:,} V1 rows, "
            f"found {len(dataframe):,}."
        )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        v1.TARGET_COLUMN,
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "V1 dataframe is missing required "
            f"columns: {sorted(missing)}"
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "__match_key"
    ] = dataframe[
        "match_id"
    ].apply(
        canonical_match_id
    )

    if dataframe[
        "__match_key"
    ].duplicated().any():

        duplicates = (
            dataframe.loc[
                dataframe[
                    "__match_key"
                ].duplicated(
                    keep=False
                ),
                [
                    "match_id",
                    "date",
                    "home_team",
                    "away_team",
                ],
            ]
        )

        raise RuntimeError(
            "Duplicate V1 match IDs found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return (
        dataframe,
        control_features,
    )


# ============================================================
# LOAD UEFA FEATURES
# ============================================================


def load_uefa_features() -> pd.DataFrame:
    if not UEFA_FEATURES_PATH.exists():

        raise FileNotFoundError(
            UEFA_FEATURES_PATH
        )

    print(
        "Loading Candidate A UEFA features..."
    )

    dataframe = pd.read_csv(
        UEFA_FEATURES_PATH,
        low_memory=False,
    )

    required = {
        "match_id",
        *CANDIDATE_UEFA_FEATURES,
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "UEFA feature layer missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataframe
    ) != EXPECTED_MATCH_ROWS:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_MATCH_ROWS:,} UEFA rows, "
            f"found {len(dataframe):,}."
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "__match_key"
    ] = dataframe[
        "match_id"
    ].apply(
        canonical_match_id
    )

    if dataframe[
        "__match_key"
    ].duplicated().any():

        duplicates = (
            dataframe.loc[
                dataframe[
                    "__match_key"
                ].duplicated(
                    keep=False
                ),
                [
                    "match_id",
                ],
            ]
        )

        raise RuntimeError(
            "Duplicate UEFA match IDs found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    missing_features = (
        dataframe[
            CANDIDATE_UEFA_FEATURES
        ]
        .isna()
        .sum()
    )

    if (
        missing_features
        > 0
    ).any():

        raise RuntimeError(
            "Missing UEFA candidate values:\n"
            f"{missing_features.to_string()}"
        )

    return dataframe


# ============================================================
# MERGE
# ============================================================


def merge_candidate_features(
    base: pd.DataFrame,
    uefa: pd.DataFrame,
) -> pd.DataFrame:
    print(
        "Joining UEFA features to frozen V1 rows..."
    )

    collisions = [
        feature
        for feature in CANDIDATE_UEFA_FEATURES
        if feature in base.columns
    ]

    if collisions:

        raise RuntimeError(
            "UEFA candidate columns already exist "
            "inside V1 matrix: "
            f"{collisions}"
        )

    base_keys = set(
        base[
            "__match_key"
        ]
    )

    uefa_keys = set(
        uefa[
            "__match_key"
        ]
    )

    missing_from_uefa = (
        base_keys
        - uefa_keys
    )

    extra_in_uefa = (
        uefa_keys
        - base_keys
    )

    if missing_from_uefa:

        examples = sorted(
            missing_from_uefa
        )[:20]

        raise RuntimeError(
            "V1 matches missing from UEFA layer. "
            f"Examples: {examples}"
        )

    if extra_in_uefa:

        examples = sorted(
            extra_in_uefa
        )[:20]

        raise RuntimeError(
            "UEFA layer contains matches not found "
            "in V1 matrix. "
            f"Examples: {examples}"
        )

    left = (
        base.copy()
    )

    left[
        "__row_order"
    ] = np.arange(
        len(
            left
        )
    )

    right = (
        uefa[
            [
                "__match_key",
                *CANDIDATE_UEFA_FEATURES,
            ]
        ]
        .copy()
    )

    merged = left.merge(
        right,
        on="__match_key",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    merged = (
        merged.sort_values(
            "__row_order"
        )
        .drop(
            columns=[
                "__row_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        merged
    ) != len(
        base
    ):

        raise RuntimeError(
            "Merge changed historical row count."
        )

    missing_candidate = (
        merged[
            CANDIDATE_UEFA_FEATURES
        ]
        .isna()
        .sum()
    )

    if (
        missing_candidate
        > 0
    ).any():

        raise RuntimeError(
            "UEFA values became missing after "
            "the merge:\n"
            f"{missing_candidate.to_string()}"
        )

    original_order = (
        base[
            "__match_key"
        ]
        .tolist()
    )

    merged_order = (
        merged[
            "__match_key"
        ]
        .tolist()
    )

    if (
        original_order
        != merged_order
    ):

        raise RuntimeError(
            "Historical match order changed "
            "during UEFA merge."
        )

    return merged


# ============================================================
# MODEL EXECUTION
# ============================================================


def run_model(
    *,
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict:
    print()
    print(
        "-" * 80
    )

    print(
        name
    )

    print(
        "-" * 80
    )

    print(
        f"Features: {len(features):,}"
    )

    X_train = (
        train[
            features
        ]
    )

    y_train = (
        train[
            v1.TARGET_COLUMN
        ]
    )

    X_test = (
        test[
            features
        ]
    )

    y_test = (
        test[
            v1.TARGET_COLUMN
        ]
    )

    train_availability = (
        1.0
        -
        X_train.isna()
        .mean()
        .mean()
    )

    test_availability = (
        1.0
        -
        X_test.isna()
        .mean()
        .mean()
    )

    print(
        "Feature availability:"
    )

    print(
        f"  Train: "
        f"{train_availability * 100:.2f}%"
    )

    print(
        f"  Test:  "
        f"{test_availability * 100:.2f}%"
    )

    print(
        "Training exact locked CatBoost "
        "specification..."
    )

    model = (
        v1.build_final_model()
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    probabilities = (
        normalize_probabilities(
            probabilities
        )
    )

    (
        metrics,
        confusion,
        predictions,
    ) = (
        v1.evaluate_final_test(
            y_true=y_test,
            probabilities=probabilities,
        )
    )

    return {
        "name": name,
        "features": features,
        "model": model,
        "metrics": metrics,
        "confusion": confusion,
        "predictions": np.asarray(
            predictions,
            dtype=int,
        ),
        "probabilities": probabilities,
        "y_true": (
            y_test.to_numpy(
                dtype=int
            )
        ),
        "train_availability": float(
            train_availability
        ),
        "test_availability": float(
            test_availability
        ),
    }


# ============================================================
# CALIBRATION DIAGNOSTIC
# ============================================================


def build_calibration_bins(
    *,
    model_name: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> tuple[
    pd.DataFrame,
    float,
]:
    """
    Top-label Expected Calibration Error.

    This is an additional research diagnostic.

    It is not being used to tune Candidate A.
    """

    confidence = np.max(
        probabilities,
        axis=1,
    )

    correct = (
        predictions
        ==
        y_true
    ).astype(
        float
    )

    rows = []

    ece = 0.0

    total = len(
        y_true
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    for index in range(
        bins
    ):
        lower = float(
            edges[
                index
            ]
        )

        upper = float(
            edges[
                index + 1
            ]
        )

        if index == (
            bins - 1
        ):

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:

            rows.append(
                {
                    "model": (
                        model_name
                    ),
                    "bin": (
                        index + 1
                    ),
                    "lower_bound": (
                        lower
                    ),
                    "upper_bound": (
                        upper
                    ),
                    "matches": 0,
                    "mean_confidence": (
                        np.nan
                    ),
                    "accuracy": (
                        np.nan
                    ),
                    "absolute_gap": (
                        np.nan
                    ),
                }
            )

            continue

        mean_confidence = float(
            confidence[
                mask
            ].mean()
        )

        accuracy = float(
            correct[
                mask
            ].mean()
        )

        gap = abs(
            accuracy
            -
            mean_confidence
        )

        weight = (
            count
            / total
        )

        ece += (
            weight
            *
            gap
        )

        rows.append(
            {
                "model": (
                    model_name
                ),
                "bin": (
                    index + 1
                ),
                "lower_bound": (
                    lower
                ),
                "upper_bound": (
                    upper
                ),
                "matches": (
                    count
                ),
                "mean_confidence": (
                    mean_confidence
                ),
                "accuracy": (
                    accuracy
                ),
                "absolute_gap": (
                    gap
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        ),
        float(
            ece
        ),
    )


# ============================================================
# PER-MATCH DIAGNOSTICS
# ============================================================


def per_match_log_loss(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        1e-15,
        1.0 - 1e-15,
    )

    true_probabilities = (
        clipped[
            np.arange(
                len(
                    y_true
                )
            ),
            y_true,
        ]
    )

    return (
        -np.log(
            true_probabilities
        )
    )


def per_match_brier(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
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

    return np.sum(
        (
            probabilities
            -
            one_hot
        )
        ** 2,
        axis=1,
    )


def result_name(
    value: int,
) -> str:
    if value not in LABEL_TO_RESULT:

        raise RuntimeError(
            "Unexpected target label: "
            f"{value}"
        )

    return LABEL_TO_RESULT[
        value
    ]


def build_prediction_report(
    *,
    test: pd.DataFrame,
    control: dict,
    candidate: dict,
) -> pd.DataFrame:
    y_true = (
        control[
            "y_true"
        ]
    )

    if not np.array_equal(
        y_true,
        candidate[
            "y_true"
        ],
    ):

        raise RuntimeError(
            "Control and Candidate A test "
            "targets differ."
        )

    control_probabilities = (
        control[
            "probabilities"
        ]
    )

    candidate_probabilities = (
        candidate[
            "probabilities"
        ]
    )

    control_predictions = (
        control[
            "predictions"
        ]
    )

    candidate_predictions = (
        candidate[
            "predictions"
        ]
    )

    control_log_loss = (
        per_match_log_loss(
            y_true=y_true,
            probabilities=(
                control_probabilities
            ),
        )
    )

    candidate_log_loss = (
        per_match_log_loss(
            y_true=y_true,
            probabilities=(
                candidate_probabilities
            ),
        )
    )

    control_brier = (
        per_match_brier(
            y_true=y_true,
            probabilities=(
                control_probabilities
            ),
        )
    )

    candidate_brier = (
        per_match_brier(
            y_true=y_true,
            probabilities=(
                candidate_probabilities
            ),
        )
    )

    report = pd.DataFrame(
        {
            "match_id": (
                test[
                    "match_id"
                ].to_numpy()
            ),
            "date": (
                test[
                    "date"
                ].to_numpy()
            ),
            "season": (
                test[
                    "season"
                ].to_numpy()
            ),
            "home_team": (
                test[
                    "home_team"
                ].to_numpy()
            ),
            "away_team": (
                test[
                    "away_team"
                ].to_numpy()
            ),
            "actual_label": (
                y_true
            ),
            "actual_result": [
                result_name(
                    int(
                        value
                    )
                )
                for value in y_true
            ],

            "control_p_home": (
                control_probabilities[
                    :,
                    0
                ]
            ),
            "control_p_draw": (
                control_probabilities[
                    :,
                    1
                ]
            ),
            "control_p_away": (
                control_probabilities[
                    :,
                    2
                ]
            ),
            "control_prediction": [
                result_name(
                    int(
                        value
                    )
                )
                for value
                in control_predictions
            ],
            "control_correct": (
                control_predictions
                ==
                y_true
            ).astype(
                int
            ),
            "control_match_log_loss": (
                control_log_loss
            ),
            "control_match_brier": (
                control_brier
            ),

            "candidate_p_home": (
                candidate_probabilities[
                    :,
                    0
                ]
            ),
            "candidate_p_draw": (
                candidate_probabilities[
                    :,
                    1
                ]
            ),
            "candidate_p_away": (
                candidate_probabilities[
                    :,
                    2
                ]
            ),
            "candidate_prediction": [
                result_name(
                    int(
                        value
                    )
                )
                for value
                in candidate_predictions
            ],
            "candidate_correct": (
                candidate_predictions
                ==
                y_true
            ).astype(
                int
            ),
            "candidate_match_log_loss": (
                candidate_log_loss
            ),
            "candidate_match_brier": (
                candidate_brier
            ),

            # Positive means Candidate A was
            # better for that specific fixture.
            "log_loss_improvement": (
                control_log_loss
                -
                candidate_log_loss
            ),
            "brier_improvement": (
                control_brier
                -
                candidate_brier
            ),
        }
    )

    return report


# ============================================================
# METRIC COMPARISON
# ============================================================


def build_metric_comparison(
    *,
    control_metrics: dict,
    candidate_metrics: dict,
    control_ece: float,
    candidate_ece: float,
) -> pd.DataFrame:
    metrics = [
        (
            "log_loss",
            "lower",
        ),
        (
            "brier_score",
            "lower",
        ),
        (
            "accuracy",
            "higher",
        ),
        (
            "macro_f1",
            "higher",
        ),
        (
            "weighted_f1",
            "higher",
        ),
        (
            "home_recall",
            "higher",
        ),
        (
            "draw_recall",
            "higher",
        ),
        (
            "away_recall",
            "higher",
        ),
    ]

    rows = []

    for (
        metric,
        direction,
    ) in metrics:

        if metric not in control_metrics:

            raise RuntimeError(
                "V1 evaluator did not return "
                f"{metric!r}."
            )

        if metric not in candidate_metrics:

            raise RuntimeError(
                "Candidate evaluator did not return "
                f"{metric!r}."
            )

        control_value = float(
            control_metrics[
                metric
            ]
        )

        candidate_value = float(
            candidate_metrics[
                metric
            ]
        )

        raw_delta = (
            candidate_value
            -
            control_value
        )

        if direction == "lower":

            improvement = (
                control_value
                -
                candidate_value
            )

        else:

            improvement = (
                candidate_value
                -
                control_value
            )

        if improvement > 0:

            winner = (
                "CANDIDATE_A"
            )

        elif improvement < 0:

            winner = (
                "CONTROL"
            )

        else:

            winner = (
                "TIE"
            )

        rows.append(
            {
                "metric": (
                    metric
                ),
                "better_direction": (
                    direction
                ),
                "control_v1": (
                    control_value
                ),
                "candidate_a": (
                    candidate_value
                ),
                "candidate_minus_control": (
                    raw_delta
                ),
                "candidate_improvement": (
                    improvement
                ),
                "winner": (
                    winner
                ),
            }
        )

    ece_improvement = (
        control_ece
        -
        candidate_ece
    )

    if ece_improvement > 0:

        ece_winner = (
            "CANDIDATE_A"
        )

    elif ece_improvement < 0:

        ece_winner = (
            "CONTROL"
        )

    else:

        ece_winner = (
            "TIE"
        )

    rows.append(
        {
            "metric": (
                "top_label_ece_10bin"
            ),
            "better_direction": (
                "lower"
            ),
            "control_v1": (
                control_ece
            ),
            "candidate_a": (
                candidate_ece
            ),
            "candidate_minus_control": (
                candidate_ece
                -
                control_ece
            ),
            "candidate_improvement": (
                ece_improvement
            ),
            "winner": (
                ece_winner
            ),
        }
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINTING
# ============================================================


def print_metric_table(
    comparison: pd.DataFrame,
) -> None:
    printable = (
        comparison.copy()
    )

    for column in [
        "control_v1",
        "candidate_a",
        "candidate_minus_control",
        "candidate_improvement",
    ]:

        printable[
            column
        ] = printable[
            column
        ].map(
            lambda value: (
                f"{value:.6f}"
            )
        )

    print(
        printable.to_string(
            index=False
        )
    )


def print_biggest_changes(
    predictions: pd.DataFrame,
) -> None:
    print()
    print(
        "=" * 100
    )

    print(
        "FIXTURES CANDIDATE A HELPED MOST"
    )

    print(
        "=" * 100
    )

    print()

    best = (
        predictions.sort_values(
            "log_loss_improvement",
            ascending=False,
        )
        .head(
            10
        )
    )

    print(
        best[
            [
                "date",
                "home_team",
                "away_team",
                "actual_result",
                "control_prediction",
                "candidate_prediction",
                "log_loss_improvement",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FIXTURES CANDIDATE A HURT MOST"
    )

    print(
        "=" * 100
    )

    print()

    worst = (
        predictions.sort_values(
            "log_loss_improvement",
            ascending=True,
        )
        .head(
            10
        )
    )

    print(
        worst[
            [
                "date",
                "home_team",
                "away_team",
                "actual_result",
                "control_prediction",
                "candidate_prediction",
                "log_loss_improvement",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=" * 100
    )

    print(
        "UCL PREDICTOR V2 - CANDIDATE A"
    )

    print(
        "UEFA STRENGTH RESEARCH EXPERIMENT"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "2025/26 has already been consumed historically."
    )

    print(
        "This is a controlled research comparison, "
        "not a new untouched final test."
    )

    print(
        "No hyperparameters will be tuned here."
    )

    print(
        "No production model or production feature "
        "matrix will be modified."
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        base,
        control_features,
    ) = load_v1_dataframe()

    uefa = (
        load_uefa_features()
    )

    dataframe = (
        merge_candidate_features(
            base,
            uefa,
        )
    )

    candidate_features = [
        *control_features,
        *CANDIDATE_UEFA_FEATURES,
    ]

    if len(
        candidate_features
    ) != CANDIDATE_FEATURE_COUNT:

        raise RuntimeError(
            "Expected Candidate A to contain "
            f"{CANDIDATE_FEATURE_COUNT} features, "
            f"found {len(candidate_features)}."
        )

    if len(
        set(
            candidate_features
        )
    ) != len(
        candidate_features
    ):

        raise RuntimeError(
            "Candidate A contains duplicate "
            "feature names."
        )

    # --------------------------------------------------------
    # LABEL CONTRACT
    # --------------------------------------------------------

    target_labels = [
        int(
            value
        )
        for value
        in v1.TARGET_LABELS
    ]

    if target_labels != [
        0,
        1,
        2,
    ]:

        raise RuntimeError(
            "This experiment expects the frozen "
            "label order [0, 1, 2] = H/D/A. "
            f"V1 returned {target_labels}."
        )

    # --------------------------------------------------------
    # SAME V1 CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    (
        train,
        test,
    ) = (
        v1.build_final_split(
            dataframe
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "EXPERIMENT SPLIT"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Training seasons: "
        f"{train['season'].min()} "
        f"to "
        f"{train['season'].max()}"
    )

    print(
        f"Training dates: "
        f"{train['date'].min().date()} "
        f"to "
        f"{train['date'].max().date()}"
    )

    print()

    print(
        f"Comparison rows: "
        f"{len(test):,}"
    )

    print(
        f"Comparison season: "
        f"{v1.FINAL_TEST_SEASON}"
    )

    print(
        f"Comparison dates: "
        f"{test['date'].min().date()} "
        f"to "
        f"{test['date'].max().date()}"
    )

    print()

    print(
        f"Control features: "
        f"{len(control_features):,}"
    )

    print(
        f"Candidate A features: "
        f"{len(candidate_features):,}"
    )

    print()

    print(
        "Candidate A additions:"
    )

    for feature in (
        CANDIDATE_UEFA_FEATURES
    ):

        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # CONTROL
    # --------------------------------------------------------

    control = (
        run_model(
            name=(
                "CONTROL - FROZEN V1 FEATURE SET"
            ),
            train=train,
            test=test,
            features=control_features,
        )
    )

    # --------------------------------------------------------
    # CANDIDATE A
    # --------------------------------------------------------

    candidate = (
        run_model(
            name=(
                "CANDIDATE A - V1 + UEFA STRENGTH"
            ),
            train=train,
            test=test,
            features=candidate_features,
        )
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    (
        control_calibration,
        control_ece,
    ) = (
        build_calibration_bins(
            model_name="CONTROL_V1",
            y_true=control[
                "y_true"
            ],
            predictions=control[
                "predictions"
            ],
            probabilities=control[
                "probabilities"
            ],
        )
    )

    (
        candidate_calibration,
        candidate_ece,
    ) = (
        build_calibration_bins(
            model_name="CANDIDATE_A",
            y_true=candidate[
                "y_true"
            ],
            predictions=candidate[
                "predictions"
            ],
            probabilities=candidate[
                "probabilities"
            ],
        )
    )

    calibration = pd.concat(
        [
            control_calibration,
            candidate_calibration,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    comparison = (
        build_metric_comparison(
            control_metrics=(
                control[
                    "metrics"
                ]
            ),
            candidate_metrics=(
                candidate[
                    "metrics"
                ]
            ),
            control_ece=(
                control_ece
            ),
            candidate_ece=(
                candidate_ece
            ),
        )
    )

    predictions = (
        build_prediction_report(
            test=test,
            control=control,
            candidate=candidate,
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    calibration.to_csv(
        CALIBRATION_PATH,
        index=False,
    )

    control[
        "confusion"
    ].to_csv(
        CONTROL_CONFUSION_PATH,
    )

    candidate[
        "confusion"
    ].to_csv(
        CANDIDATE_CONFUSION_PATH,
    )

    summary = {
        "experiment": (
            "candidate_a_uefa_strength"
        ),
        "status": (
            "research_only"
        ),
        "production_modified": False,
        "comparison_season_already_consumed": True,
        "comparison_season": int(
            v1.FINAL_TEST_SEASON
        ),
        "training_rows": int(
            len(
                train
            )
        ),
        "comparison_rows": int(
            len(
                test
            )
        ),
        "control_feature_count": int(
            len(
                control_features
            )
        ),
        "candidate_feature_count": int(
            len(
                candidate_features
            )
        ),
        "added_features": (
            CANDIDATE_UEFA_FEATURES
        ),
        "catboost_specification": {
            "iterations": 700,
            "learning_rate": 0.025,
            "depth": 5,
            "loss_function": (
                "MultiClass"
            ),
            "l2_leaf_reg": 5.0,
            "random_seed_source": (
                "src.ml.evaluate_final_test."
                "RANDOM_STATE"
            ),
            "imputer": (
                "median_with_missing_indicator"
            ),
        },
        "control_metrics": {
            key: float(
                value
            )
            for key, value
            in control[
                "metrics"
            ].items()
            if np.isscalar(
                value
            )
        },
        "candidate_metrics": {
            key: float(
                value
            )
            for key, value
            in candidate[
                "metrics"
            ].items()
            if np.isscalar(
                value
            )
        },
        "control_top_label_ece_10bin": (
            float(
                control_ece
            )
        ),
        "candidate_top_label_ece_10bin": (
            float(
                candidate_ece
            )
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "CANDIDATE A VS FROZEN V1"
    )

    print(
        "=" * 100
    )

    print()

    print_metric_table(
        comparison
    )

    print()
    print(
        "Positive candidate_improvement means "
        "Candidate A performed better."
    )

    print_biggest_changes(
        predictions
    )

    # --------------------------------------------------------
    # CORE DECISION SIGNALS
    # --------------------------------------------------------

    control_log_loss = float(
        control[
            "metrics"
        ][
            "log_loss"
        ]
    )

    candidate_log_loss = float(
        candidate[
            "metrics"
        ][
            "log_loss"
        ]
    )

    control_brier = float(
        control[
            "metrics"
        ][
            "brier_score"
        ]
    )

    candidate_brier = float(
        candidate[
            "metrics"
        ][
            "brier_score"
        ]
    )

    control_accuracy = float(
        control[
            "metrics"
        ][
            "accuracy"
        ]
    )

    candidate_accuracy = float(
        candidate[
            "metrics"
        ][
            "accuracy"
        ]
    )

    control_macro_f1 = float(
        control[
            "metrics"
        ][
            "macro_f1"
        ]
    )

    candidate_macro_f1 = float(
        candidate[
            "metrics"
        ][
            "macro_f1"
        ]
    )

    control_draw_recall = float(
        control[
            "metrics"
        ][
            "draw_recall"
        ]
    )

    candidate_draw_recall = float(
        candidate[
            "metrics"
        ][
            "draw_recall"
        ]
    )

    print()
    print(
        "=" * 100
    )

    print(
        "CORE DELTAS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Log Loss:"
    )

    print(
        f"  Control:     "
        f"{control_log_loss:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{candidate_log_loss:.6f}"
    )

    print(
        f"  Improvement: "
        f"{control_log_loss - candidate_log_loss:+.6f}"
    )

    print()

    print(
        "Brier Score:"
    )

    print(
        f"  Control:     "
        f"{control_brier:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{candidate_brier:.6f}"
    )

    print(
        f"  Improvement: "
        f"{control_brier - candidate_brier:+.6f}"
    )

    print()

    print(
        "Accuracy:"
    )

    print(
        f"  Control:     "
        f"{control_accuracy:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{candidate_accuracy:.6f}"
    )

    print(
        f"  Improvement: "
        f"{candidate_accuracy - control_accuracy:+.6f}"
    )

    print()

    print(
        "Macro F1:"
    )

    print(
        f"  Control:     "
        f"{control_macro_f1:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{candidate_macro_f1:.6f}"
    )

    print(
        f"  Improvement: "
        f"{candidate_macro_f1 - control_macro_f1:+.6f}"
    )

    print()

    print(
        "Draw Recall:"
    )

    print(
        f"  Control:     "
        f"{control_draw_recall:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{candidate_draw_recall:.6f}"
    )

    print(
        f"  Improvement: "
        f"{candidate_draw_recall - control_draw_recall:+.6f}"
    )

    print()

    print(
        "Top-label ECE (lower is better):"
    )

    print(
        f"  Control:     "
        f"{control_ece:.6f}"
    )

    print(
        f"  Candidate A: "
        f"{candidate_ece:.6f}"
    )

    print(
        f"  Improvement: "
        f"{control_ece - candidate_ece:+.6f}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "EXPERIMENT VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Frozen V1 control features: "
        f"{len(control_features)} PASS"
    )

    print(
        "Candidate A features: "
        f"{len(candidate_features)} PASS"
    )

    print(
        "Same training rows: PASS"
    )

    print(
        "Same comparison rows: PASS"
    )

    print(
        "Same chronological split: PASS"
    )

    print(
        "Same CatBoost specification: PASS"
    )

    print(
        "Same V1 evaluator: PASS"
    )

    print(
        "UEFA metadata excluded from model: PASS"
    )

    print(
        "Country-floor flags excluded from model: PASS"
    )

    print(
        "No hyperparameter tuning: PASS"
    )

    print(
        "No fuzzy matching: PASS"
    )

    print(
        "Production feature matrix untouched: PASS"
    )

    print(
        "Production model untouched: PASS"
    )

    print()
    print(
        "=" * 100
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        COMPARISON_PATH
    )

    print(
        PREDICTIONS_PATH
    )

    print(
        CALIBRATION_PATH
    )

    print(
        CONTROL_CONFUSION_PATH
    )

    print(
        CANDIDATE_CONFUSION_PATH
    )

    print(
        SUMMARY_PATH
    )

    print()

    print(
        "Research experiment complete."
    )

    print(
        "Do not promote Candidate A to production "
        "from this result alone."
    )


if __name__ == "__main__":
    main()