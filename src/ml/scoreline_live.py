from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import PREDICTIONS_DIR


# ============================================================
# SETTINGS
# ============================================================

MAX_GOALS = 10

COARSE_LAMBDA_MIN = 0.05
COARSE_LAMBDA_MAX = 5.50
COARSE_STEP = 0.05

REFINE_RADIUS = 0.12
REFINE_STEP = 0.005


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions_with_confidence.csv"
)

OUTPUT_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions_complete.csv"
)

AUDIT_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_scoreline_audit.csv"
)

METADATA_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_scoreline_metadata.json"
)


# ============================================================
# HELPERS
# ============================================================


def require_file(
    path,
    description: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}"
        )


def poisson_vector(
    goal_lambda: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """
    Poisson probabilities from 0..max_goals.

    Uses recurrence instead of factorial repeatedly.
    """

    probabilities = np.zeros(
        max_goals + 1,
        dtype=float,
    )

    probabilities[0] = math.exp(
        -goal_lambda
    )

    for goals in range(
        1,
        max_goals + 1,
    ):

        probabilities[
            goals
        ] = (
            probabilities[
                goals - 1
            ]
            *
            goal_lambda
            /
            goals
        )

    return probabilities


def score_matrix(
    *,
    home_lambda: float,
    away_lambda: float,
) -> np.ndarray:

    home = poisson_vector(
        home_lambda
    )

    away = poisson_vector(
        away_lambda
    )

    matrix = np.outer(
        home,
        away,
    )

    total = matrix.sum()

    if total <= 0:

        raise RuntimeError(
            "Invalid Poisson score matrix."
        )

    # MAX_GOALS truncates a tiny amount of tail probability.
    # Normalize so our displayed score-space sums exactly to 1.
    matrix = (
        matrix
        /
        total
    )

    return matrix


def outcome_probabilities(
    matrix: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:

    # Matrix:
    # rows = home goals
    # cols = away goals

    home_probability = float(
        np.tril(
            matrix,
            k=-1,
        ).sum()
    )

    draw_probability = float(
        np.trace(
            matrix
        )
    )

    away_probability = float(
        np.triu(
            matrix,
            k=1,
        ).sum()
    )

    return (
        home_probability,
        draw_probability,
        away_probability,
    )


def fit_error(
    *,
    implied: tuple[
        float,
        float,
        float,
    ],
    target: tuple[
        float,
        float,
        float,
    ],
) -> float:

    implied_array = np.array(
        implied,
        dtype=float,
    )

    target_array = np.array(
        target,
        dtype=float,
    )

    return float(
        np.sum(
            (
                implied_array
                -
                target_array
            )
            ** 2
        )
    )


# ============================================================
# SEARCH FOR POISSON RATES
# ============================================================


def search_lambda_grid(
    *,
    target: tuple[
        float,
        float,
        float,
    ],
    home_values: np.ndarray,
    away_values: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:

    best_error = float(
        "inf"
    )

    best_home = None
    best_away = None

    for home_lambda in home_values:

        home_vector = poisson_vector(
            float(
                home_lambda
            )
        )

        for away_lambda in away_values:

            away_vector = poisson_vector(
                float(
                    away_lambda
                )
            )

            matrix = np.outer(
                home_vector,
                away_vector,
            )

            matrix_sum = matrix.sum()

            if matrix_sum <= 0:
                continue

            matrix = (
                matrix
                /
                matrix_sum
            )

            implied = outcome_probabilities(
                matrix
            )

            error = fit_error(
                implied=implied,
                target=target,
            )

            if error < best_error:

                best_error = error

                best_home = float(
                    home_lambda
                )

                best_away = float(
                    away_lambda
                )

    if (
        best_home is None
        or
        best_away is None
    ):

        raise RuntimeError(
            "Poisson lambda search failed."
        )

    return (
        best_home,
        best_away,
        best_error,
    )


def fit_poisson_bridge(
    *,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
) -> dict:

    target = (
        float(
            home_probability
        ),
        float(
            draw_probability
        ),
        float(
            away_probability
        ),
    )

    target_sum = sum(
        target
    )

    if not np.isclose(
        target_sum,
        1.0,
        atol=1e-5,
    ):

        raise RuntimeError(
            "Target H/D/A probabilities do not "
            f"sum to 1: {target_sum}"
        )

    # --------------------------------------------------------
    # COARSE SEARCH
    # --------------------------------------------------------

    coarse_values = np.arange(
        COARSE_LAMBDA_MIN,
        COARSE_LAMBDA_MAX
        +
        COARSE_STEP
        /
        2,
        COARSE_STEP,
    )

    (
        coarse_home,
        coarse_away,
        _,
    ) = search_lambda_grid(
        target=target,
        home_values=coarse_values,
        away_values=coarse_values,
    )

    # --------------------------------------------------------
    # FINE SEARCH
    # --------------------------------------------------------

    home_min = max(
        COARSE_LAMBDA_MIN,
        coarse_home
        -
        REFINE_RADIUS,
    )

    home_max = min(
        COARSE_LAMBDA_MAX,
        coarse_home
        +
        REFINE_RADIUS,
    )

    away_min = max(
        COARSE_LAMBDA_MIN,
        coarse_away
        -
        REFINE_RADIUS,
    )

    away_max = min(
        COARSE_LAMBDA_MAX,
        coarse_away
        +
        REFINE_RADIUS,
    )

    home_values = np.arange(
        home_min,
        home_max
        +
        REFINE_STEP
        /
        2,
        REFINE_STEP,
    )

    away_values = np.arange(
        away_min,
        away_max
        +
        REFINE_STEP
        /
        2,
        REFINE_STEP,
    )

    (
        home_lambda,
        away_lambda,
        squared_error,
    ) = search_lambda_grid(
        target=target,
        home_values=home_values,
        away_values=away_values,
    )

    matrix = score_matrix(
        home_lambda=home_lambda,
        away_lambda=away_lambda,
    )

    (
        implied_home,
        implied_draw,
        implied_away,
    ) = outcome_probabilities(
        matrix
    )

    target_array = np.array(
        target,
        dtype=float,
    )

    implied_array = np.array(
        [
            implied_home,
            implied_draw,
            implied_away,
        ],
        dtype=float,
    )

    absolute_errors = np.abs(
        implied_array
        -
        target_array
    )

    mean_absolute_error_pp = float(
        absolute_errors.mean()
        *
        100
    )

    max_absolute_error_pp = float(
        absolute_errors.max()
        *
        100
    )

    return {
        "home_lambda": (
            home_lambda
        ),

        "away_lambda": (
            away_lambda
        ),

        "total_lambda": (
            home_lambda
            +
            away_lambda
        ),

        "matrix": (
            matrix
        ),

        "implied_home": (
            implied_home
        ),

        "implied_draw": (
            implied_draw
        ),

        "implied_away": (
            implied_away
        ),

        "squared_error": (
            squared_error
        ),

        "mae_pp": (
            mean_absolute_error_pp
        ),

        "max_error_pp": (
            max_absolute_error_pp
        ),
    }


# ============================================================
# SCORELINE SELECTION
# ============================================================


def scoreline_outcome(
    home_goals: int,
    away_goals: int,
) -> str:

    if home_goals > away_goals:

        return "H"

    if home_goals == away_goals:

        return "D"

    return "A"


def select_scoreline(
    *,
    matrix: np.ndarray,
    predicted_result: str,
) -> dict:
    """
    Select the most probable exact scoreline that agrees
    with CatBoost's predicted H/D/A class.
    """

    best_probability = -1.0

    best_home_goals = None
    best_away_goals = None

    for home_goals in range(
        matrix.shape[
            0
        ]
    ):

        for away_goals in range(
            matrix.shape[
                1
            ]
        ):

            outcome = scoreline_outcome(
                home_goals,
                away_goals,
            )

            if outcome != predicted_result:
                continue

            probability = float(
                matrix[
                    home_goals,
                    away_goals,
                ]
            )

            if (
                probability
                >
                best_probability
            ):

                best_probability = (
                    probability
                )

                best_home_goals = (
                    home_goals
                )

                best_away_goals = (
                    away_goals
                )

    if (
        best_home_goals is None
        or
        best_away_goals is None
    ):

        raise RuntimeError(
            "Could not select class-consistent "
            "scoreline."
        )

    # --------------------------------------------------------
    # GLOBAL MODAL SCORELINE
    # --------------------------------------------------------

    global_index = np.unravel_index(
        np.argmax(
            matrix
        ),
        matrix.shape,
    )

    global_home = int(
        global_index[
            0
        ]
    )

    global_away = int(
        global_index[
            1
        ]
    )

    global_probability = float(
        matrix[
            global_home,
            global_away,
        ]
    )

    return {
        "predicted_home_goals": (
            int(
                best_home_goals
            )
        ),

        "predicted_away_goals": (
            int(
                best_away_goals
            )
        ),

        "predicted_scoreline": (
            f"{best_home_goals}-"
            f"{best_away_goals}"
        ),

        "predicted_scoreline_probability": (
            best_probability
        ),

        "global_modal_home_goals": (
            global_home
        ),

        "global_modal_away_goals": (
            global_away
        ),

        "global_modal_scoreline": (
            f"{global_home}-"
            f"{global_away}"
        ),

        "global_modal_scoreline_probability": (
            global_probability
        ),

        "global_modal_result": (
            scoreline_outcome(
                global_home,
                global_away,
            )
        ),
    }


# ============================================================
# BRIDGE QUALITY
# ============================================================


def bridge_quality_label(
    max_error_pp: float,
) -> str:

    if max_error_pp <= 1.0:

        return "STRONG"

    if max_error_pp <= 2.5:

        return "GOOD"

    if max_error_pp <= 5.0:

        return "ACCEPTABLE"

    return "WEAK"


# ============================================================
# LOAD
# ============================================================


def load_predictions() -> pd.DataFrame:

    require_file(
        INPUT_FILE,
        "Prediction + confidence file",
    )

    dataframe = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required = {
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_probability",
        "draw_probability",
        "away_probability",
        "predicted_result",
        "predicted_selection",
        "max_probability",
        "confidence",
        "confidence_percent",
        "confidence_label",
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Prediction file missing columns:\n"
            f"{sorted(missing)}"
        )

    dataframe[
        "kickoff_utc"
    ] = pd.to_datetime(
        dataframe[
            "kickoff_utc"
        ],
        errors="raise",
        utc=True,
    )

    return dataframe


# ============================================================
# QUALITY
# ============================================================


def run_quality_checks(
    *,
    source: pd.DataFrame,
    output: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE SCORELINE ENGINE QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    print(
        f"\nFixture rows: "
        f"{len(output)}"
    )

    if len(
        source
    ) != len(
        output
    ):

        raise RuntimeError(
            "Fixture row count changed."
        )

    # --------------------------------------------------------
    # RAW PROBABILITY PRESERVATION
    # --------------------------------------------------------

    for column in [
        "home_probability",
        "draw_probability",
        "away_probability",
        "confidence",
    ]:

        original = pd.to_numeric(
            source[
                column
            ],
            errors="raise",
        ).to_numpy()

        current = pd.to_numeric(
            output[
                column
            ],
            errors="raise",
        ).to_numpy()

        if not np.allclose(
            original,
            current,
            atol=1e-12,
            rtol=0,
        ):

            raise RuntimeError(
                f"{column} changed."
            )

    print(
        "Raw CatBoost probabilities preserved ✓"
    )

    print(
        "Evidence confidence preserved ✓"
    )

    # --------------------------------------------------------
    # SCORE CLASS CONSISTENCY
    # --------------------------------------------------------

    inconsistent = 0

    for row in output.itertuples(
        index=False
    ):

        score_result = scoreline_outcome(
            int(
                row.predicted_home_goals
            ),
            int(
                row.predicted_away_goals
            ),
        )

        if (
            score_result
            !=
            row.predicted_result
        ):

            inconsistent += 1

    print(
        f"Class-inconsistent predicted scores: "
        f"{inconsistent}"
    )

    if inconsistent:

        raise RuntimeError(
            "Predicted scoreline contradicts "
            "CatBoost result class."
        )

    # --------------------------------------------------------
    # BRIDGE FIT
    # --------------------------------------------------------

    print(
        f"Maximum H/D/A bridge error: "
        f"{output['score_bridge_max_error_pp'].max():.3f} pp"
    )

    print(
        f"Mean H/D/A bridge MAE: "
        f"{output['score_bridge_mae_pp'].mean():.3f} pp"
    )

    invalid_lambdas = int(
        (
            (
                output[
                    "score_home_lambda"
                ]
                <=
                0
            )
            |
            (
                output[
                    "score_away_lambda"
                ]
                <=
                0
            )
        ).sum()
    )

    print(
        f"Invalid Poisson rates: "
        f"{invalid_lambdas}"
    )

    if invalid_lambdas:

        raise RuntimeError(
            "Invalid Poisson lambda detected."
        )

    invalid_probabilities = int(
        (
            (
                output[
                    "predicted_scoreline_probability"
                ]
                <
                0
            )
            |
            (
                output[
                    "predicted_scoreline_probability"
                ]
                >
                1
            )
        ).sum()
    )

    if invalid_probabilities:

        raise RuntimeError(
            "Invalid exact-score probability."
        )

    print(
        "Predicted scores agree with result picks ✓"
    )

    print(
        "Poisson rates valid ✓"
    )

    print(
        "Exact-score probabilities valid ✓"
    )

    print(
        "Score layer remains separate from CatBoost ✓"
    )


# ============================================================
# PRINT
# ============================================================


def print_report(
    dataframe: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "UCL 2026/27 - COMPLETE LIVE FORECAST"
    )

    print(
        "=" * 75
    )

    for row in dataframe.itertuples(
        index=False
    ):

        print(
            "\n"
            + "-"
            *
            75
        )

        print(
            f"{row.home_team} "
            f"vs "
            f"{row.away_team}"
        )

        print(
            f"\nPREDICTED RESULT: "
            f"{row.predicted_selection}"
        )

        print(
            f"PREDICTED SCORE:  "
            f"{row.predicted_scoreline}"
        )

        print(
            "\nRaw outcome probabilities:"
        )

        print(
            f"  HOME: "
            f"{row.home_probability * 100:6.2f}%"
        )

        print(
            f"  DRAW: "
            f"{row.draw_probability * 100:6.2f}%"
        )

        print(
            f"  AWAY: "
            f"{row.away_probability * 100:6.2f}%"
        )

        print(
            f"\nEvidence confidence: "
            f"{row.confidence_percent:.2f}% "
            f"({row.confidence_label})"
        )

        print(
            f"Probability profile: "
            f"{row.model_decisiveness_profile}"
        )

        print(
            "\nPoisson score bridge:"
        )

        print(
            f"  Home rate: "
            f"{row.score_home_lambda:.3f}"
        )

        print(
            f"  Away rate: "
            f"{row.score_away_lambda:.3f}"
        )

        print(
            f"  Total rate: "
            f"{row.score_total_lambda:.3f}"
        )

        print(
            f"  Exact-score share: "
            f"{row.predicted_scoreline_probability * 100:.2f}%"
        )

        print(
            f"  Bridge max error: "
            f"{row.score_bridge_max_error_pp:.3f} pp"
        )

        print(
            f"  Bridge quality: "
            f"{row.score_bridge_quality}"
        )

        if (
            row.global_modal_scoreline
            !=
            row.predicted_scoreline
        ):

            print(
                "\nNote:"
            )

            print(
                f"  Global modal exact score is "
                f"{row.global_modal_scoreline}, "
                "but the displayed score is constrained "
                "to CatBoost's predicted result."
            )


# ============================================================
# MAIN
# ============================================================


def build_live_scorelines() -> None:

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - LIVE SCORELINE ENGINE"
    )

    print(
        "=" * 75
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a Poisson score bridge."
    )

    print(
        "It is NOT a separately trained goal model."
    )

    print(
        "CatBoost probabilities will NOT be altered."
    )

    print(
        "Evidence confidence will NOT be altered."
    )

    source = load_predictions()

    rows = []

    for prediction in source.to_dict(
        orient="records"
    ):

        prediction = pd.Series(
            prediction
        )

        print(
            "\nFitting scoreline: "
            f"{prediction['home_team']} vs "
            f"{prediction['away_team']}"
        )

        bridge = fit_poisson_bridge(
            home_probability=float(
                prediction[
                    "home_probability"
                ]
            ),

            draw_probability=float(
                prediction[
                    "draw_probability"
                ]
            ),

            away_probability=float(
                prediction[
                    "away_probability"
                ]
            ),
        )

        selected = select_scoreline(
            matrix=bridge[
                "matrix"
            ],

            predicted_result=str(
                prediction[
                    "predicted_result"
                ]
            ),
        )

        rows.append(
            {
                **prediction.to_dict(),

                "score_home_lambda": (
                    bridge[
                        "home_lambda"
                    ]
                ),

                "score_away_lambda": (
                    bridge[
                        "away_lambda"
                    ]
                ),

                "score_total_lambda": (
                    bridge[
                        "total_lambda"
                    ]
                ),

                "score_bridge_home_probability": (
                    bridge[
                        "implied_home"
                    ]
                ),

                "score_bridge_draw_probability": (
                    bridge[
                        "implied_draw"
                    ]
                ),

                "score_bridge_away_probability": (
                    bridge[
                        "implied_away"
                    ]
                ),

                "score_bridge_mae_pp": (
                    bridge[
                        "mae_pp"
                    ]
                ),

                "score_bridge_max_error_pp": (
                    bridge[
                        "max_error_pp"
                    ]
                ),

                "score_bridge_quality": (
                    bridge_quality_label(
                        bridge[
                            "max_error_pp"
                        ]
                    )
                ),

                **selected,

                "scoreline_method": (
                    "POISSON_BRIDGE_TO_FROZEN_HDA"
                ),
            }
        )

    output = pd.DataFrame(
        rows
    )

    output = output.sort_values(
        by=[
            "kickoff_utc",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    run_quality_checks(
        source=source,
        output=output,
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    audit_columns = [
        "kickoff_utc",
        "home_team",
        "away_team",
        "predicted_selection",
        "predicted_result",
        "predicted_scoreline",
        "predicted_scoreline_probability",
        "global_modal_scoreline",
        "global_modal_scoreline_probability",
        "global_modal_result",
        "home_probability",
        "draw_probability",
        "away_probability",
        "confidence",
        "confidence_label",
        "score_home_lambda",
        "score_away_lambda",
        "score_total_lambda",
        "score_bridge_home_probability",
        "score_bridge_draw_probability",
        "score_bridge_away_probability",
        "score_bridge_mae_pp",
        "score_bridge_max_error_pp",
        "score_bridge_quality",
        "scoreline_method",
    ]

    output[
        audit_columns
    ].to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "engine": (
            "poisson_score_bridge_v1"
        ),

        "input": (
            str(
                INPUT_FILE
            )
        ),

        "fixture_count": int(
            len(
                output
            )
        ),

        "max_goals_modeled": (
            MAX_GOALS
        ),

        "lambda_search": {
            "minimum": (
                COARSE_LAMBDA_MIN
            ),

            "maximum": (
                COARSE_LAMBDA_MAX
            ),

            "coarse_step": (
                COARSE_STEP
            ),

            "refine_radius": (
                REFINE_RADIUS
            ),

            "refine_step": (
                REFINE_STEP
            ),
        },

        "rules": {
            "trained_goal_model": False,

            "scoreline_is_auxiliary_estimate": True,

            "catboost_probabilities_modified": False,

            "evidence_confidence_modified": False,

            "scoreline_constrained_to_catboost_result": True,

            "independent_poisson_assumption": True,

            "exact_score_probability_is_not_model_confidence": True,
        },
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_report(
        output
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE SCORELINE ENGINE COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nComplete prediction output:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nScoreline audit:"
    )

    print(
        AUDIT_FILE
    )

    print(
        "\nMetadata:"
    )

    print(
        METADATA_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The score is a display estimate derived "
        "from the frozen H/D/A distribution."
    )

    print(
        "Do not describe it as a separately trained "
        "score prediction model."
    )


if __name__ == "__main__":
    build_live_scorelines()