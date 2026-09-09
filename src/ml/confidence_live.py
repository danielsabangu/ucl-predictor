from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import (
    INTERIM_DATA_DIR,
    PREDICTIONS_DIR,
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_ELO_FEATURES = 6
EXPECTED_UCL_FORM_FEATURES = 81
EXPECTED_DOMESTIC_FEATURES = 102
EXPECTED_SQUAD_FEATURES = 101
EXPECTED_TOTAL_FEATURES = 290


# Confidence is deliberately evidence-based.
#
# Raw model probability / predicted class is NOT part of this
# confidence score.
#
# Squad evidence receives substantial weight because it is the
# largest known live-data weakness at the moment.
CONFIDENCE_WEIGHTS = {
    "elo": 0.10,
    "ucl_form": 0.25,
    "domestic": 0.20,
    "squad_model": 0.30,
    "current_roster": 0.15,
}

assert math.isclose(
    sum(CONFIDENCE_WEIGHTS.values()),
    1.0,
    abs_tol=1e-12,
)


# ============================================================
# PATHS
# ============================================================

PREDICTIONS_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions.csv"
)

CURRENT_SQUAD_CONTEXT_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "ucl_team_squad_context_live_current_36.csv"
)

OUTPUT_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions_with_confidence.csv"
)

AUDIT_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_confidence_audit.csv"
)

METADATA_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_confidence_metadata.json"
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


def clamp01(
    value,
) -> float:

    numeric = pd.to_numeric(
        pd.Series(
            [value]
        ),
        errors="coerce",
    ).iloc[0]

    if pd.isna(numeric):

        return 0.0

    return float(
        min(
            1.0,
            max(
                0.0,
                numeric,
            ),
        )
    )


def safe_ratio(
    numerator,
    denominator,
) -> float:

    numerator = pd.to_numeric(
        pd.Series(
            [numerator]
        ),
        errors="coerce",
    ).iloc[0]

    denominator = pd.to_numeric(
        pd.Series(
            [denominator]
        ),
        errors="coerce",
    ).iloc[0]

    if (
        pd.isna(numerator)
        or
        pd.isna(denominator)
        or
        denominator <= 0
    ):

        return 0.0

    return clamp01(
        numerator
        /
        denominator
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================


def load_predictions() -> pd.DataFrame:

    require_file(
        PREDICTIONS_FILE,
        "Live production predictions",
    )

    dataframe = pd.read_csv(
        PREDICTIONS_FILE,
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
        "elo_features_available",
        "ucl_form_features_available",
        "domestic_features_available",
        "squad_features_available",
        "overall_features_available",
        "overall_feature_availability",
        "squad_data_mode",
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

    if dataframe.empty:

        raise RuntimeError(
            "Prediction file contains no rows."
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
# CURRENT ROSTER CONTEXT
# ============================================================


def load_roster_context() -> pd.DataFrame:

    require_file(
        CURRENT_SQUAD_CONTEXT_FILE,
        "Current squad context",
    )

    dataframe = pd.read_csv(
        CURRENT_SQUAD_CONTEXT_FILE,
        low_memory=False,
    )

    required = {
        "team",
        "current_squad_player_rows",
        "current_confirmed_squad_rows",
        "current_fallback_pool_rows",
        "current_tm_mapped_players",
        "current_valued_players",
        "current_value_coverage",
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
            "Current squad context missing columns:\n"
            f"{sorted(missing)}"
        )

    if dataframe[
        "team"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate teams in squad context."
        )

    return dataframe


# ============================================================
# ROSTER RELIABILITY
# ============================================================


def calculate_team_roster_quality(
    row: pd.Series,
) -> dict:

    total_players = pd.to_numeric(
        row[
            "current_squad_player_rows"
        ],
        errors="coerce",
    )

    confirmed_players = pd.to_numeric(
        row[
            "current_confirmed_squad_rows"
        ],
        errors="coerce",
    )

    fallback_players = pd.to_numeric(
        row[
            "current_fallback_pool_rows"
        ],
        errors="coerce",
    )

    mapped_players = pd.to_numeric(
        row[
            "current_tm_mapped_players"
        ],
        errors="coerce",
    )

    confirmed_rate = safe_ratio(
        confirmed_players,
        total_players,
    )

    fallback_rate = safe_ratio(
        fallback_players,
        total_players,
    )

    mapping_rate = safe_ratio(
        mapped_players,
        total_players,
    )

    valuation_coverage = clamp01(
        row[
            "current_value_coverage"
        ]
    )

    # --------------------------------------------------------
    # CURRENT ROSTER QUALITY
    #
    # 45% - authoritative current-squad confirmation
    # 25% - deterministic Transfermarkt identity bridge
    # 30% - current market-value coverage
    #
    # This score is context quality only.
    # It does NOT alter model probabilities.
    # --------------------------------------------------------

    quality = (
        0.45
        *
        confirmed_rate
        +
        0.25
        *
        mapping_rate
        +
        0.30
        *
        valuation_coverage
    )

    # Transfermarkt-only fallback candidate pools are not
    # registration truth. Even with excellent ID/value coverage,
    # they must not look as trustworthy as confirmed squads.
    if fallback_rate >= 0.50:

        quality = min(
            quality,
            0.55,
        )

    return {
        "roster_quality": (
            clamp01(
                quality
            )
        ),

        "confirmed_rate": (
            confirmed_rate
        ),

        "fallback_rate": (
            fallback_rate
        ),

        "mapping_rate": (
            mapping_rate
        ),

        "valuation_coverage": (
            valuation_coverage
        ),
    }


def build_roster_quality_table(
    context: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for row in context.itertuples(
        index=False
    ):

        row_series = pd.Series(
            row._asdict()
        )

        metrics = (
            calculate_team_roster_quality(
                row_series
            )
        )

        rows.append(
            {
                "team": (
                    row_series[
                        "team"
                    ]
                ),

                **metrics,
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# MODEL DECISIVENESS
# ============================================================


def calculate_model_decisiveness(
    *,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
) -> dict:
    """
    Describe how concentrated the model's own probability
    distribution is.

    IMPORTANT:
    This is NOT confidence and is NOT used in the evidence
    confidence score.
    """

    probabilities = np.array(
        [
            home_probability,
            draw_probability,
            away_probability,
        ],
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    probabilities = (
        probabilities
        /
        probabilities.sum()
    )

    sorted_probabilities = np.sort(
        probabilities
    )[
        ::-1
    ]

    top_probability = float(
        sorted_probabilities[
            0
        ]
    )

    second_probability = float(
        sorted_probabilities[
            1
        ]
    )

    top_two_margin = (
        top_probability
        -
        second_probability
    )

    entropy = float(
        -np.sum(
            probabilities
            *
            np.log(
                probabilities
            )
        )
    )

    maximum_entropy = math.log(
        3
    )

    normalized_entropy = (
        entropy
        /
        maximum_entropy
    )

    decisiveness = (
        1.0
        -
        normalized_entropy
    )

    if top_two_margin >= 0.25:

        profile = "CLEAR_LEAN"

    elif top_two_margin >= 0.12:

        profile = "LEAN"

    else:

        profile = "TIGHT"

    return {
        "model_top_two_margin": (
            top_two_margin
        ),

        "model_probability_entropy": (
            normalized_entropy
        ),

        "model_decisiveness": (
            decisiveness
        ),

        "model_decisiveness_profile": (
            profile
        ),
    }


# ============================================================
# CONFIDENCE LABEL
# ============================================================


def confidence_label(
    confidence: float,
) -> str:

    if confidence >= 0.70:

        return "MODERATE_HIGH"

    if confidence >= 0.55:

        return "MODERATE"

    if confidence >= 0.40:

        return "LOW_MODERATE"

    return "LOW"


# ============================================================
# BUILD FIXTURE CONFIDENCE
# ============================================================


def build_fixture_confidence(
    *,
    prediction: pd.Series,
    roster_quality: pd.DataFrame,
) -> dict:

    home_team = str(
        prediction[
            "home_team"
        ]
    )

    away_team = str(
        prediction[
            "away_team"
        ]
    )

    home_match = roster_quality[
        roster_quality[
            "team"
        ].astype(
            str
        )
        ==
        home_team
    ]

    away_match = roster_quality[
        roster_quality[
            "team"
        ].astype(
            str
        )
        ==
        away_team
    ]

    if len(
        home_match
    ) != 1:

        raise RuntimeError(
            f"Roster quality missing "
            f"home team: {home_team}"
        )

    if len(
        away_match
    ) != 1:

        raise RuntimeError(
            f"Roster quality missing "
            f"away team: {away_team}"
        )

    home = home_match.iloc[
        0
    ]

    away = away_match.iloc[
        0
    ]

    home_roster_quality = clamp01(
        home[
            "roster_quality"
        ]
    )

    away_roster_quality = clamp01(
        away[
            "roster_quality"
        ]
    )

    current_roster_quality = (
        home_roster_quality
        +
        away_roster_quality
    ) / 2.0

    # --------------------------------------------------------
    # FEATURE FAMILY COVERAGE
    # --------------------------------------------------------

    elo_rate = safe_ratio(
        prediction[
            "elo_features_available"
        ],
        EXPECTED_ELO_FEATURES,
    )

    ucl_form_rate = safe_ratio(
        prediction[
            "ucl_form_features_available"
        ],
        EXPECTED_UCL_FORM_FEATURES,
    )

    domestic_rate = safe_ratio(
        prediction[
            "domestic_features_available"
        ],
        EXPECTED_DOMESTIC_FEATURES,
    )

    squad_model_rate = safe_ratio(
        prediction[
            "squad_features_available"
        ],
        EXPECTED_SQUAD_FEATURES,
    )

    # --------------------------------------------------------
    # RAW EVIDENCE CONFIDENCE
    # --------------------------------------------------------

    raw_confidence = (
        CONFIDENCE_WEIGHTS[
            "elo"
        ]
        *
        elo_rate

        +
        CONFIDENCE_WEIGHTS[
            "ucl_form"
        ]
        *
        ucl_form_rate

        +
        CONFIDENCE_WEIGHTS[
            "domestic"
        ]
        *
        domestic_rate

        +
        CONFIDENCE_WEIGHTS[
            "squad_model"
        ]
        *
        squad_model_rate

        +
        CONFIDENCE_WEIGHTS[
            "current_roster"
        ]
        *
        current_roster_quality
    )

    raw_confidence = clamp01(
        raw_confidence
    )

    confidence = raw_confidence

    caps = []

    # --------------------------------------------------------
    # KNOWN LIVE-SQUAD LIMITATION
    # --------------------------------------------------------

    squad_data_mode = str(
        prediction[
            "squad_data_mode"
        ]
    )

    if (
        squad_data_mode
        ==
        "SAFE_PARTIAL_NO_LIVE_USAGE"
    ):

        confidence = min(
            confidence,
            0.72,
        )

        caps.append(
            "NO_COMPLETE_LIVE_PLAYER_USAGE"
        )

    # --------------------------------------------------------
    # FALLBACK CURRENT ROSTER
    # --------------------------------------------------------

    weaker_roster = min(
        home_roster_quality,
        away_roster_quality,
    )

    if weaker_roster < 0.55:

        confidence = min(
            confidence,
            0.58,
        )

        caps.append(
            "LOW_CURRENT_ROSTER_SOURCE_QUALITY"
        )

    # --------------------------------------------------------
    # VERY LOW TOTAL FEATURE COVERAGE
    # --------------------------------------------------------

    overall_feature_rate = clamp01(
        prediction[
            "overall_feature_availability"
        ]
    )

    if overall_feature_rate < 0.45:

        confidence = min(
            confidence,
            0.55,
        )

        caps.append(
            "LOW_OVERALL_FEATURE_AVAILABILITY"
        )

    confidence = clamp01(
        confidence
    )

    decisiveness = (
        calculate_model_decisiveness(
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
    )

    return {
        "elo_evidence_rate": (
            elo_rate
        ),

        "ucl_form_evidence_rate": (
            ucl_form_rate
        ),

        "domestic_evidence_rate": (
            domestic_rate
        ),

        "squad_model_evidence_rate": (
            squad_model_rate
        ),

        "home_roster_quality": (
            home_roster_quality
        ),

        "away_roster_quality": (
            away_roster_quality
        ),

        "current_roster_quality": (
            current_roster_quality
        ),

        "raw_evidence_confidence": (
            raw_confidence
        ),

        "confidence": (
            confidence
        ),

        "confidence_percent": (
            confidence
            *
            100.0
        ),

        "confidence_label": (
            confidence_label(
                confidence
            )
        ),

        "confidence_caps": (
            "|".join(
                caps
            )
            if caps
            else
            "NONE"
        ),

        **decisiveness,
    }


# ============================================================
# QUALITY CHECK
# ============================================================


def run_quality_checks(
    dataframe: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE CONFIDENCE ENGINE QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    print(
        f"\nFixture rows: "
        f"{len(dataframe)}"
    )

    required_numeric = [
        "confidence",
        "raw_evidence_confidence",
        "elo_evidence_rate",
        "ucl_form_evidence_rate",
        "domestic_evidence_rate",
        "squad_model_evidence_rate",
        "home_roster_quality",
        "away_roster_quality",
        "current_roster_quality",
        "model_top_two_margin",
        "model_decisiveness",
    ]

    missing_values = int(
        dataframe[
            required_numeric
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing confidence values: "
        f"{missing_values}"
    )

    if missing_values:

        raise RuntimeError(
            "Missing confidence-engine values."
        )

    out_of_bounds = 0

    for column in required_numeric:

        invalid = (
            (
                dataframe[
                    column
                ]
                <
                0
            )
            |
            (
                dataframe[
                    column
                ]
                >
                1
            )
        )

        out_of_bounds += int(
            invalid.sum()
        )

    print(
        f"Out-of-range [0,1] values: "
        f"{out_of_bounds}"
    )

    if out_of_bounds:

        raise RuntimeError(
            "Confidence values outside [0,1]."
        )

    probability_columns = [
        "home_probability",
        "draw_probability",
        "away_probability",
    ]

    if not all(
        column in dataframe.columns
        for column in probability_columns
    ):

        raise RuntimeError(
            "Raw probability columns missing."
        )

    probability_sums = dataframe[
        probability_columns
    ].sum(
        axis=1
    )

    probability_error = int(
        (
            ~np.isclose(
                probability_sums,
                1.0,
                atol=1e-6,
            )
        ).sum()
    )

    print(
        f"Probability sum errors: "
        f"{probability_error}"
    )

    if probability_error:

        raise RuntimeError(
            "Raw probability sums changed."
        )

    if (
        dataframe[
            "confidence"
        ]
        >
        0.72
    ).any():

        raise RuntimeError(
            "Confidence exceeded live squad "
            "usage safety cap."
        )

    print(
        "\nRaw model probabilities preserved ✓"
    )

    print(
        "Confidence bounded independently ✓"
    )

    print(
        "Model decisiveness kept separate ✓"
    )

    print(
        "Missing live usage safety cap applied ✓"
    )


# ============================================================
# PRINT
# ============================================================


def print_confidence_report(
    dataframe: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "UCL 2026/27 - PREDICTION + EVIDENCE CONFIDENCE"
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
            f"\nModel pick: "
            f"{row.predicted_selection}"
        )

        print(
            f"Raw probability: "
            f"{row.max_probability * 100:.2f}%"
        )

        print(
            f"Evidence confidence: "
            f"{row.confidence_percent:.2f}% "
            f"({row.confidence_label})"
        )

        print(
            f"Model probability profile: "
            f"{row.model_decisiveness_profile}"
        )

        print(
            f"Top-two probability margin: "
            f"{row.model_top_two_margin * 100:.2f} pp"
        )

        print(
            "\nEvidence:"
        )

        print(
            f"  Elo:          "
            f"{row.elo_evidence_rate * 100:6.2f}%"
        )

        print(
            f"  UCL form:     "
            f"{row.ucl_form_evidence_rate * 100:6.2f}%"
        )

        print(
            f"  Domestic:     "
            f"{row.domestic_evidence_rate * 100:6.2f}%"
        )

        print(
            f"  Squad model:  "
            f"{row.squad_model_evidence_rate * 100:6.2f}%"
        )

        print(
            f"  Roster context:"
            f" {row.current_roster_quality * 100:6.2f}%"
        )

        print(
            f"\nConfidence caps: "
            f"{row.confidence_caps}"
        )


# ============================================================
# MAIN
# ============================================================


def build_live_confidence() -> None:

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - LIVE CONFIDENCE ENGINE"
    )

    print(
        "=" * 75
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Raw CatBoost probabilities will NOT be altered."
    )

    print(
        "Confidence measures evidence quality only."
    )

    print(
        "Model decisiveness is reported separately."
    )

    predictions = (
        load_predictions()
    )

    roster_context = (
        load_roster_context()
    )

    roster_quality = (
        build_roster_quality_table(
            roster_context
        )
    )

    confidence_rows = []

    for prediction in predictions.to_dict(
        orient="records"
    ):

        prediction = pd.Series(
            prediction
        )

        metrics = (
            build_fixture_confidence(
                prediction=prediction,
                roster_quality=roster_quality,
            )
        )

        confidence_rows.append(
            {
                **prediction.to_dict(),
                **metrics,
            }
        )

    output = pd.DataFrame(
        confidence_rows
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
        output
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
        "max_probability",
        "confidence",
        "confidence_label",
        "confidence_caps",
        "elo_evidence_rate",
        "ucl_form_evidence_rate",
        "domestic_evidence_rate",
        "squad_model_evidence_rate",
        "home_roster_quality",
        "away_roster_quality",
        "current_roster_quality",
        "model_top_two_margin",
        "model_decisiveness",
        "model_decisiveness_profile",
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
            "evidence_confidence_v1"
        ),

        "raw_probability_modified": False,

        "confidence_uses_model_probability": False,

        "model_decisiveness_combined_into_confidence": False,

        "weights": (
            CONFIDENCE_WEIGHTS
        ),

        "feature_family_totals": {
            "elo": (
                EXPECTED_ELO_FEATURES
            ),

            "ucl_form": (
                EXPECTED_UCL_FORM_FEATURES
            ),

            "domestic": (
                EXPECTED_DOMESTIC_FEATURES
            ),

            "squad": (
                EXPECTED_SQUAD_FEATURES
            ),

            "overall": (
                EXPECTED_TOTAL_FEATURES
            ),
        },

        "safety_caps": {
            "missing_complete_live_player_usage": (
                0.72
            ),

            "low_current_roster_source_quality": (
                0.58
            ),

            "overall_feature_availability_below_45_percent": (
                0.55
            ),
        },

        "confidence_labels": {
            "MODERATE_HIGH": (
                ">= 0.70"
            ),

            "MODERATE": (
                ">= 0.55 and < 0.70"
            ),

            "LOW_MODERATE": (
                ">= 0.40 and < 0.55"
            ),

            "LOW": (
                "< 0.40"
            ),
        },

        "rules": {
            "probability_is_not_confidence": True,

            "confidence_is_evidence_quality": True,

            "live_squad_usage_missingness_penalized": True,

            "fallback_candidate_squad_penalized": True,

            "current_roster_mapping_quality_used": True,

            "current_roster_value_coverage_used": True,

            "model_probability_margin_reported_separately": True,

            "historical_model_modified": False,

            "production_probabilities_modified": False,
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

    print_confidence_report(
        output
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE CONFIDENCE ENGINE COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nPrediction + confidence output:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nConfidence audit:"
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


if __name__ == "__main__":
    build_live_confidence()