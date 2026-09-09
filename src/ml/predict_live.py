from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    INTERIM_DATA_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    TRAINED_MODELS_DIR,
)

from src.features import domestic_form
from src.features import elo
from src.features import squad_strength as squad
from src.features import team_form


# ============================================================
# SETTINGS
# ============================================================

CURRENT_SEASON = 2026

EXPECTED_PRODUCTION_FEATURES = 290

EXPECTED_FEATURE_FAMILIES = {
    "elo": 6,
    "ucl_form": 81,
    "domestic": 102,
    "squad": 101,
}

CLASS_TO_RESULT = {
    0: "H",
    1: "D",
    2: "A",
}


# ============================================================
# PATHS
# ============================================================

PRODUCTION_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "production_catboost.joblib"
)

FIXTURES_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_fixtures_resolved.csv"
)

LIVE_UCL_REPLAY_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "live_2026_27"
    / "ucl_matches_master_live_replay.csv"
)

LIVE_DOMESTIC_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "live_2026_27"
    / "domestic_matches_live.csv"
)

LIVE_SQUAD_STATE_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "ucl_team_squad_state_live_safe_current_36.csv"
)

LIVE_SQUAD_CONTEXT_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "ucl_team_squad_context_live_current_36.csv"
)

HISTORICAL_STAGE_FILES = {
    "master": (
        PROCESSED_DATA_DIR
        / "ucl_matches_master.csv"
    ),

    "elo": (
        PROCESSED_DATA_DIR
        / "ucl_matches_with_elo.csv"
    ),

    "ucl_form": (
        PROCESSED_DATA_DIR
        / "ucl_matches_with_form.csv"
    ),

    "domestic": (
        PROCESSED_DATA_DIR
        / "ucl_matches_with_domestic_form.csv"
    ),

    "squad": (
        PROCESSED_DATA_DIR
        / "ucl_matches_with_squad_strength.csv"
    ),
}

OUTPUT_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions.csv"
)

FEATURE_MATRIX_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_features.csv"
)

METADATA_FILE = (
    PREDICTIONS_DIR
    / "ucl_2026_27_next_matchday_predictions_metadata.json"
)


# ============================================================
# BASIC HELPERS
# ============================================================


def require_file(
    path: Path,
    description: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}"
        )


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    *,
    description: str,
    required: bool = True,
) -> str | None:

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    if required:

        raise RuntimeError(
            f"Could not identify {description}.\n"
            f"Candidates tried:\n"
            f"{candidates}\n"
            f"Available columns:\n"
            f"{list(dataframe.columns)}"
        )

    return None


def safe_numeric(
    value,
) -> float:

    return pd.to_numeric(
        pd.Series(
            [
                value
            ]
        ),
        errors="coerce",
    ).iloc[
        0
    ]


# ============================================================
# PRODUCTION MODEL
# ============================================================


def load_production_model():

    require_file(
        PRODUCTION_MODEL_FILE,
        "Production CatBoost model",
    )

    model = joblib.load(
        PRODUCTION_MODEL_FILE
    )

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is None:

        if hasattr(
            model,
            "named_steps",
        ):

            for step in model.named_steps.values():

                feature_names = getattr(
                    step,
                    "feature_names_in_",
                    None,
                )

                if feature_names is not None:
                    break

    if feature_names is None:

        raise RuntimeError(
            "Could not recover frozen production "
            "feature names from model."
        )

    feature_names = [
        str(
            feature
        )
        for feature in feature_names
    ]

    if (
        len(
            feature_names
        )
        !=
        EXPECTED_PRODUCTION_FEATURES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PRODUCTION_FEATURES} "
            f"production features, found "
            f"{len(feature_names)}."
        )

    if (
        len(
            set(
                feature_names
            )
        )
        !=
        len(
            feature_names
        )
    ):

        raise RuntimeError(
            "Duplicate production feature names found."
        )

    return (
        model,
        feature_names,
    )


# ============================================================
# RECONSTRUCT EXACT FROZEN FEATURE FAMILIES
# ============================================================


def load_columns(
    path: Path,
) -> list[str]:

    require_file(
        path,
        f"Historical stage file {path.name}",
    )

    return list(
        pd.read_csv(
            path,
            nrows=1,
        ).columns
    )


def reconstruct_feature_families(
    production_features: list[str],
) -> dict[str, list[str]]:

    production_set = set(
        production_features
    )

    master_columns = set(
        load_columns(
            HISTORICAL_STAGE_FILES[
                "master"
            ]
        )
    )

    elo_columns = set(
        load_columns(
            HISTORICAL_STAGE_FILES[
                "elo"
            ]
        )
    )

    form_columns = set(
        load_columns(
            HISTORICAL_STAGE_FILES[
                "ucl_form"
            ]
        )
    )

    domestic_columns = set(
        load_columns(
            HISTORICAL_STAGE_FILES[
                "domestic"
            ]
        )
    )

    squad_columns = set(
        load_columns(
            HISTORICAL_STAGE_FILES[
                "squad"
            ]
        )
    )

    raw_stage_features = {
        "elo": (
            elo_columns
            -
            master_columns
        ),

        "ucl_form": (
            form_columns
            -
            elo_columns
        ),

        "domestic": (
            domestic_columns
            -
            form_columns
        ),

        "squad": (
            squad_columns
            -
            domestic_columns
        ),
    }

    families = {}

    classified = set()

    for family, generated in raw_stage_features.items():

        used = [
            feature
            for feature in production_features
            if (
                feature
                in
                generated
            )
        ]

        families[
            family
        ] = used

        classified.update(
            used
        )

    unclassified = (
        production_set
        -
        classified
    )

    if unclassified:

        raise RuntimeError(
            "Production features could not be "
            "classified:\n"
            f"{sorted(unclassified)}"
        )

    for (
        family,
        expected_count,
    ) in EXPECTED_FEATURE_FAMILIES.items():

        actual_count = len(
            families[
                family
            ]
        )

        if (
            actual_count
            !=
            expected_count
        ):

            raise RuntimeError(
                f"{family} expected "
                f"{expected_count} features, "
                f"found {actual_count}."
            )

    return families


# ============================================================
# FIXTURES
# ============================================================


def load_next_matchday_fixtures() -> pd.DataFrame:

    require_file(
        FIXTURES_FILE,
        "Resolved 2026/27 UCL fixtures",
    )

    dataframe = pd.read_csv(
        FIXTURES_FILE,
        low_memory=False,
    )

    required = {
        "date_utc",
        "status",
        "home_team",
        "away_team",
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
            "Fixture file missing columns:\n"
            f"{sorted(missing)}"
        )

    dataframe[
        "date_utc"
    ] = pd.to_datetime(
        dataframe[
            "date_utc"
        ],
        errors="raise",
        utc=True,
    )

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    upcoming = dataframe[
        (
            dataframe[
                "status"
            ].astype(
                str
            )
            !=
            "FINISHED"
        )
        &
        (
            dataframe[
                "date_utc"
            ]
            >
            now_utc
        )
    ].copy()

    if upcoming.empty:

        raise RuntimeError(
            "No future UCL fixtures found."
        )

    earliest_date = (
        upcoming[
            "date_utc"
        ].min().date()
    )

    next_matchday = upcoming[
        upcoming[
            "date_utc"
        ].dt.date
        ==
        earliest_date
    ].copy()

    next_matchday = next_matchday.sort_values(
        by=[
            "date_utc",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    if next_matchday.empty:

        raise RuntimeError(
            "Next matchday selection returned no fixtures."
        )

    return next_matchday


# ============================================================
# LIVE UCL BASE
# ============================================================


def load_live_ucl_replay() -> tuple[
    pd.DataFrame,
    str,
]:

    require_file(
        LIVE_UCL_REPLAY_FILE,
        "Live UCL replay master",
    )

    dataframe = pd.read_csv(
        LIVE_UCL_REPLAY_FILE,
        low_memory=False,
    )

    date_column = first_existing_column(
        dataframe,
        [
            "date",
            "match_date",
            "date_utc",
            "utc_date",
        ],
        description="UCL replay date column",
    )

    dataframe[
        date_column
    ] = pd.to_datetime(
        dataframe[
            date_column
        ],
        errors="raise",
        utc=True,
    )

    required = {
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
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
            "Live UCL replay missing required "
            "columns:\n"
            f"{sorted(missing)}"
        )

    dataframe[
        "season"
    ] = pd.to_numeric(
        dataframe[
            "season"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe = dataframe.sort_values(
        by=[
            date_column,
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    return (
        dataframe,
        date_column,
    )


# ============================================================
# PLACEHOLDER FUTURE ROW
# ============================================================


def build_future_placeholder(
    *,
    base: pd.DataFrame,
    date_column: str,
    fixture: pd.Series,
) -> pd.DataFrame:
    """
    Build one synthetic result row solely to allow the existing
    historical feature functions to calculate PRE-MATCH state.

    The fake result is never used as a model feature and each
    future fixture is processed independently, so it cannot
    contaminate another future fixture.
    """

    row = {
        column: np.nan
        for column in base.columns
    }

    row[
        date_column
    ] = fixture[
        "date_utc"
    ]

    row[
        "season"
    ] = CURRENT_SEASON

    row[
        "home_team"
    ] = str(
        fixture[
            "home_team"
        ]
    )

    row[
        "away_team"
    ] = str(
        fixture[
            "away_team"
        ]
    )

    # Placeholder outcome used only because historical
    # transformers update their state after creating pre-match
    # features. It is not a production model input.
    row[
        "home_goals"
    ] = 0

    row[
        "away_goals"
    ] = 0

    row[
        "result"
    ] = "D"

    if (
        "scoreline"
        in
        row
    ):

        row[
            "scoreline"
        ] = "0-0"

    if (
        "source"
        in
        row
    ):

        row[
            "source"
        ] = (
            "LIVE_PREDICTION_PLACEHOLDER"
        )

    if (
        "status"
        in
        row
    ):

        row[
            "status"
        ] = "TIMED"

    if (
        "stage"
        in
        row
        and
        "stage"
        in
        fixture.index
    ):

        row[
            "stage"
        ] = fixture[
            "stage"
        ]

    if (
        "matchday"
        in
        row
        and
        "matchday"
        in
        fixture.index
    ):

        row[
            "matchday"
        ] = fixture[
            "matchday"
        ]

    if (
        "match_id"
        in
        row
    ):

        match_id_column = None

        for candidate in [
            "match_id",
            "football_data_match_id",
            "id",
        ]:

            if candidate in fixture.index:

                match_id_column = (
                    candidate
                )

                break

        if match_id_column is not None:

            row[
                "match_id"
            ] = fixture[
                match_id_column
            ]

        else:

            row[
                "match_id"
            ] = (
                "LIVE_PREDICTION"
            )

    future = pd.DataFrame(
        [
            row
        ],
        columns=base.columns,
    )

    future[
        date_column
    ] = pd.to_datetime(
        future[
            date_column
        ],
        errors="raise",
        utc=True,
    )

    future[
        "season"
    ] = CURRENT_SEASON

    future[
        "home_goals"
    ] = 0

    future[
        "away_goals"
    ] = 0

    future[
        "result"
    ] = "D"

    return future


# ============================================================
# DOMESTIC HISTORIES
# ============================================================


def load_live_domestic_histories():

    require_file(
        LIVE_DOMESTIC_FILE,
        "Live domestic dataset",
    )

    domestic = pd.read_csv(
        LIVE_DOMESTIC_FILE,
        low_memory=False,
    )

    if (
        "date"
        not in
        domestic.columns
    ):

        raise RuntimeError(
            "Live domestic dataset missing date column."
        )

    domestic[
        "date"
    ] = pd.to_datetime(
        domestic[
            "date"
        ],
        errors="raise",
    )

    (
        histories,
        dates_lookup,
    ) = (
        domestic_form.build_domestic_histories(
            domestic
        )
    )

    return (
        domestic,
        histories,
        dates_lookup,
    )


# ============================================================
# SAFE LIVE SQUAD STATE
# ============================================================


def load_live_squad_state() -> pd.DataFrame:

    require_file(
        LIVE_SQUAD_STATE_FILE,
        "Safe live squad state",
    )

    dataframe = pd.read_csv(
        LIVE_SQUAD_STATE_FILE,
        low_memory=False,
    )

    if (
        "team"
        not in
        dataframe.columns
    ):

        raise RuntimeError(
            "Safe live squad state missing team column."
        )

    if (
        len(
            dataframe
        )
        !=
        36
    ):

        raise RuntimeError(
            f"Expected 36 safe squad state rows, "
            f"found {len(dataframe)}."
        )

    if (
        dataframe[
            "team"
        ].duplicated().any()
    ):

        raise RuntimeError(
            "Duplicate team rows in safe squad state."
        )

    return dataframe


def verify_squad_reference_date(
    *,
    squad_state: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> None:

    if (
        "reference_timestamp_utc"
        not in
        squad_state.columns
    ):

        raise RuntimeError(
            "Safe squad state has no "
            "reference_timestamp_utc."
        )

    references = pd.to_datetime(
        squad_state[
            "reference_timestamp_utc"
        ],
        errors="coerce",
        utc=True,
    ).dropna()

    if references.empty:

        raise RuntimeError(
            "Safe squad state reference timestamp "
            "is missing."
        )

    reference_dates = set(
        references.dt.date
    )

    fixture_dates = set(
        fixtures[
            "date_utc"
        ].dt.date
    )

    if (
        len(
            reference_dates
        )
        !=
        1
    ):

        raise RuntimeError(
            "Safe squad state contains multiple "
            "reference dates."
        )

    reference_date = next(
        iter(
            reference_dates
        )
    )

    if fixture_dates != {
        reference_date
    }:

        raise RuntimeError(
            "Safe squad state was generated for a "
            "different match date.\n"
            f"Squad reference date: "
            f"{reference_date}\n"
            f"Prediction fixture dates: "
            f"{sorted(fixture_dates)}\n"
            "Rerun:\n"
            "python -m src.features.live_squad_state"
        )


# ============================================================
# SQUAD FEATURE ATTACHMENT
# ============================================================


def infer_difference_feature_mapping(
    squad_production_features: list[str],
) -> dict[str, str]:

    team_side_features = set()

    for feature in squad.TEAM_FEATURES:

        team_side_features.add(
            f"home_{feature}"
        )

        team_side_features.add(
            f"away_{feature}"
        )

    production_set = set(
        squad_production_features
    )

    missing_side_features = (
        team_side_features
        -
        production_set
    )

    if missing_side_features:

        raise RuntimeError(
            "Production squad architecture does not "
            "contain expected home/away team features:\n"
            f"{sorted(missing_side_features)}"
        )

    remaining = (
        production_set
        -
        team_side_features
    )

    if (
        len(
            remaining
        )
        !=
        len(
            squad.DIFFERENCE_FEATURES
        )
    ):

        raise RuntimeError(
            "Could not isolate the 17 squad "
            "difference features.\n"
            f"Remaining production features:\n"
            f"{sorted(remaining)}"
        )

    mapping = {}

    unused = set(
        remaining
    )

    for base_feature in squad.DIFFERENCE_FEATURES:

        exact_candidates = [
            f"{base_feature}_diff",
            f"{base_feature}_difference",
        ]

        candidate = None

        for exact in exact_candidates:

            if exact in unused:

                candidate = exact
                break

        if candidate is None:

            search_token = base_feature.replace(
                "squad_",
                "",
            )

            candidates = [
                feature
                for feature in unused
                if (
                    base_feature
                    in
                    feature
                    or
                    search_token
                    in
                    feature
                )
            ]

            if (
                len(
                    candidates
                )
                !=
                1
            ):

                raise RuntimeError(
                    "Could not uniquely map difference "
                    f"feature '{base_feature}'.\n"
                    f"Candidates: {sorted(candidates)}\n"
                    f"Unused production differences:\n"
                    f"{sorted(unused)}"
                )

            candidate = candidates[
                0
            ]

        mapping[
            base_feature
        ] = candidate

        unused.remove(
            candidate
        )

    if unused:

        raise RuntimeError(
            "Unused squad production difference "
            f"features remain:\n"
            f"{sorted(unused)}"
        )

    return mapping


def attach_safe_squad_features(
    *,
    feature_row: dict,
    home_team: str,
    away_team: str,
    squad_state: pd.DataFrame,
    squad_production_features: list[str],
    difference_mapping: dict[str, str],
) -> None:

    home_match = squad_state[
        squad_state[
            "team"
        ].astype(
            str
        )
        ==
        home_team
    ]

    away_match = squad_state[
        squad_state[
            "team"
        ].astype(
            str
        )
        ==
        away_team
    ]

    if (
        len(
            home_match
        )
        !=
        1
    ):

        raise RuntimeError(
            f"Safe squad state missing unique "
            f"home team: {home_team}"
        )

    if (
        len(
            away_match
        )
        !=
        1
    ):

        raise RuntimeError(
            f"Safe squad state missing unique "
            f"away team: {away_team}"
        )

    home_state = home_match.iloc[
        0
    ]

    away_state = away_match.iloc[
        0
    ]

    assigned = set()

    for team_feature in squad.TEAM_FEATURES:

        home_column = (
            f"home_{team_feature}"
        )

        away_column = (
            f"away_{team_feature}"
        )

        feature_row[
            home_column
        ] = safe_numeric(
            home_state.get(
                team_feature,
                np.nan,
            )
        )

        feature_row[
            away_column
        ] = safe_numeric(
            away_state.get(
                team_feature,
                np.nan,
            )
        )

        assigned.add(
            home_column
        )

        assigned.add(
            away_column
        )

    for base_feature in squad.DIFFERENCE_FEATURES:

        difference_column = (
            difference_mapping[
                base_feature
            ]
        )

        home_value = feature_row[
            f"home_{base_feature}"
        ]

        away_value = feature_row[
            f"away_{base_feature}"
        ]

        feature_row[
            difference_column
        ] = squad.safe_difference(
            home_value,
            away_value,
        )

        assigned.add(
            difference_column
        )

    expected = set(
        squad_production_features
    )

    if assigned != expected:

        missing = (
            expected
            -
            assigned
        )

        extra = (
            assigned
            -
            expected
        )

        raise RuntimeError(
            "Safe squad production assignment "
            "mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )


# ============================================================
# FEATURE BUILDING FOR ONE FIXTURE
# ============================================================


def build_fixture_features(
    *,
    fixture: pd.Series,
    base_replay: pd.DataFrame,
    date_column: str,
    production_features: list[str],
    families: dict[str, list[str]],
    domestic_histories,
    domestic_dates,
    squad_state: pd.DataFrame,
    squad_difference_mapping: dict[str, str],
) -> tuple[
    dict,
    dict,
]:

    home_team = str(
        fixture[
            "home_team"
        ]
    )

    away_team = str(
        fixture[
            "away_team"
        ]
    )

    fixture_date = pd.Timestamp(
        fixture[
            "date_utc"
        ]
    )

    # --------------------------------------------------------
    # APPEND ONE FUTURE FIXTURE ONLY
    # --------------------------------------------------------

    future = build_future_placeholder(
        base=base_replay,
        date_column=date_column,
        fixture=fixture,
    )

    combined = pd.concat(
        [
            base_replay.copy(),
            future,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        by=[
            date_column,
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    future_mask = (
        (
            combined[
                date_column
            ]
            ==
            fixture_date
        )
        &
        (
            combined[
                "home_team"
            ].astype(
                str
            )
            ==
            home_team
        )
        &
        (
            combined[
                "away_team"
            ].astype(
                str
            )
            ==
            away_team
        )
    )

    if (
        int(
            future_mask.sum()
        )
        !=
        1
    ):

        raise RuntimeError(
            "Could not uniquely identify future "
            f"fixture row: "
            f"{home_team} vs {away_team}"
        )

    # --------------------------------------------------------
    # ELO
    # --------------------------------------------------------

    elo_frame, _ = (
        elo.build_elo_features(
            combined.copy()
        )
    )

    # --------------------------------------------------------
    # UCL FORM
    # --------------------------------------------------------

    form_frame, _ = (
        team_form.build_form_features(
            elo_frame.copy()
        )
    )

    # --------------------------------------------------------
    # DOMESTIC
    # --------------------------------------------------------

    domestic_frame = (
        domestic_form.build_domestic_form_features(
            form_frame.copy(),
            domestic_histories,
            domestic_dates,
        )
    )

    final_mask = (
        (
            pd.to_datetime(
                domestic_frame[
                    date_column
                ],
                errors="raise",
                utc=True,
            )
            ==
            fixture_date
        )
        &
        (
            domestic_frame[
                "home_team"
            ].astype(
                str
            )
            ==
            home_team
        )
        &
        (
            domestic_frame[
                "away_team"
            ].astype(
                str
            )
            ==
            away_team
        )
    )

    if (
        int(
            final_mask.sum()
        )
        !=
        1
    ):

        raise RuntimeError(
            "Could not recover unique final "
            "pre-match feature row for "
            f"{home_team} vs {away_team}."
        )

    generated = domestic_frame.loc[
        final_mask
    ].iloc[
        0
    ]

    feature_row = {
        feature: np.nan
        for feature in production_features
    }

    # --------------------------------------------------------
    # COPY EXACT FROZEN ELO / FORM / DOMESTIC FEATURES
    # --------------------------------------------------------

    for family in [
        "elo",
        "ucl_form",
        "domestic",
    ]:

        for feature in families[
            family
        ]:

            if feature not in generated.index:

                raise RuntimeError(
                    f"Generated fixture row is missing "
                    f"{family} feature:\n"
                    f"{feature}"
                )

            feature_row[
                feature
            ] = safe_numeric(
                generated[
                    feature
                ]
            )

    # --------------------------------------------------------
    # SAFE SQUAD FEATURES
    # --------------------------------------------------------

    attach_safe_squad_features(
        feature_row=feature_row,
        home_team=home_team,
        away_team=away_team,
        squad_state=squad_state,
        squad_production_features=families[
            "squad"
        ],
        difference_mapping=squad_difference_mapping,
    )

    # --------------------------------------------------------
    # AVAILABILITY
    # --------------------------------------------------------

    availability = {}

    for family, features in families.items():

        values = [
            feature_row[
                feature
            ]
            for feature in features
        ]

        available = int(
            pd.Series(
                values
            ).notna().sum()
        )

        availability[
            family
        ] = {
            "available": (
                available
            ),

            "total": (
                len(
                    features
                )
            ),

            "rate": (
                available
                /
                len(
                    features
                )
            ),
        }

    total_available = int(
        pd.Series(
            [
                feature_row[
                    feature
                ]
                for feature
                in
                production_features
            ]
        ).notna().sum()
    )

    availability[
        "overall"
    ] = {
        "available": (
            total_available
        ),

        "total": (
            len(
                production_features
            )
        ),

        "rate": (
            total_available
            /
            len(
                production_features
            )
        ),
    }

    return (
        feature_row,
        availability,
    )


# ============================================================
# MODEL PREDICTION
# ============================================================


def predict_probabilities(
    *,
    model,
    production_features: list[str],
    feature_row: dict,
) -> tuple[
    float,
    float,
    float,
    int,
]:

    X = pd.DataFrame(
        [
            feature_row
        ],
        columns=production_features,
    )

    for column in X.columns:

        X[
            column
        ] = pd.to_numeric(
            X[
                column
            ],
            errors="coerce",
        )

    probabilities = model.predict_proba(
        X
    )[
        0
    ]

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is None:

        classes = np.array(
            [
                0,
                1,
                2,
            ]
        )

    classes = [
        int(
            value
        )
        for value in classes
    ]

    probability_by_class = {
        class_value: float(
            probability
        )
        for (
            class_value,
            probability,
        ) in zip(
            classes,
            probabilities,
        )
    }

    missing_classes = (
        {
            0,
            1,
            2,
        }
        -
        set(
            probability_by_class
        )
    )

    if missing_classes:

        raise RuntimeError(
            "Production model probability output "
            "does not contain all H/D/A classes.\n"
            f"Missing: {sorted(missing_classes)}"
        )

    home_probability = (
        probability_by_class[
            0
        ]
    )

    draw_probability = (
        probability_by_class[
            1
        ]
    )

    away_probability = (
        probability_by_class[
            2
        ]
    )

    predicted_class = max(
        probability_by_class,
        key=probability_by_class.get,
    )

    return (
        home_probability,
        draw_probability,
        away_probability,
        predicted_class,
    )


# ============================================================
# SUMMARY / LABEL
# ============================================================


def predicted_selection(
    *,
    predicted_class: int,
    home_team: str,
    away_team: str,
) -> str:

    if predicted_class == 0:

        return home_team

    if predicted_class == 1:

        return "Draw"

    if predicted_class == 2:

        return away_team

    raise RuntimeError(
        f"Unknown predicted class: "
        f"{predicted_class}"
    )


# ============================================================
# QUALITY CHECK
# ============================================================


def run_quality_checks(
    *,
    prediction_rows: list[dict],
    feature_matrix: pd.DataFrame,
    production_features: list[str],
    families: dict[str, list[str]],
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE PRODUCTION PREDICTION QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    print(
        f"\nPrediction rows: "
        f"{len(prediction_rows)}"
    )

    print(
        f"Feature matrix rows: "
        f"{len(feature_matrix)}"
    )

    print(
        f"Feature matrix columns: "
        f"{len(feature_matrix.columns)}"
    )

    if (
        len(
            feature_matrix.columns
        )
        !=
        EXPECTED_PRODUCTION_FEATURES
    ):

        raise RuntimeError(
            "Feature matrix does not contain "
            "exactly 290 columns."
        )

    if (
        list(
            feature_matrix.columns
        )
        !=
        production_features
    ):

        raise RuntimeError(
            "Live feature matrix order does not match "
            "the frozen production model."
        )

    for (
        family,
        expected_count,
    ) in EXPECTED_FEATURE_FAMILIES.items():

        actual_count = len(
            families[
                family
            ]
        )

        print(
            f"{family:10s}: "
            f"{actual_count}"
        )

        if (
            actual_count
            !=
            expected_count
        ):

            raise RuntimeError(
                f"{family} architecture mismatch."
            )

    infinite_count = int(
        np.isinf(
            feature_matrix.to_numpy(
                dtype=float,
            )
        ).sum()
    )

    print(
        f"\nInfinite feature values: "
        f"{infinite_count}"
    )

    if infinite_count:

        raise RuntimeError(
            "Infinite feature values detected."
        )

    for row in prediction_rows:

        probability_sum = (
            row[
                "home_probability"
            ]
            +
            row[
                "draw_probability"
            ]
            +
            row[
                "away_probability"
            ]
        )

        if not np.isclose(
            probability_sum,
            1.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                "Prediction probabilities do not "
                "sum to 1."
            )

    print(
        "\nFrozen 290-feature order preserved ✓"
    )

    print(
        "No infinite values ✓"
    )

    print(
        "H/D/A probabilities sum to 1 ✓"
    )

    print(
        "Future placeholder outcomes are not "
        "production features ✓"
    )


# ============================================================
# PRINT RESULTS
# ============================================================


def print_predictions(
    predictions: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "UCL 2026/27 - NEXT MATCHDAY PRODUCTION FORECAST"
    )

    print(
        "=" * 75
    )

    for row in predictions.itertuples(
        index=False
    ):

        print(
            "\n"
            + "-"
            *
            75
        )

        print(
            f"{row.kickoff_utc}"
        )

        print(
            f"{row.home_team} "
            f"vs "
            f"{row.away_team}"
        )

        print(
            "\nProbabilities:"
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
            f"\nModel pick: "
            f"{row.predicted_selection}"
        )

        print(
            f"Maximum probability: "
            f"{row.max_probability * 100:.2f}%"
        )

        print(
            "\nFeature availability:"
        )

        print(
            f"  Elo:      "
            f"{row.elo_features_available}/6"
        )

        print(
            f"  UCL form: "
            f"{row.ucl_form_features_available}/81"
        )

        print(
            f"  Domestic: "
            f"{row.domestic_features_available}/102"
        )

        print(
            f"  Squad:    "
            f"{row.squad_features_available}/101"
        )

        print(
            f"  Overall:  "
            f"{row.overall_features_available}/290 "
            f"({row.overall_feature_availability * 100:.2f}%)"
        )

        if (
            row.squad_features_available
            <
            50
        ):

            print(
                "\n  NOTE: Squad usage coverage is "
                "intentionally limited."
            )

            print(
                "  Treat raw probability separately "
                "from confidence."
            )


# ============================================================
# MAIN
# ============================================================


def predict_live() -> None:

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - LIVE PRODUCTION PREDICTIONS"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD FROZEN MODEL
    # --------------------------------------------------------

    (
        model,
        production_features,
    ) = load_production_model()

    print(
        f"\nProduction model loaded:"
    )

    print(
        PRODUCTION_MODEL_FILE
    )

    print(
        f"\nFrozen production features: "
        f"{len(production_features)}"
    )

    # --------------------------------------------------------
    # EXACT FEATURE FAMILY ARCHITECTURE
    # --------------------------------------------------------

    families = (
        reconstruct_feature_families(
            production_features
        )
    )

    print(
        "\nFrozen feature families:"
    )

    for family in [
        "elo",
        "ucl_form",
        "domestic",
        "squad",
    ]:

        print(
            f"  {family:10s}: "
            f"{len(families[family])}"
        )

    # --------------------------------------------------------
    # NEXT MATCHDAY
    # --------------------------------------------------------

    fixtures = (
        load_next_matchday_fixtures()
    )

    print(
        f"\nNext matchday date: "
        f"{fixtures['date_utc'].min().date()}"
    )

    print(
        f"Upcoming fixtures selected: "
        f"{len(fixtures)}"
    )

    for row in fixtures.itertuples(
        index=False
    ):

        print(
            f"  {row.date_utc} | "
            f"{row.home_team} vs "
            f"{row.away_team}"
        )

    # --------------------------------------------------------
    # LIVE BASES
    # --------------------------------------------------------

    (
        base_replay,
        date_column,
    ) = load_live_ucl_replay()

    (
        live_domestic,
        domestic_histories,
        domestic_dates,
    ) = load_live_domestic_histories()

    squad_state = (
        load_live_squad_state()
    )

    verify_squad_reference_date(
        squad_state=squad_state,
        fixtures=fixtures,
    )

    squad_difference_mapping = (
        infer_difference_feature_mapping(
            families[
                "squad"
            ]
        )
    )

    print(
        f"\nLive UCL replay matches: "
        f"{len(base_replay):,}"
    )

    print(
        f"Live domestic matches: "
        f"{len(live_domestic):,}"
    )

    print(
        f"Safe squad state teams: "
        f"{len(squad_state)}"
    )

    # --------------------------------------------------------
    # BUILD + PREDICT
    # --------------------------------------------------------

    feature_rows = []

    prediction_rows = []

    for fixture in fixtures.to_dict(
        orient="records"
    ):

        fixture = pd.Series(
            fixture
        )

        home_team = str(
            fixture[
                "home_team"
            ]
        )

        away_team = str(
            fixture[
                "away_team"
            ]
        )

        kickoff = pd.Timestamp(
            fixture[
                "date_utc"
            ]
        )

        print(
            "\n"
            + "-"
            *
            75
        )

        print(
            f"Building features: "
            f"{home_team} vs {away_team}"
        )

        (
            feature_row,
            availability,
        ) = build_fixture_features(
            fixture=fixture,
            base_replay=base_replay,
            date_column=date_column,
            production_features=production_features,
            families=families,
            domestic_histories=domestic_histories,
            domestic_dates=domestic_dates,
            squad_state=squad_state,
            squad_difference_mapping=squad_difference_mapping,
        )

        (
            home_probability,
            draw_probability,
            away_probability,
            predicted_class,
        ) = predict_probabilities(
            model=model,
            production_features=production_features,
            feature_row=feature_row,
        )

        max_probability = max(
            home_probability,
            draw_probability,
            away_probability,
        )

        feature_rows.append(
            feature_row
        )

        prediction_rows.append(
            {
                "kickoff_utc": (
                    kickoff
                ),

                "home_team": (
                    home_team
                ),

                "away_team": (
                    away_team
                ),

                "home_probability": (
                    home_probability
                ),

                "draw_probability": (
                    draw_probability
                ),

                "away_probability": (
                    away_probability
                ),

                "predicted_class": (
                    predicted_class
                ),

                "predicted_result": (
                    CLASS_TO_RESULT[
                        predicted_class
                    ]
                ),

                "predicted_selection": (
                    predicted_selection(
                        predicted_class=predicted_class,
                        home_team=home_team,
                        away_team=away_team,
                    )
                ),

                "max_probability": (
                    max_probability
                ),

                "elo_features_available": (
                    availability[
                        "elo"
                    ][
                        "available"
                    ]
                ),

                "ucl_form_features_available": (
                    availability[
                        "ucl_form"
                    ][
                        "available"
                    ]
                ),

                "domestic_features_available": (
                    availability[
                        "domestic"
                    ][
                        "available"
                    ]
                ),

                "squad_features_available": (
                    availability[
                        "squad"
                    ][
                        "available"
                    ]
                ),

                "overall_features_available": (
                    availability[
                        "overall"
                    ][
                        "available"
                    ]
                ),

                "overall_feature_availability": (
                    availability[
                        "overall"
                    ][
                        "rate"
                    ]
                ),

                "squad_data_mode": (
                    "SAFE_PARTIAL_NO_LIVE_USAGE"
                ),
            }
        )

    # --------------------------------------------------------
    # EXACT 290-COLUMN FEATURE MATRIX
    # --------------------------------------------------------

    feature_matrix = pd.DataFrame(
        feature_rows,
        columns=production_features,
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    predictions = predictions.sort_values(
        by=[
            "kickoff_utc",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    run_quality_checks(
        prediction_rows=prediction_rows,
        feature_matrix=feature_matrix,
        production_features=production_features,
        families=families,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    feature_matrix.to_csv(
        FEATURE_MATRIX_FILE,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "model": (
            "ucl_predictor_v1"
        ),

        "model_file": str(
            PRODUCTION_MODEL_FILE
        ),

        "feature_count": (
            EXPECTED_PRODUCTION_FEATURES
        ),

        "feature_families": {
            family: len(
                features
            )
            for (
                family,
                features,
            ) in families.items()
        },

        "fixture_count": int(
            len(
                predictions
            )
        ),

        "prediction_date_utc": (
            predictions[
                "kickoff_utc"
            ]
            .min()
            .date()
            .isoformat()
        ),

        "ucl_history_rows": int(
            len(
                base_replay
            )
        ),

        "domestic_history_rows": int(
            len(
                live_domestic
            )
        ),

        "squad_data_mode": (
            "SAFE_PARTIAL_NO_LIVE_USAGE"
        ),

        "rules": {
            "production_model_modified": False,

            "production_feature_order_modified": False,

            "historical_training_data_modified": False,

            "future_results_used_as_features": False,

            "future_placeholder_result_is_model_input": False,

            "each_future_fixture_processed_independently": True,

            "elo_logic_reused": True,

            "ucl_form_logic_reused": True,

            "domestic_logic_reused": True,

            "synthetic_live_player_usage_used": False,

            "current_roster_context_substituted_for_usage_features": False,

            "missing_squad_usage_left_for_frozen_imputer": True,

            "probability_is_not_confidence": True,
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

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_predictions(
        predictions
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE PRODUCTION PREDICTIONS COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nPredictions:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nExact 290-column feature matrix:"
    )

    print(
        FEATURE_MATRIX_FILE
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
        "These are raw CatBoost probabilities."
    )

    print(
        "Probability is NOT the same as confidence."
    )

    print(
        "The confidence engine has not yet been applied."
    )


if __name__ == "__main__":
    predict_live()