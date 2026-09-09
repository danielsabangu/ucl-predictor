from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from src.config import (
    BASE_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TRAINED_MODELS_DIR,
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_PRODUCTION_FEATURES = 290

# Correct frozen production architecture.
#
# Historical feature-engineering stages:
#
#   Elo        =   6
#   UCL form   =  81
#   Domestic   = 102
#   Squad      = 101
#   ----------------
#   Total      = 290
#
# Transfer features are intentionally excluded from the
# production probability model.
EXPECTED_FEATURE_FAMILY_COUNTS = {
    "elo": 6,
    "ucl_form": 81,
    "domestic": 102,
    "squad": 101,
}

EXPECTED_CURRENT_UCL_CLUBS = 36

LOCAL_TIMEZONE = ZoneInfo(
    "Africa/Johannesburg"
)


# ============================================================
# PATHS
# ============================================================

PRODUCTION_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "production_catboost.joblib"
)

MASTER_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

ELO_MATCH_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_elo.csv"
)

FORM_MATCH_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_form.csv"
)

DOMESTIC_FORM_MATCH_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_domestic_form.csv"
)

SQUAD_MATCH_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_squad_strength.csv"
)

TRANSFER_MATCH_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_transfer_impact.csv"
)

ELO_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_elo_latest.csv"
)

FORM_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_form_latest.csv"
)

DOMESTIC_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_domestic_form_latest.csv"
)

SQUAD_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_squad_strength_latest.csv"
)

CURRENT_FIXTURE_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_fixtures_resolved.csv"
)

CURRENT_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
)

CURRENT_SQUAD_POOL_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "ucl_current_squad_pool.csv"
)

DOMESTIC_MATCH_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "domestic_matches.csv"
)

OUTPUT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_live_feature_readiness.csv"
)

METADATA_FILE = (
    BASE_DIR
    / "models"
    / "metadata"
    / "live_feature_readiness.json"
)


# ============================================================
# FILE HELPERS
# ============================================================


def require_file(
    path: Path,
    description: str,
) -> None:
    """
    Ensure a required local artifact exists.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}"
        )


def load_columns(
    path: Path,
) -> set[str]:
    """
    Read only a CSV header.
    """

    require_file(
        path,
        "Required feature-stage file",
    )

    dataframe = pd.read_csv(
        path,
        nrows=0,
    )

    return set(
        dataframe.columns
    )


# ============================================================
# PRODUCTION MODEL FEATURE NAMES
# ============================================================


def load_production_features() -> list[str]:
    """
    Recover the exact fitted input feature schema from the
    frozen production sklearn/CatBoost pipeline.
    """

    require_file(
        PRODUCTION_MODEL_FILE,
        "Production model",
    )

    model = joblib.load(
        PRODUCTION_MODEL_FILE
    )

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    # --------------------------------------------------------
    # FALLBACK:
    # Search fitted pipeline steps if the outer Pipeline does
    # not expose feature_names_in_ directly.
    # --------------------------------------------------------

    if feature_names is None:

        named_steps = getattr(
            model,
            "named_steps",
            None,
        )

        if named_steps is not None:

            for step in named_steps.values():

                candidate = getattr(
                    step,
                    "feature_names_in_",
                    None,
                )

                if candidate is not None:

                    feature_names = (
                        candidate
                    )

                    break

    if feature_names is None:

        raise RuntimeError(
            "Could not recover the production input feature "
            "names from production_catboost.joblib."
        )

    features = [
        str(
            feature
        )
        for feature in feature_names
    ]

    if (
        len(
            features
        )
        !=
        EXPECTED_PRODUCTION_FEATURES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PRODUCTION_FEATURES} "
            f"production features, found "
            f"{len(features)}."
        )

    duplicate_features = (
        len(
            features
        )
        -
        len(
            set(
                features
            )
        )
    )

    if duplicate_features:

        raise RuntimeError(
            "Duplicate feature names found in the frozen "
            "production input schema."
        )

    return features


# ============================================================
# FEATURE FAMILY RECONSTRUCTION
# ============================================================


def reconstruct_feature_families(
    production_features: list[str],
) -> dict[str, set[str]]:
    """
    Reconstruct the exact production feature families using
    historical pipeline stage differences.

    This is safer than guessing feature families using names
    or prefixes.
    """

    master_columns = load_columns(
        MASTER_FILE
    )

    elo_columns = load_columns(
        ELO_MATCH_FILE
    )

    form_columns = load_columns(
        FORM_MATCH_FILE
    )

    domestic_columns = load_columns(
        DOMESTIC_FORM_MATCH_FILE
    )

    squad_columns = load_columns(
        SQUAD_MATCH_FILE
    )

    transfer_columns = load_columns(
        TRANSFER_MATCH_FILE
    )

    generated_stage_features = {
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

        "transfer": (
            transfer_columns
            -
            squad_columns
        ),
    }

    production_set = set(
        production_features
    )

    families = {
        family: (
            columns
            &
            production_set
        )
        for (
            family,
            columns
        ) in generated_stage_features.items()
    }

    classified = set()

    for columns in families.values():

        classified.update(
            columns
        )

    families[
        "unclassified"
    ] = (
        production_set
        -
        classified
    )

    return families


# ============================================================
# VERIFY PRODUCTION ARCHITECTURE
# ============================================================


def verify_production_architecture(
    families: dict[str, set[str]],
) -> None:
    """
    Verify that the frozen model's 290 features match the
    exact architecture selected before final-test evaluation.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FROZEN PRODUCTION FEATURE ARCHITECTURE"
    )

    print(
        "=" * 75
    )

    total = 0

    for family in [
        "elo",
        "ucl_form",
        "domestic",
        "squad",
        "transfer",
        "unclassified",
    ]:

        count = len(
            families[
                family
            ]
        )

        total += count

        print(
            f"{family:15s}: "
            f"{count}"
        )

    print(
        f"\nTotal classified production features: "
        f"{total}"
    )

    # --------------------------------------------------------
    # EXACT FAMILY CHECKS
    # --------------------------------------------------------

    for (
        family,
        expected_count,
    ) in EXPECTED_FEATURE_FAMILY_COUNTS.items():

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
                f"Frozen feature-family mismatch for "
                f"{family}: expected "
                f"{expected_count}, found "
                f"{actual_count}."
            )

    # --------------------------------------------------------
    # TRANSFER MUST REMAIN EXCLUDED
    # --------------------------------------------------------

    transfer_count = len(
        families[
            "transfer"
        ]
    )

    if transfer_count:

        raise RuntimeError(
            f"Found {transfer_count} transfer features in "
            "the frozen production model. Transfer features "
            "must remain excluded from the probability model."
        )

    # --------------------------------------------------------
    # EVERY FEATURE MUST BE CLASSIFIED
    # --------------------------------------------------------

    if families[
        "unclassified"
    ]:

        raise RuntimeError(
            "Some production features could not be assigned "
            "to the historical feature pipeline:\n"
            f"{sorted(families['unclassified'])}"
        )

    expected_total = sum(
        EXPECTED_FEATURE_FAMILY_COUNTS.values()
    )

    if (
        expected_total
        !=
        EXPECTED_PRODUCTION_FEATURES
    ):

        raise RuntimeError(
            "Expected feature-family counts do not sum "
            "to 290."
        )

    if (
        total
        !=
        EXPECTED_PRODUCTION_FEATURES
    ):

        raise RuntimeError(
            f"Feature-family reconstruction produced "
            f"{total} features rather than "
            f"{EXPECTED_PRODUCTION_FEATURES}."
        )

    print(
        "\nFrozen 290-feature architecture verified ✓"
    )


# ============================================================
# CURRENT UCL TEAM MAP
# ============================================================


def load_current_teams() -> pd.DataFrame:
    """
    Load the resolved 36-team 2026/27 UCL universe.
    """

    require_file(
        CURRENT_TEAM_MAP_FILE,
        "Current UCL team map",
    )

    dataframe = pd.read_csv(
        CURRENT_TEAM_MAP_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "identity_status",
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
            "Current UCL team map missing columns: "
            f"{sorted(missing)}"
        )

    if (
        len(
            dataframe
        )
        !=
        EXPECTED_CURRENT_UCL_CLUBS
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_CURRENT_UCL_CLUBS} "
            f"current UCL clubs, found "
            f"{len(dataframe)}."
        )

    duplicate_teams = int(
        dataframe[
            "canonical_team"
        ].duplicated().sum()
    )

    if duplicate_teams:

        raise RuntimeError(
            "Duplicate canonical teams found in the "
            "current UCL team map."
        )

    return dataframe


# ============================================================
# TEAM COLUMN DETECTION
# ============================================================


def detect_team_column(
    dataframe: pd.DataFrame,
) -> str:
    """
    Detect the team-name column used by historical latest
    feature-state artifacts.
    """

    candidates = [
        "team",
        "canonical_team",
        "team_name",
        "club",
        "club_name",
    ]

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    raise RuntimeError(
        "Could not identify a team-name column.\n"
        f"Available columns:\n"
        f"{list(dataframe.columns)}"
    )


# ============================================================
# HISTORICAL LATEST-STATE COVERAGE
# ============================================================


def latest_team_coverage(
    *,
    path: Path,
    current_teams: set[str],
) -> tuple[
    int,
    list[str],
]:
    """
    Count current 2026/27 UCL teams already represented in a
    historical latest-state table.
    """

    require_file(
        path,
        "Latest feature-state table",
    )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    team_column = detect_team_column(
        dataframe
    )

    available_teams = set(
        dataframe[
            team_column
        ]
        .dropna()
        .astype(
            str
        )
    )

    covered = (
        current_teams
        &
        available_teams
    )

    missing = sorted(
        current_teams
        -
        available_teams
    )

    return (
        len(
            covered
        ),
        missing,
    )


# ============================================================
# DATE HELPERS
# ============================================================


def detect_date_column(
    dataframe: pd.DataFrame,
) -> str:
    """
    Detect a match/date column.
    """

    candidates = [
        "date",
        "match_date",
        "date_utc",
        "utc_date",
    ]

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    raise RuntimeError(
        "Could not identify a date column.\n"
        f"Available columns:\n"
        f"{list(dataframe.columns)}"
    )


def max_date_from_csv(
    path: Path,
) -> pd.Timestamp | None:
    """
    Return the latest valid timestamp in a CSV.
    """

    require_file(
        path,
        "Date-audit source",
    )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    date_column = detect_date_column(
        dataframe
    )

    dates = pd.to_datetime(
        dataframe[
            date_column
        ],
        errors="coerce",
        utc=True,
    )

    valid = dates.dropna()

    if valid.empty:

        return None

    return valid.max()


# ============================================================
# CURRENT FIXTURE STATUS
# ============================================================


def inspect_current_fixtures() -> dict:
    """
    Inspect the resolved 2026/27 UCL fixture state.
    """

    require_file(
        CURRENT_FIXTURE_FILE,
        "Resolved current fixture file",
    )

    fixtures = pd.read_csv(
        CURRENT_FIXTURE_FILE,
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
            fixtures.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Current fixture file missing columns: "
            f"{sorted(missing)}"
        )

    fixtures[
        "date_utc"
    ] = pd.to_datetime(
        fixtures[
            "date_utc"
        ],
        errors="coerce",
        utc=True,
    )

    finished = fixtures[
        fixtures[
            "status"
        ]
        ==
        "FINISHED"
    ].copy()

    unfinished = fixtures[
        fixtures[
            "status"
        ]
        !=
        "FINISHED"
    ].copy()

    latest_finished_date = None
    next_fixture_date = None

    if not finished.empty:

        latest_finished_date = (
            finished[
                "date_utc"
            ].max()
        )

    if not unfinished.empty:

        next_fixture_date = (
            unfinished[
                "date_utc"
            ].min()
        )

    return {
        "fixture_rows": int(
            len(
                fixtures
            )
        ),

        "finished_rows": int(
            len(
                finished
            )
        ),

        "unfinished_rows": int(
            len(
                unfinished
            )
        ),

        "latest_finished_date": (
            latest_finished_date
        ),

        "next_fixture_date": (
            next_fixture_date
        ),
    }


# ============================================================
# CURRENT SQUAD POOL
# ============================================================


def inspect_current_squad_pool() -> dict:
    """
    Inspect the unified 36-team current squad pool.
    """

    require_file(
        CURRENT_SQUAD_POOL_FILE,
        "Current squad pool",
    )

    pool = pd.read_csv(
        CURRENT_SQUAD_POOL_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "market_value_eur",
        "squad_source",
    }

    missing = (
        required
        -
        set(
            pool.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Current squad pool missing columns: "
            f"{sorted(missing)}"
        )

    player_rows = int(
        len(
            pool
        )
    )

    clubs = int(
        pool[
            "canonical_team"
        ].nunique()
    )

    valued_players = int(
        pool[
            "market_value_eur"
        ].notna().sum()
    )

    valuation_coverage = (
        valued_players
        /
        player_rows
        if player_rows
        else 0.0
    )

    fallback_rows = int(
        (
            pool[
                "squad_source"
            ]
            ==
            "transfermarkt_current_club_fallback"
        ).sum()
    )

    return {
        "player_rows": (
            player_rows
        ),

        "clubs": (
            clubs
        ),

        "valued_players": (
            valued_players
        ),

        "valuation_coverage": (
            valuation_coverage
        ),

        "fallback_rows": (
            fallback_rows
        ),
    }


# ============================================================
# READINESS ROW
# ============================================================


def readiness_row(
    *,
    family: str,
    feature_count: int,
    historical_team_coverage: int,
    status: str,
    blocker: str,
    required_action: str,
) -> dict:
    """
    Standardize one feature-family readiness result.
    """

    return {
        "feature_family": (
            family
        ),

        "production_feature_count": int(
            feature_count
        ),

        "current_ucl_teams_covered_by_historical_latest_state": int(
            historical_team_coverage
        ),

        "current_ucl_team_count": (
            EXPECTED_CURRENT_UCL_CLUBS
        ),

        "historical_state_coverage": (
            historical_team_coverage
            /
            EXPECTED_CURRENT_UCL_CLUBS
        ),

        "live_readiness_status": (
            status
        ),

        "current_blocker": (
            blocker
        ),

        "required_action": (
            required_action
        ),
    }


# ============================================================
# BUILD LIVE READINESS AUDIT
# ============================================================


def build_readiness_audit(
    *,
    families: dict[str, set[str]],
    current_teams: pd.DataFrame,
    fixture_status: dict,
    squad_status: dict,
) -> pd.DataFrame:
    """
    Determine exactly what still needs refreshing before the
    290 production features can be generated for live fixtures.
    """

    current_team_names = set(
        current_teams[
            "canonical_team"
        ].astype(
            str
        )
    )

    (
        elo_coverage,
        elo_missing,
    ) = latest_team_coverage(
        path=ELO_LATEST_FILE,
        current_teams=current_team_names,
    )

    (
        form_coverage,
        form_missing,
    ) = latest_team_coverage(
        path=FORM_LATEST_FILE,
        current_teams=current_team_names,
    )

    (
        domestic_coverage,
        domestic_missing,
    ) = latest_team_coverage(
        path=DOMESTIC_LATEST_FILE,
        current_teams=current_team_names,
    )

    (
        squad_coverage,
        squad_missing,
    ) = latest_team_coverage(
        path=SQUAD_LATEST_FILE,
        current_teams=current_team_names,
    )

    rows = []

    # --------------------------------------------------------
    # ELO
    # --------------------------------------------------------

    rows.append(
        readiness_row(
            family="elo",
            feature_count=len(
                families[
                    "elo"
                ]
            ),
            historical_team_coverage=elo_coverage,
            status="REFRESH_REQUIRED",
            blocker=(
                f"{fixture_status['finished_rows']} "
                "2026/27 UCL matches have already finished. "
                "The historical latest Elo state therefore "
                "does not yet include those results."
            ),
            required_action=(
                "Replay finished 2026/27 UCL results "
                "chronologically through the frozen Elo logic."
            ),
        )
    )

    # --------------------------------------------------------
    # UCL FORM
    # --------------------------------------------------------

    rows.append(
        readiness_row(
            family="ucl_form",
            feature_count=len(
                families[
                    "ucl_form"
                ]
            ),
            historical_team_coverage=form_coverage,
            status="REFRESH_REQUIRED",
            blocker=(
                f"{fixture_status['finished_rows']} "
                "new UCL matches are not yet included in the "
                "rolling 5/10-match team histories."
            ),
            required_action=(
                "Replay finished 2026/27 UCL results using "
                "the frozen rolling UCL-form definitions."
            ),
        )
    )

    # --------------------------------------------------------
    # DOMESTIC
    # --------------------------------------------------------

    domestic_latest_date = (
        max_date_from_csv(
            DOMESTIC_MATCH_FILE
        )
    )

    if domestic_latest_date is None:

        domestic_date_text = (
            "unknown"
        )

    else:

        domestic_date_text = (
            domestic_latest_date
            .date()
            .isoformat()
        )

    rows.append(
        readiness_row(
            family="domestic",
            feature_count=len(
                families[
                    "domestic"
                ]
            ),
            historical_team_coverage=domestic_coverage,
            status="REFRESH_REQUIRED",
            blocker=(
                "The currently collected domestic match source "
                f"ends at {domestic_date_text}. "
                "Domestic form and known-competitive-rest "
                "features must be brought up to the fixture "
                "prediction timestamp."
            ),
            required_action=(
                "Refresh current domestic results and rebuild "
                "the same 102 domestic features using only "
                "matches strictly before each UCL fixture."
            ),
        )
    )

    # --------------------------------------------------------
    # SQUAD
    # --------------------------------------------------------

    squad_status_label = (
        "PARTIAL"
    )

    if (
        squad_status[
            "clubs"
        ]
        !=
        EXPECTED_CURRENT_UCL_CLUBS
    ):

        squad_status_label = (
            "REFRESH_REQUIRED"
        )

    rows.append(
        readiness_row(
            family="squad",
            feature_count=len(
                families[
                    "squad"
                ]
            ),
            historical_team_coverage=squad_coverage,
            status=squad_status_label,
            blocker=(
                "The current identity/value squad pool now "
                f"covers {squad_status['clubs']}/36 clubs "
                f"with "
                f"{squad_status['valuation_coverage'] * 100:.2f}% "
                "valuation coverage. However, the frozen "
                "101 squad features also rely on player usage, "
                "recent appearances, lineups, core XI and "
                "continuity information."
            ),
            required_action=(
                "Build or refresh the live player-usage / "
                "lineup state and then recreate the frozen "
                "101 squad features."
            ),
        )
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT AUDIT DETAILS
# ============================================================


def print_details(
    *,
    audit: pd.DataFrame,
    fixture_status: dict,
    squad_status: dict,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE 2026/27 DATA STATE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nCurrent fixture rows: "
        f"{fixture_status['fixture_rows']}"
    )

    print(
        f"Finished 2026/27 UCL matches: "
        f"{fixture_status['finished_rows']}"
    )

    print(
        f"Upcoming/unplayed fixtures: "
        f"{fixture_status['unfinished_rows']}"
    )

    if (
        fixture_status[
            "latest_finished_date"
        ]
        is not None
    ):

        print(
            "Latest finished UCL fixture: "
            f"{fixture_status['latest_finished_date']}"
        )

    if (
        fixture_status[
            "next_fixture_date"
        ]
        is not None
    ):

        print(
            "Next scheduled UCL fixture: "
            f"{fixture_status['next_fixture_date']}"
        )

    print(
        f"\nCurrent squad pool rows: "
        f"{squad_status['player_rows']:,}"
    )

    print(
        f"Current squad clubs: "
        f"{squad_status['clubs']}"
    )

    print(
        f"Current valued players: "
        f"{squad_status['valued_players']:,}"
    )

    print(
        "Current valuation coverage: "
        f"{squad_status['valuation_coverage'] * 100:.2f}%"
    )

    print(
        f"Fallback squad-pool rows: "
        f"{squad_status['fallback_rows']}"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE FEATURE READINESS"
    )

    print(
        "=" * 75
    )

    print(
        "\n"
        +
        audit[
            [
                "feature_family",
                "production_feature_count",
                "current_ucl_teams_covered_by_historical_latest_state",
                "live_readiness_status",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nBlockers / next actions:"
    )

    for row in audit.itertuples(
        index=False
    ):

        print(
            f"\n{row.feature_family.upper()} "
            f"({row.production_feature_count} features)"
        )

        print(
            f"  Status: "
            f"{row.live_readiness_status}"
        )

        print(
            f"  Blocker: "
            f"{row.current_blocker}"
        )

        print(
            f"  Next: "
            f"{row.required_action}"
        )


# ============================================================
# SAVE OUTPUTS
# ============================================================


def save_outputs(
    *,
    audit: pd.DataFrame,
    fixture_status: dict,
    squad_status: dict,
    families: dict[str, set[str]],
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    now = datetime.now(
        tz=LOCAL_TIMEZONE
    )

    metadata = {
        "created_at_local": (
            now.isoformat()
        ),

        "timezone": (
            "Africa/Johannesburg"
        ),

        "production_model": (
            "ucl_predictor_v1"
        ),

        "production_feature_count": (
            EXPECTED_PRODUCTION_FEATURES
        ),

        "feature_family_counts": {
            family: int(
                len(
                    features
                )
            )
            for (
                family,
                features
            ) in families.items()
        },

        "expected_feature_family_counts": (
            EXPECTED_FEATURE_FAMILY_COUNTS
        ),

        "current_fixture_state": {
            "fixture_rows": (
                fixture_status[
                    "fixture_rows"
                ]
            ),

            "finished_rows": (
                fixture_status[
                    "finished_rows"
                ]
            ),

            "unfinished_rows": (
                fixture_status[
                    "unfinished_rows"
                ]
            ),

            "latest_finished_date": (
                fixture_status[
                    "latest_finished_date"
                ].isoformat()
                if fixture_status[
                    "latest_finished_date"
                ]
                is not None
                else None
            ),

            "next_fixture_date": (
                fixture_status[
                    "next_fixture_date"
                ].isoformat()
                if fixture_status[
                    "next_fixture_date"
                ]
                is not None
                else None
            ),
        },

        "current_squad_state": (
            squad_status
        ),

        "architecture_locked": True,

        "transfer_features_excluded": True,

        "historical_training_data_modified": False,

        "important_rules": {
            "exact_290_feature_schema_required": True,

            "no_post_test_model_tuning": True,

            "live_features_must_use_only_pre_fixture_information": True,

            "missing_feature_values_may_be_preserved_for_model_imputation": True,

            "feature_definitions_must_not_change_for_live_prediction": True,
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


# ============================================================
# MAIN
# ============================================================


def audit_live_feature_readiness() -> None:

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - LIVE FEATURE READINESS AUDIT"
    )

    print(
        "=" * 75
    )

    production_features = (
        load_production_features()
    )

    print(
        f"\nFrozen production features: "
        f"{len(production_features)}"
    )

    families = (
        reconstruct_feature_families(
            production_features
        )
    )

    verify_production_architecture(
        families
    )

    current_teams = (
        load_current_teams()
    )

    fixture_status = (
        inspect_current_fixtures()
    )

    squad_status = (
        inspect_current_squad_pool()
    )

    audit = build_readiness_audit(
        families=families,
        current_teams=current_teams,
        fixture_status=fixture_status,
        squad_status=squad_status,
    )

    print_details(
        audit=audit,
        fixture_status=fixture_status,
        squad_status=squad_status,
    )

    save_outputs(
        audit=audit,
        fixture_status=fixture_status,
        squad_status=squad_status,
        families=families,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE FEATURE READINESS AUDIT COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nReadiness report:"
    )

    print(
        OUTPUT_FILE
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
        "The production model remains frozen."
    )

    print(
        "This audit does not modify training data, "
        "feature definitions or model parameters."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    audit_live_feature_readiness()