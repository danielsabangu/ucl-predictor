from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    INTERIM_DATA_DIR,
    REPORTS_DIR,
)

from src.features import squad_strength as squad


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_CURRENT_TEAMS = 36
EXPECTED_TEAM_FEATURES = 42
EXPECTED_DIFFERENCE_FEATURES = 17
EXPECTED_PRODUCTION_SQUAD_FEATURES = 101


# ============================================================
# PATHS
# ============================================================

CURRENT_FIXTURES_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_fixtures_resolved.csv"
)

CURRENT_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
)

EXTENDED_CLUB_MAP_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map_2026_27.csv"
)

CURRENT_SQUAD_POOL_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
    / "ucl_current_squad_pool.csv"
)

OUTPUT_DIR = (
    INTERIM_DATA_DIR
    / "players"
    / "current_2026_27"
)

LIVE_SQUAD_STATE_FILE = (
    OUTPUT_DIR
    / "ucl_team_squad_state_live_safe_current_36.csv"
)

LIVE_SQUAD_CONTEXT_FILE = (
    OUTPUT_DIR
    / "ucl_team_squad_context_live_current_36.csv"
)

AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_live_squad_state_audit.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "ucl_live_squad_state_metadata.json"
)


# ============================================================
# SAFE FEATURE POLICY
# ============================================================

# These can still be represented honestly from historical
# evidence availability.
SAFE_HISTORY_FEATURES = {
    "squad_appearance_history_available",
    "squad_lineup_history_available",
    "squad_valuation_history_available",
    "squad_days_since_last_player_evidence",
}

# This MUST be forced to 0 because our current live usage source
# is incomplete after the historical snapshot cutoffs.
FORCED_ZERO_FEATURES = {
    "squad_recent_history_available",
}

# Every remaining TEAM_FEATURES field is dynamic usage /
# continuity / active-player valuation context and therefore
# becomes NaN rather than being fabricated.
DYNAMIC_FEATURES = (
    set(squad.TEAM_FEATURES)
    -
    SAFE_HISTORY_FEATURES
    -
    FORCED_ZERO_FEATURES
)


# ============================================================
# HELPERS
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
    description: str,
) -> str:

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    raise RuntimeError(
        f"Could not identify {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns:\n"
        f"{list(dataframe.columns)}"
    )


def json_safe_value(value):

    if isinstance(
        value,
        pd.Timestamp,
    ):

        if pd.isna(value):

            return None

        return value.isoformat()

    if isinstance(
        value,
        np.generic,
    ):

        return value.item()

    if pd.isna(value):

        return None

    return value


# ============================================================
# ARCHITECTURE CHECK
# ============================================================


def verify_frozen_squad_architecture() -> None:

    team_feature_count = len(
        squad.TEAM_FEATURES
    )

    difference_feature_count = len(
        squad.DIFFERENCE_FEATURES
    )

    production_count = (
        team_feature_count
        *
        2
        +
        difference_feature_count
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FROZEN SQUAD FEATURE ARCHITECTURE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nTeam features: "
        f"{team_feature_count}"
    )

    print(
        f"Difference features: "
        f"{difference_feature_count}"
    )

    print(
        f"Production squad features: "
        f"{production_count}"
    )

    if (
        team_feature_count
        !=
        EXPECTED_TEAM_FEATURES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TEAM_FEATURES} "
            f"team squad features, "
            f"found {team_feature_count}."
        )

    if (
        difference_feature_count
        !=
        EXPECTED_DIFFERENCE_FEATURES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_DIFFERENCE_FEATURES} "
            f"difference features, "
            f"found {difference_feature_count}."
        )

    if (
        production_count
        !=
        EXPECTED_PRODUCTION_SQUAD_FEATURES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PRODUCTION_SQUAD_FEATURES} "
            f"production squad features, "
            f"reconstructed {production_count}."
        )

    print(
        "\nFrozen squad architecture verified ✓"
    )


# ============================================================
# REFERENCE TIMESTAMP
# ============================================================


def determine_reference_timestamp() -> pd.Timestamp:
    require_file(
        CURRENT_FIXTURES_FILE,
        "Resolved current UCL fixtures",
    )

    fixtures = pd.read_csv(
        CURRENT_FIXTURES_FILE,
        low_memory=False,
    )

    if (
        "date_utc"
        not in
        fixtures.columns
    ):
        raise RuntimeError(
            "Current fixture file has no date_utc column."
        )

    fixtures[
        "date_utc"
    ] = pd.to_datetime(
        fixtures[
            "date_utc"
        ],
        errors="raise",
        utc=True,
    )

    if (
        "status"
        not in
        fixtures.columns
    ):
        raise RuntimeError(
            "Current fixture file has no status column."
        )

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    status = (
        fixtures[
            "status"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.upper()
    )

    upcoming = fixtures[
        (
            fixtures[
                "date_utc"
            ]
            >
            now_utc
        )
        &
        (
            status
            !=
            "FINISHED"
        )
    ].copy()

    if upcoming.empty:
        raise RuntimeError(
            "No future UCL fixtures found. "
            f"Current UTC time: {now_utc.isoformat()}"
        )

    reference = (
        upcoming[
            "date_utc"
        ]
        .sort_values()
        .iloc[0]
    )

    return pd.Timestamp(
        reference
    )

def load_current_teams() -> pd.DataFrame:

    require_file(
        CURRENT_TEAM_MAP_FILE,
        "Current UCL team map",
    )

    dataframe = pd.read_csv(
        CURRENT_TEAM_MAP_FILE,
        low_memory=False,
    )

    if (
        "canonical_team"
        not in
        dataframe.columns
    ):

        raise RuntimeError(
            "Current team map missing canonical_team."
        )

    dataframe = dataframe.drop_duplicates(
        subset=[
            "canonical_team"
        ]
    ).copy()

    if (
        len(
            dataframe
        )
        !=
        EXPECTED_CURRENT_TEAMS
    ):

        raise RuntimeError(
            f"Expected 36 current teams, "
            f"found {len(dataframe)}."
        )

    return dataframe


# ============================================================
# TRANSFERMARKT CLUB MAP
# ============================================================


def load_extended_club_map() -> dict[str, int]:

    require_file(
        EXTENDED_CLUB_MAP_FILE,
        "Extended Transfermarkt UCL club map",
    )

    dataframe = pd.read_csv(
        EXTENDED_CLUB_MAP_FILE,
        low_memory=False,
    )

    team_column = first_existing_column(
        dataframe,
        [
            "canonical_team",
            "team",
            "canonical_name",
        ],
        "canonical team column",
    )

    club_id_column = first_existing_column(
        dataframe,
        [
            "transfermarkt_club_id",
            "club_id",
            "transfermarkt_id",
        ],
        "Transfermarkt club ID column",
    )

    mapping = {}

    for row in dataframe.itertuples(
        index=False
    ):

        row_dict = row._asdict()

        team = str(
            row_dict[
                team_column
            ]
        )

        club_id = pd.to_numeric(
            row_dict[
                club_id_column
            ],
            errors="coerce",
        )

        if pd.isna(
            club_id
        ):

            continue

        mapping[
            team
        ] = int(
            club_id
        )

    return mapping


# ============================================================
# CURRENT SQUAD CONTEXT
# ============================================================


def load_current_squad_pool() -> pd.DataFrame:

    require_file(
        CURRENT_SQUAD_POOL_FILE,
        "Unified current squad pool",
    )

    dataframe = pd.read_csv(
        CURRENT_SQUAD_POOL_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "transfermarkt_club_id",
        "transfermarkt_player_id",
        "player_name",
        "market_value_eur",
        "confirmed_current_squad",
        "transfermarkt_fallback_pool",
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
            "Current squad pool missing columns:\n"
            f"{sorted(missing)}"
        )

    return dataframe


def build_current_squad_context(
    *,
    pool: pd.DataFrame,
    current_teams: set[str],
) -> pd.DataFrame:

    rows = []

    for team in sorted(
        current_teams
    ):

        team_pool = pool[
            pool[
                "canonical_team"
            ].astype(
                str
            )
            ==
            team
        ].copy()

        values = pd.to_numeric(
            team_pool[
                "market_value_eur"
            ],
            errors="coerce",
        )

        positive_values = values[
            values
            >
            0
        ]

        total_players = len(
            team_pool
        )

        valued_players = int(
            positive_values.notna().sum()
        )

        value_coverage = (
            valued_players
            /
            total_players
            if total_players
            else
            0.0
        )

        confirmed = int(
            pd.to_numeric(
                team_pool[
                    "confirmed_current_squad"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        fallback = int(
            pd.to_numeric(
                team_pool[
                    "transfermarkt_fallback_pool"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        mapped_players = int(
            team_pool[
                "transfermarkt_player_id"
            ].notna().sum()
        )

        top5_value = (
            float(
                positive_values
                .nlargest(
                    5
                )
                .sum()
            )
            if not positive_values.empty
            else np.nan
        )

        rows.append(
            {
                "team": (
                    team
                ),

                "current_squad_player_rows": (
                    total_players
                ),

                "current_confirmed_squad_rows": (
                    confirmed
                ),

                "current_fallback_pool_rows": (
                    fallback
                ),

                "current_tm_mapped_players": (
                    mapped_players
                ),

                "current_valued_players": (
                    valued_players
                ),

                "current_value_coverage": (
                    value_coverage
                ),

                "current_total_market_value_m_eur": (
                    float(
                        positive_values.sum()
                        /
                        1_000_000
                    )
                    if not positive_values.empty
                    else np.nan
                ),

                "current_mean_market_value_m_eur": (
                    float(
                        positive_values.mean()
                        /
                        1_000_000
                    )
                    if not positive_values.empty
                    else np.nan
                ),

                "current_median_market_value_m_eur": (
                    float(
                        positive_values.median()
                        /
                        1_000_000
                    )
                    if not positive_values.empty
                    else np.nan
                ),

                "current_top5_market_value_m_eur": (
                    top5_value
                    /
                    1_000_000
                    if not pd.isna(
                        top5_value
                    )
                    else np.nan
                ),
            }
        )

    context = pd.DataFrame(
        rows
    )

    if (
        len(
            context
        )
        !=
        EXPECTED_CURRENT_TEAMS
    ):

        raise RuntimeError(
            "Current squad context does not contain "
            "36 team rows."
        )

    return context


# ============================================================
# HISTORICAL PLAYER EVIDENCE
# ============================================================


def load_historical_histories(
    club_ids: set[int],
):
    """
    Load the existing frozen Transfermarkt histories.

    These are used only to identify whether historical evidence
    exists and when the latest evidence occurred.

    They are NOT treated as complete current-season usage.
    """

    appearances = squad.load_appearances()
    lineups = squad.load_lineups()
    valuations = squad.load_valuations()

    (
        appearance_histories,
        lineup_histories,
    ) = squad.build_club_histories(
        appearances=appearances,
        lineups=lineups,
        club_ids=club_ids,
    )

    valuation_histories = (
        squad.build_valuation_histories(
            valuations
        )
    )

    return (
        appearances,
        lineups,
        valuations,
        appearance_histories,
        lineup_histories,
        valuation_histories,
    )


# ============================================================
# SAFE TEAM SNAPSHOT
# ============================================================


def build_safe_team_snapshot(
    *,
    team: str,
    club_id: int,
    reference_timestamp: pd.Timestamp,
    appearance_histories,
    lineup_histories,
    valuation_histories,
) -> dict:

    # Historical squad module uses match-date semantics.
    # Use a naive UTC timestamp to match its historical tables.
    reference_naive = (
        reference_timestamp
        .tz_convert(
            "UTC"
        )
        .tz_localize(
            None
        )
    )

    raw = squad.calculate_team_snapshot(
        club_id=club_id,
        match_date=reference_naive,
        appearance_histories=appearance_histories,
        lineup_histories=lineup_histories,
        valuation_histories=valuation_histories,
    )

    safe = {
        "team": (
            team
        ),

        "club_id": (
            club_id
        ),

        "reference_timestamp_utc": (
            reference_timestamp
        ),
    }

    for feature in squad.TEAM_FEATURES:

        if feature in SAFE_HISTORY_FEATURES:

            safe[
                feature
            ] = raw.get(
                feature,
                np.nan,
            )

        elif feature in FORCED_ZERO_FEATURES:

            safe[
                feature
            ] = 0

        else:

            safe[
                feature
            ] = np.nan

    safe[
        "historical_snapshot_latest_player_evidence_date"
    ] = raw.get(
        "squad_latest_player_evidence_date",
        pd.NaT,
    )

    safe[
        "historical_snapshot_latest_valuation_date_used"
    ] = raw.get(
        "squad_latest_valuation_date_used",
        pd.NaT,
    )

    safe[
        "live_usage_source_complete"
    ] = 0

    safe[
        "live_usage_policy"
    ] = (
        "RECENT_USAGE_MASKED_NO_COMPLETE_LIVE_SOURCE"
    )

    return safe


# ============================================================
# BUILD ALL 36 TEAM STATES
# ============================================================


def build_live_safe_state(
    *,
    current_teams: pd.DataFrame,
    club_map: dict[str, int],
    reference_timestamp: pd.Timestamp,
    appearance_histories,
    lineup_histories,
    valuation_histories,
) -> pd.DataFrame:

    rows = []

    missing_clubs = []

    for team in current_teams[
        "canonical_team"
    ].astype(
        str
    ):

        if team not in club_map:

            missing_clubs.append(
                team
            )

            continue

        club_id = club_map[
            team
        ]

        rows.append(
            build_safe_team_snapshot(
                team=team,
                club_id=club_id,
                reference_timestamp=reference_timestamp,
                appearance_histories=appearance_histories,
                lineup_histories=lineup_histories,
                valuation_histories=valuation_histories,
            )
        )

    if missing_clubs:

        raise RuntimeError(
            "Current teams missing from extended "
            "Transfermarkt club map:\n"
            f"{sorted(missing_clubs)}"
        )

    output = pd.DataFrame(
        rows
    )

    output = output.sort_values(
        "team"
    ).reset_index(
        drop=True
    )

    if (
        len(
            output
        )
        !=
        EXPECTED_CURRENT_TEAMS
    ):

        raise RuntimeError(
            f"Expected 36 live squad state rows, "
            f"found {len(output)}."
        )

    return output


# ============================================================
# AUDIT
# ============================================================


def build_audit(
    *,
    state: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:

    audit = state[
        [
            "team",
            "club_id",
            "reference_timestamp_utc",
            "squad_appearance_history_available",
            "squad_lineup_history_available",
            "squad_recent_history_available",
            "squad_valuation_history_available",
            "squad_days_since_last_player_evidence",
            "historical_snapshot_latest_player_evidence_date",
            "historical_snapshot_latest_valuation_date_used",
            "live_usage_source_complete",
            "live_usage_policy",
        ]
    ].merge(
        context,
        on="team",
        how="left",
        validate="one_to_one",
    )

    return audit


# ============================================================
# QUALITY CHECK
# ============================================================


def run_quality_checks(
    *,
    state: pd.DataFrame,
    context: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    valuations: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE SQUAD STATE QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    print(
        f"\nCurrent team rows: "
        f"{len(state)}/36"
    )

    missing_team_features = [
        feature
        for feature in squad.TEAM_FEATURES
        if feature not in state.columns
    ]

    print(
        f"Missing TEAM_FEATURE columns: "
        f"{len(missing_team_features)}"
    )

    if missing_team_features:

        raise RuntimeError(
            "Missing squad team features:\n"
            f"{missing_team_features}"
        )

    dynamic_non_null = int(
        state[
            sorted(
                DYNAMIC_FEATURES
            )
        ]
        .notna()
        .sum()
        .sum()
    )

    print(
        f"Dynamic usage values accidentally populated: "
        f"{dynamic_non_null}"
    )

    if dynamic_non_null:

        raise RuntimeError(
            "Dynamic squad features were populated even "
            "though no complete live usage source exists."
        )

    recent_history_values = set(
        pd.to_numeric(
            state[
                "squad_recent_history_available"
            ],
            errors="coerce",
        )
        .fillna(
            -1
        )
        .astype(
            int
        )
    )

    print(
        f"Recent-history availability values: "
        f"{sorted(recent_history_values)}"
    )

    if recent_history_values != {
        0
    }:

        raise RuntimeError(
            "squad_recent_history_available must "
            "be forced to 0."
        )

    if (
        len(
            context
        )
        !=
        EXPECTED_CURRENT_TEAMS
    ):

        raise RuntimeError(
            "Current squad context does not cover "
            "all 36 clubs."
        )

    appearance_dates = pd.to_datetime(
        appearances[
            "date"
        ],
        errors="coerce",
    ).dropna()

    lineup_dates = pd.to_datetime(
        lineups[
            "date"
        ],
        errors="coerce",
    ).dropna()

    valuation_dates = pd.to_datetime(
        valuations[
            "date"
        ],
        errors="coerce",
    ).dropna()

    print(
        f"\nHistorical appearances latest: "
        f"{appearance_dates.max().date()}"
    )

    print(
        f"Historical lineups latest: "
        f"{lineup_dates.max().date()}"
    )

    print(
        f"Historical valuations latest: "
        f"{valuation_dates.max().date()}"
    )

    print(
        "\nNo synthetic starts created ✓"
    )

    print(
        "No synthetic appearances created ✓"
    )

    print(
        "No synthetic minutes created ✓"
    )

    print(
        "No fake core XI created ✓"
    )

    print(
        "Current roster context kept outside "
        "production feature semantics ✓"
    )


# ============================================================
# SUMMARY
# ============================================================


def print_summary(
    *,
    state: pd.DataFrame,
    context: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 SAFE LIVE SQUAD STATE SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nReference timestamp: "
        f"{reference_timestamp}"
    )

    print(
        f"Current team rows: "
        f"{len(state)}"
    )

    print(
        f"Production team features present: "
        f"{len(squad.TEAM_FEATURES)}"
    )

    print(
        f"Difference features defined: "
        f"{len(squad.DIFFERENCE_FEATURES)}"
    )

    print(
        f"Resulting fixture squad architecture: "
        f"{2 * len(squad.TEAM_FEATURES) + len(squad.DIFFERENCE_FEATURES)}"
    )

    populated_counts = (
        state[
            squad.TEAM_FEATURES
        ]
        .notna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nNon-null values by model team feature:"
    )

    print(
        populated_counts.to_string()
    )

    print(
        "\nCurrent squad context:"
    )

    display_columns = [
        "team",
        "current_squad_player_rows",
        "current_confirmed_squad_rows",
        "current_fallback_pool_rows",
        "current_tm_mapped_players",
        "current_valued_players",
        "current_value_coverage",
        "current_total_market_value_m_eur",
        "current_top5_market_value_m_eur",
    ]

    print(
        context[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Recent usage-dependent squad features are "
        "NaN by design."
    )

    print(
        "The production model's existing imputer "
        "will handle those missing values."
    )

    print(
        "Current roster/value context remains available "
        "for confidence and explanations."
    )


# ============================================================
# SAVE
# ============================================================


def save_outputs(
    *,
    state: pd.DataFrame,
    context: pd.DataFrame,
    audit: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    valuations: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state.to_csv(
        LIVE_SQUAD_STATE_FILE,
        index=False,
        encoding="utf-8",
    )

    context.to_csv(
        LIVE_SQUAD_CONTEXT_FILE,
        index=False,
        encoding="utf-8",
    )

    audit.to_csv(
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

        "reference_timestamp_utc": (
            reference_timestamp.isoformat()
        ),

        "team_feature_count": (
            len(
                squad.TEAM_FEATURES
            )
        ),

        "difference_feature_count": (
            len(
                squad.DIFFERENCE_FEATURES
            )
        ),

        "production_squad_feature_count": (
            2
            *
            len(
                squad.TEAM_FEATURES
            )
            +
            len(
                squad.DIFFERENCE_FEATURES
            )
        ),

        "safe_history_features": sorted(
            SAFE_HISTORY_FEATURES
        ),

        "forced_zero_features": sorted(
            FORCED_ZERO_FEATURES
        ),

        "masked_dynamic_features": sorted(
            DYNAMIC_FEATURES
        ),

        "historical_source_cutoffs": {
            "appearances": (
                pd.to_datetime(
                    appearances[
                        "date"
                    ],
                    errors="coerce",
                )
                .max()
                .date()
                .isoformat()
            ),

            "lineups": (
                pd.to_datetime(
                    lineups[
                        "date"
                    ],
                    errors="coerce",
                )
                .max()
                .date()
                .isoformat()
            ),

            "valuations": (
                pd.to_datetime(
                    valuations[
                        "date"
                    ],
                    errors="coerce",
                )
                .max()
                .date()
                .isoformat()
            ),
        },

        "rules": {
            "synthetic_appearances_created": False,
            "synthetic_starts_created": False,
            "synthetic_minutes_created": False,
            "synthetic_core_xi_created": False,
            "fuzzy_player_matching_used": False,
            "recent_usage_marked_available": False,
            "current_roster_context_used_as_raw_model_usage": False,
            "production_model_modified": False,
            "historical_training_data_modified": False,
            "missing_values_left_for_existing_model_imputer": True,
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
            default=json_safe_value,
        )


# ============================================================
# MAIN
# ============================================================


def build_live_squad_state() -> None:

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - BUILD SAFE LIVE SQUAD STATE"
    )

    print(
        "=" * 75
    )

    verify_frozen_squad_architecture()

    reference_timestamp = (
        determine_reference_timestamp()
    )

    print(
        f"\nNext UCL fixture timestamp: "
        f"{reference_timestamp}"
    )

    current_teams = (
        load_current_teams()
    )

    current_team_names = set(
        current_teams[
            "canonical_team"
        ].astype(
            str
        )
    )

    club_map = (
        load_extended_club_map()
    )

    current_club_ids = {
        club_map[
            team
        ]
        for team in current_team_names
        if team in club_map
    }

    if (
        len(
            current_club_ids
        )
        !=
        EXPECTED_CURRENT_TEAMS
    ):

        missing = (
            current_team_names
            -
            set(
                club_map.keys()
            )
        )

        raise RuntimeError(
            "Current Transfermarkt club identity "
            "coverage is incomplete:\n"
            f"{sorted(missing)}"
        )

    squad_pool = (
        load_current_squad_pool()
    )

    context = (
        build_current_squad_context(
            pool=squad_pool,
            current_teams=current_team_names,
        )
    )

    (
        appearances,
        lineups,
        valuations,
        appearance_histories,
        lineup_histories,
        valuation_histories,
    ) = load_historical_histories(
        current_club_ids
    )

    state = (
        build_live_safe_state(
            current_teams=current_teams,
            club_map=club_map,
            reference_timestamp=reference_timestamp,
            appearance_histories=appearance_histories,
            lineup_histories=lineup_histories,
            valuation_histories=valuation_histories,
        )
    )

    audit = build_audit(
        state=state,
        context=context,
    )

    run_quality_checks(
        state=state,
        context=context,
        appearances=appearances,
        lineups=lineups,
        valuations=valuations,
    )

    print_summary(
        state=state,
        context=context,
        reference_timestamp=reference_timestamp,
    )

    save_outputs(
        state=state,
        context=context,
        audit=audit,
        reference_timestamp=reference_timestamp,
        appearances=appearances,
        lineups=lineups,
        valuations=valuations,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SAFE LIVE SQUAD STATE BUILT SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nSafe production squad team-state:"
    )

    print(
        LIVE_SQUAD_STATE_FILE
    )

    print(
        "\nCurrent roster/value context:"
    )

    print(
        LIVE_SQUAD_CONTEXT_FILE
    )

    print(
        "\nAudit:"
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
    build_live_squad_state()