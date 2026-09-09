from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

from src.features import elo
from src.features import team_form


# ============================================================
# SETTINGS
# ============================================================

CURRENT_SEASON = 2026

EXPECTED_HISTORICAL_MATCHES = 2089

EXPECTED_CURRENT_UCL_CLUBS = 36


# ============================================================
# PATHS
# ============================================================

HISTORICAL_MASTER_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

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

OUTPUT_DIR = (
    INTERIM_DATA_DIR
    / "matches"
    / "live_2026_27"
)

COMBINED_MASTER_FILE = (
    OUTPUT_DIR
    / "ucl_matches_master_live_replay.csv"
)

LIVE_ELO_MATCH_FILE = (
    OUTPUT_DIR
    / "ucl_matches_with_live_elo.csv"
)

LIVE_ELO_ALL_TEAMS_FILE = (
    OUTPUT_DIR
    / "ucl_team_elo_live_all.csv"
)

LIVE_ELO_CURRENT_TEAMS_FILE = (
    OUTPUT_DIR
    / "ucl_team_elo_live_current_36.csv"
)

LIVE_FORM_MATCH_FILE = (
    OUTPUT_DIR
    / "ucl_matches_with_live_form.csv"
)

LIVE_FORM_ALL_TEAMS_FILE = (
    OUTPUT_DIR
    / "ucl_team_form_live_all.csv"
)

LIVE_FORM_CURRENT_TEAMS_FILE = (
    OUTPUT_DIR
    / "ucl_team_form_live_current_36.csv"
)

CURRENT_FINISHED_MATCH_AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_finished_match_replay_audit.csv"
)

CURRENT_STATE_AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_live_ucl_state_audit.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "ucl_live_state_metadata.json"
)


# ============================================================
# BASIC HELPERS
# ============================================================


def require_file(
    path: Path,
    description: str,
) -> None:
    """
    Require a local pipeline artifact.
    """

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
    """
    Return the first compatible column name.
    """

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    if required:

        raise RuntimeError(
            f"Could not identify {description} column.\n"
            f"Candidates tried:\n"
            f"{candidates}\n"
            f"Available columns:\n"
            f"{list(dataframe.columns)}"
        )

    return None


# ============================================================
# HISTORICAL MASTER
# ============================================================


def load_historical_master() -> pd.DataFrame:
    """
    Load the frozen historical UCL match master.

    The file itself is never modified.
    """

    require_file(
        HISTORICAL_MASTER_FILE,
        "Historical UCL master",
    )

    dataframe = pd.read_csv(
        HISTORICAL_MASTER_FILE,
        low_memory=False,
    )

    if (
        len(dataframe)
        !=
        EXPECTED_HISTORICAL_MATCHES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_HISTORICAL_MATCHES:,} "
            f"historical UCL matches, found "
            f"{len(dataframe):,}."
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
            "Historical master is missing required "
            "feature-engineering columns:\n"
            f"{sorted(missing)}"
        )

    date_column = first_existing_column(
        dataframe,
        [
            "date",
            "match_date",
            "date_utc",
            "utc_date",
        ],
        description="historical match date",
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

    dataframe[
        "home_goals"
    ] = pd.to_numeric(
        dataframe[
            "home_goals"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "away_goals"
    ] = pd.to_numeric(
        dataframe[
            "away_goals"
        ],
        errors="raise",
    ).astype(
        int
    )

    return dataframe


# ============================================================
# CURRENT TEAM UNIVERSE
# ============================================================


def load_current_team_map() -> pd.DataFrame:
    """
    Load all 36 canonical current UCL teams.
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
            "Current team map missing columns:\n"
            f"{sorted(missing)}"
        )

    if (
        len(dataframe)
        !=
        EXPECTED_CURRENT_UCL_CLUBS
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_CURRENT_UCL_CLUBS} "
            f"current UCL teams, found "
            f"{len(dataframe)}."
        )

    if dataframe[
        "canonical_team"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate canonical teams found in the "
            "current UCL team map."
        )

    return dataframe


# ============================================================
# CURRENT FINISHED FIXTURES
# ============================================================


def result_from_goals(
    home_goals: int,
    away_goals: int,
) -> str:
    """
    Convert scoreline to H/D/A target.
    """

    if (
        home_goals
        >
        away_goals
    ):

        return "H"

    if (
        home_goals
        <
        away_goals
    ):

        return "A"

    return "D"


def load_finished_current_fixtures() -> pd.DataFrame:
    """
    Load only FINISHED 2026/27 UCL fixtures.

    Unplayed fixtures are never included in state replay.
    """

    require_file(
        CURRENT_FIXTURES_FILE,
        "Resolved 2026/27 fixture file",
    )

    dataframe = pd.read_csv(
        CURRENT_FIXTURES_FILE,
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
            "Current fixture file missing columns:\n"
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

    finished = dataframe[
        dataframe[
            "status"
        ].astype(
            str
        )
        ==
        "FINISHED"
    ].copy()

    if finished.empty:

        raise RuntimeError(
            "No finished 2026/27 UCL fixtures found."
        )

    # --------------------------------------------------------
    # SCORE COLUMN DETECTION
    # --------------------------------------------------------

    home_goal_column = (
        first_existing_column(
            finished,
            [
                "home_goals",
                "home_score",
                "score_home",
                "full_time_home",
                "fulltime_home",
                "home_full_time",
                "home_team_score",
            ],
            description="current home goals",
        )
    )

    away_goal_column = (
        first_existing_column(
            finished,
            [
                "away_goals",
                "away_score",
                "score_away",
                "full_time_away",
                "fulltime_away",
                "away_full_time",
                "away_team_score",
            ],
            description="current away goals",
        )
    )

    finished[
        "_home_goals"
    ] = pd.to_numeric(
        finished[
            home_goal_column
        ],
        errors="coerce",
    )

    finished[
        "_away_goals"
    ] = pd.to_numeric(
        finished[
            away_goal_column
        ],
        errors="coerce",
    )

    missing_scores = finished[
        finished[
            "_home_goals"
        ].isna()
        |
        finished[
            "_away_goals"
        ].isna()
    ]

    if not missing_scores.empty:

        raise RuntimeError(
            "Some FINISHED current fixtures do not have "
            "usable final scores:\n"
            +
            missing_scores[
                [
                    "date_utc",
                    "home_team",
                    "away_team",
                    home_goal_column,
                    away_goal_column,
                ]
            ].to_string(
                index=False
            )
        )

    finished[
        "_home_goals"
    ] = finished[
        "_home_goals"
    ].astype(
        int
    )

    finished[
        "_away_goals"
    ] = finished[
        "_away_goals"
    ].astype(
        int
    )

    finished[
        "_result"
    ] = [
        result_from_goals(
            int(home_goals),
            int(away_goals),
        )
        for (
            home_goals,
            away_goals,
        ) in zip(
            finished[
                "_home_goals"
            ],
            finished[
                "_away_goals"
            ],
        )
    ]

    # --------------------------------------------------------
    # BASIC INTEGRITY
    # --------------------------------------------------------

    same_team = finished[
        finished[
            "home_team"
        ]
        ==
        finished[
            "away_team"
        ]
    ]

    if not same_team.empty:

        raise RuntimeError(
            "Same-team fixture found in current results."
        )

    finished = finished.sort_values(
        by=[
            "date_utc",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    return finished


# ============================================================
# CURRENT FIXTURE -> HISTORICAL MASTER ADAPTER
# ============================================================


def populate_optional_column(
    *,
    target: pd.DataFrame,
    target_column: str,
    source: pd.DataFrame,
    source_candidates: list[str],
) -> None:
    """
    Fill an optional historical master field when a matching
    current fixture field exists.
    """

    source_column = first_existing_column(
        source,
        source_candidates,
        description=target_column,
        required=False,
    )

    if source_column is not None:

        target[
            target_column
        ] = source[
            source_column
        ].values


def convert_finished_to_master_schema(
    *,
    historical_master: pd.DataFrame,
    finished: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert current finished fixtures to the historical master
    schema required by the existing feature modules.
    """

    rows = pd.DataFrame(
        index=range(
            len(
                finished
            )
        ),
        columns=historical_master.columns,
    )

    # --------------------------------------------------------
    # CORE FIELDS
    # --------------------------------------------------------

    historical_date_column = (
        first_existing_column(
            historical_master,
            [
                "date",
                "match_date",
                "date_utc",
                "utc_date",
            ],
            description="historical date",
        )
    )

    rows[
        historical_date_column
    ] = finished[
        "date_utc"
    ].values

    rows[
        "season"
    ] = CURRENT_SEASON

    rows[
        "home_team"
    ] = finished[
        "home_team"
    ].astype(
        str
    ).values

    rows[
        "away_team"
    ] = finished[
        "away_team"
    ].astype(
        str
    ).values

    rows[
        "home_goals"
    ] = finished[
        "_home_goals"
    ].values

    rows[
        "away_goals"
    ] = finished[
        "_away_goals"
    ].values

    rows[
        "result"
    ] = finished[
        "_result"
    ].values

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    if (
        "source"
        in
        rows.columns
    ):

        rows[
            "source"
        ] = (
            "football-data.org_live_2026_27"
        )

    # --------------------------------------------------------
    # COMPETITION / STAGE
    # --------------------------------------------------------

    if (
        "stage"
        in
        rows.columns
    ):

        if (
            "stage"
            in
            finished.columns
        ):

            rows[
                "stage"
            ] = finished[
                "stage"
            ].values

        else:

            rows[
                "stage"
            ] = "LEAGUE_STAGE"

    # --------------------------------------------------------
    # MATCH ID
    # --------------------------------------------------------

    if (
        "match_id"
        in
        rows.columns
    ):

        source_match_id = (
            first_existing_column(
                finished,
                [
                    "match_id",
                    "football_data_match_id",
                    "id",
                ],
                description="current match ID",
                required=False,
            )
        )

        if source_match_id is not None:

            rows[
                "match_id"
            ] = finished[
                source_match_id
            ].values

        else:

            rows[
                "match_id"
            ] = [
                (
                    "LIVE_2026_"
                    +
                    str(index + 1)
                )
                for index in range(
                    len(
                        rows
                    )
                )
            ]

    # --------------------------------------------------------
    # OPTIONAL FIELDS
    # --------------------------------------------------------

    optional_adapters = {
        "competition": [
            "competition",
            "competition_name",
        ],

        "competition_code": [
            "competition_code",
        ],

        "round": [
            "round",
            "matchday",
        ],

        "matchday": [
            "matchday",
            "round",
        ],

        "status": [
            "status",
        ],

        "home_team_id": [
            "home_team_id",
            "home_football_data_team_id",
        ],

        "away_team_id": [
            "away_team_id",
            "away_football_data_team_id",
        ],
    }

    for (
        target_column,
        source_candidates,
    ) in optional_adapters.items():

        if (
            target_column
            in
            rows.columns
        ):

            populate_optional_column(
                target=rows,
                target_column=target_column,
                source=finished,
                source_candidates=source_candidates,
            )

    # --------------------------------------------------------
    # TYPES
    # --------------------------------------------------------

    rows[
        historical_date_column
    ] = pd.to_datetime(
        rows[
            historical_date_column
        ],
        errors="raise",
        utc=True,
    )

    rows[
        "season"
    ] = pd.to_numeric(
        rows[
            "season"
        ],
        errors="raise",
    ).astype(
        int
    )

    rows[
        "home_goals"
    ] = pd.to_numeric(
        rows[
            "home_goals"
        ],
        errors="raise",
    ).astype(
        int
    )

    rows[
        "away_goals"
    ] = pd.to_numeric(
        rows[
            "away_goals"
        ],
        errors="raise",
    ).astype(
        int
    )

    return rows


# ============================================================
# BUILD COMBINED REPLAY MASTER
# ============================================================


def build_live_master(
    *,
    historical_master: pd.DataFrame,
    current_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append only finished current fixtures to historical
    matches and sort chronologically.
    """

    date_column = first_existing_column(
        historical_master,
        [
            "date",
            "match_date",
            "date_utc",
            "utc_date",
        ],
        description="historical date",
    )

    historical = (
        historical_master.copy()
    )

    historical[
        date_column
    ] = pd.to_datetime(
        historical[
            date_column
        ],
        errors="raise",
        utc=True,
    )

    current = (
        current_rows.copy()
    )

    current[
        date_column
    ] = pd.to_datetime(
        current[
            date_column
        ],
        errors="raise",
        utc=True,
    )

    # --------------------------------------------------------
    # DUPLICATE MATCH CHECK
    # --------------------------------------------------------

    historical_keys = set(
        zip(
            historical[
                date_column
            ],
            historical[
                "home_team"
            ].astype(
                str
            ),
            historical[
                "away_team"
            ].astype(
                str
            ),
        )
    )

    duplicate_current = []

    for row in current.itertuples(
        index=False
    ):

        row_dict = row._asdict()

        key = (
            pd.Timestamp(
                row_dict[
                    date_column
                ]
            ),
            str(
                row_dict[
                    "home_team"
                ]
            ),
            str(
                row_dict[
                    "away_team"
                ]
            ),
        )

        if key in historical_keys:

            duplicate_current.append(
                key
            )

    if duplicate_current:

        raise RuntimeError(
            "Current finished fixtures already appear in "
            "historical master:\n"
            f"{duplicate_current}"
        )

    combined = pd.concat(
        [
            historical,
            current,
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

    # --------------------------------------------------------
    # CHRONOLOGICAL CHECK
    # --------------------------------------------------------

    if not combined[
        date_column
    ].is_monotonic_increasing:

        raise RuntimeError(
            "Combined live master is not chronological."
        )

    return combined


# ============================================================
# BUILD ELO CURRENT-TEAM TABLE
# ============================================================


def build_current_elo_table(
    *,
    ratings: dict[str, float],
    current_teams: pd.DataFrame,
    historical_teams: set[str],
    finished_teams: set[str],
) -> pd.DataFrame:
    """
    Produce one current Elo state row for all 36 UCL clubs.

    Genuinely unseen clubs that have not yet played receive the
    exact historical default INITIAL_ELO = 1500.
    """

    rows = []

    for team_row in current_teams.itertuples(
        index=False
    ):

        team = str(
            team_row.canonical_team
        )

        identity_status = str(
            team_row.identity_status
        )

        if team in ratings:

            rating = float(
                ratings[
                    team
                ]
            )

        else:

            rating = float(
                elo.INITIAL_ELO
            )

        if team in finished_teams:

            if team in historical_teams:

                state_source = (
                    "2026_RESULT_REPLAYED"
                )

            else:

                state_source = (
                    "2026_RESULT_REPLAYED_FROM_INITIAL_1500"
                )

        elif team in historical_teams:

            state_source = (
                "HISTORICAL_STATE_WITH_2026_SEASON_CARRYOVER"
            )

        else:

            state_source = (
                "NEW_CLUB_INITIAL_1500"
            )

        rows.append(
            {
                "team": (
                    team
                ),

                "identity_status": (
                    identity_status
                ),

                "ucl_elo": (
                    rating
                ),

                "elo_state_source": (
                    state_source
                ),

                "played_2026_27_ucl_match": int(
                    team
                    in
                    finished_teams
                ),

                "historical_ucl_identity": int(
                    team
                    in
                    historical_teams
                ),
            }
        )

    output = pd.DataFrame(
        rows
    )

    output[
        "rank"
    ] = output[
        "ucl_elo"
    ].rank(
        method="min",
        ascending=False,
    ).astype(
        int
    )

    output = output[
        [
            "rank",
            "team",
            "identity_status",
            "ucl_elo",
            "elo_state_source",
            "played_2026_27_ucl_match",
            "historical_ucl_identity",
        ]
    ]

    output = output.sort_values(
        by=[
            "rank",
            "team",
        ]
    ).reset_index(
        drop=True
    )

    return output


# ============================================================
# BUILD CURRENT FORM TABLE
# ============================================================


def build_current_form_table(
    *,
    latest_form: pd.DataFrame,
    current_teams: pd.DataFrame,
    history: dict,
) -> pd.DataFrame:
    """
    Create a 36-team current UCL-form state table.

    Teams without UCL history remain genuine cold starts.
    """

    if (
        "team"
        not in
        latest_form.columns
    ):

        raise RuntimeError(
            "Latest UCL form table does not contain "
            "a team column."
        )

    team_base = current_teams[
        [
            "canonical_team",
            "identity_status",
        ]
    ].rename(
        columns={
            "canonical_team": (
                "team"
            ),
        }
    )

    output = team_base.merge(
        latest_form,
        on="team",
        how="left",
        validate="one_to_one",
    )

    output[
        "ucl_form_history_available"
    ] = output[
        "team"
    ].map(
        lambda team: int(
            (
                team
                in
                history
            )
            and
            (
                len(
                    history[
                        team
                    ]
                )
                >
                0
            )
        )
    )

    output[
        "ucl_form_state_source"
    ] = np.where(
        output[
            "ucl_form_history_available"
        ]
        ==
        1,
        "HISTORICAL_OR_REPLAYED_UCL_HISTORY",
        "COLD_START_NO_PRIOR_UCL_HISTORY",
    )

    return output


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    *,
    historical_master: pd.DataFrame,
    finished: pd.DataFrame,
    combined: pd.DataFrame,
    elo_matches: pd.DataFrame,
    current_elo: pd.DataFrame,
    form_matches: pd.DataFrame,
    current_form: pd.DataFrame,
    current_teams: pd.DataFrame,
) -> None:
    """
    Verify the live replay does not violate chronology or
    team coverage.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE UCL STATE QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    print(
        f"\nHistorical matches: "
        f"{len(historical_master):,}"
    )

    print(
        f"Finished 2026/27 matches replayed: "
        f"{len(finished):,}"
    )

    print(
        f"Combined replay matches: "
        f"{len(combined):,}"
    )

    expected_combined = (
        len(
            historical_master
        )
        +
        len(
            finished
        )
    )

    if (
        len(
            combined
        )
        !=
        expected_combined
    ):

        raise RuntimeError(
            "Unexpected combined replay row count."
        )

    if (
        len(
            elo_matches
        )
        !=
        len(
            combined
        )
    ):

        raise RuntimeError(
            "Elo output row count changed."
        )

    if (
        len(
            form_matches
        )
        !=
        len(
            combined
        )
    ):

        raise RuntimeError(
            "UCL-form output row count changed."
        )

    # --------------------------------------------------------
    # CURRENT TEAM COVERAGE
    # --------------------------------------------------------

    expected_current_teams = set(
        current_teams[
            "canonical_team"
        ]
    )

    elo_current_teams = set(
        current_elo[
            "team"
        ]
    )

    form_current_teams = set(
        current_form[
            "team"
        ]
    )

    missing_elo = (
        expected_current_teams
        -
        elo_current_teams
    )

    missing_form_rows = (
        expected_current_teams
        -
        form_current_teams
    )

    print(
        f"\nCurrent Elo state coverage: "
        f"{len(elo_current_teams)}/"
        f"{len(expected_current_teams)}"
    )

    print(
        f"Current form table row coverage: "
        f"{len(form_current_teams)}/"
        f"{len(expected_current_teams)}"
    )

    if missing_elo:

        raise RuntimeError(
            "Current Elo state is missing teams:\n"
            f"{sorted(missing_elo)}"
        )

    if missing_form_rows:

        raise RuntimeError(
            "Current form state table is missing teams:\n"
            f"{sorted(missing_form_rows)}"
        )

    # --------------------------------------------------------
    # FORM HISTORY AVAILABILITY
    # --------------------------------------------------------

    teams_with_form_history = int(
        current_form[
            "ucl_form_history_available"
        ].sum()
    )

    cold_start_teams = current_form.loc[
        current_form[
            "ucl_form_history_available"
        ]
        ==
        0,
        "team",
    ].tolist()

    print(
        f"Current teams with prior UCL form history: "
        f"{teams_with_form_history}/"
        f"{EXPECTED_CURRENT_UCL_CLUBS}"
    )

    print(
        f"Current UCL form cold starts: "
        f"{len(cold_start_teams)}"
    )

    if cold_start_teams:

        print(
            "Cold-start teams:"
        )

        for team in cold_start_teams:

            print(
                f"  {team}"
            )

    # --------------------------------------------------------
    # ELO VALIDITY
    # --------------------------------------------------------

    missing_elo_values = int(
        current_elo[
            "ucl_elo"
        ].isna().sum()
    )

    if missing_elo_values:

        raise RuntimeError(
            "Missing current Elo ratings detected."
        )

    invalid_results = set(
        finished[
            "_result"
        ]
    ) - {
        "H",
        "D",
        "A",
    }

    if invalid_results:

        raise RuntimeError(
            f"Invalid current results: "
            f"{sorted(invalid_results)}"
        )

    print(
        "\nCurrent Elo values complete ✓"
    )

    print(
        "Current results H/D/A valid ✓"
    )

    print(
        "Historical files remain untouched ✓"
    )


# ============================================================
# BUILD AUDITS
# ============================================================


def build_finished_match_audit(
    finished: pd.DataFrame,
) -> pd.DataFrame:
    """
    Small human-readable audit of the matches replayed.
    """

    audit = finished[
        [
            "date_utc",
            "home_team",
            "away_team",
            "_home_goals",
            "_away_goals",
            "_result",
        ]
    ].copy()

    audit = audit.rename(
        columns={
            "_home_goals": (
                "home_goals"
            ),

            "_away_goals": (
                "away_goals"
            ),

            "_result": (
                "result"
            ),
        }
    )

    return audit


def build_current_state_audit(
    *,
    current_elo: pd.DataFrame,
    current_form: pd.DataFrame,
) -> pd.DataFrame:
    """
    Current 36-team state coverage audit.
    """

    form_columns = [
        "team",
        "ucl_form_history_available",
        "ucl_form_state_source",
    ]

    if (
        "ucl_matches"
        in
        current_form.columns
    ):

        form_columns.append(
            "ucl_matches"
        )

    if (
        "last_match_date"
        in
        current_form.columns
    ):

        form_columns.append(
            "last_match_date"
        )

    audit = current_elo.merge(
        current_form[
            form_columns
        ],
        on="team",
        how="left",
        validate="one_to_one",
    )

    return audit


# ============================================================
# SAVE OUTPUTS
# ============================================================


def save_outputs(
    *,
    combined: pd.DataFrame,
    elo_matches: pd.DataFrame,
    elo_all: pd.DataFrame,
    current_elo: pd.DataFrame,
    form_matches: pd.DataFrame,
    form_all: pd.DataFrame,
    current_form: pd.DataFrame,
    finished_audit: pd.DataFrame,
    current_state_audit: pd.DataFrame,
    finished: pd.DataFrame,
) -> None:
    """
    Save live artifacts separately from frozen historical
    processed outputs.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.joinpath(
        "metrics"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        COMBINED_MASTER_FILE,
        index=False,
        encoding="utf-8",
    )

    elo_matches.to_csv(
        LIVE_ELO_MATCH_FILE,
        index=False,
        encoding="utf-8",
    )

    elo_all.to_csv(
        LIVE_ELO_ALL_TEAMS_FILE,
        index=False,
        encoding="utf-8",
    )

    current_elo.to_csv(
        LIVE_ELO_CURRENT_TEAMS_FILE,
        index=False,
        encoding="utf-8",
    )

    form_matches.to_csv(
        LIVE_FORM_MATCH_FILE,
        index=False,
        encoding="utf-8",
    )

    form_all.to_csv(
        LIVE_FORM_ALL_TEAMS_FILE,
        index=False,
        encoding="utf-8",
    )

    current_form.to_csv(
        LIVE_FORM_CURRENT_TEAMS_FILE,
        index=False,
        encoding="utf-8",
    )

    finished_audit.to_csv(
        CURRENT_FINISHED_MATCH_AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    current_state_audit.to_csv(
        CURRENT_STATE_AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    latest_finished_date = (
        finished[
            "date_utc"
        ].max()
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "current_season": (
            CURRENT_SEASON
        ),

        "historical_matches": (
            EXPECTED_HISTORICAL_MATCHES
        ),

        "current_finished_matches_replayed": int(
            len(
                finished
            )
        ),

        "combined_match_rows": int(
            len(
                combined
            )
        ),

        "latest_finished_current_match_utc": (
            latest_finished_date.isoformat()
        ),

        "elo_constants": {
            "initial_elo": (
                elo.INITIAL_ELO
            ),

            "k_factor": (
                elo.K_FACTOR
            ),

            "home_advantage": (
                elo.HOME_ADVANTAGE
            ),

            "elo_scale": (
                elo.ELO_SCALE
            ),

            "season_carryover": (
                elo.SEASON_CARRYOVER
            ),
        },

        "form_windows": list(
            team_form.FORM_WINDOWS
        ),

        "rules": {
            "historical_processed_files_overwritten": False,

            "only_finished_current_matches_replayed": True,

            "future_matches_used_as_results": False,

            "elo_logic_reused_from_existing_module": True,

            "ucl_form_logic_reused_from_existing_module": True,

            "new_club_elo_default": (
                elo.INITIAL_ELO
            ),

            "ucl_form_cold_start_preserved": True,

            "strict_chronological_state_replay": True,
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
# SUMMARY
# ============================================================


def print_summary(
    *,
    finished: pd.DataFrame,
    current_elo: pd.DataFrame,
    current_form: pd.DataFrame,
) -> None:
    """
    Print current state summary.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2026/27 LIVE UCL STATE SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nFinished matches replayed: "
        f"{len(finished)}"
    )

    print(
        "\nReplayed current matches:"
    )

    print(
        finished[
            [
                "date_utc",
                "home_team",
                "away_team",
                "_home_goals",
                "_away_goals",
                "_result",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nCurrent Elo state:"
    )

    print(
        current_elo[
            [
                "rank",
                "team",
                "ucl_elo",
                "elo_state_source",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    history_available = int(
        current_form[
            "ucl_form_history_available"
        ].sum()
    )

    print(
        f"\nCurrent clubs with UCL form history: "
        f"{history_available}/"
        f"{EXPECTED_CURRENT_UCL_CLUBS}"
    )

    cold_starts = current_form[
        current_form[
            "ucl_form_history_available"
        ]
        ==
        0
    ]

    print(
        f"Current cold-start clubs: "
        f"{len(cold_starts)}"
    )

    if not cold_starts.empty:

        print(
            "\nCold-start clubs:"
        )

        print(
            cold_starts[
                [
                    "team",
                    "identity_status",
                    "ucl_form_state_source",
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# MAIN LIVE STATE BUILDER
# ============================================================


def build_live_ucl_state() -> None:
    """
    Rebuild current 2026/27 UCL Elo and UCL-form states by
    replaying finished current matches through the exact
    historical feature implementations.

    Historical processed files are never overwritten.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - BUILD LIVE 2026/27 UCL STATE"
    )

    print(
        "=" * 75
    )

    historical_master = (
        load_historical_master()
    )

    current_teams = (
        load_current_team_map()
    )

    finished = (
        load_finished_current_fixtures()
    )

    print(
        f"\nHistorical matches loaded: "
        f"{len(historical_master):,}"
    )

    print(
        f"Finished current fixtures: "
        f"{len(finished):,}"
    )

    current_rows = (
        convert_finished_to_master_schema(
            historical_master=historical_master,
            finished=finished,
        )
    )

    combined = build_live_master(
        historical_master=historical_master,
        current_rows=current_rows,
    )

    # --------------------------------------------------------
    # EXISTING ELO IMPLEMENTATION
    # --------------------------------------------------------

    print(
        "\nReplaying Elo through existing "
        "src.features.elo logic..."
    )

    (
        elo_matches,
        ratings,
    ) = elo.build_elo_features(
        combined.copy()
    )

    elo_all = (
        elo.build_latest_elo_table(
            ratings
        )
    )

    # --------------------------------------------------------
    # HISTORICAL TEAM UNIVERSE
    # --------------------------------------------------------

    historical_teams = (
        set(
            historical_master[
                "home_team"
            ].astype(
                str
            )
        )
        |
        set(
            historical_master[
                "away_team"
            ].astype(
                str
            )
        )
    )

    finished_teams = (
        set(
            finished[
                "home_team"
            ].astype(
                str
            )
        )
        |
        set(
            finished[
                "away_team"
            ].astype(
                str
            )
        )
    )

    current_elo = (
        build_current_elo_table(
            ratings=ratings,
            current_teams=current_teams,
            historical_teams=historical_teams,
            finished_teams=finished_teams,
        )
    )

    # --------------------------------------------------------
    # EXISTING UCL FORM IMPLEMENTATION
    # --------------------------------------------------------

    print(
        "Replaying UCL form through existing "
        "src.features.team_form logic..."
    )

    (
        form_matches,
        history,
    ) = team_form.build_form_features(
        elo_matches.copy()
    )

    form_all = (
        team_form.build_latest_form_table(
            history
        )
    )

    current_form = (
        build_current_form_table(
            latest_form=form_all,
            current_teams=current_teams,
            history=history,
        )
    )

    # --------------------------------------------------------
    # AUDITS
    # --------------------------------------------------------

    run_quality_checks(
        historical_master=historical_master,
        finished=finished,
        combined=combined,
        elo_matches=elo_matches,
        current_elo=current_elo,
        form_matches=form_matches,
        current_form=current_form,
        current_teams=current_teams,
    )

    finished_audit = (
        build_finished_match_audit(
            finished
        )
    )

    current_state_audit = (
        build_current_state_audit(
            current_elo=current_elo,
            current_form=current_form,
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_outputs(
        combined=combined,
        elo_matches=elo_matches,
        elo_all=elo_all,
        current_elo=current_elo,
        form_matches=form_matches,
        form_all=form_all,
        current_form=current_form,
        finished_audit=finished_audit,
        current_state_audit=current_state_audit,
        finished=finished,
    )

    print_summary(
        finished=finished,
        current_elo=current_elo,
        current_form=current_form,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE 2026/27 UCL STATE BUILT SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nCombined replay master:"
    )

    print(
        COMBINED_MASTER_FILE
    )

    print(
        "\nCurrent 36-team Elo state:"
    )

    print(
        LIVE_ELO_CURRENT_TEAMS_FILE
    )

    print(
        "\nCurrent 36-team UCL-form state:"
    )

    print(
        LIVE_FORM_CURRENT_TEAMS_FILE
    )

    print(
        "\nState coverage audit:"
    )

    print(
        CURRENT_STATE_AUDIT_FILE
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
        "Historical processed data was not overwritten."
    )

    print(
        "Only FINISHED 2026/27 matches affected live state."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    build_live_ucl_state()