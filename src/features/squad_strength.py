from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
)


# ============================================================
# PATHS
# ============================================================

UCL_INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_domestic_form.csv"
)

CLUB_MAP_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)

APPEARANCES_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_appearances_ucl_clubs.csv"
)

LINEUPS_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_lineups_ucl_clubs.csv"
)

VALUATIONS_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_player_valuations_ucl_players.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_squad_strength.csv"
)

LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_squad_strength_latest.csv"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_UCL_MATCHES = 2089
EXPECTED_UCL_CLUBS = 117


# ============================================================
# WINDOWS
# ============================================================

RECENT_30_DAYS = 30
RECENT_90_DAYS = 90
RECENT_180_DAYS = 180

PREVIOUS_BLOCK_START_DAYS = 360
PREVIOUS_BLOCK_END_DAYS = 180

VALUATION_FRESH_DAYS = 365

CORE_PLAYER_COUNT = 11


# ============================================================
# MODEL FEATURE NAMES
# ============================================================

TEAM_FEATURES = [
    "squad_appearance_history_available",
    "squad_lineup_history_available",
    "squad_recent_history_available",
    "squad_valuation_history_available",

    "squad_days_since_last_player_evidence",

    "squad_games_last30d",
    "squad_games_last90d",
    "squad_games_last180d",

    "squad_players_seen_last30d",
    "squad_players_seen_last90d",
    "squad_players_seen_last180d",

    "squad_appearances_last30d",
    "squad_appearances_last90d",
    "squad_appearances_last180d",

    "squad_starts_last30d",
    "squad_starts_last90d",
    "squad_starts_last180d",

    "squad_minutes_last30d",
    "squad_minutes_last90d",
    "squad_minutes_last180d",

    "squad_minutes_per_game_last90d",
    "squad_minutes_per_game_last180d",

    "squad_core11_count",
    "squad_core11_minutes_share_90d",
    "squad_core11_start_share_180d",
    "squad_starter_stability_180d",

    "squad_core11_continuity",
    "squad_player_pool_continuity",

    "squad_active_value_m_eur",
    "squad_active_value_log",
    "squad_active_valuation_coverage",
    "squad_active_fresh_valuation_rate",
    "squad_active_mean_valuation_age_days",

    "squad_core11_value_m_eur",
    "squad_core11_value_log",
    "squad_core11_avg_value_m_eur",
    "squad_core11_median_value_m_eur",
    "squad_core11_valuation_coverage",
    "squad_core11_fresh_valuation_rate",
    "squad_core11_mean_valuation_age_days",

    "squad_top5_value_m_eur",

    "squad_data_completeness",
]


DIFFERENCE_FEATURES = [
    "squad_games_last90d",
    "squad_players_seen_last180d",

    "squad_minutes_per_game_last90d",

    "squad_core11_minutes_share_90d",
    "squad_core11_start_share_180d",
    "squad_starter_stability_180d",

    "squad_core11_continuity",
    "squad_player_pool_continuity",

    "squad_active_value_m_eur",
    "squad_active_value_log",
    "squad_active_valuation_coverage",

    "squad_core11_value_m_eur",
    "squad_core11_value_log",
    "squad_core11_avg_value_m_eur",
    "squad_core11_valuation_coverage",

    "squad_top5_value_m_eur",

    "squad_data_completeness",
]


# ============================================================
# HELPERS
# ============================================================


def require_file(
    path: Path,
    description: str,
) -> None:
    """
    Fail early when a required dataset is unavailable.
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


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe ratio preserving missing/undefined states.
    """

    if denominator is None:
        return np.nan

    if pd.isna(
        denominator
    ):
        return np.nan

    if denominator <= 0:
        return np.nan

    return float(
        numerator
        / denominator
    )


def safe_mean(
    values: list[float],
) -> float:
    """
    Mean of non-missing values.
    """

    clean = [
        float(
            value
        )
        for value in values
        if value is not None
        and not pd.isna(
            value
        )
    ]

    if not clean:
        return np.nan

    return float(
        np.mean(
            clean
        )
    )


def safe_difference(
    home_value,
    away_value,
):
    """
    Preserve NaN if either side is unavailable.
    """

    if pd.isna(
        home_value
    ) or pd.isna(
        away_value
    ):
        return np.nan

    return float(
        home_value
        - away_value
    )


def timestamp_to_numpy(
    value: pd.Timestamp,
) -> np.datetime64:
    """
    Convert pandas timestamp into nanosecond numpy datetime.
    """

    return np.datetime64(
        pd.Timestamp(
            value
        ).to_datetime64()
    )


# ============================================================
# LOAD UCL
# ============================================================


def load_ucl_matches() -> pd.DataFrame:
    """
    Load the already engineered UCL match dataset.
    """

    print(
        "\nLoading UCL feature dataset:"
    )

    print(
        UCL_INPUT_FILE
    )

    require_file(
        UCL_INPUT_FILE,
        "UCL domestic-form dataset",
    )

    dataframe = pd.read_csv(
        UCL_INPUT_FILE,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "home_team",
        "away_team",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"UCL dataset missing columns: "
            f"{sorted(missing)}"
        )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    dataframe = dataframe.sort_values(
        by=[
            "date",
            "match_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"UCL matches loaded: "
        f"{len(dataframe):,}"
    )

    return dataframe


# ============================================================
# LOAD CLUB MAP
# ============================================================


def load_club_map() -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[int, str],
]:
    """
    Load canonical UCL -> Transfermarkt club IDs.
    """

    require_file(
        CLUB_MAP_FILE,
        "Final Transfermarkt club map",
    )

    mapping = pd.read_csv(
        CLUB_MAP_FILE,
        low_memory=False,
    )

    required = {
        "ucl_team",
        "club_id",
    }

    missing = (
        required
        - set(
            mapping.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Club map missing columns: "
            f"{sorted(missing)}"
        )

    mapping[
        "club_id"
    ] = pd.to_numeric(
        mapping[
            "club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    if len(
        mapping
    ) != EXPECTED_UCL_CLUBS:

        raise RuntimeError(
            f"Expected {EXPECTED_UCL_CLUBS} "
            f"mapped clubs, found "
            f"{len(mapping)}."
        )

    if mapping[
        "ucl_team"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate canonical UCL team "
            "in Transfermarkt map."
        )

    if mapping[
        "club_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Transfermarkt club ID "
            "in final map."
        )

    team_to_id = (
        mapping
        .set_index(
            "ucl_team"
        )[
            "club_id"
        ]
        .to_dict()
    )

    id_to_team = (
        mapping
        .set_index(
            "club_id"
        )[
            "ucl_team"
        ]
        .to_dict()
    )

    return (
        mapping,
        team_to_id,
        id_to_team,
    )


# ============================================================
# LOAD APPEARANCES
# ============================================================


def load_appearances() -> pd.DataFrame:
    """
    Load only fields needed for usage features.
    """

    require_file(
        APPEARANCES_FILE,
        "Filtered historical appearances",
    )

    print(
        "\nLoading historical appearances..."
    )

    dataframe = pd.read_csv(
        APPEARANCES_FILE,
        usecols=[
            "game_id",
            "player_id",
            "player_club_id",
            "date",
            "minutes_played",
        ],
        low_memory=False,
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    dataframe[
        "player_id"
    ] = pd.to_numeric(
        dataframe[
            "player_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "player_club_id"
    ] = pd.to_numeric(
        dataframe[
            "player_club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "minutes_played"
    ] = pd.to_numeric(
        dataframe[
            "minutes_played"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    dataframe = dataframe.sort_values(
        by=[
            "player_club_id",
            "date",
            "game_id",
            "player_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Appearance rows loaded: "
        f"{len(dataframe):,}"
    )

    return dataframe


# ============================================================
# LOAD LINEUPS
# ============================================================


def load_lineups() -> pd.DataFrame:
    """
    Load historical selected starting/bench players.
    """

    require_file(
        LINEUPS_FILE,
        "Filtered historical lineups",
    )

    print(
        "\nLoading historical lineups..."
    )

    dataframe = pd.read_csv(
        LINEUPS_FILE,
        usecols=[
            "game_id",
            "player_id",
            "club_id",
            "type",
            "date",
        ],
        low_memory=False,
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    dataframe[
        "player_id"
    ] = pd.to_numeric(
        dataframe[
            "player_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "club_id"
    ] = pd.to_numeric(
        dataframe[
            "club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "type"
    ] = (
        dataframe[
            "type"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe = dataframe.sort_values(
        by=[
            "club_id",
            "date",
            "game_id",
            "player_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Lineup rows loaded: "
        f"{len(dataframe):,}"
    )

    return dataframe


# ============================================================
# LOAD VALUATIONS
# ============================================================


def load_valuations() -> pd.DataFrame:
    """
    Load historical player valuations.

    current_club fields are intentionally not used.
    """

    require_file(
        VALUATIONS_FILE,
        "Filtered historical player valuations",
    )

    print(
        "\nLoading historical valuations..."
    )

    dataframe = pd.read_csv(
        VALUATIONS_FILE,
        usecols=[
            "player_id",
            "date",
            "market_value_in_eur",
        ],
        low_memory=False,
    )

    dataframe[
        "player_id"
    ] = pd.to_numeric(
        dataframe[
            "player_id"
        ],
        errors="raise",
    ).astype(
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

    dataframe[
        "market_value_in_eur"
    ] = pd.to_numeric(
        dataframe[
            "market_value_in_eur"
        ],
        errors="raise",
    )

    dataframe = dataframe.sort_values(
        by=[
            "player_id",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Valuation rows loaded: "
        f"{len(dataframe):,}"
    )

    return dataframe


# ============================================================
# CLUB HISTORY INDEX
# ============================================================


def build_club_histories(
    *,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    club_ids: set[int],
) -> tuple[
    dict[int, dict],
    dict[int, dict],
]:
    """
    Build date-sorted per-club history objects.

    Searchsorted then allows strict historical slicing without
    repeatedly scanning the entire dataset.
    """

    appearance_histories = {}

    lineup_histories = {}

    appearance_groups = {
        int(
            club_id
        ): group.reset_index(
            drop=True
        )
        for (
            club_id,
            group,
        ) in appearances.groupby(
            "player_club_id",
            sort=False,
        )
    }

    lineup_groups = {
        int(
            club_id
        ): group.reset_index(
            drop=True
        )
        for (
            club_id,
            group,
        ) in lineups.groupby(
            "club_id",
            sort=False,
        )
    }

    for club_id in club_ids:

        appearance_frame = (
            appearance_groups.get(
                club_id
            )
        )

        if appearance_frame is None:

            appearance_frame = pd.DataFrame(
                columns=appearances.columns
            )

        appearance_histories[
            club_id
        ] = {
            "frame": appearance_frame,
            "dates": (
                appearance_frame[
                    "date"
                ].to_numpy(
                    dtype="datetime64[ns]"
                )
                if not appearance_frame.empty
                else np.array(
                    [],
                    dtype="datetime64[ns]",
                )
            ),
        }

        lineup_frame = (
            lineup_groups.get(
                club_id
            )
        )

        if lineup_frame is None:

            lineup_frame = pd.DataFrame(
                columns=lineups.columns
            )

        lineup_histories[
            club_id
        ] = {
            "frame": lineup_frame,
            "dates": (
                lineup_frame[
                    "date"
                ].to_numpy(
                    dtype="datetime64[ns]"
                )
                if not lineup_frame.empty
                else np.array(
                    [],
                    dtype="datetime64[ns]",
                )
            ),
        }

    return (
        appearance_histories,
        lineup_histories,
    )


# ============================================================
# VALUATION INDEX
# ============================================================


def build_valuation_histories(
    valuations: pd.DataFrame,
) -> dict[int, dict]:
    """
    Build player -> chronological valuation arrays.
    """

    histories = {}

    for (
        player_id,
        group,
    ) in valuations.groupby(
        "player_id",
        sort=False,
    ):

        group = group.sort_values(
            by="date"
        )

        histories[
            int(
                player_id
            )
        ] = {
            "dates": group[
                "date"
            ].to_numpy(
                dtype="datetime64[ns]"
            ),
            "values": group[
                "market_value_in_eur"
            ].to_numpy(
                dtype=float
            ),
        }

    return histories


# ============================================================
# HISTORY SLICING
# ============================================================


def history_slice(
    history: dict,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Slice history where:

        start_date <= observation_date < end_date

    The strict upper bound is our leakage barrier.
    """

    dates = history[
        "dates"
    ]

    if len(
        dates
    ) == 0:

        return history[
            "frame"
        ].iloc[
            0:0
        ]

    start_np = timestamp_to_numpy(
        start_date
    )

    end_np = timestamp_to_numpy(
        end_date
    )

    left = int(
        np.searchsorted(
            dates,
            start_np,
            side="left",
        )
    )

    right = int(
        np.searchsorted(
            dates,
            end_np,
            side="left",
        )
    )

    return history[
        "frame"
    ].iloc[
        left:right
    ]


def full_prior_slice(
    history: dict,
    *,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return every observation strictly before end_date.
    """

    dates = history[
        "dates"
    ]

    if len(
        dates
    ) == 0:

        return history[
            "frame"
        ].iloc[
            0:0
        ]

    end_np = timestamp_to_numpy(
        end_date
    )

    right = int(
        np.searchsorted(
            dates,
            end_np,
            side="left",
        )
    )

    return history[
        "frame"
    ].iloc[
        :right
    ]


# ============================================================
# PLAYER AGGREGATES
# ============================================================


def appearance_player_stats(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    """
    Per-player minutes and appearance counts.
    """

    if dataframe.empty:

        return (
            pd.Series(
                dtype=float
            ),
            pd.Series(
                dtype=float
            ),
        )

    minutes = (
        dataframe
        .groupby(
            "player_id"
        )[
            "minutes_played"
        ]
        .sum()
    )

    appearances = (
        dataframe
        .groupby(
            "player_id"
        )[
            "game_id"
        ]
        .nunique()
    )

    return (
        minutes,
        appearances,
    )


def lineup_player_stats(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    """
    Per-player starting counts and total selection counts.
    """

    if dataframe.empty:

        return (
            pd.Series(
                dtype=float
            ),
            pd.Series(
                dtype=float
            ),
        )

    starters = dataframe[
        dataframe[
            "type"
        ]
        ==
        "starting_lineup"
    ]

    start_counts = (
        starters
        .groupby(
            "player_id"
        )[
            "game_id"
        ]
        .nunique()
    )

    selected_counts = (
        dataframe
        .groupby(
            "player_id"
        )[
            "game_id"
        ]
        .nunique()
    )

    return (
        start_counts,
        selected_counts,
    )


# ============================================================
# CORE PLAYER RANKING
# ============================================================


def build_core_players(
    *,
    appearances_90: pd.DataFrame,
    appearances_180: pd.DataFrame,
    lineups_90: pd.DataFrame,
    lineups_180: pd.DataFrame,
) -> list[int]:
    """
    Estimate the pre-match core XI.

    Priority:
        recent starts
        recent minutes
        180-day starts
        180-day minutes
        total selections

    No match-day lineup is used.
    """

    (
        minutes_90,
        apps_90,
    ) = appearance_player_stats(
        appearances_90
    )

    (
        minutes_180,
        apps_180,
    ) = appearance_player_stats(
        appearances_180
    )

    (
        starts_90,
        selections_90,
    ) = lineup_player_stats(
        lineups_90
    )

    (
        starts_180,
        selections_180,
    ) = lineup_player_stats(
        lineups_180
    )

    player_ids = set()

    for series in [
        minutes_90,
        apps_90,
        minutes_180,
        apps_180,
        starts_90,
        selections_90,
        starts_180,
        selections_180,
    ]:

        player_ids.update(
            int(
                player_id
            )
            for player_id in series.index
        )

    if not player_ids:

        return []

    ranking = pd.DataFrame(
        index=sorted(
            player_ids
        )
    )

    ranking[
        "starts_90"
    ] = starts_90.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "minutes_90"
    ] = minutes_90.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "starts_180"
    ] = starts_180.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "minutes_180"
    ] = minutes_180.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "selections_180"
    ] = selections_180.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "appearances_180"
    ] = apps_180.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "player_id_sort"
    ] = ranking.index

    ranking = ranking.sort_values(
        by=[
            "starts_90",
            "minutes_90",
            "starts_180",
            "minutes_180",
            "selections_180",
            "appearances_180",
            "player_id_sort",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
    )

    return [
        int(
            player_id
        )
        for player_id in ranking.head(
            CORE_PLAYER_COUNT
        ).index
    ]


def build_block_core_players(
    *,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
) -> list[int]:
    """
    Core players for an older comparison window.

    Used only for continuity.
    """

    (
        minutes,
        appearances_count,
    ) = appearance_player_stats(
        appearances
    )

    (
        starts,
        selections,
    ) = lineup_player_stats(
        lineups
    )

    player_ids = set()

    for series in [
        minutes,
        appearances_count,
        starts,
        selections,
    ]:

        player_ids.update(
            int(
                player_id
            )
            for player_id in series.index
        )

    if not player_ids:

        return []

    ranking = pd.DataFrame(
        index=sorted(
            player_ids
        )
    )

    ranking[
        "starts"
    ] = starts.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "minutes"
    ] = minutes.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "selections"
    ] = selections.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "appearances"
    ] = appearances_count.reindex(
        ranking.index
    ).fillna(
        0
    )

    ranking[
        "player_id_sort"
    ] = ranking.index

    ranking = ranking.sort_values(
        by=[
            "starts",
            "minutes",
            "selections",
            "appearances",
            "player_id_sort",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    )

    return [
        int(
            player_id
        )
        for player_id in ranking.head(
            CORE_PLAYER_COUNT
        ).index
    ]


# ============================================================
# PLAYER SETS
# ============================================================


def player_set_from_evidence(
    *,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
) -> set[int]:
    """
    Union of players observed in either historical source.
    """

    players = set()

    if not appearances.empty:

        players.update(
            int(
                player_id
            )
            for player_id in appearances[
                "player_id"
            ].unique()
        )

    if not lineups.empty:

        players.update(
            int(
                player_id
            )
            for player_id in lineups[
                "player_id"
            ].unique()
        )

    return players


# ============================================================
# VALUATION LOOKUP
# ============================================================


def latest_valuation_before(
    *,
    player_id: int,
    match_date: pd.Timestamp,
    valuation_histories: dict[int, dict],
) -> tuple[
    float | None,
    pd.Timestamp | None,
]:
    """
    Latest valuation strictly before match_date.
    """

    history = valuation_histories.get(
        int(
            player_id
        )
    )

    if history is None:

        return (
            None,
            None,
        )

    dates = history[
        "dates"
    ]

    if len(
        dates
    ) == 0:

        return (
            None,
            None,
        )

    match_np = timestamp_to_numpy(
        match_date
    )

    index = int(
        np.searchsorted(
            dates,
            match_np,
            side="left",
        )
        - 1
    )

    if index < 0:

        return (
            None,
            None,
        )

    value = float(
        history[
            "values"
        ][
            index
        ]
    )

    valuation_date = pd.Timestamp(
        dates[
            index
        ]
    )

    return (
        value,
        valuation_date,
    )


# ============================================================
# VALUE FEATURES
# ============================================================


def calculate_value_features(
    *,
    player_ids: set[int] | list[int],
    match_date: pd.Timestamp,
    valuation_histories: dict[int, dict],
) -> dict:
    """
    Aggregate latest strictly pre-match valuations.
    """

    unique_players = list(
        dict.fromkeys(
            int(
                player_id
            )
            for player_id in player_ids
        )
    )

    if not unique_players:

        return {
            "player_count": 0,
            "valued_player_count": 0,
            "coverage": np.nan,
            "fresh_rate": np.nan,
            "mean_age_days": np.nan,
            "sum_eur": np.nan,
            "mean_eur": np.nan,
            "median_eur": np.nan,
            "top5_sum_eur": np.nan,
            "latest_valuation_date": pd.NaT,
        }

    values = []

    ages = []

    valuation_dates = []

    fresh_count = 0

    for player_id in unique_players:

        (
            value,
            valuation_date,
        ) = latest_valuation_before(
            player_id=player_id,
            match_date=match_date,
            valuation_histories=valuation_histories,
        )

        if (
            value is None
            or valuation_date is None
        ):

            continue

        age_days = int(
            (
                match_date
                - valuation_date
            ).days
        )

        if age_days <= 0:

            raise RuntimeError(
                "Valuation leakage detected: "
                f"player_id={player_id}, "
                f"valuation_date={valuation_date}, "
                f"match_date={match_date}"
            )

        values.append(
            float(
                value
            )
        )

        ages.append(
            float(
                age_days
            )
        )

        valuation_dates.append(
            valuation_date
        )

        if (
            age_days
            <=
            VALUATION_FRESH_DAYS
        ):

            fresh_count += 1

    player_count = len(
        unique_players
    )

    valued_count = len(
        values
    )

    if valued_count == 0:

        return {
            "player_count": player_count,
            "valued_player_count": 0,
            "coverage": 0.0,
            "fresh_rate": np.nan,
            "mean_age_days": np.nan,
            "sum_eur": np.nan,
            "mean_eur": np.nan,
            "median_eur": np.nan,
            "top5_sum_eur": np.nan,
            "latest_valuation_date": pd.NaT,
        }

    values_array = np.array(
        values,
        dtype=float,
    )

    descending_values = np.sort(
        values_array
    )[
        ::-1
    ]

    return {
        "player_count": player_count,

        "valued_player_count": (
            valued_count
        ),

        "coverage": (
            valued_count
            / player_count
        ),

        "fresh_rate": (
            fresh_count
            / valued_count
        ),

        "mean_age_days": float(
            np.mean(
                ages
            )
        ),

        "sum_eur": float(
            values_array.sum()
        ),

        "mean_eur": float(
            values_array.mean()
        ),

        "median_eur": float(
            np.median(
                values_array
            )
        ),

        "top5_sum_eur": float(
            descending_values[
                :5
            ].sum()
        ),

        "latest_valuation_date": max(
            valuation_dates
        ),
    }


# ============================================================
# EVIDENCE DATE
# ============================================================


def latest_prior_evidence_date(
    *,
    appearance_history: dict,
    lineup_history: dict,
    match_date: pd.Timestamp,
) -> pd.Timestamp | None:
    """
    Latest player evidence date strictly before the match.
    """

    candidates = []

    match_np = timestamp_to_numpy(
        match_date
    )

    for history in [
        appearance_history,
        lineup_history,
    ]:

        dates = history[
            "dates"
        ]

        if len(
            dates
        ) == 0:

            continue

        index = int(
            np.searchsorted(
                dates,
                match_np,
                side="left",
            )
            - 1
        )

        if index >= 0:

            candidates.append(
                pd.Timestamp(
                    dates[
                        index
                    ]
                )
            )

    if not candidates:

        return None

    return max(
        candidates
    )


# ============================================================
# TEAM SNAPSHOT
# ============================================================


def calculate_team_snapshot(
    *,
    club_id: int,
    match_date: pd.Timestamp,
    appearance_histories: dict[int, dict],
    lineup_histories: dict[int, dict],
    valuation_histories: dict[int, dict],
) -> dict:
    """
    Build one leakage-safe squad snapshot for one club.

    All observations obey:

        observation_date < match_date
    """

    appearance_history = (
        appearance_histories[
            club_id
        ]
    )

    lineup_history = (
        lineup_histories[
            club_id
        ]
    )

    # --------------------------------------------------------
    # FULL PRIOR HISTORY
    # --------------------------------------------------------

    prior_appearances = full_prior_slice(
        appearance_history,
        end_date=match_date,
    )

    prior_lineups = full_prior_slice(
        lineup_history,
        end_date=match_date,
    )

    appearance_available = int(
        not prior_appearances.empty
    )

    lineup_available = int(
        not prior_lineups.empty
    )

    # --------------------------------------------------------
    # RECENT WINDOWS
    # --------------------------------------------------------

    date_30 = (
        match_date
        - pd.Timedelta(
            days=RECENT_30_DAYS
        )
    )

    date_90 = (
        match_date
        - pd.Timedelta(
            days=RECENT_90_DAYS
        )
    )

    date_180 = (
        match_date
        - pd.Timedelta(
            days=RECENT_180_DAYS
        )
    )

    date_360 = (
        match_date
        - pd.Timedelta(
            days=PREVIOUS_BLOCK_START_DAYS
        )
    )

    appearances_30 = history_slice(
        appearance_history,
        start_date=date_30,
        end_date=match_date,
    )

    appearances_90 = history_slice(
        appearance_history,
        start_date=date_90,
        end_date=match_date,
    )

    appearances_180 = history_slice(
        appearance_history,
        start_date=date_180,
        end_date=match_date,
    )

    appearances_previous = history_slice(
        appearance_history,
        start_date=date_360,
        end_date=date_180,
    )

    lineups_30 = history_slice(
        lineup_history,
        start_date=date_30,
        end_date=match_date,
    )

    lineups_90 = history_slice(
        lineup_history,
        start_date=date_90,
        end_date=match_date,
    )

    lineups_180 = history_slice(
        lineup_history,
        start_date=date_180,
        end_date=match_date,
    )

    lineups_previous = history_slice(
        lineup_history,
        start_date=date_360,
        end_date=date_180,
    )

    # --------------------------------------------------------
    # RECENT PLAYER SETS
    # --------------------------------------------------------

    players_30 = player_set_from_evidence(
        appearances=appearances_30,
        lineups=lineups_30,
    )

    players_90 = player_set_from_evidence(
        appearances=appearances_90,
        lineups=lineups_90,
    )

    players_180 = player_set_from_evidence(
        appearances=appearances_180,
        lineups=lineups_180,
    )

    previous_players = (
        player_set_from_evidence(
            appearances=appearances_previous,
            lineups=lineups_previous,
        )
    )

    recent_history_available = int(
        bool(
            players_180
        )
    )

    # --------------------------------------------------------
    # UNIQUE GAMES
    # --------------------------------------------------------

    def unique_games(
        appearances_block: pd.DataFrame,
        lineups_block: pd.DataFrame,
    ) -> set:
        games = set()

        if not appearances_block.empty:

            games.update(
                appearances_block[
                    "game_id"
                ].dropna()
            )

        if not lineups_block.empty:

            games.update(
                lineups_block[
                    "game_id"
                ].dropna()
            )

        return games

    games_30 = unique_games(
        appearances_30,
        lineups_30,
    )

    games_90 = unique_games(
        appearances_90,
        lineups_90,
    )

    games_180 = unique_games(
        appearances_180,
        lineups_180,
    )

    # --------------------------------------------------------
    # APPEARANCE / START COUNTS
    # --------------------------------------------------------

    appearance_count_30 = len(
        appearances_30
    )

    appearance_count_90 = len(
        appearances_90
    )

    appearance_count_180 = len(
        appearances_180
    )

    starts_30_frame = lineups_30[
        lineups_30[
            "type"
        ]
        ==
        "starting_lineup"
    ]

    starts_90_frame = lineups_90[
        lineups_90[
            "type"
        ]
        ==
        "starting_lineup"
    ]

    starts_180_frame = lineups_180[
        lineups_180[
            "type"
        ]
        ==
        "starting_lineup"
    ]

    starts_count_30 = len(
        starts_30_frame
    )

    starts_count_90 = len(
        starts_90_frame
    )

    starts_count_180 = len(
        starts_180_frame
    )

    minutes_30 = float(
        appearances_30[
            "minutes_played"
        ].sum()
    )

    minutes_90 = float(
        appearances_90[
            "minutes_played"
        ].sum()
    )

    minutes_180 = float(
        appearances_180[
            "minutes_played"
        ].sum()
    )

    # --------------------------------------------------------
    # CORE XI
    # --------------------------------------------------------

    core_players = build_core_players(
        appearances_90=appearances_90,
        appearances_180=appearances_180,
        lineups_90=lineups_90,
        lineups_180=lineups_180,
    )

    previous_core = build_block_core_players(
        appearances=appearances_previous,
        lineups=lineups_previous,
    )

    core_set = set(
        core_players
    )

    previous_core_set = set(
        previous_core
    )

    # --------------------------------------------------------
    # CORE MINUTE SHARE
    # --------------------------------------------------------

    if (
        not appearances_90.empty
        and minutes_90 > 0
        and core_set
    ):

        core_minutes_90 = float(
            appearances_90[
                appearances_90[
                    "player_id"
                ].isin(
                    core_set
                )
            ][
                "minutes_played"
            ].sum()
        )

        core_minutes_share_90 = (
            core_minutes_90
            / minutes_90
        )

    else:

        core_minutes_share_90 = np.nan

    # --------------------------------------------------------
    # CORE START SHARE
    # --------------------------------------------------------

    if (
        starts_count_180 > 0
        and core_set
    ):

        core_starts_180 = len(
            starts_180_frame[
                starts_180_frame[
                    "player_id"
                ].isin(
                    core_set
                )
            ]
        )

        core_start_share_180 = (
            core_starts_180
            / starts_count_180
        )

    else:

        core_start_share_180 = np.nan

    # --------------------------------------------------------
    # STARTER STABILITY
    # --------------------------------------------------------

    if starts_count_180 > 0:

        player_start_counts = (
            starts_180_frame
            .groupby(
                "player_id"
            )[
                "game_id"
            ]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        top11_starts = float(
            player_start_counts.head(
                CORE_PLAYER_COUNT
            ).sum()
        )

        starter_stability = (
            top11_starts
            / starts_count_180
        )

    else:

        starter_stability = np.nan

    # --------------------------------------------------------
    # CONTINUITY
    # --------------------------------------------------------

    if (
        core_set
        and previous_core_set
    ):

        denominator = min(
            CORE_PLAYER_COUNT,
            len(
                core_set
            ),
            len(
                previous_core_set
            ),
        )

        if denominator > 0:

            core_continuity = (
                len(
                    core_set
                    &
                    previous_core_set
                )
                / denominator
            )

        else:

            core_continuity = np.nan

    else:

        core_continuity = np.nan

    if (
        players_180
        and previous_players
    ):

        union = (
            players_180
            |
            previous_players
        )

        player_pool_continuity = (
            len(
                players_180
                &
                previous_players
            )
            / len(
                union
            )
        )

    else:

        player_pool_continuity = np.nan

    # --------------------------------------------------------
    # VALUATIONS
    # --------------------------------------------------------

    active_values = calculate_value_features(
        player_ids=players_180,
        match_date=match_date,
        valuation_histories=valuation_histories,
    )

    core_values = calculate_value_features(
        player_ids=core_players,
        match_date=match_date,
        valuation_histories=valuation_histories,
    )

    valuation_available = int(
        active_values[
            "valued_player_count"
        ]
        > 0
    )

    # --------------------------------------------------------
    # LATEST EVIDENCE
    # --------------------------------------------------------

    latest_evidence = (
        latest_prior_evidence_date(
            appearance_history=appearance_history,
            lineup_history=lineup_history,
            match_date=match_date,
        )
    )

    if latest_evidence is None:

        days_since_evidence = np.nan

    else:

        days_since_evidence = float(
            (
                match_date
                - latest_evidence
            ).days
        )

        if days_since_evidence <= 0:

            raise RuntimeError(
                "Squad evidence leakage detected."
            )

    latest_valuation_dates = [
        date
        for date in [
            active_values[
                "latest_valuation_date"
            ],
            core_values[
                "latest_valuation_date"
            ],
        ]
        if not pd.isna(
            date
        )
    ]

    latest_valuation_used = (
        max(
            latest_valuation_dates
        )
        if latest_valuation_dates
        else pd.NaT
    )

    if (
        not pd.isna(
            latest_valuation_used
        )
        and
        latest_valuation_used
        >= match_date
    ):

        raise RuntimeError(
            "Valuation date leakage detected."
        )

    # --------------------------------------------------------
    # DATA COMPLETENESS
    # --------------------------------------------------------

    completeness_parts = [
        float(
            appearance_available
        ),
        float(
            lineup_available
        ),
    ]

    if not pd.isna(
        core_values[
            "coverage"
        ]
    ):

        completeness_parts.append(
            float(
                core_values[
                    "coverage"
                ]
            )
        )

    data_completeness = safe_mean(
        completeness_parts
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    active_sum_eur = active_values[
        "sum_eur"
    ]

    core_sum_eur = core_values[
        "sum_eur"
    ]

    top5_sum_eur = core_values[
        "top5_sum_eur"
    ]

    return {
        "squad_appearance_history_available": (
            appearance_available
        ),

        "squad_lineup_history_available": (
            lineup_available
        ),

        "squad_recent_history_available": (
            recent_history_available
        ),

        "squad_valuation_history_available": (
            valuation_available
        ),

        "squad_days_since_last_player_evidence": (
            days_since_evidence
        ),

        "squad_games_last30d": len(
            games_30
        ),

        "squad_games_last90d": len(
            games_90
        ),

        "squad_games_last180d": len(
            games_180
        ),

        "squad_players_seen_last30d": len(
            players_30
        ),

        "squad_players_seen_last90d": len(
            players_90
        ),

        "squad_players_seen_last180d": len(
            players_180
        ),

        "squad_appearances_last30d": (
            appearance_count_30
        ),

        "squad_appearances_last90d": (
            appearance_count_90
        ),

        "squad_appearances_last180d": (
            appearance_count_180
        ),

        "squad_starts_last30d": (
            starts_count_30
        ),

        "squad_starts_last90d": (
            starts_count_90
        ),

        "squad_starts_last180d": (
            starts_count_180
        ),

        "squad_minutes_last30d": (
            minutes_30
        ),

        "squad_minutes_last90d": (
            minutes_90
        ),

        "squad_minutes_last180d": (
            minutes_180
        ),

        "squad_minutes_per_game_last90d": (
            safe_ratio(
                minutes_90,
                len(
                    games_90
                ),
            )
        ),

        "squad_minutes_per_game_last180d": (
            safe_ratio(
                minutes_180,
                len(
                    games_180
                ),
            )
        ),

        "squad_core11_count": len(
            core_players
        ),

        "squad_core11_minutes_share_90d": (
            core_minutes_share_90
        ),

        "squad_core11_start_share_180d": (
            core_start_share_180
        ),

        "squad_starter_stability_180d": (
            starter_stability
        ),

        "squad_core11_continuity": (
            core_continuity
        ),

        "squad_player_pool_continuity": (
            player_pool_continuity
        ),

        "squad_active_value_m_eur": (
            active_sum_eur
            / 1_000_000
            if not pd.isna(
                active_sum_eur
            )
            else np.nan
        ),

        "squad_active_value_log": (
            float(
                np.log1p(
                    active_sum_eur
                )
            )
            if not pd.isna(
                active_sum_eur
            )
            else np.nan
        ),

        "squad_active_valuation_coverage": (
            active_values[
                "coverage"
            ]
        ),

        "squad_active_fresh_valuation_rate": (
            active_values[
                "fresh_rate"
            ]
        ),

        "squad_active_mean_valuation_age_days": (
            active_values[
                "mean_age_days"
            ]
        ),

        "squad_core11_value_m_eur": (
            core_sum_eur
            / 1_000_000
            if not pd.isna(
                core_sum_eur
            )
            else np.nan
        ),

        "squad_core11_value_log": (
            float(
                np.log1p(
                    core_sum_eur
                )
            )
            if not pd.isna(
                core_sum_eur
            )
            else np.nan
        ),

        "squad_core11_avg_value_m_eur": (
            core_values[
                "mean_eur"
            ]
            / 1_000_000
            if not pd.isna(
                core_values[
                    "mean_eur"
                ]
            )
            else np.nan
        ),

        "squad_core11_median_value_m_eur": (
            core_values[
                "median_eur"
            ]
            / 1_000_000
            if not pd.isna(
                core_values[
                    "median_eur"
                ]
            )
            else np.nan
        ),

        "squad_core11_valuation_coverage": (
            core_values[
                "coverage"
            ]
        ),

        "squad_core11_fresh_valuation_rate": (
            core_values[
                "fresh_rate"
            ]
        ),

        "squad_core11_mean_valuation_age_days": (
            core_values[
                "mean_age_days"
            ]
        ),

        "squad_top5_value_m_eur": (
            top5_sum_eur
            / 1_000_000
            if not pd.isna(
                top5_sum_eur
            )
            else np.nan
        ),

        "squad_data_completeness": (
            data_completeness
        ),

        # -----------------------------------------------
        # AUDIT-ONLY
        # -----------------------------------------------

        "squad_latest_player_evidence_date": (
            latest_evidence
            if latest_evidence is not None
            else pd.NaT
        ),

        "squad_latest_valuation_date_used": (
            latest_valuation_used
        ),
    }


# ============================================================
# ATTACH FEATURES
# ============================================================


def attach_squad_features(
    *,
    matches: pd.DataFrame,
    team_to_id: dict[str, int],
    appearance_histories: dict[int, dict],
    lineup_histories: dict[int, dict],
    valuation_histories: dict[int, dict],
) -> pd.DataFrame:
    """
    Calculate home/away squad features for every UCL match.
    """

    output = matches.copy()

    cache = {}

    rows = []

    print(
        "\nAttaching strictly pre-match "
        "squad/player features..."
    )

    total = len(
        output
    )

    for index, row in output.iterrows():

        match_date = row[
            "date"
        ]

        home_team = row[
            "home_team"
        ]

        away_team = row[
            "away_team"
        ]

        if (
            home_team
            not in team_to_id
        ):

            raise RuntimeError(
                f"No Transfermarkt club ID "
                f"for {home_team}."
            )

        if (
            away_team
            not in team_to_id
        ):

            raise RuntimeError(
                f"No Transfermarkt club ID "
                f"for {away_team}."
            )

        home_club_id = int(
            team_to_id[
                home_team
            ]
        )

        away_club_id = int(
            team_to_id[
                away_team
            ]
        )

        home_cache_key = (
            home_club_id,
            match_date,
        )

        away_cache_key = (
            away_club_id,
            match_date,
        )

        if (
            home_cache_key
            not in cache
        ):

            cache[
                home_cache_key
            ] = calculate_team_snapshot(
                club_id=home_club_id,
                match_date=match_date,
                appearance_histories=appearance_histories,
                lineup_histories=lineup_histories,
                valuation_histories=valuation_histories,
            )

        if (
            away_cache_key
            not in cache
        ):

            cache[
                away_cache_key
            ] = calculate_team_snapshot(
                club_id=away_club_id,
                match_date=match_date,
                appearance_histories=appearance_histories,
                lineup_histories=lineup_histories,
                valuation_histories=valuation_histories,
            )

        home_snapshot = cache[
            home_cache_key
        ]

        away_snapshot = cache[
            away_cache_key
        ]

        feature_row = {}

        for (
            feature_name,
            value,
        ) in home_snapshot.items():

            feature_row[
                f"home_{feature_name}"
            ] = value

        for (
            feature_name,
            value,
        ) in away_snapshot.items():

            feature_row[
                f"away_{feature_name}"
            ] = value

        for feature_name in DIFFERENCE_FEATURES:

            feature_row[
                f"{feature_name}_diff"
            ] = safe_difference(
                home_snapshot[
                    feature_name
                ],
                away_snapshot[
                    feature_name
                ],
            )

        rows.append(
            feature_row
        )

        if (
            (
                index + 1
            )
            % 250
            ==
            0
        ):

            print(
                f"  Processed "
                f"{index + 1:,} / "
                f"{total:,} UCL matches"
            )

    feature_frame = pd.DataFrame(
        rows
    )

    output = pd.concat(
        [
            output.reset_index(
                drop=True
            ),
            feature_frame.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return output


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate row integrity, ranges and temporal leakage.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SQUAD STRENGTH DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # ROWS
    # --------------------------------------------------------

    if len(
        dataframe
    ) != EXPECTED_UCL_MATCHES:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_UCL_MATCHES:,} UCL rows, "
            f"found {len(dataframe):,}."
        )

    print(
        f"\nMatch rows: "
        f"{len(dataframe):,} ✓"
    )

    # --------------------------------------------------------
    # MATCH IDS
    # --------------------------------------------------------

    if dataframe[
        "match_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate match IDs detected."
        )

    print(
        "Unique match IDs: valid ✓"
    )

    # --------------------------------------------------------
    # CHRONOLOGY
    # --------------------------------------------------------

    if not dataframe[
        "date"
    ].is_monotonic_increasing:

        raise RuntimeError(
            "Output is not chronological."
        )

    print(
        "Chronological ordering: valid ✓"
    )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    for side in [
        "home",
        "away",
    ]:

        evidence_column = (
            f"{side}_"
            f"squad_latest_player_evidence_date"
        )

        valuation_column = (
            f"{side}_"
            f"squad_latest_valuation_date_used"
        )

        evidence_dates = pd.to_datetime(
            dataframe[
                evidence_column
            ],
            errors="coerce",
        )

        valuation_dates = pd.to_datetime(
            dataframe[
                valuation_column
            ],
            errors="coerce",
        )

        evidence_bad = (
            evidence_dates.notna()
            &
            (
                evidence_dates
                >=
                dataframe[
                    "date"
                ]
            )
        )

        valuation_bad = (
            valuation_dates.notna()
            &
            (
                valuation_dates
                >=
                dataframe[
                    "date"
                ]
            )
        )

        if evidence_bad.any():

            raise RuntimeError(
                f"{side} player evidence "
                f"leakage detected."
            )

        if valuation_bad.any():

            raise RuntimeError(
                f"{side} valuation "
                f"leakage detected."
            )

    print(
        "Strict pre-match player-date "
        "leakage check: passed ✓"
    )

    # --------------------------------------------------------
    # AVAILABILITY FLAGS
    # --------------------------------------------------------

    flag_features = [
        "squad_appearance_history_available",
        "squad_lineup_history_available",
        "squad_recent_history_available",
        "squad_valuation_history_available",
    ]

    for side in [
        "home",
        "away",
    ]:

        for feature in flag_features:

            column = (
                f"{side}_{feature}"
            )

            values = set(
                dataframe[
                    column
                ]
                .dropna()
                .unique()
            )

            if not values.issubset(
                {
                    0,
                    1,
                }
            ):

                raise RuntimeError(
                    f"Invalid availability values "
                    f"in {column}: "
                    f"{values}"
                )

    print(
        "Availability flags: valid ✓"
    )

    # --------------------------------------------------------
    # CORE COUNT
    # --------------------------------------------------------

    for side in [
        "home",
        "away",
    ]:

        column = (
            f"{side}_"
            f"squad_core11_count"
        )

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
                CORE_PLAYER_COUNT
            )
        )

        if invalid.any():

            raise RuntimeError(
                f"Invalid core XI counts "
                f"in {column}."
            )

    print(
        "Core XI counts: valid ✓"
    )

    # --------------------------------------------------------
    # RATIO FEATURES
    # --------------------------------------------------------

    ratio_features = [
        "squad_core11_minutes_share_90d",
        "squad_core11_start_share_180d",
        "squad_starter_stability_180d",
        "squad_core11_continuity",
        "squad_player_pool_continuity",
        "squad_active_valuation_coverage",
        "squad_active_fresh_valuation_rate",
        "squad_core11_valuation_coverage",
        "squad_core11_fresh_valuation_rate",
        "squad_data_completeness",
    ]

    for side in [
        "home",
        "away",
    ]:

        for feature in ratio_features:

            column = (
                f"{side}_{feature}"
            )

            values = dataframe[
                column
            ].dropna()

            if (
                (
                    values
                    <
                    -1e-9
                ).any()
                or
                (
                    values
                    >
                    1 + 1e-9
                ).any()
            ):

                raise RuntimeError(
                    f"Ratio feature outside "
                    f"[0, 1]: {column}"
                )

    print(
        "Coverage/stability ranges: valid ✓"
    )

    # --------------------------------------------------------
    # NONNEGATIVE FEATURES
    # --------------------------------------------------------

    nonnegative_patterns = [
        "_games_last",
        "_players_seen_last",
        "_appearances_last",
        "_starts_last",
        "_minutes_last",
        "_minutes_per_game",
        "_value_m_eur",
        "_mean_valuation_age_days",
        "_days_since_last_player_evidence",
    ]

    for column in dataframe.columns:

        if (
            not column.startswith(
                (
                    "home_",
                    "away_",
                )
            )
        ):

            continue

        if not any(
            pattern in column
            for pattern in nonnegative_patterns
        ):

            continue

        values = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        ).dropna()

        if (
            values
            <
            0
        ).any():

            raise RuntimeError(
                f"Negative values detected "
                f"in {column}."
            )

    print(
        "Count/value ranges: valid ✓"
    )

    # --------------------------------------------------------
    # COLD START COVERAGE
    # --------------------------------------------------------

    both_recent = (
        (
            dataframe[
                "home_squad_recent_history_available"
            ]
            ==
            1
        )
        &
        (
            dataframe[
                "away_squad_recent_history_available"
            ]
            ==
            1
        )
    )

    one_recent = (
        (
            dataframe[
                "home_squad_recent_history_available"
            ]
            +
            dataframe[
                "away_squad_recent_history_available"
            ]
        )
        ==
        1
    )

    neither_recent = (
        (
            dataframe[
                "home_squad_recent_history_available"
            ]
            ==
            0
        )
        &
        (
            dataframe[
                "away_squad_recent_history_available"
            ]
            ==
            0
        )
    )

    print(
        "\nRecent squad-history coverage:"
    )

    print(
        f"Both teams available: "
        f"{int(both_recent.sum()):,}"
    )

    print(
        f"One team available:  "
        f"{int(one_recent.sum()):,}"
    )

    print(
        f"Neither available:   "
        f"{int(neither_recent.sum()):,}"
    )

    print(
        f"Both-team coverage:  "
        f"{both_recent.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CORE VALUATION COVERAGE
    # --------------------------------------------------------

    both_core_valued = (
        (
            dataframe[
                "home_squad_core11_valuation_coverage"
            ]
            >=
            0.75
        )
        &
        (
            dataframe[
                "away_squad_core11_valuation_coverage"
            ]
            >=
            0.75
        )
    )

    print(
        "\nMatches where both teams have "
        ">=75% core-XI valuation coverage:"
    )

    print(
        f"{int(both_core_valued.sum()):,} "
        f"/ {len(dataframe):,} "
        f"({both_core_valued.mean() * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # EXPECTED NANs
    # --------------------------------------------------------

    squad_columns = [
        column
        for column in dataframe.columns
        if (
            column.startswith(
                "home_squad_"
            )
            or
            column.startswith(
                "away_squad_"
            )
            or (
                column.startswith(
                    "squad_"
                )
                and column.endswith(
                    "_diff"
                )
            )
        )
    ]

    total_missing = int(
        dataframe[
            squad_columns
        ].isna().sum().sum()
    )

    print(
        f"\nExpected squad cold-start/"
        f"coverage NaNs: "
        f"{total_missing:,}"
    )

    print(
        "Missing player history preserved "
        "intentionally ✓"
    )


# ============================================================
# SAVE OUTPUT
# ============================================================


def save_output(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save match-level player/squad feature history.
    """

    output = dataframe.copy()

    date_columns = [
        column
        for column in output.columns
        if (
            column == "date"
            or column.endswith(
                "_date"
            )
            or column.endswith(
                "_date_used"
            )
        )
    ]

    for column in date_columns:

        output[
            column
        ] = pd.to_datetime(
            output[
                column
            ],
            errors="coerce",
        ).dt.strftime(
            "%Y-%m-%d"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )


# ============================================================
# LATEST TRAINING SNAPSHOT
# ============================================================


def build_latest_team_table(
    *,
    matches: pd.DataFrame,
    mapping: pd.DataFrame,
    appearance_histories: dict[int, dict],
    lineup_histories: dict[int, dict],
    valuation_histories: dict[int, dict],
) -> pd.DataFrame:
    """
    Build one final training-era snapshot per UCL club.

    Reference date is one day after the final UCL match in the
    master training history.

    This is NOT our final live 2026/27 squad source.
    """

    reference_date = (
        matches[
            "date"
        ].max()
        +
        pd.Timedelta(
            days=1
        )
    )

    rows = []

    for row in mapping.itertuples(
        index=False
    ):

        snapshot = calculate_team_snapshot(
            club_id=int(
                row.club_id
            ),
            match_date=reference_date,
            appearance_histories=appearance_histories,
            lineup_histories=lineup_histories,
            valuation_histories=valuation_histories,
        )

        output_row = {
            "team": row.ucl_team,
            "club_id": int(
                row.club_id
            ),
            "reference_date": (
                reference_date
            ),
        }

        output_row.update(
            snapshot
        )

        rows.append(
            output_row
        )

    latest = pd.DataFrame(
        rows
    )

    latest = latest.sort_values(
        by=[
            "squad_core11_value_m_eur",
            "squad_active_value_m_eur",
        ],
        ascending=False,
        na_position="last",
    ).reset_index(
        drop=True
    )

    output = latest.copy()

    date_columns = [
        "reference_date",
        "squad_latest_player_evidence_date",
        "squad_latest_valuation_date_used",
    ]

    for column in date_columns:

        output[
            column
        ] = pd.to_datetime(
            output[
                column
            ],
            errors="coerce",
        ).dt.strftime(
            "%Y-%m-%d"
        )

    output.to_csv(
        LATEST_FILE,
        index=False,
        encoding="utf-8",
    )

    return latest


# ============================================================
# DISPLAY SUMMARY
# ============================================================


def print_summary(
    *,
    dataframe: pd.DataFrame,
    original_columns: list[str],
    latest: pd.DataFrame,
) -> None:
    """
    Print useful sanity checks.
    """

    new_columns = [
        column
        for column in dataframe.columns
        if column not in original_columns
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SQUAD STRENGTH FEATURE SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nMatches processed: "
        f"{len(dataframe):,}"
    )

    print(
        f"New squad/player/context "
        f"features created: "
        f"{len(new_columns):,}"
    )

    display_columns = [
        "team",
        "squad_players_seen_last180d",
        "squad_core11_count",
        "squad_core11_value_m_eur",
        "squad_core11_valuation_coverage",
        "squad_active_value_m_eur",
        "squad_starter_stability_180d",
        "squad_core11_continuity",
        "squad_data_completeness",
    ]

    print(
        "\nTop 15 UCL clubs by latest "
        "training-era core-XI value:"
    )

    print(
        latest[
            display_columns
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
    )

    match_display_columns = [
        "date",
        "home_team",
        "away_team",
        "result",

        "home_squad_core11_value_m_eur",
        "away_squad_core11_value_m_eur",

        "home_squad_core11_valuation_coverage",
        "away_squad_core11_valuation_coverage",

        "home_squad_starter_stability_180d",
        "away_squad_starter_stability_180d",

        "home_squad_core11_continuity",
        "away_squad_core11_continuity",

        "squad_core11_value_m_eur_diff",
    ]

    existing_display = [
        column
        for column in match_display_columns
        if column in dataframe.columns
    ]

    print(
        "\nFirst 5 UCL matches:"
    )

    print(
        dataframe[
            existing_display
        ]
        .head(
            5
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nLast 5 UCL matches:"
    )

    print(
        dataframe[
            existing_display
        ]
        .tail(
            5
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nFeature history saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nLatest training-era squad table saved to:"
    )

    print(
        LATEST_FILE
    )


# ============================================================
# MAIN
# ============================================================


def build_squad_strength_features() -> None:
    """
    Build leakage-safe historical player/squad features.

    Membership:
        historical appearances + lineups

    Player quality:
        latest valuation strictly before UCL match date

    Core XI:
        estimated only from prior usage/starts

    No current-match lineup is used.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - SQUAD STRENGTH FEATURE ENGINEERING"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    matches = load_ucl_matches()

    original_columns = list(
        matches.columns
    )

    (
        mapping,
        team_to_id,
        _,
    ) = load_club_map()

    appearances = load_appearances()

    lineups = load_lineups()

    valuations = load_valuations()

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    print(
        "\nBuilding chronological player-history indexes..."
    )

    club_ids = set(
        mapping[
            "club_id"
        ].astype(
            int
        )
    )

    (
        appearance_histories,
        lineup_histories,
    ) = build_club_histories(
        appearances=appearances,
        lineups=lineups,
        club_ids=club_ids,
    )

    valuation_histories = (
        build_valuation_histories(
            valuations
        )
    )

    print(
        f"Club appearance histories: "
        f"{len(appearance_histories):,}"
    )

    print(
        f"Club lineup histories: "
        f"{len(lineup_histories):,}"
    )

    print(
        f"Player valuation histories: "
        f"{len(valuation_histories):,}"
    )

    # --------------------------------------------------------
    # ATTACH
    # --------------------------------------------------------

    featured = attach_squad_features(
        matches=matches,
        team_to_id=team_to_id,
        appearance_histories=appearance_histories,
        lineup_histories=lineup_histories,
        valuation_histories=valuation_histories,
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    run_quality_checks(
        featured
    )

    # --------------------------------------------------------
    # LATEST TABLE
    # --------------------------------------------------------

    latest = build_latest_team_table(
        matches=matches,
        mapping=mapping,
        appearance_histories=appearance_histories,
        lineup_histories=lineup_histories,
        valuation_histories=valuation_histories,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_output(
        featured
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_summary(
        dataframe=featured,
        original_columns=original_columns,
        latest=latest,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SQUAD STRENGTH FEATURES CREATED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nLeakage barriers enforced:"
    )

    print(
        "  appearance_date < UCL_match_date"
    )

    print(
        "  lineup_date < UCL_match_date"
    )

    print(
        "  valuation_date < UCL_match_date"
    )

    print(
        "\nCurrent-match XI is never used."
    )

    print(
        "Current player club fields are never used "
        "for historical membership."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    build_squad_strength_features()