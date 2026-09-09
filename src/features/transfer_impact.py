from __future__ import annotations

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
    / "ucl_matches_with_squad_strength.csv"
)

TRANSFER_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_transfers_ucl_clubs.csv"
)

CLUB_MAP_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_transfer_impact.csv"
)

LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_transfer_impact_latest.csv"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_UCL_MATCHES = 2089
EXPECTED_UCL_CLUBS = 117


# ============================================================
# WINDOWS
# ============================================================

WINDOW_30D = 30
WINDOW_90D = 90
WINDOW_180D = 180
WINDOW_365D = 365


# ============================================================
# TEAM FEATURES
# ============================================================

TEAM_FEATURES = [
    "transfer_history_available",

    "transfer_days_since_last_event",
    "transfer_days_since_last_incoming",
    "transfer_days_since_last_outgoing",

    "transfer_incoming_count_30d",
    "transfer_incoming_count_90d",
    "transfer_incoming_count_180d",
    "transfer_incoming_count_365d",

    "transfer_outgoing_count_30d",
    "transfer_outgoing_count_90d",
    "transfer_outgoing_count_180d",
    "transfer_outgoing_count_365d",

    "transfer_net_player_count_30d",
    "transfer_net_player_count_90d",
    "transfer_net_player_count_180d",
    "transfer_net_player_count_365d",

    "transfer_turnover_count_30d",
    "transfer_turnover_count_90d",
    "transfer_turnover_count_180d",
    "transfer_turnover_count_365d",

    "transfer_turnover_per_active_player_180d",

    "transfer_incoming_value_m_eur_90d",
    "transfer_incoming_value_m_eur_180d",
    "transfer_incoming_value_m_eur_365d",

    "transfer_outgoing_value_m_eur_90d",
    "transfer_outgoing_value_m_eur_180d",
    "transfer_outgoing_value_m_eur_365d",

    "transfer_net_value_m_eur_90d",
    "transfer_net_value_m_eur_180d",
    "transfer_net_value_m_eur_365d",

    "transfer_value_turnover_m_eur_180d",
    "transfer_value_turnover_m_eur_365d",

    "transfer_value_turnover_to_squad_value_180d",

    "transfer_incoming_value_coverage_180d",
    "transfer_outgoing_value_coverage_180d",
    "transfer_value_coverage_180d",

    "transfer_incoming_fee_m_eur_180d",
    "transfer_outgoing_fee_m_eur_180d",
    "transfer_net_spend_m_eur_180d",
    "transfer_fee_coverage_180d",

    "transfer_arrival_avg_age_days_180d",
    "transfer_arrival_value_weighted_age_days_180d",

    "transfer_largest_arrival_value_m_eur_365d",
    "transfer_days_since_largest_arrival_365d",

    "transfer_largest_departure_value_m_eur_365d",
    "transfer_days_since_largest_departure_365d",

    "transfer_recent_arrival_pressure",
    "transfer_recent_departure_pressure",
    "transfer_recent_turnover_pressure",

    "transfer_data_completeness",
]


# ============================================================
# DIFFERENCE FEATURES
# ============================================================

DIFFERENCE_FEATURES = [
    "transfer_incoming_count_90d",
    "transfer_outgoing_count_90d",

    "transfer_net_player_count_90d",
    "transfer_net_player_count_180d",

    "transfer_turnover_count_180d",
    "transfer_turnover_per_active_player_180d",

    "transfer_incoming_value_m_eur_180d",
    "transfer_outgoing_value_m_eur_180d",

    "transfer_net_value_m_eur_90d",
    "transfer_net_value_m_eur_180d",
    "transfer_net_value_m_eur_365d",

    "transfer_value_turnover_m_eur_180d",
    "transfer_value_turnover_to_squad_value_180d",

    "transfer_arrival_avg_age_days_180d",
    "transfer_arrival_value_weighted_age_days_180d",

    "transfer_largest_arrival_value_m_eur_365d",
    "transfer_largest_departure_value_m_eur_365d",

    "transfer_recent_arrival_pressure",
    "transfer_recent_departure_pressure",
    "transfer_recent_turnover_pressure",

    "transfer_data_completeness",
]


# ============================================================
# HELPERS
# ============================================================


def require_file(
    path,
    description: str,
) -> None:
    """
    Fail early if a required source is missing.
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


def safe_difference(
    home_value,
    away_value,
):
    """
    Preserve missingness when either side is unavailable.
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


def safe_ratio(
    numerator,
    denominator,
):
    """
    Safe non-zero ratio.
    """

    if pd.isna(
        numerator
    ) or pd.isna(
        denominator
    ):

        return np.nan

    if denominator <= 0:

        return np.nan

    return float(
        numerator
        / denominator
    )


def timestamp_to_numpy(
    value: pd.Timestamp,
) -> np.datetime64:
    """
    Convert timestamp into numpy datetime64[ns].
    """

    return np.datetime64(
        pd.Timestamp(
            value
        ).to_datetime64()
    )


# ============================================================
# LOAD UCL FEATURES
# ============================================================


def load_ucl_matches() -> pd.DataFrame:
    """
    Load UCL matches including squad-strength features.
    """

    print(
        "\nLoading UCL squad-strength dataset:"
    )

    print(
        UCL_INPUT_FILE
    )

    require_file(
        UCL_INPUT_FILE,
        "UCL squad-strength dataset",
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

        "home_squad_players_seen_last180d",
        "away_squad_players_seen_last180d",

        "home_squad_active_value_m_eur",
        "away_squad_active_value_m_eur",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "UCL squad-strength dataset missing columns: "
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
]:
    """
    Load canonical UCL -> Transfermarkt IDs.
    """

    require_file(
        CLUB_MAP_FILE,
        "Final Transfermarkt club map",
    )

    dataframe = pd.read_csv(
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
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Club map missing columns: "
            f"{sorted(missing)}"
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

    if len(
        dataframe
    ) != EXPECTED_UCL_CLUBS:

        raise RuntimeError(
            f"Expected {EXPECTED_UCL_CLUBS} "
            f"mapped clubs but found "
            f"{len(dataframe)}."
        )

    if dataframe[
        "ucl_team"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate canonical team in club map."
        )

    if dataframe[
        "club_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Transfermarkt club ID in club map."
        )

    team_to_id = (
        dataframe
        .set_index(
            "ucl_team"
        )[
            "club_id"
        ]
        .to_dict()
    )

    return (
        dataframe,
        team_to_id,
    )


# ============================================================
# LOAD TRANSFERS
# ============================================================


def load_transfer_history() -> pd.DataFrame:
    """
    Load prepared UCL-club transfer history.

    Future records remain in this dataframe, but all feature
    lookups will use a strict upper date bound.
    """

    print(
        "\nLoading prepared transfer history:"
    )

    print(
        TRANSFER_FILE
    )

    require_file(
        TRANSFER_FILE,
        "Prepared transfer history",
    )

    dataframe = pd.read_csv(
        TRANSFER_FILE,
        low_memory=False,
    )

    required = {
        "player_id",
        "transfer_date",
        "from_club_id",
        "to_club_id",
        "from_ucl_club",
        "to_ucl_club",
        "movement_type",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Transfer history missing columns: "
            f"{sorted(missing)}"
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
        "transfer_date"
    ] = pd.to_datetime(
        dataframe[
            "transfer_date"
        ],
        errors="raise",
    )

    for column in [
        "from_club_id",
        "to_club_id",
    ]:

        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        ).astype(
            "Int64"
        )

    for column in [
        "market_value_in_eur",
        "transfer_fee",
    ]:

        if column not in dataframe.columns:

            dataframe[
                column
            ] = np.nan

        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        )

    dataframe = dataframe.sort_values(
        by=[
            "transfer_date",
            "player_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Transfer rows loaded: "
        f"{len(dataframe):,}"
    )

    print(
        f"Transfer date range: "
        f"{dataframe['transfer_date'].min().date()} "
        f"to "
        f"{dataframe['transfer_date'].max().date()}"
    )

    return dataframe


# ============================================================
# EXPAND INTO CLUB EVENTS
# ============================================================


def build_club_events(
    *,
    transfers: pd.DataFrame,
    club_ids: set[int],
) -> pd.DataFrame:
    """
    Convert each transfer into one event per mapped club side.

    Example:

        Arsenal -> Barcelona

    creates:

        Arsenal: outgoing
        Barcelona: incoming

    This makes per-club chronological slicing straightforward.
    """

    rows = []

    for row in transfers.itertuples(
        index=False
    ):

        from_club_id = (
            int(
                row.from_club_id
            )
            if not pd.isna(
                row.from_club_id
            )
            else None
        )

        to_club_id = (
            int(
                row.to_club_id
            )
            if not pd.isna(
                row.to_club_id
            )
            else None
        )

        common = {
            "player_id": int(
                row.player_id
            ),

            "transfer_date": (
                row.transfer_date
            ),

            "market_value_in_eur": (
                row.market_value_in_eur
            ),

            "transfer_fee": (
                row.transfer_fee
            ),

            "movement_type": (
                row.movement_type
            ),

            "from_club_id": (
                from_club_id
            ),

            "to_club_id": (
                to_club_id
            ),
        }

        if (
            from_club_id
            in club_ids
        ):

            event = common.copy()

            event[
                "club_id"
            ] = from_club_id

            event[
                "direction"
            ] = "outgoing"

            rows.append(
                event
            )

        if (
            to_club_id
            in club_ids
        ):

            event = common.copy()

            event[
                "club_id"
            ] = to_club_id

            event[
                "direction"
            ] = "incoming"

            rows.append(
                event
            )

    events = pd.DataFrame(
        rows
    )

    if events.empty:

        raise RuntimeError(
            "No mapped transfer events were created."
        )

    events = events.sort_values(
        by=[
            "club_id",
            "transfer_date",
            "player_id",
            "direction",
        ]
    ).reset_index(
        drop=True
    )

    return events


# ============================================================
# BUILD CLUB EVENT HISTORIES
# ============================================================


def build_club_histories(
    *,
    events: pd.DataFrame,
    club_ids: set[int],
) -> dict[int, dict]:
    """
    Build sorted transfer history for each mapped club.
    """

    grouped = {
        int(
            club_id
        ): group.reset_index(
            drop=True
        )
        for (
            club_id,
            group,
        ) in events.groupby(
            "club_id",
            sort=False,
        )
    }

    histories = {}

    for club_id in club_ids:

        frame = grouped.get(
            club_id
        )

        if frame is None:

            frame = pd.DataFrame(
                columns=events.columns
            )

        histories[
            club_id
        ] = {
            "frame": frame,

            "dates": (
                frame[
                    "transfer_date"
                ].to_numpy(
                    dtype="datetime64[ns]"
                )
                if not frame.empty
                else np.array(
                    [],
                    dtype="datetime64[ns]",
                )
            ),
        }

    return histories


# ============================================================
# HISTORY SLICING
# ============================================================


def full_prior_slice(
    history: dict,
    match_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    All transfer events strictly before match_date.
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

    match_np = timestamp_to_numpy(
        match_date
    )

    right = int(
        np.searchsorted(
            dates,
            match_np,
            side="left",
        )
    )

    return history[
        "frame"
    ].iloc[
        :right
    ]


def window_slice(
    history: dict,
    *,
    match_date: pd.Timestamp,
    days: int,
) -> pd.DataFrame:
    """
    Return:

        match_date - days <= transfer_date < match_date
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

    start_date = (
        match_date
        -
        pd.Timedelta(
            days=days
        )
    )

    start_np = timestamp_to_numpy(
        start_date
    )

    match_np = timestamp_to_numpy(
        match_date
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
            match_np,
            side="left",
        )
    )

    return history[
        "frame"
    ].iloc[
        left:right
    ]


# ============================================================
# EVENT HELPERS
# ============================================================


def direction_frame(
    dataframe: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """
    Select incoming or outgoing events.
    """

    if dataframe.empty:

        return dataframe

    return dataframe[
        dataframe[
            "direction"
        ]
        ==
        direction
    ]


def value_summary(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Market-value coverage and sum for a transfer subset.
    """

    total = len(
        dataframe
    )

    if total == 0:

        return {
            "count": 0,
            "known_count": 0,
            "coverage": np.nan,
            "sum_eur": 0.0,
        }

    known = dataframe[
        "market_value_in_eur"
    ].notna()

    known_count = int(
        known.sum()
    )

    return {
        "count": total,

        "known_count": known_count,

        "coverage": (
            known_count
            / total
        ),

        "sum_eur": float(
            dataframe.loc[
                known,
                "market_value_in_eur",
            ].sum()
        ),
    }


def fee_summary(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Transfer-fee coverage and sum.

    Zero is kept as a known numeric value but is not
    semantically re-labelled as a free transfer.
    """

    total = len(
        dataframe
    )

    if total == 0:

        return {
            "count": 0,
            "known_count": 0,
            "coverage": np.nan,
            "sum_eur": 0.0,
        }

    known = dataframe[
        "transfer_fee"
    ].notna()

    known_count = int(
        known.sum()
    )

    return {
        "count": total,

        "known_count": known_count,

        "coverage": (
            known_count
            / total
        ),

        "sum_eur": float(
            dataframe.loc[
                known,
                "transfer_fee",
            ].sum()
        ),
    }


def days_since_latest(
    dataframe: pd.DataFrame,
    match_date: pd.Timestamp,
) -> float:
    """
    Days since most recent event in the supplied dataframe.
    """

    if dataframe.empty:

        return np.nan

    latest = dataframe[
        "transfer_date"
    ].max()

    days = int(
        (
            match_date
            - latest
        ).days
    )

    if days <= 0:

        raise RuntimeError(
            "Transfer-date leakage detected."
        )

    return float(
        days
    )


# ============================================================
# ARRIVAL INTEGRATION
# ============================================================


def arrival_age_features(
    *,
    incoming: pd.DataFrame,
    match_date: pd.Timestamp,
) -> tuple[
    float,
    float,
]:
    """
    Estimate integration age of recent arrivals.

    Returns:
        simple average days at club
        market-value weighted average days at club
    """

    if incoming.empty:

        return (
            np.nan,
            np.nan,
        )

    ages = (
        match_date
        -
        incoming[
            "transfer_date"
        ]
    ).dt.days.astype(
        float
    )

    if (
        ages
        <=
        0
    ).any():

        raise RuntimeError(
            "Arrival integration leakage detected."
        )

    simple_average = float(
        ages.mean()
    )

    valued = incoming[
        "market_value_in_eur"
    ].notna()

    if not valued.any():

        weighted_average = np.nan

    else:

        weights = (
            incoming.loc[
                valued,
                "market_value_in_eur",
            ]
            .astype(float)
            .to_numpy()
        )

        valued_ages = (
            ages.loc[
                valued
            ]
            .astype(float)
            .to_numpy()
        )

        positive_weight = float(
            weights.sum()
        )

        if positive_weight <= 0:

            weighted_average = float(
                valued_ages.mean()
            )

        else:

            weighted_average = float(
                np.average(
                    valued_ages,
                    weights=weights,
                )
            )

    return (
        simple_average,
        weighted_average,
    )


# ============================================================
# LARGEST PLAYER MOVEMENT
# ============================================================


def largest_value_event(
    *,
    dataframe: pd.DataFrame,
    match_date: pd.Timestamp,
) -> tuple[
    float,
    float,
]:
    """
    Find highest-value player movement in a window.

    Returns:
        value in EUR
        days since event
    """

    if dataframe.empty:

        return (
            np.nan,
            np.nan,
        )

    valued = dataframe[
        dataframe[
            "market_value_in_eur"
        ].notna()
    ]

    if valued.empty:

        return (
            np.nan,
            np.nan,
        )

    index = valued[
        "market_value_in_eur"
    ].idxmax()

    row = valued.loc[
        index
    ]

    value = float(
        row[
            "market_value_in_eur"
        ]
    )

    days = int(
        (
            match_date
            -
            row[
                "transfer_date"
            ]
        ).days
    )

    if days <= 0:

        raise RuntimeError(
            "Largest transfer event leakage detected."
        )

    return (
        value,
        float(
            days
        ),
    )


# ============================================================
# PRESSURE SCORES
# ============================================================


def pressure_score(
    *,
    count_30: int,
    count_90: int,
    count_180: int,
) -> float:
    """
    Simple bounded recent-transfer pressure metric.

    More weight is given to very recent activity.

    This is not a prediction probability.
    """

    raw = (
        (
            count_30
            * 1.0
        )
        +
        (
            max(
                count_90
                -
                count_30,
                0,
            )
            * 0.5
        )
        +
        (
            max(
                count_180
                -
                count_90,
                0,
            )
            * 0.2
        )
    )

    return float(
        1.0
        -
        np.exp(
            -raw
            / 5.0
        )
    )


# ============================================================
# TEAM SNAPSHOT
# ============================================================


def calculate_team_snapshot(
    *,
    club_id: int,
    match_date: pd.Timestamp,
    history: dict,
    active_player_count,
    active_squad_value_m_eur,
) -> dict:
    """
    Calculate all transfer features for one club strictly
    before a UCL match.
    """

    prior = full_prior_slice(
        history,
        match_date,
    )

    events_30 = window_slice(
        history,
        match_date=match_date,
        days=WINDOW_30D,
    )

    events_90 = window_slice(
        history,
        match_date=match_date,
        days=WINDOW_90D,
    )

    events_180 = window_slice(
        history,
        match_date=match_date,
        days=WINDOW_180D,
    )

    events_365 = window_slice(
        history,
        match_date=match_date,
        days=WINDOW_365D,
    )

    history_available = int(
        not prior.empty
    )

    # --------------------------------------------------------
    # DIRECTIONS
    # --------------------------------------------------------

    in_30 = direction_frame(
        events_30,
        "incoming",
    )

    in_90 = direction_frame(
        events_90,
        "incoming",
    )

    in_180 = direction_frame(
        events_180,
        "incoming",
    )

    in_365 = direction_frame(
        events_365,
        "incoming",
    )

    out_30 = direction_frame(
        events_30,
        "outgoing",
    )

    out_90 = direction_frame(
        events_90,
        "outgoing",
    )

    out_180 = direction_frame(
        events_180,
        "outgoing",
    )

    out_365 = direction_frame(
        events_365,
        "outgoing",
    )

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    incoming_count_30 = len(
        in_30
    )

    incoming_count_90 = len(
        in_90
    )

    incoming_count_180 = len(
        in_180
    )

    incoming_count_365 = len(
        in_365
    )

    outgoing_count_30 = len(
        out_30
    )

    outgoing_count_90 = len(
        out_90
    )

    outgoing_count_180 = len(
        out_180
    )

    outgoing_count_365 = len(
        out_365
    )

    turnover_30 = (
        incoming_count_30
        +
        outgoing_count_30
    )

    turnover_90 = (
        incoming_count_90
        +
        outgoing_count_90
    )

    turnover_180 = (
        incoming_count_180
        +
        outgoing_count_180
    )

    turnover_365 = (
        incoming_count_365
        +
        outgoing_count_365
    )

    # --------------------------------------------------------
    # VALUE SUMMARIES
    # --------------------------------------------------------

    in_value_90 = value_summary(
        in_90
    )

    in_value_180 = value_summary(
        in_180
    )

    in_value_365 = value_summary(
        in_365
    )

    out_value_90 = value_summary(
        out_90
    )

    out_value_180 = value_summary(
        out_180
    )

    out_value_365 = value_summary(
        out_365
    )

    # --------------------------------------------------------
    # FEE SUMMARIES
    # --------------------------------------------------------

    in_fee_180 = fee_summary(
        in_180
    )

    out_fee_180 = fee_summary(
        out_180
    )

    fee_event_count = (
        len(
            in_180
        )
        +
        len(
            out_180
        )
    )

    fee_known_count = (
        in_fee_180[
            "known_count"
        ]
        +
        out_fee_180[
            "known_count"
        ]
    )

    fee_coverage_180 = (
        fee_known_count
        / fee_event_count
        if fee_event_count > 0
        else np.nan
    )

    # --------------------------------------------------------
    # COMBINED VALUE COVERAGE
    # --------------------------------------------------------

    value_event_count_180 = (
        len(
            in_180
        )
        +
        len(
            out_180
        )
    )

    value_known_count_180 = (
        in_value_180[
            "known_count"
        ]
        +
        out_value_180[
            "known_count"
        ]
    )

    value_coverage_180 = (
        value_known_count_180
        / value_event_count_180
        if value_event_count_180 > 0
        else np.nan
    )

    # --------------------------------------------------------
    # LAST EVENT RECENCY
    # --------------------------------------------------------

    days_since_event = days_since_latest(
        prior,
        match_date,
    )

    prior_incoming = direction_frame(
        prior,
        "incoming",
    )

    prior_outgoing = direction_frame(
        prior,
        "outgoing",
    )

    days_since_incoming = days_since_latest(
        prior_incoming,
        match_date,
    )

    days_since_outgoing = days_since_latest(
        prior_outgoing,
        match_date,
    )

    # --------------------------------------------------------
    # ARRIVAL INTEGRATION
    # --------------------------------------------------------

    (
        arrival_average_age,
        arrival_weighted_age,
    ) = arrival_age_features(
        incoming=in_180,
        match_date=match_date,
    )

    # --------------------------------------------------------
    # LARGEST MOVEMENTS
    # --------------------------------------------------------

    (
        largest_arrival_eur,
        days_since_largest_arrival,
    ) = largest_value_event(
        dataframe=in_365,
        match_date=match_date,
    )

    (
        largest_departure_eur,
        days_since_largest_departure,
    ) = largest_value_event(
        dataframe=out_365,
        match_date=match_date,
    )

    # --------------------------------------------------------
    # NORMALIZED TURNOVER
    # --------------------------------------------------------

    turnover_per_active_player = safe_ratio(
        turnover_180,
        active_player_count,
    )

    value_turnover_180_eur = (
        in_value_180[
            "sum_eur"
        ]
        +
        out_value_180[
            "sum_eur"
        ]
    )

    if (
        pd.isna(
            active_squad_value_m_eur
        )
        or
        active_squad_value_m_eur
        <=
        0
    ):

        value_turnover_to_squad = np.nan

    else:

        value_turnover_to_squad = (
            (
                value_turnover_180_eur
                /
                1_000_000
            )
            /
            active_squad_value_m_eur
        )

    # --------------------------------------------------------
    # PRESSURE
    # --------------------------------------------------------

    arrival_pressure = pressure_score(
        count_30=incoming_count_30,
        count_90=incoming_count_90,
        count_180=incoming_count_180,
    )

    departure_pressure = pressure_score(
        count_30=outgoing_count_30,
        count_90=outgoing_count_90,
        count_180=outgoing_count_180,
    )

    turnover_pressure = pressure_score(
        count_30=turnover_30,
        count_90=turnover_90,
        count_180=turnover_180,
    )

    # --------------------------------------------------------
    # DATA COMPLETENESS
    # --------------------------------------------------------

    completeness_parts = [
        float(
            history_available
        ),
    ]

    if not pd.isna(
        value_coverage_180
    ):

        completeness_parts.append(
            float(
                value_coverage_180
            )
        )

    if not pd.isna(
        fee_coverage_180
    ):

        completeness_parts.append(
            float(
                fee_coverage_180
            )
        )

    data_completeness = float(
        np.mean(
            completeness_parts
        )
    )

    # --------------------------------------------------------
    # AUDIT DATE
    # --------------------------------------------------------

    latest_transfer_date_used = (
        prior[
            "transfer_date"
        ].max()
        if not prior.empty
        else pd.NaT
    )

    if (
        not pd.isna(
            latest_transfer_date_used
        )
        and
        latest_transfer_date_used
        >=
        match_date
    ):

        raise RuntimeError(
            "Transfer chronology leakage detected."
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    return {
        "transfer_history_available": (
            history_available
        ),

        "transfer_days_since_last_event": (
            days_since_event
        ),

        "transfer_days_since_last_incoming": (
            days_since_incoming
        ),

        "transfer_days_since_last_outgoing": (
            days_since_outgoing
        ),

        "transfer_incoming_count_30d": (
            incoming_count_30
        ),

        "transfer_incoming_count_90d": (
            incoming_count_90
        ),

        "transfer_incoming_count_180d": (
            incoming_count_180
        ),

        "transfer_incoming_count_365d": (
            incoming_count_365
        ),

        "transfer_outgoing_count_30d": (
            outgoing_count_30
        ),

        "transfer_outgoing_count_90d": (
            outgoing_count_90
        ),

        "transfer_outgoing_count_180d": (
            outgoing_count_180
        ),

        "transfer_outgoing_count_365d": (
            outgoing_count_365
        ),

        "transfer_net_player_count_30d": (
            incoming_count_30
            -
            outgoing_count_30
        ),

        "transfer_net_player_count_90d": (
            incoming_count_90
            -
            outgoing_count_90
        ),

        "transfer_net_player_count_180d": (
            incoming_count_180
            -
            outgoing_count_180
        ),

        "transfer_net_player_count_365d": (
            incoming_count_365
            -
            outgoing_count_365
        ),

        "transfer_turnover_count_30d": (
            turnover_30
        ),

        "transfer_turnover_count_90d": (
            turnover_90
        ),

        "transfer_turnover_count_180d": (
            turnover_180
        ),

        "transfer_turnover_count_365d": (
            turnover_365
        ),

        "transfer_turnover_per_active_player_180d": (
            turnover_per_active_player
        ),

        "transfer_incoming_value_m_eur_90d": (
            in_value_90[
                "sum_eur"
            ]
            /
            1_000_000
        ),

        "transfer_incoming_value_m_eur_180d": (
            in_value_180[
                "sum_eur"
            ]
            /
            1_000_000
        ),

        "transfer_incoming_value_m_eur_365d": (
            in_value_365[
                "sum_eur"
            ]
            /
            1_000_000
        ),

        "transfer_outgoing_value_m_eur_90d": (
            out_value_90[
                "sum_eur"
            ]
            /
            1_000_000
        ),

        "transfer_outgoing_value_m_eur_180d": (
            out_value_180[
                "sum_eur"
            ]
            /
            1_000_000
        ),

        "transfer_outgoing_value_m_eur_365d": (
            out_value_365[
                "sum_eur"
            ]
            /
            1_000_000
        ),

        "transfer_net_value_m_eur_90d": (
            (
                in_value_90[
                    "sum_eur"
                ]
                -
                out_value_90[
                    "sum_eur"
                ]
            )
            /
            1_000_000
        ),

        "transfer_net_value_m_eur_180d": (
            (
                in_value_180[
                    "sum_eur"
                ]
                -
                out_value_180[
                    "sum_eur"
                ]
            )
            /
            1_000_000
        ),

        "transfer_net_value_m_eur_365d": (
            (
                in_value_365[
                    "sum_eur"
                ]
                -
                out_value_365[
                    "sum_eur"
                ]
            )
            /
            1_000_000
        ),

        "transfer_value_turnover_m_eur_180d": (
            value_turnover_180_eur
            /
            1_000_000
        ),

        "transfer_value_turnover_m_eur_365d": (
            (
                in_value_365[
                    "sum_eur"
                ]
                +
                out_value_365[
                    "sum_eur"
                ]
            )
            /
            1_000_000
        ),

        "transfer_value_turnover_to_squad_value_180d": (
            value_turnover_to_squad
        ),

        "transfer_incoming_value_coverage_180d": (
            in_value_180[
                "coverage"
            ]
        ),

        "transfer_outgoing_value_coverage_180d": (
            out_value_180[
                "coverage"
            ]
        ),

        "transfer_value_coverage_180d": (
            value_coverage_180
        ),

        "transfer_incoming_fee_m_eur_180d": (
            in_fee_180[
                "sum_eur"
            ]
            /
            1_000_000
        ),

        "transfer_outgoing_fee_m_eur_180d": (
            out_fee_180[
                "sum_eur"
            ]
            /
            1_000_000
        ),

        "transfer_net_spend_m_eur_180d": (
            (
                in_fee_180[
                    "sum_eur"
                ]
                -
                out_fee_180[
                    "sum_eur"
                ]
            )
            /
            1_000_000
        ),

        "transfer_fee_coverage_180d": (
            fee_coverage_180
        ),

        "transfer_arrival_avg_age_days_180d": (
            arrival_average_age
        ),

        "transfer_arrival_value_weighted_age_days_180d": (
            arrival_weighted_age
        ),

        "transfer_largest_arrival_value_m_eur_365d": (
            largest_arrival_eur
            /
            1_000_000
            if not pd.isna(
                largest_arrival_eur
            )
            else np.nan
        ),

        "transfer_days_since_largest_arrival_365d": (
            days_since_largest_arrival
        ),

        "transfer_largest_departure_value_m_eur_365d": (
            largest_departure_eur
            /
            1_000_000
            if not pd.isna(
                largest_departure_eur
            )
            else np.nan
        ),

        "transfer_days_since_largest_departure_365d": (
            days_since_largest_departure
        ),

        "transfer_recent_arrival_pressure": (
            arrival_pressure
        ),

        "transfer_recent_departure_pressure": (
            departure_pressure
        ),

        "transfer_recent_turnover_pressure": (
            turnover_pressure
        ),

        "transfer_data_completeness": (
            data_completeness
        ),

        # -----------------------------------------------
        # AUDIT ONLY
        # -----------------------------------------------

        "transfer_latest_date_used": (
            latest_transfer_date_used
        ),
    }


# ============================================================
# ATTACH FEATURES
# ============================================================


def attach_transfer_features(
    *,
    matches: pd.DataFrame,
    team_to_id: dict[str, int],
    histories: dict[int, dict],
) -> pd.DataFrame:
    """
    Attach home/away transfer features to every UCL match.
    """

    rows = []

    cache = {}

    total = len(
        matches
    )

    print(
        "\nAttaching strictly pre-match "
        "transfer-impact features..."
    )

    for index, row in matches.iterrows():

        match_date = row[
            "date"
        ]

        home_team = row[
            "home_team"
        ]

        away_team = row[
            "away_team"
        ]

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

        home_key = (
            home_club_id,
            match_date,
            row[
                "home_squad_players_seen_last180d"
            ],
            row[
                "home_squad_active_value_m_eur"
            ],
        )

        away_key = (
            away_club_id,
            match_date,
            row[
                "away_squad_players_seen_last180d"
            ],
            row[
                "away_squad_active_value_m_eur"
            ],
        )

        if home_key not in cache:

            cache[
                home_key
            ] = calculate_team_snapshot(
                club_id=home_club_id,
                match_date=match_date,
                history=histories[
                    home_club_id
                ],
                active_player_count=row[
                    "home_squad_players_seen_last180d"
                ],
                active_squad_value_m_eur=row[
                    "home_squad_active_value_m_eur"
                ],
            )

        if away_key not in cache:

            cache[
                away_key
            ] = calculate_team_snapshot(
                club_id=away_club_id,
                match_date=match_date,
                history=histories[
                    away_club_id
                ],
                active_player_count=row[
                    "away_squad_players_seen_last180d"
                ],
                active_squad_value_m_eur=row[
                    "away_squad_active_value_m_eur"
                ],
            )

        home_snapshot = cache[
            home_key
        ]

        away_snapshot = cache[
            away_key
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
                index
                +
                1
            )
            %
            250
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
            matches.reset_index(
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
    Validate chronology, ranges and feature integrity.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFER IMPACT DATA QUALITY CHECK"
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
            f"{EXPECTED_UCL_MATCHES:,} rows, "
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
            "Output not chronologically sorted."
        )

    print(
        "Chronological ordering: valid ✓"
    )

    # --------------------------------------------------------
    # LEAKAGE
    # --------------------------------------------------------

    for side in [
        "home",
        "away",
    ]:

        column = (
            f"{side}_"
            f"transfer_latest_date_used"
        )

        latest_dates = pd.to_datetime(
            dataframe[
                column
            ],
            errors="coerce",
        )

        leakage = (
            latest_dates.notna()
            &
            (
                latest_dates
                >=
                dataframe[
                    "date"
                ]
            )
        )

        if leakage.any():

            bad = dataframe[
                leakage
            ][
                [
                    "date",
                    "home_team",
                    "away_team",
                    column,
                ]
            ].head(
                20
            )

            print(
                bad.to_string(
                    index=False
                )
            )

            raise RuntimeError(
                f"{side} transfer-date leakage "
                f"detected."
            )

    print(
        "Strict pre-match transfer-date "
        "leakage check: passed ✓"
    )

    # --------------------------------------------------------
    # FLAGS
    # --------------------------------------------------------

    for side in [
        "home",
        "away",
    ]:

        column = (
            f"{side}_"
            f"transfer_history_available"
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
                f"Invalid history flags in "
                f"{column}."
            )

    print(
        "Transfer-history flags: valid ✓"
    )

    # --------------------------------------------------------
    # COVERAGE / PRESSURE RANGES
    # --------------------------------------------------------

    bounded_features = [
        "transfer_incoming_value_coverage_180d",
        "transfer_outgoing_value_coverage_180d",
        "transfer_value_coverage_180d",
        "transfer_fee_coverage_180d",
        "transfer_recent_arrival_pressure",
        "transfer_recent_departure_pressure",
        "transfer_recent_turnover_pressure",
        "transfer_data_completeness",
    ]

    for side in [
        "home",
        "away",
    ]:

        for feature in bounded_features:

            column = (
                f"{side}_{feature}"
            )

            values = dataframe[
                column
            ].dropna()

            invalid = (
                (
                    values
                    <
                    -1e-9
                )
                |
                (
                    values
                    >
                    1
                    +
                    1e-9
                )
            )

            if invalid.any():

                raise RuntimeError(
                    f"{column} outside [0,1]."
                )

    print(
        "Coverage/pressure ranges: valid ✓"
    )

    # --------------------------------------------------------
    # NONNEGATIVE COUNTS
    # --------------------------------------------------------

    for column in dataframe.columns:

        if (
            not column.startswith(
                (
                    "home_transfer_",
                    "away_transfer_",
                )
            )
        ):

            continue

        if (
            "_count_" in column
            and
            "_net_" not in column
        ):

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
                    f"Negative transfer count "
                    f"in {column}."
                )

    print(
        "Transfer counts: valid ✓"
    )

    # --------------------------------------------------------
    # RECENCY
    # --------------------------------------------------------

    recency_markers = [
        "days_since_last_event",
        "days_since_last_incoming",
        "days_since_last_outgoing",
        "days_since_largest_arrival_365d",
        "days_since_largest_departure_365d",
        "arrival_avg_age_days_180d",
        "arrival_value_weighted_age_days_180d",
    ]

    for side in [
        "home",
        "away",
    ]:

        for marker in recency_markers:

            column = (
                f"{side}_"
                f"transfer_{marker}"
            )

            values = dataframe[
                column
            ].dropna()

            if (
                values
                <=
                0
            ).any():

                raise RuntimeError(
                    f"Non-positive transfer recency "
                    f"in {column}."
                )

    print(
        "Transfer recency values: valid ✓"
    )

    # --------------------------------------------------------
    # COVERAGE SUMMARY
    # --------------------------------------------------------

    both_history = (
        (
            dataframe[
                "home_transfer_history_available"
            ]
            ==
            1
        )
        &
        (
            dataframe[
                "away_transfer_history_available"
            ]
            ==
            1
        )
    )

    one_history = (
        (
            dataframe[
                "home_transfer_history_available"
            ]
            +
            dataframe[
                "away_transfer_history_available"
            ]
        )
        ==
        1
    )

    neither_history = (
        (
            dataframe[
                "home_transfer_history_available"
            ]
            ==
            0
        )
        &
        (
            dataframe[
                "away_transfer_history_available"
            ]
            ==
            0
        )
    )

    print(
        "\nTransfer-history coverage:"
    )

    print(
        f"Both teams available: "
        f"{int(both_history.sum()):,}"
    )

    print(
        f"One team available:  "
        f"{int(one_history.sum()):,}"
    )

    print(
        f"Neither available:   "
        f"{int(neither_history.sum()):,}"
    )

    print(
        f"Both-team coverage:  "
        f"{both_history.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # RECENT MOVEMENT SUMMARY
    # --------------------------------------------------------

    both_recent_turnover = (
        (
            dataframe[
                "home_transfer_turnover_count_180d"
            ]
            >
            0
        )
        &
        (
            dataframe[
                "away_transfer_turnover_count_180d"
            ]
            >
            0
        )
    )

    print(
        "\nMatches where both teams had >=1 "
        "transfer event in prior 180 days:"
    )

    print(
        f"{int(both_recent_turnover.sum()):,} "
        f"/ {len(dataframe):,} "
        f"({both_recent_turnover.mean() * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    transfer_columns = [
        column
        for column in dataframe.columns
        if (
            column.startswith(
                "home_transfer_"
            )
            or
            column.startswith(
                "away_transfer_"
            )
            or (
                column.startswith(
                    "transfer_"
                )
                and
                column.endswith(
                    "_diff"
                )
            )
        )
    ]

    total_missing = int(
        dataframe[
            transfer_columns
        ].isna().sum().sum()
    )

    print(
        f"\nExpected transfer coverage NaNs: "
        f"{total_missing:,}"
    )

    print(
        "Missing transfer metadata preserved "
        "intentionally ✓"
    )


# ============================================================
# SAVE FEATURE HISTORY
# ============================================================


def save_output(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save match-level transfer features.
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
# BUILD LATEST TRAINING SNAPSHOT
# ============================================================


def build_latest_table(
    *,
    matches: pd.DataFrame,
    mapping: pd.DataFrame,
    histories: dict[int, dict],
) -> pd.DataFrame:
    """
    Build a training-era reference snapshot for each club.

    This is not the final live 2026/27 squad state.
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

    latest_match_rows = {}

    for team in mapping[
        "ucl_team"
    ]:

        relevant = matches[
            (
                matches[
                    "home_team"
                ]
                ==
                team
            )
            |
            (
                matches[
                    "away_team"
                ]
                ==
                team
            )
        ]

        if relevant.empty:

            continue

        latest_match_rows[
            team
        ] = relevant.iloc[
            -1
        ]

    rows = []

    for row in mapping.itertuples(
        index=False
    ):

        team = row.ucl_team

        club_id = int(
            row.club_id
        )

        latest_match = latest_match_rows.get(
            team
        )

        if latest_match is None:

            active_players = np.nan
            active_value = np.nan

        elif (
            latest_match[
                "home_team"
            ]
            ==
            team
        ):

            active_players = latest_match[
                "home_squad_players_seen_last180d"
            ]

            active_value = latest_match[
                "home_squad_active_value_m_eur"
            ]

        else:

            active_players = latest_match[
                "away_squad_players_seen_last180d"
            ]

            active_value = latest_match[
                "away_squad_active_value_m_eur"
            ]

        snapshot = calculate_team_snapshot(
            club_id=club_id,
            match_date=reference_date,
            history=histories[
                club_id
            ],
            active_player_count=active_players,
            active_squad_value_m_eur=active_value,
        )

        output_row = {
            "team": team,

            "club_id": club_id,

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

    dataframe = pd.DataFrame(
        rows
    )

    dataframe = dataframe.sort_values(
        by=[
            "transfer_net_value_m_eur_180d",
            "transfer_incoming_value_m_eur_180d",
        ],
        ascending=False,
        na_position="last",
    ).reset_index(
        drop=True
    )

    output = dataframe.copy()

    for column in [
        "reference_date",
        "transfer_latest_date_used",
    ]:

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

    return dataframe


# ============================================================
# PRINT SUMMARY
# ============================================================


def print_summary(
    *,
    featured: pd.DataFrame,
    original_columns: list[str],
    latest: pd.DataFrame,
) -> None:
    """
    Print useful transfer feature examples.
    """

    new_columns = [
        column
        for column in featured.columns
        if column not in original_columns
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFER IMPACT FEATURE SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nMatches processed: "
        f"{len(featured):,}"
    )

    print(
        f"New transfer/context features created: "
        f"{len(new_columns):,}"
    )

    latest_columns = [
        "team",

        "transfer_incoming_count_180d",
        "transfer_outgoing_count_180d",

        "transfer_incoming_value_m_eur_180d",
        "transfer_outgoing_value_m_eur_180d",
        "transfer_net_value_m_eur_180d",

        "transfer_turnover_per_active_player_180d",

        "transfer_arrival_avg_age_days_180d",

        "transfer_recent_turnover_pressure",

        "transfer_data_completeness",
    ]

    print(
        "\nTop 15 clubs by training-era "
        "180-day net incoming player value:"
    )

    print(
        latest[
            latest_columns
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
    )

    match_columns = [
        "date",
        "home_team",
        "away_team",
        "result",

        "home_transfer_incoming_count_180d",
        "away_transfer_incoming_count_180d",

        "home_transfer_outgoing_count_180d",
        "away_transfer_outgoing_count_180d",

        "home_transfer_net_value_m_eur_180d",
        "away_transfer_net_value_m_eur_180d",

        "home_transfer_arrival_avg_age_days_180d",
        "away_transfer_arrival_avg_age_days_180d",

        "home_transfer_recent_turnover_pressure",
        "away_transfer_recent_turnover_pressure",

        "transfer_net_value_m_eur_180d_diff",
    ]

    print(
        "\nFirst 5 UCL matches:"
    )

    print(
        featured[
            match_columns
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
        featured[
            match_columns
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
        "\nLatest training-era transfer table saved to:"
    )

    print(
        LATEST_FILE
    )


# ============================================================
# MAIN
# ============================================================


def build_transfer_impact_features() -> None:
    """
    Build strictly pre-match transfer-impact features.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - TRANSFER IMPACT FEATURE ENGINEERING"
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
    ) = load_club_map()

    transfers = load_transfer_history()

    # --------------------------------------------------------
    # CLUB EVENTS
    # --------------------------------------------------------

    print(
        "\nBuilding per-club transfer event history..."
    )

    club_ids = set(
        mapping[
            "club_id"
        ].astype(
            int
        )
    )

    events = build_club_events(
        transfers=transfers,
        club_ids=club_ids,
    )

    print(
        f"Club transfer-event rows: "
        f"{len(events):,}"
    )

    print(
        "\nEvent direction distribution:"
    )

    print(
        events[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    histories = build_club_histories(
        events=events,
        club_ids=club_ids,
    )

    print(
        f"Club transfer histories built: "
        f"{len(histories):,}"
    )

    # --------------------------------------------------------
    # ATTACH
    # --------------------------------------------------------

    featured = attach_transfer_features(
        matches=matches,
        team_to_id=team_to_id,
        histories=histories,
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    run_quality_checks(
        featured
    )

    # --------------------------------------------------------
    # LATEST
    # --------------------------------------------------------

    latest = build_latest_table(
        matches=matches,
        mapping=mapping,
        histories=histories,
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
        featured=featured,
        original_columns=original_columns,
        latest=latest,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFER IMPACT FEATURES CREATED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nLeakage barrier enforced:"
    )

    print(
        "  transfer_date < UCL_match_date"
    )

    print(
        "\nScheduled future transfers remain in the "
        "source table but are invisible to earlier matches."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    build_transfer_impact_features()