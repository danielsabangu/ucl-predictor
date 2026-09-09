from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)


# ============================================================
# PATHS
# ============================================================

TRANSFERMARKT_DIR = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
)

RAW_TRANSFERS_FILE = (
    TRANSFERMARKT_DIR
    / "transfers.csv"
)

CLUB_MAP_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)

UCL_MASTER_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

OUTPUT_DIR = (
    INTERIM_DATA_DIR
    / "players"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "transfermarkt_transfers_ucl_clubs.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "transfermarkt_transfer_history_metadata.json"
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_UCL_CLUBS = 117


# ============================================================
# SOURCE COLUMNS
# ============================================================

REQUIRED_TRANSFER_COLUMNS = {
    "player_id",
    "transfer_date",
    "from_club_id",
    "to_club_id",
}

PREFERRED_TRANSFER_COLUMNS = [
    "player_id",
    "transfer_date",
    "transfer_season",
    "from_club_id",
    "to_club_id",
    "from_club_name",
    "to_club_name",
    "transfer_fee",
    "market_value_in_eur",
    "player_name",
]


# ============================================================
# FILE CHECK
# ============================================================


def require_file(
    path: Path,
    description: str,
) -> None:
    """
    Fail early when a required file is missing.
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
# LOAD CLUB MAP
# ============================================================


def load_club_map() -> tuple[
    pd.DataFrame,
    set[int],
    dict[int, str],
]:
    """
    Load the final canonical 117-team Transfermarkt map.
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
            f"mapped UCL clubs, found "
            f"{len(mapping)}."
        )

    if mapping[
        "club_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Transfermarkt IDs "
            "detected in final club map."
        )

    club_ids = set(
        mapping[
            "club_id"
        ]
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
        club_ids,
        id_to_team,
    )


# ============================================================
# LOAD TRAINING DATE RANGE
# ============================================================


def load_ucl_training_dates() -> tuple[
    pd.Timestamp,
    pd.Timestamp,
]:
    """
    Load historical UCL training range.

    This is used only for source auditing.
    """

    require_file(
        UCL_MASTER_FILE,
        "UCL master dataset",
    )

    dataframe = pd.read_csv(
        UCL_MASTER_FILE,
        usecols=[
            "date",
        ],
        low_memory=False,
    )

    dates = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    return (
        dates.min(),
        dates.max(),
    )


# ============================================================
# LOAD RAW TRANSFERS
# ============================================================


def load_raw_transfers() -> pd.DataFrame:
    """
    Load Transfermarkt transfer history.
    """

    require_file(
        RAW_TRANSFERS_FILE,
        "Transfermarkt transfers.csv",
    )

    header = pd.read_csv(
        RAW_TRANSFERS_FILE,
        nrows=0,
        low_memory=False,
    )

    missing = (
        REQUIRED_TRANSFER_COLUMNS
        - set(
            header.columns
        )
    )

    if missing:

        raise RuntimeError(
            "transfers.csv missing required columns: "
            f"{sorted(missing)}"
        )

    usecols = [
        column
        for column in PREFERRED_TRANSFER_COLUMNS
        if column in header.columns
    ]

    dataframe = pd.read_csv(
        RAW_TRANSFERS_FILE,
        usecols=usecols,
        low_memory=False,
    )

    print(
        f"\nRaw transfer rows loaded: "
        f"{len(dataframe):,}"
    )

    return dataframe


# ============================================================
# NORMALIZE TRANSFERS
# ============================================================


def normalize_transfers(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize IDs, dates and numeric fields.

    Missing club IDs are allowed because transfers can involve
    clubs/entities outside the mapped dataset.
    """

    df = dataframe.copy()

    df[
        "player_id"
    ] = pd.to_numeric(
        df[
            "player_id"
        ],
        errors="coerce",
    )

    df[
        "transfer_date"
    ] = pd.to_datetime(
        df[
            "transfer_date"
        ],
        errors="coerce",
    )

    for column in [
        "from_club_id",
        "to_club_id",
    ]:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        ).astype(
            "Int64"
        )

    for column in [
        "transfer_fee",
        "market_value_in_eur",
    ]:

        if column in df.columns:

            df[
                column
            ] = pd.to_numeric(
                df[
                    column
                ],
                errors="coerce",
            )

    return df


# ============================================================
# FILTER TO OUR CLUB UNIVERSE
# ============================================================


def filter_relevant_transfers(
    *,
    dataframe: pd.DataFrame,
    club_ids: set[int],
    id_to_team: dict[int, str],
) -> pd.DataFrame:
    """
    Keep every transfer where either the source club or
    destination club belongs to our 117-team UCL universe.
    """

    from_mapped = dataframe[
        "from_club_id"
    ].isin(
        club_ids
    )

    to_mapped = dataframe[
        "to_club_id"
    ].isin(
        club_ids
    )

    relevant = dataframe[
        from_mapped
        |
        to_mapped
    ].copy()

    relevant[
        "from_ucl_club"
    ] = relevant[
        "from_club_id"
    ].isin(
        club_ids
    ).astype(
        int
    )

    relevant[
        "to_ucl_club"
    ] = relevant[
        "to_club_id"
    ].isin(
        club_ids
    ).astype(
        int
    )

    relevant[
        "mapped_from_team"
    ] = relevant[
        "from_club_id"
    ].map(
        id_to_team
    )

    relevant[
        "mapped_to_team"
    ] = relevant[
        "to_club_id"
    ].map(
        id_to_team
    )

    def movement_type(
        row,
    ) -> str:

        from_inside = (
            row[
                "from_ucl_club"
            ]
            ==
            1
        )

        to_inside = (
            row[
                "to_ucl_club"
            ]
            ==
            1
        )

        if (
            from_inside
            and to_inside
        ):

            return "mapped_to_mapped"

        if to_inside:

            return "incoming_to_mapped"

        return "outgoing_from_mapped"

    relevant[
        "movement_type"
    ] = relevant.apply(
        movement_type,
        axis=1,
    )

    return relevant


# ============================================================
# REMOVE EXACT DUPLICATE ROWS
# ============================================================


def remove_exact_duplicates(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    int,
]:
    """
    Remove only completely identical records.

    We do NOT deduplicate on player/date alone because a player
    can legitimately have multiple transfer events around the
    same date.
    """

    before = len(
        dataframe
    )

    dataframe = dataframe.drop_duplicates().copy()

    removed = (
        before
        - len(
            dataframe
        )
    )

    return (
        dataframe,
        removed,
    )


# ============================================================
# ADD AUDIT FLAGS
# ============================================================


def add_audit_flags(
    *,
    dataframe: pd.DataFrame,
    training_end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Add source-audit fields.

    Scheduled future transfers are retained intentionally.

    Later feature engineering will enforce:

        transfer_date < UCL_match_date
    """

    output = dataframe.copy()

    output[
        "after_training_history"
    ] = (
        output[
            "transfer_date"
        ]
        >
        training_end_date
    ).astype(
        int
    )

    return output


# ============================================================
# QUALITY CHECK
# ============================================================


def run_quality_checks(
    *,
    dataframe: pd.DataFrame,
    club_ids: set[int],
    id_to_team: dict[int, str],
    training_start_date: pd.Timestamp,
    training_end_date: pd.Timestamp,
    exact_duplicates_removed: int,
) -> dict:
    """
    Audit the filtered transfer dataset.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFER HISTORY DATA QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # CORE VALUES
    # --------------------------------------------------------

    missing_player_or_date = int(
        dataframe[
            [
                "player_id",
                "transfer_date",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"\nRows missing player/date: "
        f"{missing_player_or_date:,}"
    )

    if missing_player_or_date > 0:

        raise RuntimeError(
            "Relevant transfers contain missing "
            "player/date values."
        )

    # --------------------------------------------------------
    # MAPPED-SIDE GUARANTEE
    # --------------------------------------------------------

    invalid_mapped_side = (
        (
            dataframe[
                "from_ucl_club"
            ]
            ==
            0
        )
        &
        (
            dataframe[
                "to_ucl_club"
            ]
            ==
            0
        )
    )

    if invalid_mapped_side.any():

        raise RuntimeError(
            "Transfer rows outside mapped club "
            "universe survived filtering."
        )

    print(
        "Every row touches >=1 mapped UCL club ✓"
    )

    # --------------------------------------------------------
    # SELF TRANSFERS
    # --------------------------------------------------------

    self_transfer_mask = (
        dataframe[
            "from_club_id"
        ].notna()
        &
        dataframe[
            "to_club_id"
        ].notna()
        &
        (
            dataframe[
                "from_club_id"
            ]
            ==
            dataframe[
                "to_club_id"
            ]
        )
    )

    self_transfers = int(
        self_transfer_mask.sum()
    )

    print(
        f"Same-club transfer rows: "
        f"{self_transfers:,}"
    )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    minimum_date = dataframe[
        "transfer_date"
    ].min()

    maximum_date = dataframe[
        "transfer_date"
    ].max()

    print(
        f"Transfer date range: "
        f"{minimum_date.date()} "
        f"to "
        f"{maximum_date.date()}"
    )

    print(
        f"UCL training date range: "
        f"{training_start_date.date()} "
        f"to "
        f"{training_end_date.date()}"
    )

    future_vs_training = int(
        dataframe[
            "after_training_history"
        ].sum()
    )

    print(
        f"Transfers after final training match: "
        f"{future_vs_training:,}"
    )

    print(
        "Future/scheduled rows retained intentionally ✓"
    )

    # --------------------------------------------------------
    # NUMERIC RANGE
    # --------------------------------------------------------

    for column in [
        "transfer_fee",
        "market_value_in_eur",
    ]:

        if column not in dataframe.columns:
            continue

        negative_count = int(
            (
                dataframe[
                    column
                ].dropna()
                <
                0
            ).sum()
        )

        print(
            f"Negative {column}: "
            f"{negative_count:,}"
        )

        if negative_count > 0:

            raise RuntimeError(
                f"Negative values detected "
                f"in {column}."
            )

    # --------------------------------------------------------
    # MOVEMENT DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nMovement distribution:"
    )

    movement_counts = (
        dataframe[
            "movement_type"
        ]
        .value_counts()
    )

    print(
        movement_counts.to_string()
    )

    # --------------------------------------------------------
    # CLUB COVERAGE
    # --------------------------------------------------------

    inbound_ids = set(
        dataframe.loc[
            dataframe[
                "to_ucl_club"
            ]
            ==
            1,
            "to_club_id",
        ]
        .dropna()
        .astype(
            int
        )
    )

    outbound_ids = set(
        dataframe.loc[
            dataframe[
                "from_ucl_club"
            ]
            ==
            1,
            "from_club_id",
        ]
        .dropna()
        .astype(
            int
        )
    )

    any_history_ids = (
        inbound_ids
        |
        outbound_ids
    )

    missing_any = (
        club_ids
        - any_history_ids
    )

    missing_inbound = (
        club_ids
        - inbound_ids
    )

    missing_outbound = (
        club_ids
        - outbound_ids
    )

    print(
        "\nMapped-club transfer coverage:"
    )

    print(
        f"Clubs with any transfer history: "
        f"{len(any_history_ids):,} / "
        f"{len(club_ids):,}"
    )

    print(
        f"Clubs with inbound history: "
        f"{len(inbound_ids):,} / "
        f"{len(club_ids):,}"
    )

    print(
        f"Clubs with outbound history: "
        f"{len(outbound_ids):,} / "
        f"{len(club_ids):,}"
    )

    if missing_any:

        print(
            "\nMapped clubs with no transfer rows:"
        )

        for club_id in sorted(
            missing_any
        ):

            print(
                f"  "
                f"{id_to_team.get(club_id, '?')} "
                f"| club_id={club_id}"
            )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    print(
        f"\nExact duplicate rows removed: "
        f"{exact_duplicates_removed:,}"
    )

    # --------------------------------------------------------
    # SAME PLAYER/DATE/ROUTE
    # --------------------------------------------------------

    route_duplicates = int(
        dataframe.duplicated(
            subset=[
                "player_id",
                "transfer_date",
                "from_club_id",
                "to_club_id",
            ]
        ).sum()
    )

    print(
        "Repeated player/date/from/to keys "
        f"remaining: {route_duplicates:,}"
    )

    # We deliberately do not fail here.
    # Same-day data can contain multi-step loan/return activity
    # or differing metadata in upstream transfer records.

    print(
        "\nTransfer-history integrity checks: passed ✓"
    )

    return {
        "rows": int(
            len(
                dataframe
            )
        ),

        "min_date": (
            minimum_date
            .date()
            .isoformat()
        ),

        "max_date": (
            maximum_date
            .date()
            .isoformat()
        ),

        "training_start_date": (
            training_start_date
            .date()
            .isoformat()
        ),

        "training_end_date": (
            training_end_date
            .date()
            .isoformat()
        ),

        "transfers_after_training_history": (
            future_vs_training
        ),

        "same_club_rows": (
            self_transfers
        ),

        "exact_duplicates_removed": (
            exact_duplicates_removed
        ),

        "repeated_player_date_route_keys": (
            route_duplicates
        ),

        "clubs_with_any_transfer_history": len(
            any_history_ids
        ),

        "clubs_with_inbound_history": len(
            inbound_ids
        ),

        "clubs_with_outbound_history": len(
            outbound_ids
        ),

        "clubs_without_any_transfer_history": [
            {
                "club_id": int(
                    club_id
                ),
                "team": id_to_team.get(
                    club_id
                ),
            }
            for club_id in sorted(
                missing_any
            )
        ],

        "movement_distribution": {
            str(
                key
            ): int(
                value
            )
            for (
                key,
                value,
            ) in movement_counts.items()
        },
    }


# ============================================================
# SAVE DATA
# ============================================================


def save_transfer_history(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save prepared transfer history.
    """

    output = dataframe.copy()

    output[
        "transfer_date"
    ] = output[
        "transfer_date"
    ].dt.strftime(
        "%Y-%m-%d"
    )

    output = output.sort_values(
        by=[
            "transfer_date",
            "player_id",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    output[
        "transfer_record_id"
    ] = range(
        1,
        len(
            output
        )
        + 1,
    )

    preferred_order = [
        "transfer_record_id",

        "player_id",
        "player_name",

        "transfer_date",
        "transfer_season",

        "from_club_id",
        "from_club_name",
        "mapped_from_team",
        "from_ucl_club",

        "to_club_id",
        "to_club_name",
        "mapped_to_team",
        "to_ucl_club",

        "movement_type",

        "transfer_fee",
        "market_value_in_eur",

        "after_training_history",
    ]

    ordered = [
        column
        for column in preferred_order
        if column in output.columns
    ]

    remaining = [
        column
        for column in output.columns
        if column not in ordered
    ]

    output = output[
        ordered
        +
        remaining
    ]

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nPrepared transfer history saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# SAVE METADATA
# ============================================================


def save_metadata(
    *,
    audit: dict,
    source_rows: int,
    filtered_rows_before_dedupe: int,
) -> None:
    """
    Save transfer data provenance and modeling rules.
    """

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "source_file": str(
            RAW_TRANSFERS_FILE
        ),

        "output_file": str(
            OUTPUT_FILE
        ),

        "source_rows": int(
            source_rows
        ),

        "filtered_rows_before_exact_dedupe": int(
            filtered_rows_before_dedupe
        ),

        "prepared_rows": int(
            audit[
                "rows"
            ]
        ),

        "club_filter": (
            "from_club_id or to_club_id belongs "
            "to final 117 UCL Transfermarkt IDs"
        ),

        "future_transfer_policy": (
            "Future/scheduled transfers remain in the "
            "prepared source table. Feature engineering "
            "must use only transfer_date < UCL_match_date."
        ),

        "historical_feature_rule": (
            "For match date t, no transfer with "
            "transfer_date >= t may contribute."
        ),

        "transfer_fee_rule": (
            "Missing fee is preserved as missing. "
            "Zero fee is not automatically interpreted "
            "as a free transfer because it can represent "
            "other transfer mechanics."
        ),

        "audit": audit,
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

    print(
        "\nTransfer metadata saved to:"
    )

    print(
        METADATA_FILE
    )


# ============================================================
# SAMPLE DISPLAY
# ============================================================


def print_samples(
    dataframe: pd.DataFrame,
) -> None:
    """
    Show historical and latest transfer examples.
    """

    display_columns = [
        column
        for column in [
            "player_name",
            "transfer_date",
            "from_club_name",
            "mapped_from_team",
            "to_club_name",
            "mapped_to_team",
            "movement_type",
            "transfer_fee",
            "market_value_in_eur",
            "after_training_history",
        ]
        if column in dataframe.columns
    ]

    print(
        "\nFirst 10 relevant transfers:"
    )

    print(
        dataframe[
            display_columns
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nLast 10 relevant transfers:"
    )

    print(
        dataframe[
            display_columns
        ]
        .tail(
            10
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================


def prepare_transfer_history() -> None:
    """
    Prepare Transfermarkt transfer history for the 117 mapped
    UCL club identities.

    This step prepares source data only.

    Transfer-impact feature engineering comes next and will
    enforce strict match-date chronology.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - TRANSFER HISTORY PREPARATION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # LOAD MAPPING
    # --------------------------------------------------------

    (
        mapping,
        club_ids,
        id_to_team,
    ) = load_club_map()

    print(
        f"\nMapped UCL clubs: "
        f"{len(mapping):,}"
    )

    print(
        f"Unique Transfermarkt IDs: "
        f"{len(club_ids):,}"
    )

    # --------------------------------------------------------
    # TRAINING RANGE
    # --------------------------------------------------------

    (
        training_start_date,
        training_end_date,
    ) = load_ucl_training_dates()

    # --------------------------------------------------------
    # RAW TRANSFERS
    # --------------------------------------------------------

    raw = load_raw_transfers()

    source_rows = len(
        raw
    )

    normalized = normalize_transfers(
        raw
    )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    relevant = filter_relevant_transfers(
        dataframe=normalized,
        club_ids=club_ids,
        id_to_team=id_to_team,
    )

    filtered_rows_before_dedupe = len(
        relevant
    )

    print(
        f"Relevant transfer rows before dedupe: "
        f"{filtered_rows_before_dedupe:,}"
    )

    # --------------------------------------------------------
    # EXACT DUPLICATES
    # --------------------------------------------------------

    (
        relevant,
        duplicates_removed,
    ) = remove_exact_duplicates(
        relevant
    )

    # --------------------------------------------------------
    # AUDIT FLAGS
    # --------------------------------------------------------

    relevant = add_audit_flags(
        dataframe=relevant,
        training_end_date=training_end_date,
    )

    relevant = relevant.sort_values(
        by=[
            "transfer_date",
            "player_id",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    audit = run_quality_checks(
        dataframe=relevant,
        club_ids=club_ids,
        id_to_team=id_to_team,
        training_start_date=training_start_date,
        training_end_date=training_end_date,
        exact_duplicates_removed=duplicates_removed,
    )

    # --------------------------------------------------------
    # SAMPLES
    # --------------------------------------------------------

    print_samples(
        relevant
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_transfer_history(
        relevant
    )

    save_metadata(
        audit=audit,
        source_rows=source_rows,
        filtered_rows_before_dedupe=(
            filtered_rows_before_dedupe
        ),
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFER HISTORY PREPARATION SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nRaw Transfermarkt transfers: "
        f"{source_rows:,}"
    )

    print(
        f"Relevant UCL-club transfers: "
        f"{len(relevant):,}"
    )

    print(
        f"Mapped clubs with transfer history: "
        f"{audit['clubs_with_any_transfer_history']:,} "
        f"/ {EXPECTED_UCL_CLUBS}"
    )

    print(
        f"Transfers after training history: "
        f"{audit['transfers_after_training_history']:,}"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFER HISTORY PREPARED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nNext leakage barrier:"
    )

    print(
        "    transfer_date < UCL_match_date"
    )

    print(
        "\nFuture scheduled transfers remain stored "
        "but cannot influence earlier predictions."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    prepare_transfer_history()