from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
)

from src.utils.team_names import (
    normalize_team_name,
    alias_text,
)


# ============================================================
# PATHS
# ============================================================

GAMES_FILE = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
    / "games.csv"
)

UNRESOLVED_FILE = (
    REPORTS_DIR
    / "metrics"
    / "transfermarkt_clubs_unresolved.csv"
)

OUTPUT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "transfermarkt_game_club_audit.csv"
)


# ============================================================
# SETTINGS
# ============================================================

UCL_COMPETITION_ID = "CL"

LIKELY_ALIAS_THRESHOLD = 0.85

REVIEW_THRESHOLD = 0.55

TOP_SUGGESTIONS = 5


# ============================================================
# SIMILARITY
# ============================================================


def similarity(
    first: str,
    second: str,
) -> float:
    """
    Advisory fuzzy similarity only.

    No fuzzy result is automatically accepted.
    """

    return SequenceMatcher(
        None,
        alias_text(
            first
        ),
        alias_text(
            second
        ),
    ).ratio()


# ============================================================
# LOAD UNRESOLVED UCL CLUBS
# ============================================================


def load_unresolved_clubs() -> list[str]:
    """
    Load clubs that remain unresolved after clubs.csv
    mapping.
    """

    if not UNRESOLVED_FILE.exists():

        raise FileNotFoundError(
            f"Unresolved club report not found:\n"
            f"{UNRESOLVED_FILE}"
        )

    dataframe = pd.read_csv(
        UNRESOLVED_FILE,
        low_memory=False,
    )

    if "ucl_team" not in dataframe.columns:

        raise RuntimeError(
            "Unresolved report does not contain "
            "'ucl_team'."
        )

    teams = sorted(
        dataframe[
            "ucl_team"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    return teams


# ============================================================
# LOAD TRANSFERMARKT UCL GAMES
# ============================================================


def load_ucl_games() -> pd.DataFrame:
    """
    Load competition-proper Champions League matches.
    """

    if not GAMES_FILE.exists():

        raise FileNotFoundError(
            f"Transfermarkt games file not found:\n"
            f"{GAMES_FILE}"
        )

    usecols = [
        "game_id",
        "competition_id",
        "season",
        "date",
        "home_club_id",
        "home_club_name",
        "away_club_id",
        "away_club_name",
    ]

    dataframe = pd.read_csv(
        GAMES_FILE,
        usecols=usecols,
        low_memory=False,
    )

    dataframe[
        "competition_id"
    ] = (
        dataframe[
            "competition_id"
        ]
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        dataframe[
            "competition_id"
        ]
        ==
        UCL_COMPETITION_ID
    ].copy()

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="raise",
    )

    return dataframe


# ============================================================
# BUILD UNIQUE UCL CLUB TABLE
# ============================================================


def build_ucl_game_clubs(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build unique Transfermarkt club ID/name pairs observed in
    actual Champions League matches.
    """

    home = (
        games[
            [
                "home_club_id",
                "home_club_name",
            ]
        ]
        .rename(
            columns={
                "home_club_id": "club_id",
                "home_club_name": "transfermarkt_name",
            }
        )
    )

    away = (
        games[
            [
                "away_club_id",
                "away_club_name",
            ]
        ]
        .rename(
            columns={
                "away_club_id": "club_id",
                "away_club_name": "transfermarkt_name",
            }
        )
    )

    clubs = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    clubs = clubs.dropna(
        subset=[
            "club_id",
            "transfermarkt_name",
        ]
    ).copy()

    clubs[
        "club_id"
    ] = pd.to_numeric(
        clubs[
            "club_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    clubs[
        "transfermarkt_name"
    ] = (
        clubs[
            "transfermarkt_name"
        ]
        .astype(str)
        .str.strip()
    )

    clubs = clubs.drop_duplicates(
        subset=[
            "club_id",
            "transfermarkt_name",
        ]
    )

    # --------------------------------------------------------
    # NORMALIZE USING OUR EXISTING PROJECT RULES
    # --------------------------------------------------------

    clubs[
        "normalized_name"
    ] = clubs[
        "transfermarkt_name"
    ].map(
        normalize_team_name
    )

    return clubs.sort_values(
        by=[
            "normalized_name",
            "club_id",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# FIRST / LAST UCL APPEARANCE
# ============================================================


def build_club_date_lookup(
    games: pd.DataFrame,
) -> dict[int, dict]:
    """
    Determine first and last observed CL match per club ID.
    """

    records = []

    for row in games.itertuples(
        index=False
    ):

        records.append(
            {
                "club_id": int(
                    row.home_club_id
                ),
                "date": row.date,
            }
        )

        records.append(
            {
                "club_id": int(
                    row.away_club_id
                ),
                "date": row.date,
            }
        )

    dataframe = pd.DataFrame(
        records
    )

    grouped = (
        dataframe
        .groupby(
            "club_id"
        )[
            "date"
        ]
        .agg(
            [
                "min",
                "max",
                "count",
            ]
        )
    )

    lookup = {}

    for (
        club_id,
        row,
    ) in grouped.iterrows():

        lookup[
            int(
                club_id
            )
        ] = {
            "first_ucl_date": (
                row[
                    "min"
                ]
            ),
            "last_ucl_date": (
                row[
                    "max"
                ]
            ),
            "ucl_game_count": int(
                row[
                    "count"
                ]
            ),
        }

    return lookup


# ============================================================
# AUDIT SINGLE TEAM
# ============================================================


def audit_team(
    *,
    ucl_team: str,
    clubs: pd.DataFrame,
    date_lookup: dict[int, dict],
) -> dict:
    """
    Audit one unresolved canonical team against clubs observed
    in Transfermarkt UCL games.
    """

    # --------------------------------------------------------
    # EXACT NORMALIZED MATCH
    # --------------------------------------------------------

    exact = clubs[
        clubs[
            "normalized_name"
        ]
        ==
        ucl_team
    ]

    if len(
        exact
    ) == 1:

        row = exact.iloc[
            0
        ]

        club_id = int(
            row[
                "club_id"
            ]
        )

        dates = date_lookup.get(
            club_id,
            {},
        )

        return {
            "ucl_team": ucl_team,
            "status": "MATCHED",
            "club_id": club_id,
            "transfermarkt_name": row[
                "transfermarkt_name"
            ],
            "normalized_name": row[
                "normalized_name"
            ],
            "match_score": 1.0,
            "first_ucl_date": dates.get(
                "first_ucl_date"
            ),
            "last_ucl_date": dates.get(
                "last_ucl_date"
            ),
            "ucl_game_count": dates.get(
                "ucl_game_count",
                0,
            ),
            "suggestion_1": "",
            "suggestion_1_club_id": "",
            "suggestion_1_score": "",
            "suggestion_2": "",
            "suggestion_2_club_id": "",
            "suggestion_2_score": "",
            "suggestion_3": "",
            "suggestion_3_club_id": "",
            "suggestion_3_score": "",
            "suggestion_4": "",
            "suggestion_4_club_id": "",
            "suggestion_4_score": "",
            "suggestion_5": "",
            "suggestion_5_club_id": "",
            "suggestion_5_score": "",
        }

    # --------------------------------------------------------
    # MULTIPLE EXACT NORMALIZED MATCHES
    # --------------------------------------------------------

    if len(
        exact
    ) > 1:

        row = {
            "ucl_team": ucl_team,
            "status": "MULTIPLE_MATCHES",
            "club_id": "",
            "transfermarkt_name": "",
            "normalized_name": ucl_team,
            "match_score": 1.0,
            "first_ucl_date": "",
            "last_ucl_date": "",
            "ucl_game_count": "",
        }

        candidates = exact.head(
            TOP_SUGGESTIONS
        )

        for index in range(
            TOP_SUGGESTIONS
        ):

            number = (
                index
                + 1
            )

            if index < len(
                candidates
            ):

                candidate = candidates.iloc[
                    index
                ]

                row[
                    f"suggestion_{number}"
                ] = candidate[
                    "transfermarkt_name"
                ]

                row[
                    f"suggestion_{number}_club_id"
                ] = int(
                    candidate[
                        "club_id"
                    ]
                )

                row[
                    f"suggestion_{number}_score"
                ] = 1.0

            else:

                row[
                    f"suggestion_{number}"
                ] = ""

                row[
                    f"suggestion_{number}_club_id"
                ] = ""

                row[
                    f"suggestion_{number}_score"
                ] = ""

        return row

    # --------------------------------------------------------
    # FUZZY ADVISORY SUGGESTIONS
    # --------------------------------------------------------

    scored = []

    for candidate in clubs.itertuples(
        index=False
    ):

        score = similarity(
            ucl_team,
            candidate.normalized_name,
        )

        scored.append(
            {
                "club_id": int(
                    candidate.club_id
                ),
                "transfermarkt_name": (
                    candidate.transfermarkt_name
                ),
                "normalized_name": (
                    candidate.normalized_name
                ),
                "score": float(
                    score
                ),
            }
        )

    scored.sort(
        key=lambda item: item[
            "score"
        ],
        reverse=True,
    )

    suggestions = scored[
        :TOP_SUGGESTIONS
    ]

    best_score = (
        suggestions[
            0
        ][
            "score"
        ]
        if suggestions
        else 0.0
    )

    if (
        best_score
        >= LIKELY_ALIAS_THRESHOLD
    ):

        status = "LIKELY_ALIAS"

    elif (
        best_score
        >= REVIEW_THRESHOLD
    ):

        status = "REVIEW"

    else:

        status = "NO_MATCH"

    row = {
        "ucl_team": ucl_team,
        "status": status,
        "club_id": "",
        "transfermarkt_name": "",
        "normalized_name": ucl_team,
        "match_score": "",
        "first_ucl_date": "",
        "last_ucl_date": "",
        "ucl_game_count": "",
    }

    for index in range(
        TOP_SUGGESTIONS
    ):

        number = (
            index
            + 1
        )

        if index < len(
            suggestions
        ):

            suggestion = suggestions[
                index
            ]

            row[
                f"suggestion_{number}"
            ] = suggestion[
                "transfermarkt_name"
            ]

            row[
                f"suggestion_{number}_club_id"
            ] = suggestion[
                "club_id"
            ]

            row[
                f"suggestion_{number}_score"
            ] = round(
                suggestion[
                    "score"
                ],
                3,
            )

        else:

            row[
                f"suggestion_{number}"
            ] = ""

            row[
                f"suggestion_{number}_club_id"
            ] = ""

            row[
                f"suggestion_{number}_score"
            ] = ""

    return row


# ============================================================
# BUILD AUDIT
# ============================================================


def build_audit(
    *,
    unresolved_teams: list[str],
    clubs: pd.DataFrame,
    date_lookup: dict[int, dict],
) -> pd.DataFrame:
    """
    Audit all remaining unresolved clubs.
    """

    rows = []

    for team in unresolved_teams:

        rows.append(
            audit_team(
                ucl_team=team,
                clubs=clubs,
                date_lookup=date_lookup,
            )
        )

    dataframe = pd.DataFrame(
        rows
    )

    status_order = {
        "MATCHED": 0,
        "MULTIPLE_MATCHES": 1,
        "LIKELY_ALIAS": 2,
        "REVIEW": 3,
        "NO_MATCH": 4,
    }

    dataframe[
        "_status_order"
    ] = dataframe[
        "status"
    ].map(
        status_order
    )

    dataframe = (
        dataframe
        .sort_values(
            by=[
                "_status_order",
                "ucl_team",
            ]
        )
        .drop(
            columns=[
                "_status_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


# ============================================================
# PRINT AUDIT
# ============================================================


def print_audit(
    dataframe: pd.DataFrame,
) -> None:
    """
    Print exact and advisory matches.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT UCL GAME CLUB AUDIT"
    )

    print(
        "=" * 75
    )

    for row in dataframe.itertuples(
        index=False
    ):

        print(
            f"\n{row.ucl_team}"
        )

        print(
            f"Status: "
            f"{row.status}"
        )

        if row.status == "MATCHED":

            print(
                f"  Transfermarkt: "
                f"{row.transfermarkt_name}"
            )

            print(
                f"  club_id: "
                f"{row.club_id}"
            )

            print(
                f"  UCL games: "
                f"{row.ucl_game_count}"
            )

            print(
                f"  observed: "
                f"{row.first_ucl_date} "
                f"to "
                f"{row.last_ucl_date}"
            )

            continue

        for number in range(
            1,
            TOP_SUGGESTIONS + 1,
        ):

            name = getattr(
                row,
                f"suggestion_{number}",
            )

            club_id = getattr(
                row,
                f"suggestion_{number}_club_id",
            )

            score = getattr(
                row,
                f"suggestion_{number}_score",
            )

            if (
                pd.isna(
                    name
                )
                or str(
                    name
                ).strip() == ""
            ):

                continue

            print(
                f"  {number}. "
                f"{name}"
                f" | club_id={club_id}"
                f" | score={score}"
            )


# ============================================================
# SUMMARY
# ============================================================


def print_summary(
    dataframe: pd.DataFrame,
) -> None:
    """
    Print status summary.
    """

    counts = dataframe[
        "status"
    ].value_counts()

    print(
        "\n"
        + "=" * 75
    )

    print(
        "GAME-BASED CLUB AUDIT SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nMATCHED:          "
        f"{counts.get('MATCHED', 0):,}"
    )

    print(
        f"MULTIPLE_MATCHES: "
        f"{counts.get('MULTIPLE_MATCHES', 0):,}"
    )

    print(
        f"LIKELY_ALIAS:     "
        f"{counts.get('LIKELY_ALIAS', 0):,}"
    )

    print(
        f"REVIEW:           "
        f"{counts.get('REVIEW', 0):,}"
    )

    print(
        f"NO_MATCH:         "
        f"{counts.get('NO_MATCH', 0):,}"
    )

    print(
        "\nAudit saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nNo mappings were modified."
    )


# ============================================================
# MAIN
# ============================================================


def audit_transfermarkt_game_clubs() -> None:
    """
    Audit unresolved UCL identities using actual
    Transfermarkt Champions League match records.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - TRANSFERMARKT GAME CLUB AUDIT"
    )

    print(
        "=" * 75
    )

    unresolved = load_unresolved_clubs()

    games = load_ucl_games()

    clubs = build_ucl_game_clubs(
        games
    )

    date_lookup = build_club_date_lookup(
        games
    )

    print(
        f"\nRemaining unresolved UCL clubs: "
        f"{len(unresolved):,}"
    )

    print(
        f"Transfermarkt CL games: "
        f"{len(games):,}"
    )

    print(
        f"Unique CL club IDs: "
        f"{clubs['club_id'].nunique():,}"
    )

    audit = build_audit(
        unresolved_teams=unresolved,
        clubs=clubs,
        date_lookup=date_lookup,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print_audit(
        audit
    )

    print_summary(
        audit
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    audit_transfermarkt_game_clubs()