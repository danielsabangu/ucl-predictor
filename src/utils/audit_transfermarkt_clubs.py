from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
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

UCL_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

TRANSFERMARKT_CLUBS_FILE = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
    / "clubs.csv"
)

AUDIT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "transfermarkt_club_identity_audit.csv"
)

MAPPING_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LIKELY_ALIAS_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.60

TOP_SUGGESTIONS = 3


# ============================================================
# HELPERS
# ============================================================


def similarity(
    first: str,
    second: str,
) -> float:
    """
    Calculate a simple text similarity score.

    Fuzzy scores are used ONLY for auditing.
    They are never automatically applied.
    """

    first_key = alias_text(
        first
    )

    second_key = alias_text(
        second
    )

    return SequenceMatcher(
        None,
        first_key,
        second_key,
    ).ratio()


def get_ucl_teams(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return the canonical UCL club universe.
    """

    teams = set(
        dataframe[
            "home_team"
        ].dropna()
    )

    teams.update(
        dataframe[
            "away_team"
        ].dropna()
    )

    return sorted(
        teams
    )


# ============================================================
# LOAD UCL
# ============================================================


def load_ucl() -> tuple[
    pd.DataFrame,
    list[str],
]:
    """
    Load canonical UCL matches.
    """

    print(
        "\nLoading UCL dataset:"
    )

    print(
        UCL_FILE
    )

    if not UCL_FILE.exists():

        raise FileNotFoundError(
            f"UCL dataset not found:\n"
            f"{UCL_FILE}"
        )

    dataframe = pd.read_csv(
        UCL_FILE
    )

    required = {
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

    teams = get_ucl_teams(
        dataframe
    )

    print(
        f"UCL matches loaded: "
        f"{len(dataframe):,}"
    )

    print(
        f"Canonical UCL clubs: "
        f"{len(teams):,}"
    )

    return (
        dataframe,
        teams,
    )


# ============================================================
# LOAD TRANSFERMARKT CLUBS
# ============================================================


def load_transfermarkt_clubs() -> pd.DataFrame:
    """
    Load and normalize Transfermarkt club identities.
    """

    print(
        "\nLoading Transfermarkt clubs:"
    )

    print(
        TRANSFERMARKT_CLUBS_FILE
    )

    if not TRANSFERMARKT_CLUBS_FILE.exists():

        raise FileNotFoundError(
            f"Transfermarkt clubs file not found:\n"
            f"{TRANSFERMARKT_CLUBS_FILE}"
        )

    clubs = pd.read_csv(
        TRANSFERMARKT_CLUBS_FILE,
        low_memory=False,
    )

    required = {
        "club_id",
        "name",
    }

    missing = (
        required
        - set(
            clubs.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Transfermarkt clubs file "
            f"missing columns: "
            f"{sorted(missing)}"
        )

    clubs = clubs[
        clubs[
            "club_id"
        ].notna()
        & clubs[
            "name"
        ].notna()
    ].copy()

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
    ] = clubs[
        "name"
    ].astype(
        str
    ).str.strip()

    clubs[
        "normalized_name"
    ] = clubs[
        "transfermarkt_name"
    ].map(
        normalize_team_name
    )

    clubs = clubs.sort_values(
        by=[
            "normalized_name",
            "club_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Transfermarkt clubs loaded: "
        f"{len(clubs):,}"
    )

    print(
        f"Unique Transfermarkt club IDs: "
        f"{clubs['club_id'].nunique():,}"
    )

    return clubs


# ============================================================
# EXACT MATCH INDEX
# ============================================================


def build_exact_match_index(
    clubs: pd.DataFrame,
) -> dict[
    str,
    list[dict],
]:
    """
    Group Transfermarkt clubs by our normalized identity.
    """

    lookup: dict[
        str,
        list[dict],
    ] = {}

    for row in clubs.itertuples(
        index=False
    ):

        lookup.setdefault(
            row.normalized_name,
            [],
        ).append(
            {
                "club_id": int(
                    row.club_id
                ),
                "transfermarkt_name": (
                    row.transfermarkt_name
                ),
                "normalized_name": (
                    row.normalized_name
                ),
            }
        )

    return lookup


# ============================================================
# FUZZY SUGGESTIONS
# ============================================================


def get_fuzzy_suggestions(
    *,
    ucl_team: str,
    clubs: pd.DataFrame,
    limit: int = TOP_SUGGESTIONS,
) -> list[dict]:
    """
    Return top text suggestions.

    These are advisory only.
    """

    scored = []

    for row in clubs.itertuples(
        index=False
    ):

        score = similarity(
            ucl_team,
            row.normalized_name,
        )

        scored.append(
            {
                "club_id": int(
                    row.club_id
                ),
                "transfermarkt_name": (
                    row.transfermarkt_name
                ),
                "normalized_name": (
                    row.normalized_name
                ),
                "score": float(
                    score
                ),
            }
        )

    scored.sort(
        key=lambda item: (
            item[
                "score"
            ],
            -item[
                "club_id"
            ],
        ),
        reverse=True,
    )

    return scored[
        :limit
    ]


# ============================================================
# AUDIT ONE UCL CLUB
# ============================================================


def audit_ucl_team(
    *,
    team: str,
    exact_lookup: dict[
        str,
        list[dict],
    ],
    clubs: pd.DataFrame,
) -> dict:
    """
    Resolve one UCL club against Transfermarkt.
    """

    exact_matches = exact_lookup.get(
        team,
        [],
    )

    # --------------------------------------------------------
    # UNAMBIGUOUS EXACT MATCH
    # --------------------------------------------------------

    if len(
        exact_matches
    ) == 1:

        match = exact_matches[
            0
        ]

        return {
            "ucl_team": team,
            "status": "MATCHED",
            "club_id": match[
                "club_id"
            ],
            "transfermarkt_name": match[
                "transfermarkt_name"
            ],
            "normalized_name": match[
                "normalized_name"
            ],
            "match_score": 1.0,
            "candidate_count": 1,

            "suggestion_1": "",
            "suggestion_1_club_id": "",
            "suggestion_1_score": "",

            "suggestion_2": "",
            "suggestion_2_club_id": "",
            "suggestion_2_score": "",

            "suggestion_3": "",
            "suggestion_3_club_id": "",
            "suggestion_3_score": "",
        }

    # --------------------------------------------------------
    # MULTIPLE EXACT MATCHES
    # --------------------------------------------------------

    if len(
        exact_matches
    ) > 1:

        suggestions = exact_matches[
            :TOP_SUGGESTIONS
        ]

        row = {
            "ucl_team": team,
            "status": "MULTIPLE_MATCHES",
            "club_id": "",
            "transfermarkt_name": "",
            "normalized_name": team,
            "match_score": 1.0,
            "candidate_count": len(
                exact_matches
            ),
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

                candidate = suggestions[
                    index
                ]

                row[
                    f"suggestion_{number}"
                ] = candidate[
                    "transfermarkt_name"
                ]

                row[
                    f"suggestion_{number}_club_id"
                ] = candidate[
                    "club_id"
                ]

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
    # NO EXACT MATCH
    # --------------------------------------------------------

    suggestions = get_fuzzy_suggestions(
        ucl_team=team,
        clubs=clubs,
    )

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
        "ucl_team": team,
        "status": status,
        "club_id": "",
        "transfermarkt_name": "",
        "normalized_name": team,
        "match_score": "",
        "candidate_count": 0,
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

            candidate = suggestions[
                index
            ]

            row[
                f"suggestion_{number}"
            ] = candidate[
                "transfermarkt_name"
            ]

            row[
                f"suggestion_{number}_club_id"
            ] = candidate[
                "club_id"
            ]

            row[
                f"suggestion_{number}_score"
            ] = round(
                candidate[
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
    ucl_teams: list[str],
    clubs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Audit all 117 canonical UCL clubs.
    """

    exact_lookup = build_exact_match_index(
        clubs
    )

    rows = []

    for team in ucl_teams:

        rows.append(
            audit_ucl_team(
                team=team,
                exact_lookup=exact_lookup,
                clubs=clubs,
            )
        )

    audit = pd.DataFrame(
        rows
    )

    status_order = {
        "MULTIPLE_MATCHES": 0,
        "LIKELY_ALIAS": 1,
        "REVIEW": 2,
        "NO_MATCH": 3,
        "MATCHED": 4,
    }

    audit[
        "_status_order"
    ] = audit[
        "status"
    ].map(
        status_order
    )

    audit = audit.sort_values(
        by=[
            "_status_order",
            "ucl_team",
        ]
    ).drop(
        columns=[
            "_status_order"
        ]
    ).reset_index(
        drop=True
    )

    return audit


# ============================================================
# SAFE MAPPING
# ============================================================


def build_safe_mapping(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save ONLY unambiguous exact matches.

    Fuzzy suggestions are never automatically accepted.
    """

    mapping = audit[
        audit[
            "status"
        ]
        ==
        "MATCHED"
    ][
        [
            "ucl_team",
            "club_id",
            "transfermarkt_name",
        ]
    ].copy()

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

    mapping[
        "mapping_method"
    ] = "explicit_normalization"

    mapping[
        "mapping_confidence"
    ] = 1.0

    mapping = mapping.sort_values(
        by="ucl_team"
    ).reset_index(
        drop=True
    )

    return mapping


# ============================================================
# QUALITY CHECKS
# ============================================================


def run_quality_checks(
    *,
    audit: pd.DataFrame,
    mapping: pd.DataFrame,
    expected_ucl_teams: int,
) -> None:
    """
    Ensure mapping integrity before player features use it.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT CLUB IDENTITY QUALITY CHECK"
    )

    print(
        "=" * 75
    )

    if len(
        audit
    ) != expected_ucl_teams:

        raise RuntimeError(
            f"Expected "
            f"{expected_ucl_teams:,} "
            f"UCL clubs in audit but found "
            f"{len(audit):,}."
        )

    print(
        f"\nUCL clubs audited: "
        f"{len(audit):,} ✓"
    )

    if audit[
        "ucl_team"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate UCL teams found in audit."
        )

    print(
        "Unique UCL identities: valid ✓"
    )

    if mapping[
        "ucl_team"
    ].duplicated().any():

        raise RuntimeError(
            "One UCL team maps to multiple "
            "Transfermarkt IDs."
        )

    print(
        "One-to-one UCL mapping: valid ✓"
    )

    duplicate_club_ids = (
        mapping[
            "club_id"
        ].duplicated(
            keep=False
        )
    )

    if duplicate_club_ids.any():

        duplicates = mapping[
            duplicate_club_ids
        ].sort_values(
            by="club_id"
        )

        print(
            "\nWARNING:"
        )

        print(
            "One Transfermarkt club_id currently maps "
            "to multiple UCL canonical identities:"
        )

        print(
            duplicates.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Transfermarkt club ID collision detected."
        )

    print(
        "Unique Transfermarkt IDs: valid ✓"
    )


# ============================================================
# PRINT REVIEW SECTION
# ============================================================


def print_manual_review(
    audit: pd.DataFrame,
) -> None:
    """
    Print only clubs requiring human review.
    """

    review = audit[
        audit[
            "status"
        ]
        !=
        "MATCHED"
    ].copy()

    print(
        "\n"
        + "=" * 75
    )

    print(
        "UCL CLUBS REQUIRING REVIEW"
    )

    print(
        "=" * 75
    )

    if review.empty:

        print(
            "\nNone. Every UCL club matched "
            "unambiguously."
        )

        return

    for row in review.itertuples(
        index=False
    ):

        print(
            f"\n{row.ucl_team}"
        )

        print(
            f"Status: "
            f"{row.status}"
        )

        if (
            row.status
            ==
            "MULTIPLE_MATCHES"
        ):

            print(
                f"Exact candidates: "
                f"{row.candidate_count}"
            )

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
    *,
    audit: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    """
    Print identity coverage.
    """

    counts = audit[
        "status"
    ].value_counts()

    total = len(
        audit
    )

    matched = int(
        counts.get(
            "MATCHED",
            0,
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TRANSFERMARKT CLUB IDENTITY SUMMARY"
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
        f"\nSafe exact mappings: "
        f"{len(mapping):,} / {total:,}"
    )

    print(
        f"Safe identity coverage: "
        f"{matched / total * 100:.2f}%"
    )

    print(
        "\nAudit saved to:"
    )

    print(
        AUDIT_FILE
    )

    print(
        "\nSafe mapping saved to:"
    )

    print(
        MAPPING_FILE
    )

    print(
        "\nNo fuzzy mapping has been applied."
    )


# ============================================================
# MAIN
# ============================================================


def audit_transfermarkt_clubs() -> None:
    """
    Audit Transfermarkt club IDs against the canonical
    historical UCL club universe.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - TRANSFERMARKT CLUB IDENTITY AUDIT"
    )

    print(
        "=" * 75
    )

    (
        _,
        ucl_teams,
    ) = load_ucl()

    clubs = load_transfermarkt_clubs()

    audit = build_audit(
        ucl_teams=ucl_teams,
        clubs=clubs,
    )

    mapping = build_safe_mapping(
        audit
    )

    run_quality_checks(
        audit=audit,
        mapping=mapping,
        expected_ucl_teams=len(
            ucl_teams
        ),
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MAPPING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    mapping.to_csv(
        MAPPING_FILE,
        index=False,
        encoding="utf-8",
    )

    print_manual_review(
        audit
    )

    print_summary(
        audit=audit,
        mapping=mapping,
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    audit_transfermarkt_clubs()