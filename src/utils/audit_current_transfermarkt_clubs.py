from __future__ import annotations

import unicodedata

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
)


# ============================================================
# PATHS
# ============================================================

CURRENT_TEAM_MAP_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "ucl_2026_2027_team_map.csv"
)

TRANSFERMARKT_CLUBS_FILE = (
    RAW_DATA_DIR
    / "players"
    / "transfermarkt"
    / "clubs.csv"
)

HISTORICAL_TRANSFERMARKT_MAP_FILE = (
    INTERIM_DATA_DIR
    / "players"
    / "transfermarkt_ucl_club_map.csv"
)

OUTPUT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "ucl_2026_2027_transfermarkt_club_candidates.csv"
)


# ============================================================
# EXPECTED NEW CLUBS
# ============================================================

EXPECTED_NEW_CLUBS = {
    "Como 1907",
    "Fenerbahçe",
    "LASK Linz",
    "Real Betis Balompié",
    "Sabah FK",
    "Viking FK",
}


# ============================================================
# EXPLICIT SEARCH VARIANTS
# ============================================================

SEARCH_VARIANTS = {
    "Como 1907": [
        "Como 1907",
        "Como",
    ],

    "Fenerbahçe": [
        "Fenerbahçe",
        "Fenerbahce",
        "Fenerbahce SK",
        "Fenerbahçe SK",
    ],

    "LASK Linz": [
        "LASK Linz",
        "LASK",
    ],

    "Real Betis Balompié": [
        "Real Betis Balompié",
        "Real Betis Balompie",
        "Real Betis",
    ],

    "Sabah FK": [
        "Sabah FK",
        "Sabah FC",
        "Sabah",
    ],

    "Viking FK": [
        "Viking FK",
        "Viking",
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_text(
    value,
) -> str:
    """
    Text normalization used ONLY for candidate discovery.

    This function does not create or approve club mappings.
    """

    if pd.isna(
        value
    ):

        return ""

    text = str(
        value
    ).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(
            character
        )
    )

    replacements = {
        "ı": "i",
        "ø": "o",
        "đ": "d",
        "ß": "ss",
        "æ": "ae",
        "œ": "oe",
    }

    for (
        source,
        target,
    ) in replacements.items():

        text = text.replace(
            source,
            target,
        )

    text = (
        text
        .replace(
            ".",
            " ",
        )
        .replace(
            "-",
            " ",
        )
        .replace(
            "_",
            " ",
        )
        .replace(
            "/",
            " ",
        )
    )

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# LOAD CURRENT CLUBS
# ============================================================


def load_current_new_clubs() -> pd.DataFrame:
    """
    Load only genuinely new 2026/27 UCL identities.
    """

    if not CURRENT_TEAM_MAP_FILE.exists():

        raise FileNotFoundError(
            "Current UCL team map not found:\n"
            f"{CURRENT_TEAM_MAP_FILE}"
        )

    dataframe = pd.read_csv(
        CURRENT_TEAM_MAP_FILE,
        low_memory=False,
    )

    required = {
        "canonical_team",
        "football_data_team_id",
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
            "Current team map missing columns: "
            f"{sorted(missing)}"
        )

    new_clubs = dataframe[
        dataframe[
            "identity_status"
        ]
        ==
        "NEW_2026_27"
    ].copy()

    actual_new_clubs = set(
        new_clubs[
            "canonical_team"
        ]
    )

    if (
        actual_new_clubs
        !=
        EXPECTED_NEW_CLUBS
    ):

        raise RuntimeError(
            "\nUnexpected new-club set.\n"
            f"Expected: {sorted(EXPECTED_NEW_CLUBS)}\n"
            f"Found:    {sorted(actual_new_clubs)}"
        )

    return new_clubs


# ============================================================
# LOAD TRANSFERMARKT CLUBS
# ============================================================


def load_transfermarkt_clubs() -> pd.DataFrame:
    """
    Load local Transfermarkt club snapshot.
    """

    if not TRANSFERMARKT_CLUBS_FILE.exists():

        raise FileNotFoundError(
            "Transfermarkt clubs.csv not found:\n"
            f"{TRANSFERMARKT_CLUBS_FILE}"
        )

    dataframe = pd.read_csv(
        TRANSFERMARKT_CLUBS_FILE,
        low_memory=False,
    )

    required = {
        "club_id",
        "name",
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
            "Transfermarkt clubs.csv missing columns: "
            f"{sorted(missing)}"
        )

    if dataframe[
        "club_id"
    ].isna().any():

        raise RuntimeError(
            "Transfermarkt clubs.csv contains "
            "missing club IDs."
        )

    dataframe[
        "normalized_name"
    ] = dataframe[
        "name"
    ].map(
        normalize_text
    )

    return dataframe


# ============================================================
# LOAD EXISTING HISTORICAL MAP
# ============================================================


def load_historical_transfermarkt_map() -> pd.DataFrame:
    """
    Load existing 117-club Transfermarkt mapping.
    """

    if not HISTORICAL_TRANSFERMARKT_MAP_FILE.exists():

        raise FileNotFoundError(
            "Historical Transfermarkt club map "
            "not found:\n"
            f"{HISTORICAL_TRANSFERMARKT_MAP_FILE}"
        )

    dataframe = pd.read_csv(
        HISTORICAL_TRANSFERMARKT_MAP_FILE,
        low_memory=False,
    )

    return dataframe


# ============================================================
# FIND HISTORICAL ID COLUMN
# ============================================================


def find_historical_id_column(
    dataframe: pd.DataFrame,
) -> str:
    """
    Detect Transfermarkt club-ID column in the historical map.
    """

    candidates = [
        "transfermarkt_club_id",
        "club_id",
        "tm_club_id",
    ]

    for column in candidates:

        if column in dataframe.columns:

            return column

    raise RuntimeError(
        "Could not identify Transfermarkt club-ID column "
        "in historical map.\n"
        f"Columns: {list(dataframe.columns)}"
    )


# ============================================================
# CANDIDATE SEARCH
# ============================================================


def find_candidates_for_club(
    *,
    current_team: str,
    clubs: pd.DataFrame,
    used_historical_ids: set[int],
) -> pd.DataFrame:
    """
    Find deterministic candidate matches.

    Priority:

        1. exact raw name
        2. exact normalized name
        3. explicit search variant exact normalized match
        4. normalized containment candidate

    No candidate is automatically approved.
    """

    variants = SEARCH_VARIANTS[
        current_team
    ]

    normalized_variants = {
        normalize_text(
            value
        )
        for value in variants
    }

    rows = []

    for club in clubs.itertuples(
        index=False
    ):

        club_id = int(
            club.club_id
        )

        club_name = str(
            club.name
        )

        normalized_club = str(
            club.normalized_name
        )

        match_type = None
        priority = None
        matched_variant = None

        # ----------------------------------------------------
        # EXACT RAW
        # ----------------------------------------------------

        if club_name == current_team:

            match_type = (
                "EXACT_RAW_NAME"
            )

            priority = 1

            matched_variant = (
                current_team
            )

        # ----------------------------------------------------
        # EXACT NORMALIZED CURRENT NAME
        # ----------------------------------------------------

        elif (
            normalized_club
            ==
            normalize_text(
                current_team
            )
        ):

            match_type = (
                "EXACT_NORMALIZED_NAME"
            )

            priority = 2

            matched_variant = (
                current_team
            )

        # ----------------------------------------------------
        # EXPLICIT VARIANT
        # ----------------------------------------------------

        else:

            for variant in variants:

                if (
                    normalized_club
                    ==
                    normalize_text(
                        variant
                    )
                ):

                    match_type = (
                        "EXPLICIT_VARIANT"
                    )

                    priority = 3

                    matched_variant = (
                        variant
                    )

                    break

        # ----------------------------------------------------
        # CONTAINMENT
        # ----------------------------------------------------

        if match_type is None:

            for normalized_variant in normalized_variants:

                if (
                    normalized_variant
                    and
                    (
                        normalized_variant
                        in
                        normalized_club
                        or
                        normalized_club
                        in
                        normalized_variant
                    )
                ):

                    match_type = (
                        "CONTAINMENT_REVIEW"
                    )

                    priority = 4

                    matched_variant = (
                        normalized_variant
                    )

                    break

        if match_type is None:

            continue

        rows.append(
            {
                "current_team": (
                    current_team
                ),

                "transfermarkt_club_id": (
                    club_id
                ),

                "transfermarkt_name": (
                    club_name
                ),

                "match_type": (
                    match_type
                ),

                "priority": (
                    priority
                ),

                "matched_variant": (
                    matched_variant
                ),

                "already_used_by_historical_ucl_club": int(
                    club_id
                    in
                    used_historical_ids
                ),
            }
        )

    candidates = pd.DataFrame(
        rows
    )

    if candidates.empty:

        return pd.DataFrame(
            columns=[
                "current_team",
                "transfermarkt_club_id",
                "transfermarkt_name",
                "match_type",
                "priority",
                "matched_variant",
                "already_used_by_historical_ucl_club",
            ]
        )

    candidates = candidates.sort_values(
        by=[
            "priority",
            "already_used_by_historical_ucl_club",
            "transfermarkt_name",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return candidates


# ============================================================
# MAIN AUDIT
# ============================================================


def audit_current_transfermarkt_clubs() -> None:
    """
    Audit the six new 2026/27 UCL clubs against the local
    Transfermarkt clubs snapshot.

    This script DOES NOT modify the 117-club map.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - CURRENT TRANSFERMARKT CLUB AUDIT"
    )

    print(
        "=" * 75
    )

    current_clubs = (
        load_current_new_clubs()
    )

    transfermarkt_clubs = (
        load_transfermarkt_clubs()
    )

    historical_map = (
        load_historical_transfermarkt_map()
    )

    historical_id_column = (
        find_historical_id_column(
            historical_map
        )
    )

    used_historical_ids = set(
        pd.to_numeric(
            historical_map[
                historical_id_column
            ],
            errors="coerce",
        )
        .dropna()
        .astype(
            int
        )
    )

    print(
        f"\nNew current UCL clubs: "
        f"{len(current_clubs)}"
    )

    print(
        f"Transfermarkt clubs available: "
        f"{len(transfermarkt_clubs):,}"
    )

    print(
        f"Historical mapped Transfermarkt IDs: "
        f"{len(used_historical_ids)}"
    )

    all_candidates = []

    # --------------------------------------------------------
    # CLUB-BY-CLUB
    # --------------------------------------------------------

    for current_team in sorted(
        current_clubs[
            "canonical_team"
        ]
    ):

        candidates = (
            find_candidates_for_club(
                current_team=current_team,
                clubs=transfermarkt_clubs,
                used_historical_ids=used_historical_ids,
            )
        )

        print(
            "\n"
            + "-" * 75
        )

        print(
            current_team
        )

        print(
            "-" * 75
        )

        if candidates.empty:

            print(
                "No candidate found."
            )

            continue

        print(
            candidates[
                [
                    "transfermarkt_club_id",
                    "transfermarkt_name",
                    "match_type",
                    "already_used_by_historical_ucl_club",
                ]
            ]
            .head(
                15
            )
            .to_string(
                index=False
            )
        )

        all_candidates.append(
            candidates
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    if all_candidates:

        output = pd.concat(
            all_candidates,
            ignore_index=True,
        )

    else:

        output = pd.DataFrame(
            columns=[
                "current_team",
                "transfermarkt_club_id",
                "transfermarkt_name",
                "match_type",
                "priority",
                "matched_variant",
                "already_used_by_historical_ucl_club",
            ]
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    clubs_with_candidates = set(
        output[
            "current_team"
        ]
    )

    missing_candidates = (
        EXPECTED_NEW_CLUBS
        -
        clubs_with_candidates
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "AUDIT SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nNew clubs expected: "
        f"{len(EXPECTED_NEW_CLUBS)}"
    )

    print(
        f"Clubs with candidate(s): "
        f"{len(clubs_with_candidates)}"
    )

    print(
        f"Clubs without candidates: "
        f"{len(missing_candidates)}"
    )

    if missing_candidates:

        print(
            "\nNo candidates found for:"
        )

        for team in sorted(
            missing_candidates
        ):

            print(
                f"  {team}"
            )

    conflicts = output[
        output[
            "already_used_by_historical_ucl_club"
        ]
        ==
        1
    ]

    print(
        f"\nCandidate IDs already used by "
        f"historical map: "
        f"{len(conflicts)}"
    )

    print(
        "\nCandidate audit saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No Transfermarkt mappings were changed."
    )

    print(
        "Review candidate IDs before extending "
        "the historical club map."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    audit_current_transfermarkt_clubs()