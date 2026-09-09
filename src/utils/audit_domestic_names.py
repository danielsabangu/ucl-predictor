from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
)

from src.utils.team_names import (
    normalize_team_name,
    alias_lookup_key,
)


# ============================================================
# FILE PATHS
# ============================================================

UCL_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

DOMESTIC_FILE = (
    INTERIM_DATA_DIR
    / "matches"
    / "domestic_matches.csv"
)

OUTPUT_FILE = (
    REPORTS_DIR
    / "metrics"
    / "domestic_team_name_audit.csv"
)


# ============================================================
# SETTINGS
# ============================================================

FUZZY_DISPLAY_THRESHOLD = 0.60

HIGH_CONFIDENCE_THRESHOLD = 0.84


# ============================================================
# TEAM COLLECTION
# ============================================================


def get_team_names(
    dataframe: pd.DataFrame,
) -> set[str]:
    """
    Return all unique home and away team names.
    """

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

        raise ValueError(
            f"Missing team columns: "
            f"{sorted(missing)}"
        )

    home = set(
        dataframe[
            "home_team"
        ]
        .dropna()
        .astype(str)
    )

    away = set(
        dataframe[
            "away_team"
        ]
        .dropna()
        .astype(str)
    )

    return home.union(
        away
    )


# ============================================================
# FUZZY MATCHING
# ============================================================


def similarity_score(
    first: str,
    second: str,
) -> float:
    """
    Compare two club names after building normalization keys.

    Returns a score from 0 to 1.
    """

    first_key = alias_lookup_key(
        first
    )

    second_key = alias_lookup_key(
        second
    )

    return SequenceMatcher(
        None,
        first_key,
        second_key,
    ).ratio()


def best_ucl_matches(
    domestic_name: str,
    ucl_teams: set[str],
    limit: int = 3,
) -> list[tuple[str, float]]:
    """
    Return the closest UCL club names for one domestic club.
    """

    scores = []

    for ucl_team in ucl_teams:

        score = similarity_score(
            domestic_name,
            ucl_team,
        )

        scores.append(
            (
                ucl_team,
                score,
            )
        )

    scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scores[
        :limit
    ]


# ============================================================
# CLUB SEASON COVERAGE
# ============================================================


def build_domestic_season_lookup(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
    """
    Build a readable season range for every raw domestic club.
    """

    rows = []

    for row in dataframe.itertuples(
        index=False
    ):

        rows.append(
            (
                row.home_team,
                int(
                    row.season
                ),
            )
        )

        rows.append(
            (
                row.away_team,
                int(
                    row.season
                ),
            )
        )

    team_seasons: dict[
        str,
        set[int],
    ] = {}

    for team, season in rows:

        team = str(
            team
        )

        if team not in team_seasons:

            team_seasons[
                team
            ] = set()

        team_seasons[
            team
        ].add(
            season
        )

    result = {}

    for team, seasons in team_seasons.items():

        minimum = min(
            seasons
        )

        maximum = max(
            seasons
        )

        result[
            team
        ] = (
            f"{minimum}/"
            f"{str(minimum + 1)[-2:]}"
            f" - "
            f"{maximum}/"
            f"{str(maximum + 1)[-2:]}"
        )

    return result


# ============================================================
# AUDIT
# ============================================================


def build_name_audit(
    ucl: pd.DataFrame,
    domestic: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare every raw domestic team name against our UCL
    canonical club identities.
    """

    ucl_teams_raw = get_team_names(
        ucl
    )

    # Normalize again for safety.
    ucl_teams = {
        normalize_team_name(
            team
        )
        for team in ucl_teams_raw
    }

    domestic_teams = sorted(
        get_team_names(
            domestic
        )
    )

    domestic_seasons = (
        build_domestic_season_lookup(
            domestic
        )
    )

    rows = []

    for raw_name in domestic_teams:

        normalized_name = (
            normalize_team_name(
                raw_name
            )
        )

        exact_ucl_match = (
            normalized_name
            in ucl_teams
        )

        suggestions = best_ucl_matches(
            normalized_name,
            ucl_teams,
            limit=3,
        )

        best_name = (
            suggestions[0][0]
            if suggestions
            else ""
        )

        best_score = (
            suggestions[0][1]
            if suggestions
            else 0.0
        )

        second_name = (
            suggestions[1][0]
            if len(suggestions) > 1
            else ""
        )

        second_score = (
            suggestions[1][1]
            if len(suggestions) > 1
            else 0.0
        )

        third_name = (
            suggestions[2][0]
            if len(suggestions) > 2
            else ""
        )

        third_score = (
            suggestions[2][1]
            if len(suggestions) > 2
            else 0.0
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if exact_ucl_match:

            status = "MATCHED"

        elif (
            best_score
            >= HIGH_CONFIDENCE_THRESHOLD
        ):

            status = (
                "LIKELY_ALIAS"
            )

        elif (
            best_score
            >= FUZZY_DISPLAY_THRESHOLD
        ):

            status = (
                "REVIEW"
            )

        else:

            status = (
                "NO_UCL_MATCH"
            )

        rows.append(
            {
                "domestic_raw_name": raw_name,
                "normalized_candidate": normalized_name,
                "domestic_season_range": (
                    domestic_seasons.get(
                        raw_name,
                        ""
                    )
                ),
                "exact_ucl_match": exact_ucl_match,
                "status": status,
                "best_ucl_suggestion": best_name,
                "best_score": round(
                    best_score,
                    4,
                ),
                "second_ucl_suggestion": second_name,
                "second_score": round(
                    second_score,
                    4,
                ),
                "third_ucl_suggestion": third_name,
                "third_score": round(
                    third_score,
                    4,
                ),
            }
        )

    audit = pd.DataFrame(
        rows
    )

    return audit


# ============================================================
# CROSS-SOURCE VARIANT AUDIT
# ============================================================


def print_domestic_variants(
    audit: pd.DataFrame,
) -> None:
    """
    Print clubs where the raw domestic name changes under our
    existing normalization rules.
    """

    changed = audit[
        audit[
            "domestic_raw_name"
        ]
        != audit[
            "normalized_candidate"
        ]
    ].copy()

    print(
        "\n"
        + "=" * 75
    )

    print(
        "EXISTING NORMALIZATION RULES"
    )

    print(
        "=" * 75
    )

    print(
        f"\nDomestic names already normalized: "
        f"{len(changed):,}"
    )

    if changed.empty:

        print(
            "\nNo current aliases were applied."
        )

        return

    for row in changed.itertuples(
        index=False
    ):

        print(
            f"{row.domestic_raw_name:<40} "
            f"-> "
            f"{row.normalized_candidate}"
        )


# ============================================================
# LIKELY MISSING ALIASES
# ============================================================


def print_likely_aliases(
    audit: pd.DataFrame,
) -> None:
    """
    Print high-confidence domestic/UCL fuzzy matches that are
    not yet covered by TEAM_ALIASES.
    """

    likely = audit[
        audit[
            "status"
        ]
        ==
        "LIKELY_ALIAS"
    ].copy()

    likely = likely.sort_values(
        by="best_score",
        ascending=False,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIKELY MISSING TEAM ALIASES"
    )

    print(
        "=" * 75
    )

    print(
        f"\nHigh-confidence candidates: "
        f"{len(likely):,}"
    )

    if likely.empty:

        print(
            "\nNo high-confidence missing aliases."
        )

        return

    for row in likely.itertuples(
        index=False
    ):

        print(
            f"{row.domestic_raw_name:<38} "
            f"-> "
            f"{row.best_ucl_suggestion:<28} "
            f"{row.best_score:.3f}"
        )


# ============================================================
# REVIEW CANDIDATES
# ============================================================


def print_review_candidates(
    audit: pd.DataFrame,
) -> None:
    """
    Print medium-confidence matches requiring human review.
    """

    review = audit[
        audit[
            "status"
        ]
        ==
        "REVIEW"
    ].copy()

    review = review.sort_values(
        by="best_score",
        ascending=False,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "MANUAL REVIEW CANDIDATES"
    )

    print(
        "=" * 75
    )

    print(
        f"\nCandidates: "
        f"{len(review):,}"
    )

    if review.empty:

        print(
            "\nNo medium-confidence candidates."
        )

        return

    for row in review.itertuples(
        index=False
    ):

        print(
            f"\n{row.domestic_raw_name}"
        )

        print(
            f"  1. "
            f"{row.best_ucl_suggestion:<30} "
            f"{row.best_score:.3f}"
        )

        print(
            f"  2. "
            f"{row.second_ucl_suggestion:<30} "
            f"{row.second_score:.3f}"
        )

        print(
            f"  3. "
            f"{row.third_ucl_suggestion:<30} "
            f"{row.third_score:.3f}"
        )


# ============================================================
# MATCHED UCL COVERAGE
# ============================================================


def print_ucl_coverage(
    audit: pd.DataFrame,
    ucl: pd.DataFrame,
) -> None:
    """
    Show how many UCL canonical clubs can currently be linked
    to at least one domestic source name.
    """

    ucl_teams = {
        normalize_team_name(
            team
        )
        for team in get_team_names(
            ucl
        )
    }

    matched = set(
        audit.loc[
            audit[
                "status"
            ]
            ==
            "MATCHED",
            "normalized_candidate",
        ]
    )

    likely = set(
        audit.loc[
            audit[
                "status"
            ]
            ==
            "LIKELY_ALIAS",
            "best_ucl_suggestion",
        ]
    )

    currently_linked = (
        matched
        | likely
    )

    linked_ucl = (
        ucl_teams
        & currently_linked
    )

    missing_ucl = sorted(
        ucl_teams
        - currently_linked
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "UCL CLUB DOMESTIC COVERAGE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nUCL clubs: "
        f"{len(ucl_teams):,}"
    )

    print(
        f"Matched / likely domestic link: "
        f"{len(linked_ucl):,}"
    )

    coverage_percentage = (
        len(linked_ucl)
        / len(ucl_teams)
        * 100
    )

    print(
        f"Current identity coverage: "
        f"{coverage_percentage:.2f}%"
    )

    print(
        f"UCL clubs without a domestic link: "
        f"{len(missing_ucl):,}"
    )

    if missing_ucl:

        print(
            "\nUnlinked UCL clubs:"
        )

        for team in missing_ucl:

            print(
                f"  {team}"
            )


# ============================================================
# MAIN
# ============================================================


def run_domestic_name_audit() -> pd.DataFrame:
    """
    Audit domestic team identities against our UCL dataset.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - DOMESTIC TEAM IDENTITY AUDIT"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # FILE CHECKS
    # --------------------------------------------------------

    print(
        f"\nLoading UCL dataset:\n"
        f"{UCL_FILE}"
    )

    if not UCL_FILE.exists():

        raise FileNotFoundError(
            f"UCL dataset not found:\n"
            f"{UCL_FILE}"
        )

    print(
        f"\nLoading domestic dataset:\n"
        f"{DOMESTIC_FILE}"
    )

    if not DOMESTIC_FILE.exists():

        raise FileNotFoundError(
            f"Domestic dataset not found:\n"
            f"{DOMESTIC_FILE}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    ucl = pd.read_csv(
        UCL_FILE
    )

    domestic = pd.read_csv(
        DOMESTIC_FILE
    )

    print(
        f"\nUCL matches loaded: "
        f"{len(ucl):,}"
    )

    print(
        f"Domestic matches loaded: "
        f"{len(domestic):,}"
    )

    print(
        f"UCL team names: "
        f"{len(get_team_names(ucl)):,}"
    )

    print(
        f"Raw domestic team names: "
        f"{len(get_team_names(domestic)):,}"
    )

    # --------------------------------------------------------
    # BUILD AUDIT
    # --------------------------------------------------------

    audit = build_name_audit(
        ucl,
        domestic,
    )

    # --------------------------------------------------------
    # SUMMARY COUNTS
    # --------------------------------------------------------

    status_counts = (
        audit[
            "status"
        ]
        .value_counts()
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "AUDIT STATUS SUMMARY"
    )

    print(
        "=" * 75
    )

    for status in [
        "MATCHED",
        "LIKELY_ALIAS",
        "REVIEW",
        "NO_UCL_MATCH",
    ]:

        print(
            f"{status:<18} "
            f"{int(status_counts.get(status, 0)):,}"
        )

    # --------------------------------------------------------
    # DETAIL OUTPUT
    # --------------------------------------------------------

    print_domestic_variants(
        audit
    )

    print_likely_aliases(
        audit
    )

    print_review_candidates(
        audit
    )

    print_ucl_coverage(
        audit,
        ucl,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "AUDIT SAVED"
    )

    print(
        "=" * 75
    )

    print(
        f"\n{OUTPUT_FILE}"
    )

    print(
        "\nNo match files were modified."
    )

    return audit


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    run_domestic_name_audit()