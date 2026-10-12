from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MATCHES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ucl_matches_master.csv"
)

COEFFICIENTS_PATH = (
    ROOT
    / "data"
    / "research"
    / "uefa_club_coefficients_2010_2026.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

MATCH_AUDIT_PATH = (
    REPORT_DIR
    / "uefa_coefficient_match_coverage.csv"
)

UNMATCHED_TEAM_YEAR_PATH = (
    REPORT_DIR
    / "uefa_coefficient_unmatched_team_years.csv"
)

UNMATCHED_UNIQUE_PATH = (
    REPORT_DIR
    / "uefa_coefficient_unmatched_unique_names.csv"
)

SEASON_COVERAGE_PATH = (
    REPORT_DIR
    / "uefa_coefficient_coverage_by_season.csv"
)


# ============================================================
# GLOBAL EXPLICIT ALIASES
# ============================================================
#
# historical master name -> Kassiesa source name
#
# These are deterministic mappings only.
#
# IMPORTANT:
#
# If an alias destination is absent in a particular ranking
# year, the team remains unmatched for that year.
#
# That does NOT automatically mean the alias is wrong.
# A club may genuinely have had no club-specific five-year
# coefficient in that preseason ranking.
#
# We therefore distinguish:
#
# EXACT_NORMALIZED
# EXPLICIT_ALIAS
# YEAR_EXPLICIT_ALIAS
# GLOBAL_ALIAS_TARGET_NOT_IN_YEAR
# YEAR_ALIAS_TARGET_NOT_IN_YEAR
# UNMATCHED_NO_ALIAS
#
# No fuzzy matching is used anywhere.
# ============================================================

EXPLICIT_ALIASES: dict[str, str] = {
    # Germany
    "Bayern Munich": "Bayern München",
    "Wolfsburg": "VfL Wolfsburg",
    "Hoffenheim": "1899 Hoffenheim",

    # Spain
    "Barcelona": "FC Barcelona",
    "Girona": "Girona FC",
    "Málaga": "Málaga CF",

    # Portugal
    "Porto": "FC Porto",
    "Sporting CP": "Sporting CP Lisbon",
    "Braga": "Sporting Braga",

    # Italy
    "Inter Milan": "Internazionale",
    "Roma": "AS Roma",

    # Greece
    "Olympiacos": "Olympiakos Piraeus",

    # Russia
    "Zenit": "Zenit St. Petersburg",
    "Krasnodar": "FK Krasnodar",

    # France
    "Monaco": "AS Monaco",
    "Lyon": "Olympique Lyon",
    "Marseille": "Olympique Marseille",
    "Lille": "Lille OSC",
    "Lens": "RC Lens",
    "Rennes": "Stade Rennais",
    "Auxerre": "AJ Auxerre",

    # Switzerland
    "Basel": "FC Basel",

    # Denmark
    "Copenhagen": "FC København",
    "Midtjylland": "FC Midtjylland",
    "Nordsjælland": "FC Nordsjælland",

    # Austria
    "RB Salzburg": "FC Salzburg",

    # Czechia
    "Slavia Prague": "Slavia Praha",
    "Sparta Prague": "Sparta Praha",
    "Viktoria Plzeň": "Viktoria Plzen",

    # Cyprus
    "APOEL": "APOEL Nicosia",

    # Turkey
    "Beşiktaş": "Besiktas",
    "İstanbul Başakşehir": "Istanbul Basaksehir",

    # Sweden
    "Malmö": "Malmö FF",

    # Azerbaijan
    "Qarabağ": "Qarabag FK",

    # Belgium
    "Genk": "Racing Genk",
    "Gent": "AA Gent",

    # Bulgaria
    "Ludogorets": "Ludogorets Razgrad",

    # Scotland
    "Rangers": "Glasgow Rangers",

    # Israel
    "Maccabi Tel Aviv": "Maccabi Tel-Aviv",
    "Hapoel Tel Aviv": "Hapoel Tel-Aviv",

    # Kazakhstan
    "Astana": "FK Astana",
    "FK Kairat": "Kairat Almaty",

    # Slovakia
    "ŠK Slovan Bratislava": "Slovan Bratislava",

    # Serbia
    "Partizan": "Partizan Belgrade",

    # Netherlands
    "Twente": "FC Twente Enschede",

    # Romania
    "Steaua": "Steaua Bucuresti",
}


# ============================================================
# YEAR-SPECIFIC ALIASES
# ============================================================
#
# Some source names changed historically.
#
# The key is:
#
#     (historical master name, ranking year)
#
# This is checked BEFORE the global alias map.
#
# Dynamo Kyiv is the first confirmed case.
# Kassiesa historically used "Dinamo Kiev".
# ============================================================

YEAR_EXPLICIT_ALIASES: dict[
    tuple[str, int],
    str,
] = {
    ("Dynamo Kyiv", 2012): "Dinamo Kiev",
    ("Dynamo Kyiv", 2015): "Dinamo Kiev",
    ("Dynamo Kyiv", 2016): "Dinamo Kiev",
    ("Dynamo Kyiv", 2020): "Dinamo Kiev",
    ("Dynamo Kyiv", 2021): "Dinamo Kiev",
}


def normalize_exact_name(
    value: str,
) -> str:
    """
    Deterministic normalization only.

    This is NOT fuzzy matching.

    We:
    - normalize Unicode representation
    - normalize non-breaking spaces
    - trim surrounding whitespace
    - collapse repeated whitespace
    - casefold

    We deliberately DO NOT:
    - remove accents
    - remove punctuation
    - use edit distance
    - guess abbreviations
    - search for similar names
    """

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = text.replace(
        "\u00a0",
        " ",
    )

    text = " ".join(
        text.strip().split()
    )

    return text.casefold()


def parse_season_start(
    value,
) -> int:
    """
    Convert the historical season representation
    into the season starting year.
    """

    if pd.isna(
        value
    ):
        raise ValueError(
            "Historical match has a missing season."
        )

    if isinstance(
        value,
        (int, float),
    ):
        numeric = float(
            value
        )

        if numeric.is_integer():
            year = int(
                numeric
            )

            if 1900 <= year <= 2100:
                return year

    text = str(
        value
    ).strip()

    numeric_match = re.fullmatch(
        r"(\d{4})(?:\.0+)?",
        text,
    )

    if numeric_match:
        return int(
            numeric_match.group(1)
        )

    year_match = re.search(
        r"(?<!\d)(20\d{2})(?!\d)",
        text,
    )

    if year_match:
        return int(
            year_match.group(1)
        )

    raise ValueError(
        "Could not derive season start year "
        f"from: {value!r}"
    )


def load_matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            "Historical master not found: "
            f"{MATCHES_PATH}"
        )

    frame = pd.read_csv(
        MATCHES_PATH,
        low_memory=False,
    )

    required = {
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Historical master is missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        frame
    ) != 2089:
        raise RuntimeError(
            "Historical master row count changed. "
            f"Expected 2089, observed {len(frame)}."
        )

    frame = frame.copy()

    frame[
        "ranking_year"
    ] = frame[
        "season"
    ].apply(
        parse_season_start
    )

    return frame


def load_coefficients() -> pd.DataFrame:
    if not COEFFICIENTS_PATH.exists():
        raise FileNotFoundError(
            "Coefficient dataset not found: "
            f"{COEFFICIENTS_PATH}"
        )

    frame = pd.read_csv(
        COEFFICIENTS_PATH,
        low_memory=False,
    )

    required = {
        "ranking_year",
        "uefa_rank",
        "uefa_rank_percentile",
        "club_source",
        "country_code",
        "uefa_coefficient",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Coefficient dataset is missing columns: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    frame[
        "ranking_year"
    ] = (
        frame[
            "ranking_year"
        ]
        .astype(
            int
        )
    )

    frame[
        "source_match_key"
    ] = frame[
        "club_source"
    ].apply(
        normalize_exact_name
    )

    duplicate_mask = (
        frame.duplicated(
            subset=[
                "ranking_year",
                "source_match_key",
            ],
            keep=False,
        )
    )

    if duplicate_mask.any():
        duplicates = (
            frame.loc[
                duplicate_mask,
                [
                    "ranking_year",
                    "club_source",
                    "country_code",
                ],
            ]
            .sort_values(
                [
                    "ranking_year",
                    "club_source",
                ]
            )
        )

        raise RuntimeError(
            "Coefficient lookup contains duplicate "
            "normalized club/year keys:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return frame


def build_lookup(
    coefficients: pd.DataFrame,
) -> dict[
    tuple[int, str],
    dict,
]:
    lookup: dict[
        tuple[int, str],
        dict,
    ] = {}

    for row in coefficients.itertuples(
        index=False
    ):
        key = (
            int(
                row.ranking_year
            ),
            str(
                row.source_match_key
            ),
        )

        lookup[
            key
        ] = {
            "club_source": (
                row.club_source
            ),
            "country_code": (
                row.country_code
            ),
            "uefa_rank": int(
                row.uefa_rank
            ),
            "uefa_rank_percentile": float(
                row.uefa_rank_percentile
            ),
            "uefa_coefficient": float(
                row.uefa_coefficient
            ),
        }

    return lookup


def build_global_name_inventory(
    coefficients: pd.DataFrame,
) -> set[str]:
    return set(
        coefficients[
            "source_match_key"
        ].astype(
            str
        )
    )


def build_global_alias_lookup() -> dict[
    str,
    str,
]:
    aliases = {}

    for (
        historical_name,
        source_name,
    ) in EXPLICIT_ALIASES.items():
        key = normalize_exact_name(
            historical_name
        )

        if key in aliases:
            raise RuntimeError(
                "Duplicate normalized global alias: "
                f"{historical_name}"
            )

        aliases[
            key
        ] = source_name

    return aliases


def build_year_alias_lookup() -> dict[
    tuple[str, int],
    str,
]:
    aliases = {}

    for (
        historical_name,
        ranking_year,
    ), source_name in (
        YEAR_EXPLICIT_ALIASES.items()
    ):
        key = (
            normalize_exact_name(
                historical_name
            ),
            int(
                ranking_year
            ),
        )

        if key in aliases:
            raise RuntimeError(
                "Duplicate normalized year alias: "
                f"{historical_name}, "
                f"{ranking_year}"
            )

        aliases[
            key
        ] = source_name

    return aliases


def matched_result(
    *,
    method: str,
    source: dict,
    alias_candidate: str | None = None,
) -> dict:
    return {
        "matched": True,
        "match_method": (
            method
        ),
        "alias_candidate": (
            alias_candidate
        ),
        "resolved_source_name": (
            source[
                "club_source"
            ]
        ),
        "country_code": (
            source[
                "country_code"
            ]
        ),
        "uefa_rank": (
            source[
                "uefa_rank"
            ]
        ),
        "uefa_rank_percentile": (
            source[
                "uefa_rank_percentile"
            ]
        ),
        "uefa_coefficient": (
            source[
                "uefa_coefficient"
            ]
        ),
    }


def unmatched_result(
    *,
    method: str,
    alias_candidate: str | None = None,
) -> dict:
    return {
        "matched": False,
        "match_method": (
            method
        ),
        "alias_candidate": (
            alias_candidate
        ),
        "resolved_source_name": None,
        "country_code": None,
        "uefa_rank": None,
        "uefa_rank_percentile": None,
        "uefa_coefficient": None,
    }


def resolve_team(
    *,
    historical_name: str,
    ranking_year: int,
    coefficient_lookup: dict,
    global_alias_lookup: dict,
    year_alias_lookup: dict,
) -> dict:
    """
    Resolve one historical club against one
    coefficient year.

    Resolution order:

    1. Exact normalized name
    2. Year-specific reviewed alias
    3. Global reviewed alias
    4. Unmatched

    A configured alias whose target is absent
    in a particular ranking year does NOT crash.

    It is recorded separately because the club
    may genuinely have been unranked that year.
    """

    historical_key = (
        normalize_exact_name(
            historical_name
        )
    )

    direct_key = (
        ranking_year,
        historical_key,
    )

    direct = (
        coefficient_lookup.get(
            direct_key
        )
    )

    if direct is not None:
        return matched_result(
            method=(
                "EXACT_NORMALIZED"
            ),
            source=direct,
        )

    year_alias_name = (
        year_alias_lookup.get(
            (
                historical_key,
                ranking_year,
            )
        )
    )

    if year_alias_name is not None:
        year_alias_key = (
            ranking_year,
            normalize_exact_name(
                year_alias_name
            ),
        )

        resolved = (
            coefficient_lookup.get(
                year_alias_key
            )
        )

        if resolved is not None:
            return matched_result(
                method=(
                    "YEAR_EXPLICIT_ALIAS"
                ),
                source=resolved,
                alias_candidate=(
                    year_alias_name
                ),
            )

        return unmatched_result(
            method=(
                "YEAR_ALIAS_TARGET_NOT_IN_YEAR"
            ),
            alias_candidate=(
                year_alias_name
            ),
        )

    global_alias_name = (
        global_alias_lookup.get(
            historical_key
        )
    )

    if global_alias_name is not None:
        alias_key = (
            ranking_year,
            normalize_exact_name(
                global_alias_name
            ),
        )

        resolved = (
            coefficient_lookup.get(
                alias_key
            )
        )

        if resolved is not None:
            return matched_result(
                method=(
                    "EXPLICIT_ALIAS"
                ),
                source=resolved,
                alias_candidate=(
                    global_alias_name
                ),
            )

        return unmatched_result(
            method=(
                "GLOBAL_ALIAS_TARGET_NOT_IN_YEAR"
            ),
            alias_candidate=(
                global_alias_name
            ),
        )

    return unmatched_result(
        method=(
            "UNMATCHED_NO_ALIAS"
        ),
    )


def validate_ranking_years(
    matches: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    required_years = set(
        matches[
            "ranking_year"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    available_years = set(
        coefficients[
            "ranking_year"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    missing = (
        required_years
        - available_years
    )

    if missing:
        raise RuntimeError(
            "Historical matches require coefficient "
            "years that are unavailable: "
            f"{sorted(missing)}"
        )


def audit_alias_configuration(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check whether configured alias destinations
    exist anywhere in the coefficient dataset.

    This does not require a destination to exist
    in every year.
    """

    global_inventory = (
        build_global_name_inventory(
            coefficients
        )
    )

    rows = []

    for (
        historical_name,
        source_name,
    ) in sorted(
        EXPLICIT_ALIASES.items()
    ):
        target_key = (
            normalize_exact_name(
                source_name
            )
        )

        rows.append(
            {
                "alias_type": (
                    "GLOBAL"
                ),
                "historical_name": (
                    historical_name
                ),
                "ranking_year": None,
                "source_name": (
                    source_name
                ),
                "source_exists_somewhere": (
                    target_key
                    in global_inventory
                ),
                "source_exists_in_year": None,
            }
        )

    coefficient_lookup = (
        build_lookup(
            coefficients
        )
    )

    for (
        historical_name,
        ranking_year,
    ), source_name in sorted(
        YEAR_EXPLICIT_ALIASES.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
        ),
    ):
        target_key = (
            normalize_exact_name(
                source_name
            )
        )

        exists_in_year = (
            (
                int(
                    ranking_year
                ),
                target_key,
            )
            in coefficient_lookup
        )

        rows.append(
            {
                "alias_type": (
                    "YEAR"
                ),
                "historical_name": (
                    historical_name
                ),
                "ranking_year": int(
                    ranking_year
                ),
                "source_name": (
                    source_name
                ),
                "source_exists_somewhere": (
                    target_key
                    in global_inventory
                ),
                "source_exists_in_year": (
                    exists_in_year
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def audit_matches(
    matches: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    coefficient_lookup = (
        build_lookup(
            coefficients
        )
    )

    global_alias_lookup = (
        build_global_alias_lookup()
    )

    year_alias_lookup = (
        build_year_alias_lookup()
    )

    rows = []

    for match in matches.itertuples(
        index=False
    ):
        ranking_year = int(
            match.ranking_year
        )

        home = resolve_team(
            historical_name=str(
                match.home_team
            ),
            ranking_year=(
                ranking_year
            ),
            coefficient_lookup=(
                coefficient_lookup
            ),
            global_alias_lookup=(
                global_alias_lookup
            ),
            year_alias_lookup=(
                year_alias_lookup
            ),
        )

        away = resolve_team(
            historical_name=str(
                match.away_team
            ),
            ranking_year=(
                ranking_year
            ),
            coefficient_lookup=(
                coefficient_lookup
            ),
            global_alias_lookup=(
                global_alias_lookup
            ),
            year_alias_lookup=(
                year_alias_lookup
            ),
        )

        rows.append(
            {
                "match_id": (
                    match.match_id
                ),
                "date": (
                    match.date
                ),
                "season": (
                    match.season
                ),
                "ranking_year": (
                    ranking_year
                ),

                "home_team": (
                    match.home_team
                ),
                "away_team": (
                    match.away_team
                ),

                "home_matched": (
                    home[
                        "matched"
                    ]
                ),
                "away_matched": (
                    away[
                        "matched"
                    ]
                ),

                "fully_matched": (
                    home[
                        "matched"
                    ]
                    and away[
                        "matched"
                    ]
                ),

                "home_match_method": (
                    home[
                        "match_method"
                    ]
                ),
                "away_match_method": (
                    away[
                        "match_method"
                    ]
                ),

                "home_alias_candidate": (
                    home[
                        "alias_candidate"
                    ]
                ),
                "away_alias_candidate": (
                    away[
                        "alias_candidate"
                    ]
                ),

                "home_coefficient_source_name": (
                    home[
                        "resolved_source_name"
                    ]
                ),
                "away_coefficient_source_name": (
                    away[
                        "resolved_source_name"
                    ]
                ),

                "home_country_code": (
                    home[
                        "country_code"
                    ]
                ),
                "away_country_code": (
                    away[
                        "country_code"
                    ]
                ),

                "home_uefa_rank": (
                    home[
                        "uefa_rank"
                    ]
                ),
                "away_uefa_rank": (
                    away[
                        "uefa_rank"
                    ]
                ),

                "home_uefa_rank_percentile": (
                    home[
                        "uefa_rank_percentile"
                    ]
                ),
                "away_uefa_rank_percentile": (
                    away[
                        "uefa_rank_percentile"
                    ]
                ),

                "home_uefa_coefficient": (
                    home[
                        "uefa_coefficient"
                    ]
                ),
                "away_uefa_coefficient": (
                    away[
                        "uefa_coefficient"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_unmatched_team_years(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for row in audit.itertuples(
        index=False
    ):
        if not row.home_matched:
            rows.append(
                {
                    "ranking_year": (
                        row.ranking_year
                    ),
                    "historical_team": (
                        row.home_team
                    ),
                    "side": "HOME",
                    "resolution_reason": (
                        row.home_match_method
                    ),
                    "alias_candidate": (
                        row.home_alias_candidate
                    ),
                }
            )

        if not row.away_matched:
            rows.append(
                {
                    "ranking_year": (
                        row.ranking_year
                    ),
                    "historical_team": (
                        row.away_team
                    ),
                    "side": "AWAY",
                    "resolution_reason": (
                        row.away_match_method
                    ),
                    "alias_candidate": (
                        row.away_alias_candidate
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ranking_year",
                "historical_team",
                "resolution_reason",
                "alias_candidate",
                "appearances",
                "home_appearances",
                "away_appearances",
            ]
        )

    unmatched = pd.DataFrame(
        rows
    )

    output_rows = []

    group_columns = [
        "ranking_year",
        "historical_team",
        "resolution_reason",
        "alias_candidate",
    ]

    for keys, group in unmatched.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        (
            ranking_year,
            historical_team,
            resolution_reason,
            alias_candidate,
        ) = keys

        side_counts = Counter(
            group[
                "side"
            ].tolist()
        )

        output_rows.append(
            {
                "ranking_year": int(
                    ranking_year
                ),
                "historical_team": (
                    historical_team
                ),
                "resolution_reason": (
                    resolution_reason
                ),
                "alias_candidate": (
                    alias_candidate
                    if pd.notna(
                        alias_candidate
                    )
                    else None
                ),
                "appearances": len(
                    group
                ),
                "home_appearances": (
                    side_counts.get(
                        "HOME",
                        0,
                    )
                ),
                "away_appearances": (
                    side_counts.get(
                        "AWAY",
                        0,
                    )
                ),
            }
        )

    return (
        pd.DataFrame(
            output_rows
        )
        .sort_values(
            [
                "appearances",
                "historical_team",
                "ranking_year",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def build_unmatched_unique(
    unmatched_team_years: pd.DataFrame,
) -> pd.DataFrame:
    if unmatched_team_years.empty:
        return pd.DataFrame(
            columns=[
                "historical_team",
                "total_unmatched_appearances",
                "first_ranking_year",
                "last_ranking_year",
                "ranking_years",
                "resolution_reasons",
            ]
        )

    rows = []

    for (
        historical_team,
        group,
    ) in unmatched_team_years.groupby(
        "historical_team",
        sort=True,
    ):
        years = sorted(
            group[
                "ranking_year"
            ]
            .astype(
                int
            )
            .unique()
            .tolist()
        )

        reasons = sorted(
            group[
                "resolution_reason"
            ]
            .dropna()
            .astype(
                str
            )
            .unique()
            .tolist()
        )

        rows.append(
            {
                "historical_team": (
                    historical_team
                ),
                "total_unmatched_appearances": int(
                    group[
                        "appearances"
                    ].sum()
                ),
                "first_ranking_year": (
                    min(
                        years
                    )
                ),
                "last_ranking_year": (
                    max(
                        years
                    )
                ),
                "ranking_years": (
                    ",".join(
                        str(
                            year
                        )
                        for year
                        in years
                    )
                ),
                "resolution_reasons": (
                    ",".join(
                        reasons
                    )
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "total_unmatched_appearances",
                "historical_team",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def build_season_coverage(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        ranking_year,
        group,
    ) in audit.groupby(
        "ranking_year",
        sort=True,
    ):
        matches = len(
            group
        )

        team_sides = (
            matches
            * 2
        )

        home_matched = int(
            group[
                "home_matched"
            ].sum()
        )

        away_matched = int(
            group[
                "away_matched"
            ].sum()
        )

        matched_sides = (
            home_matched
            + away_matched
        )

        fully_matched = int(
            group[
                "fully_matched"
            ].sum()
        )

        rows.append(
            {
                "ranking_year": int(
                    ranking_year
                ),
                "matches": (
                    matches
                ),
                "team_sides": (
                    team_sides
                ),
                "matched_team_sides": (
                    matched_sides
                ),
                "team_side_coverage": (
                    matched_sides
                    / team_sides
                    if team_sides
                    else 0.0
                ),
                "fully_matched_matches": (
                    fully_matched
                ),
                "full_match_coverage": (
                    fully_matched
                    / matches
                    if matches
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_alias_configuration(
    alias_audit: pd.DataFrame,
) -> None:
    print()
    print(
        "=" * 100
    )
    print(
        "ALIAS CONFIGURATION AUDIT"
    )
    print(
        "=" * 100
    )
    print()

    print(
        f"Global aliases configured: "
        f"{len(EXPLICIT_ALIASES)}"
    )

    print(
        f"Year aliases configured: "
        f"{len(YEAR_EXPLICIT_ALIASES)}"
    )

    if alias_audit.empty:
        return

    invalid_global = alias_audit[
        (
            alias_audit[
                "alias_type"
            ]
            == "GLOBAL"
        )
        & (
            alias_audit[
                "source_exists_somewhere"
            ]
            == False
        )
    ]

    invalid_year = alias_audit[
        (
            alias_audit[
                "alias_type"
            ]
            == "YEAR"
        )
        & (
            alias_audit[
                "source_exists_in_year"
            ]
            == False
        )
    ]

    print()
    print(
        "Global alias destinations never "
        "seen in any ranking:"
    )

    if invalid_global.empty:
        print(
            "  None"
        )
    else:
        for row in invalid_global.itertuples(
            index=False
        ):
            print(
                f"  {row.historical_name} "
                f"-> {row.source_name}"
            )

    print()
    print(
        "Year aliases whose destination "
        "is absent in that exact year:"
    )

    if invalid_year.empty:
        print(
            "  None"
        )
    else:
        for row in invalid_year.itertuples(
            index=False
        ):
            print(
                f"  {row.ranking_year}: "
                f"{row.historical_name} "
                f"-> {row.source_name}"
            )


def main() -> None:
    print(
        "Loading historical UCL master..."
    )

    matches = (
        load_matches()
    )

    print(
        "Loading UEFA coefficient dataset..."
    )

    coefficients = (
        load_coefficients()
    )

    validate_ranking_years(
        matches,
        coefficients,
    )

    alias_audit = (
        audit_alias_configuration(
            coefficients
        )
    )

    print(
        "Auditing deterministic team-name coverage..."
    )

    audit = audit_matches(
        matches,
        coefficients,
    )

    unmatched_team_years = (
        build_unmatched_team_years(
            audit
        )
    )

    unmatched_unique = (
        build_unmatched_unique(
            unmatched_team_years
        )
    )

    season_coverage = (
        build_season_coverage(
            audit
        )
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        MATCH_AUDIT_PATH,
        index=False,
    )

    unmatched_team_years.to_csv(
        UNMATCHED_TEAM_YEAR_PATH,
        index=False,
    )

    unmatched_unique.to_csv(
        UNMATCHED_UNIQUE_PATH,
        index=False,
    )

    season_coverage.to_csv(
        SEASON_COVERAGE_PATH,
        index=False,
    )

    total_matches = len(
        audit
    )

    total_sides = (
        total_matches
        * 2
    )

    home_matched = int(
        audit[
            "home_matched"
        ].sum()
    )

    away_matched = int(
        audit[
            "away_matched"
        ].sum()
    )

    matched_sides = (
        home_matched
        + away_matched
    )

    fully_matched = int(
        audit[
            "fully_matched"
        ].sum()
    )

    unique_historical_clubs = (
        pd.concat(
            [
                matches[
                    "home_team"
                ],
                matches[
                    "away_team"
                ],
            ],
            ignore_index=True,
        )
        .nunique()
    )

    print()
    print(
        "=" * 100
    )
    print(
        "UEFA COEFFICIENT MATCH COVERAGE"
    )
    print(
        "=" * 100
    )
    print()

    print(
        f"Historical matches: "
        f"{total_matches:,}"
    )

    print(
        f"Historical team-sides: "
        f"{total_sides:,}"
    )

    print(
        f"Unique historical clubs: "
        f"{unique_historical_clubs:,}"
    )

    print()

    print(
        f"Matched home sides: "
        f"{home_matched:,}/"
        f"{total_matches:,} "
        f"({home_matched / total_matches:.2%})"
    )

    print(
        f"Matched away sides: "
        f"{away_matched:,}/"
        f"{total_matches:,} "
        f"({away_matched / total_matches:.2%})"
    )

    print(
        f"Matched all team-sides: "
        f"{matched_sides:,}/"
        f"{total_sides:,} "
        f"({matched_sides / total_sides:.2%})"
    )

    print(
        f"Fully matched fixtures: "
        f"{fully_matched:,}/"
        f"{total_matches:,} "
        f"({fully_matched / total_matches:.2%})"
    )

    print(
        "Unmatched unique historical names: "
        f"{len(unmatched_unique):,}"
    )

    print()
    print(
        "=" * 100
    )
    print(
        "COVERAGE BY SEASON"
    )
    print(
        "=" * 100
    )
    print()

    display_season = (
        season_coverage.copy()
    )

    display_season[
        "team_side_coverage"
    ] = display_season[
        "team_side_coverage"
    ].map(
        lambda value: (
            f"{value:.2%}"
        )
    )

    display_season[
        "full_match_coverage"
    ] = display_season[
        "full_match_coverage"
    ].map(
        lambda value: (
            f"{value:.2%}"
        )
    )

    print(
        display_season.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "UNMATCHED HISTORICAL CLUB NAMES"
    )
    print(
        "=" * 100
    )
    print()

    if unmatched_unique.empty:
        print(
            "None."
        )
    else:
        print(
            unmatched_unique.to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 100
    )
    print(
        "UNMATCHED TEAM-YEAR DETAILS"
    )
    print(
        "=" * 100
    )
    print()

    if unmatched_team_years.empty:
        print(
            "None."
        )
    else:
        print(
            unmatched_team_years.to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 100
    )
    print(
        "MATCH METHODS"
    )
    print(
        "=" * 100
    )
    print()

    methods = pd.concat(
        [
            audit[
                "home_match_method"
            ],
            audit[
                "away_match_method"
            ],
        ],
        ignore_index=True,
    )

    print(
        methods.value_counts().to_string()
    )

    print_alias_configuration(
        alias_audit
    )

    print()
    print(
        "=" * 100
    )
    print(
        "OUTPUTS"
    )
    print(
        "=" * 100
    )
    print()

    print(
        MATCH_AUDIT_PATH
    )

    print(
        UNMATCHED_TEAM_YEAR_PATH
    )

    print(
        UNMATCHED_UNIQUE_PATH
    )

    print(
        SEASON_COVERAGE_PATH
    )

    print()
    print(
        "No fuzzy matching was used."
    )

    print(
        "No country-floor values were applied yet."
    )

    print(
        "No model or production feature "
        "files were modified."
    )


if __name__ == "__main__":
    main()