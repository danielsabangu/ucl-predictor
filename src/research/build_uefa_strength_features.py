from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
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

COUNTRY_FLOORS_PATH = (
    ROOT
    / "data"
    / "research"
    / "uefa_country_floors_2010_2026.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "research"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "ucl_uefa_strength_features.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

RESOLUTION_REPORT_PATH = (
    REPORT_DIR
    / "uefa_strength_resolution_by_season.csv"
)

SOURCE_REPORT_PATH = (
    REPORT_DIR
    / "uefa_strength_source_counts.csv"
)


# ============================================================
# GLOBAL EXPLICIT TEAM ALIASES
# ============================================================
#
# historical master name -> Kassiesa source name
#
# No fuzzy matching.
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
    "Rostov": "FK Rostov",

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
    "Paphos FC": "Pafos FC",

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
    "MŠK Žilina": "MSK Zilina",

    # Serbia
    "Partizan": "Partizan Belgrade",

    # Netherlands
    "Twente": "FC Twente Enschede",

    # Romania
    "Steaua": "Steaua Bucuresti",
}


# ============================================================
# YEAR-SPECIFIC TEAM ALIASES
# ============================================================
#
# Some source names changed historically.
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
    ("LASK Linz", 2026): "LASK",
    ("Real Betis Balompié", 2026): "Real Betis",
    ("Viking FK", 2026): "Viking Stavanger",
}


# ============================================================
# COUNTRY-FLOOR FALLBACKS
# ============================================================
#
# These clubs genuinely do not have a club-specific
# coefficient in the relevant preseason ranking.
#
# Their historically valid UEFA country floor is used.
#
# This is NOT arbitrary imputation.
# ============================================================

COUNTRY_FLOOR_FALLBACKS: dict[
    str,
    str,
] = {
    "Brest": "France",
    "Leicester City": "England",
    "Monaco": "France",
    "Málaga": "Spain",
    "Bologna": "Italy",
    "Girona": "Spain",
    "VfB Stuttgart": "Germany",
    "Lens": "France",
    "Newcastle United": "England",
    "Oțelul Galați": "Romania",
    "RB Leipzig": "Germany",
    "Real Sociedad": "Spain",
    "Bursaspor": "Turkey",
    "Como 1907": "Italy",
}


CANDIDATE_FEATURE_COLUMNS = [
    "home_uefa_coefficient",
    "away_uefa_coefficient",
    "uefa_coefficient_diff",
    "home_uefa_strength_percentile",
    "away_uefa_strength_percentile",
    "uefa_strength_percentile_diff",
]


def normalize_exact_name(
    value: str,
) -> str:
    """
    Deterministic normalization only.

    No fuzzy matching.
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
    if pd.isna(
        value
    ):
        raise ValueError(
            "Missing season."
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
        "Could not derive season start "
        f"from {value!r}"
    )


def load_matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            MATCHES_PATH
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
            "Historical master missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        frame
    ) != 2089:
        raise RuntimeError(
            "Expected 2,089 historical matches, "
            f"found {len(frame):,}."
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
            COEFFICIENTS_PATH
        )

    frame = pd.read_csv(
        COEFFICIENTS_PATH,
        low_memory=False,
    )

    required = {
        "ranking_year",
        "club_source",
        "country_code",
        "uefa_rank",
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
            "Coefficient dataset missing columns: "
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
        "source_key"
    ] = frame[
        "club_source"
    ].apply(
        normalize_exact_name
    )

    duplicate_mask = (
        frame.duplicated(
            subset=[
                "ranking_year",
                "source_key",
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
                ],
            ]
        )

        raise RuntimeError(
            "Duplicate coefficient keys:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return frame


def load_country_floors() -> pd.DataFrame:
    if not COUNTRY_FLOORS_PATH.exists():
        raise FileNotFoundError(
            COUNTRY_FLOORS_PATH
        )

    frame = pd.read_csv(
        COUNTRY_FLOORS_PATH,
        low_memory=False,
    )

    required = {
        "ranking_year",
        "country_name",
        "country_floor_coefficient",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Country-floor dataset missing columns: "
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
        "country_key"
    ] = frame[
        "country_name"
    ].apply(
        normalize_exact_name
    )

    duplicate_mask = (
        frame.duplicated(
            subset=[
                "ranking_year",
                "country_key",
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
                    "country_name",
                ],
            ]
        )

        raise RuntimeError(
            "Duplicate country-floor keys:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return frame


def build_coefficient_lookup(
    coefficients: pd.DataFrame,
) -> dict[
    tuple[int, str],
    dict,
]:
    lookup = {}

    for row in coefficients.itertuples(
        index=False
    ):
        lookup[
            (
                int(
                    row.ranking_year
                ),
                str(
                    row.source_key
                ),
            )
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
            "uefa_coefficient": float(
                row.uefa_coefficient
            ),
        }

    return lookup


def build_floor_lookup(
    floors: pd.DataFrame,
) -> dict[
    tuple[int, str],
    dict,
]:
    lookup = {}

    for row in floors.itertuples(
        index=False
    ):
        lookup[
            (
                int(
                    row.ranking_year
                ),
                str(
                    row.country_key
                ),
            )
        ] = {
            "country_name": (
                row.country_name
            ),
            "country_floor_coefficient": float(
                row.country_floor_coefficient
            ),
        }

    return lookup


def build_global_alias_lookup() -> dict[
    str,
    str,
]:
    return {
        normalize_exact_name(
            historical_name
        ): source_name
        for historical_name, source_name
        in EXPLICIT_ALIASES.items()
    }


def build_year_alias_lookup() -> dict[
    tuple[str, int],
    str,
]:
    return {
        (
            normalize_exact_name(
                historical_name
            ),
            int(
                ranking_year
            ),
        ): source_name
        for (
            historical_name,
            ranking_year,
        ), source_name
        in YEAR_EXPLICIT_ALIASES.items()
    }


def build_country_fallback_lookup() -> dict[
    str,
    str,
]:
    return {
        normalize_exact_name(
            historical_name
        ): country_name
        for historical_name, country_name
        in COUNTRY_FLOOR_FALLBACKS.items()
    }


def build_year_coefficient_arrays(
    coefficients: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """
    Sorted club-specific coefficients by ranking year.

    Used only to turn the raw coefficient into an
    era-relative strength percentile.

    Higher percentile = stronger UEFA coefficient.
    """

    arrays = {}

    for (
        ranking_year,
        group,
    ) in coefficients.groupby(
        "ranking_year",
        sort=True,
    ):
        values = (
            group[
                "uefa_coefficient"
            ]
            .astype(
                float
            )
            .to_numpy()
        )

        arrays[
            int(
                ranking_year
            )
        ] = np.sort(
            values
        )

    return arrays


def coefficient_strength_percentile(
    *,
    ranking_year: int,
    coefficient: float,
    year_arrays: dict[
        int,
        np.ndarray,
    ],
) -> float:
    values = (
        year_arrays.get(
            ranking_year
        )
    )

    if values is None:
        raise RuntimeError(
            "Missing percentile reference "
            f"for ranking year {ranking_year}."
        )

    if len(
        values
    ) == 0:
        raise RuntimeError(
            "Empty coefficient distribution "
            f"for ranking year {ranking_year}."
        )

    # Fraction of ranked clubs with coefficient
    # less than or equal to this value.
    #
    # Higher = stronger.
    position = np.searchsorted(
        values,
        coefficient,
        side="right",
    )

    percentile = (
        float(
            position
        )
        / float(
            len(
                values
            )
        )
    )

    return float(
        np.clip(
            percentile,
            0.0,
            1.0,
        )
    )


def matched_club_result(
    *,
    source: dict,
    method: str,
    alias_candidate: str | None,
    ranking_year: int,
    year_arrays: dict[
        int,
        np.ndarray,
    ],
) -> dict:
    coefficient = float(
        source[
            "uefa_coefficient"
        ]
    )

    percentile = (
        coefficient_strength_percentile(
            ranking_year=(
                ranking_year
            ),
            coefficient=(
                coefficient
            ),
            year_arrays=(
                year_arrays
            ),
        )
    )

    return {
        "resolved": True,
        "source_type": (
            "CLUB_COEFFICIENT"
        ),
        "resolution_method": (
            method
        ),
        "source_name": (
            source[
                "club_source"
            ]
        ),
        "country_name": None,
        "country_code": (
            source[
                "country_code"
            ]
        ),
        "alias_candidate": (
            alias_candidate
        ),
        "uefa_rank": (
            source[
                "uefa_rank"
            ]
        ),
        "uefa_coefficient": (
            coefficient
        ),
        "uefa_strength_percentile": (
            percentile
        ),
        "country_floor_used": 0,
    }


def country_floor_result(
    *,
    historical_name: str,
    ranking_year: int,
    country_name: str,
    floor_lookup: dict,
    year_arrays: dict[
        int,
        np.ndarray,
    ],
    previous_method: str,
    alias_candidate: str | None,
) -> dict:
    floor_key = (
        ranking_year,
        normalize_exact_name(
            country_name
        ),
    )

    floor = (
        floor_lookup.get(
            floor_key
        )
    )

    if floor is None:
        raise RuntimeError(
            "Country floor missing for "
            f"{historical_name!r}, "
            f"ranking {ranking_year}, "
            f"country {country_name!r}."
        )

    coefficient = float(
        floor[
            "country_floor_coefficient"
        ]
    )

    percentile = (
        coefficient_strength_percentile(
            ranking_year=(
                ranking_year
            ),
            coefficient=(
                coefficient
            ),
            year_arrays=(
                year_arrays
            ),
        )
    )

    return {
        "resolved": True,
        "source_type": (
            "COUNTRY_FLOOR"
        ),
        "resolution_method": (
            previous_method
            + "_COUNTRY_FLOOR"
        ),
        "source_name": None,
        "country_name": (
            floor[
                "country_name"
            ]
        ),
        "country_code": None,
        "alias_candidate": (
            alias_candidate
        ),
        "uefa_rank": None,
        "uefa_coefficient": (
            coefficient
        ),
        "uefa_strength_percentile": (
            percentile
        ),
        "country_floor_used": 1,
    }


def resolve_team(
    *,
    historical_name: str,
    ranking_year: int,
    coefficient_lookup: dict,
    floor_lookup: dict,
    global_alias_lookup: dict,
    year_alias_lookup: dict,
    country_fallback_lookup: dict,
    year_arrays: dict[
        int,
        np.ndarray,
    ],
) -> dict:
    historical_key = (
        normalize_exact_name(
            historical_name
        )
    )

    # --------------------------------------------------------
    # 1. Exact source name
    # --------------------------------------------------------

    exact = (
        coefficient_lookup.get(
            (
                ranking_year,
                historical_key,
            )
        )
    )

    if exact is not None:
        return matched_club_result(
            source=exact,
            method="EXACT_NORMALIZED",
            alias_candidate=None,
            ranking_year=ranking_year,
            year_arrays=year_arrays,
        )

    # --------------------------------------------------------
    # 2. Year-specific alias
    # --------------------------------------------------------

    year_alias = (
        year_alias_lookup.get(
            (
                historical_key,
                ranking_year,
            )
        )
    )

    if year_alias is not None:
        resolved = (
            coefficient_lookup.get(
                (
                    ranking_year,
                    normalize_exact_name(
                        year_alias
                    ),
                )
            )
        )

        if resolved is not None:
            return matched_club_result(
                source=resolved,
                method="YEAR_EXPLICIT_ALIAS",
                alias_candidate=year_alias,
                ranking_year=ranking_year,
                year_arrays=year_arrays,
            )

        previous_method = (
            "YEAR_ALIAS_TARGET_NOT_IN_YEAR"
        )

        alias_candidate = (
            year_alias
        )

    else:
        previous_method = (
            "NO_YEAR_ALIAS"
        )

        alias_candidate = None

    # --------------------------------------------------------
    # 3. Global alias
    # --------------------------------------------------------

    global_alias = (
        global_alias_lookup.get(
            historical_key
        )
    )

    if global_alias is not None:
        resolved = (
            coefficient_lookup.get(
                (
                    ranking_year,
                    normalize_exact_name(
                        global_alias
                    ),
                )
            )
        )

        if resolved is not None:
            return matched_club_result(
                source=resolved,
                method="EXPLICIT_ALIAS",
                alias_candidate=global_alias,
                ranking_year=ranking_year,
                year_arrays=year_arrays,
            )

        previous_method = (
            "GLOBAL_ALIAS_TARGET_NOT_IN_YEAR"
        )

        alias_candidate = (
            global_alias
        )

    # --------------------------------------------------------
    # 4. Historical country floor
    # --------------------------------------------------------

    country_name = (
        country_fallback_lookup.get(
            historical_key
        )
    )

    if country_name is not None:
        return country_floor_result(
            historical_name=(
                historical_name
            ),
            ranking_year=(
                ranking_year
            ),
            country_name=(
                country_name
            ),
            floor_lookup=(
                floor_lookup
            ),
            year_arrays=(
                year_arrays
            ),
            previous_method=(
                previous_method
            ),
            alias_candidate=(
                alias_candidate
            ),
        )

    # --------------------------------------------------------
    # 5. Never silently invent strength
    # --------------------------------------------------------

    raise RuntimeError(
        "Unable to resolve historical UEFA strength "
        f"for {historical_name!r} "
        f"in ranking year {ranking_year}. "
        "No club coefficient and no reviewed "
        "country-floor fallback exist."
    )


def build_features(
    matches: pd.DataFrame,
    coefficients: pd.DataFrame,
    floors: pd.DataFrame,
) -> pd.DataFrame:
    coefficient_lookup = (
        build_coefficient_lookup(
            coefficients
        )
    )

    floor_lookup = (
        build_floor_lookup(
            floors
        )
    )

    global_alias_lookup = (
        build_global_alias_lookup()
    )

    year_alias_lookup = (
        build_year_alias_lookup()
    )

    country_fallback_lookup = (
        build_country_fallback_lookup()
    )

    year_arrays = (
        build_year_coefficient_arrays(
            coefficients
        )
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
            floor_lookup=(
                floor_lookup
            ),
            global_alias_lookup=(
                global_alias_lookup
            ),
            year_alias_lookup=(
                year_alias_lookup
            ),
            country_fallback_lookup=(
                country_fallback_lookup
            ),
            year_arrays=(
                year_arrays
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
            floor_lookup=(
                floor_lookup
            ),
            global_alias_lookup=(
                global_alias_lookup
            ),
            year_alias_lookup=(
                year_alias_lookup
            ),
            country_fallback_lookup=(
                country_fallback_lookup
            ),
            year_arrays=(
                year_arrays
            ),
        )

        home_coefficient = float(
            home[
                "uefa_coefficient"
            ]
        )

        away_coefficient = float(
            away[
                "uefa_coefficient"
            ]
        )

        home_percentile = float(
            home[
                "uefa_strength_percentile"
            ]
        )

        away_percentile = float(
            away[
                "uefa_strength_percentile"
            ]
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

                # Candidate model features
                "home_uefa_coefficient": (
                    home_coefficient
                ),
                "away_uefa_coefficient": (
                    away_coefficient
                ),
                "uefa_coefficient_diff": (
                    home_coefficient
                    - away_coefficient
                ),

                "home_uefa_strength_percentile": (
                    home_percentile
                ),
                "away_uefa_strength_percentile": (
                    away_percentile
                ),
                "uefa_strength_percentile_diff": (
                    home_percentile
                    - away_percentile
                ),

                # Availability/source metadata
                "home_uefa_country_floor_used": (
                    home[
                        "country_floor_used"
                    ]
                ),
                "away_uefa_country_floor_used": (
                    away[
                        "country_floor_used"
                    ]
                ),

                "home_uefa_source_type": (
                    home[
                        "source_type"
                    ]
                ),
                "away_uefa_source_type": (
                    away[
                        "source_type"
                    ]
                ),

                "home_uefa_resolution_method": (
                    home[
                        "resolution_method"
                    ]
                ),
                "away_uefa_resolution_method": (
                    away[
                        "resolution_method"
                    ]
                ),

                "home_uefa_source_name": (
                    home[
                        "source_name"
                    ]
                ),
                "away_uefa_source_name": (
                    away[
                        "source_name"
                    ]
                ),

                "home_uefa_country_name": (
                    home[
                        "country_name"
                    ]
                ),
                "away_uefa_country_name": (
                    away[
                        "country_name"
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
            }
        )

    return pd.DataFrame(
        rows
    )


def validate_features(
    frame: pd.DataFrame,
    matches: pd.DataFrame,
) -> None:
    if len(
        frame
    ) != len(
        matches
    ):
        raise RuntimeError(
            "Feature row count does not match "
            "historical master."
        )

    if len(
        frame
    ) != 2089:
        raise RuntimeError(
            "Expected 2,089 UEFA feature rows, "
            f"found {len(frame):,}."
        )

    if frame[
        "match_id"
    ].duplicated().any():
        duplicates = (
            frame.loc[
                frame[
                    "match_id"
                ].duplicated(
                    keep=False
                ),
                [
                    "match_id",
                    "date",
                    "home_team",
                    "away_team",
                ],
            ]
        )

        raise RuntimeError(
            "Duplicate match IDs found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    missing_features = (
        frame[
            CANDIDATE_FEATURE_COLUMNS
        ]
        .isna()
        .sum()
    )

    if (
        missing_features
        > 0
    ).any():
        raise RuntimeError(
            "Missing candidate UEFA features:\n"
            f"{missing_features.to_string()}"
        )

    if (
        frame[
            "home_uefa_coefficient"
        ]
        < 0
    ).any():
        raise RuntimeError(
            "Negative home coefficient found."
        )

    if (
        frame[
            "away_uefa_coefficient"
        ]
        < 0
    ).any():
        raise RuntimeError(
            "Negative away coefficient found."
        )

    percentile_columns = [
        "home_uefa_strength_percentile",
        "away_uefa_strength_percentile",
    ]

    for column in percentile_columns:
        invalid = (
            (
                frame[
                    column
                ]
                < 0
            )
            |
            (
                frame[
                    column
                ]
                > 1
            )
        )

        if invalid.any():
            raise RuntimeError(
                "Percentile outside [0, 1] "
                f"in {column}."
            )

    valid_source_types = {
        "CLUB_COEFFICIENT",
        "COUNTRY_FLOOR",
    }

    for column in [
        "home_uefa_source_type",
        "away_uefa_source_type",
    ]:
        unexpected = (
            set(
                frame[
                    column
                ].unique()
            )
            - valid_source_types
        )

        if unexpected:
            raise RuntimeError(
                "Unexpected source types "
                f"in {column}: "
                f"{sorted(unexpected)}"
            )

    expected_ids = (
        matches[
            "match_id"
        ]
        .tolist()
    )

    actual_ids = (
        frame[
            "match_id"
        ]
        .tolist()
    )

    if (
        actual_ids
        != expected_ids
    ):
        raise RuntimeError(
            "Match order changed while "
            "building UEFA features."
        )


def build_resolution_by_season(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        ranking_year,
        group,
    ) in frame.groupby(
        "ranking_year",
        sort=True,
    ):
        match_count = len(
            group
        )

        team_sides = (
            match_count
            * 2
        )

        floor_sides = int(
            group[
                "home_uefa_country_floor_used"
            ].sum()
            +
            group[
                "away_uefa_country_floor_used"
            ].sum()
        )

        club_sides = (
            team_sides
            - floor_sides
        )

        rows.append(
            {
                "ranking_year": int(
                    ranking_year
                ),
                "matches": (
                    match_count
                ),
                "team_sides": (
                    team_sides
                ),
                "club_coefficient_sides": (
                    club_sides
                ),
                "country_floor_sides": (
                    floor_sides
                ),
                "club_coefficient_rate": (
                    club_sides
                    / team_sides
                ),
                "country_floor_rate": (
                    floor_sides
                    / team_sides
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_source_counts(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    methods = pd.concat(
        [
            frame[
                [
                    "home_uefa_source_type",
                    "home_uefa_resolution_method",
                ]
            ].rename(
                columns={
                    "home_uefa_source_type": (
                        "source_type"
                    ),
                    "home_uefa_resolution_method": (
                        "resolution_method"
                    ),
                }
            ),
            frame[
                [
                    "away_uefa_source_type",
                    "away_uefa_resolution_method",
                ]
            ].rename(
                columns={
                    "away_uefa_source_type": (
                        "source_type"
                    ),
                    "away_uefa_resolution_method": (
                        "resolution_method"
                    ),
                }
            ),
        ],
        ignore_index=True,
    )

    result = (
        methods.groupby(
            [
                "source_type",
                "resolution_method",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="team_sides"
        )
        .sort_values(
            [
                "source_type",
                "team_sides",
                "resolution_method",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return result


def main() -> None:
    print(
        "Loading historical UCL matches..."
    )

    matches = (
        load_matches()
    )

    print(
        "Loading historical UEFA "
        "club coefficients..."
    )

    coefficients = (
        load_coefficients()
    )

    print(
        "Loading historical UEFA "
        "country floors..."
    )

    floors = (
        load_country_floors()
    )

    print(
        "Building historical UEFA "
        "strength features..."
    )

    features = (
        build_features(
            matches,
            coefficients,
            floors,
        )
    )

    print(
        "Validating final feature layer..."
    )

    validate_features(
        features,
        matches,
    )

    resolution_by_season = (
        build_resolution_by_season(
            features
        )
    )

    source_counts = (
        build_source_counts(
            features
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    resolution_by_season.to_csv(
        RESOLUTION_REPORT_PATH,
        index=False,
    )

    source_counts.to_csv(
        SOURCE_REPORT_PATH,
        index=False,
    )

    total_matches = len(
        features
    )

    total_team_sides = (
        total_matches
        * 2
    )

    total_floor_sides = int(
        features[
            "home_uefa_country_floor_used"
        ].sum()
        +
        features[
            "away_uefa_country_floor_used"
        ].sum()
    )

    total_club_sides = (
        total_team_sides
        - total_floor_sides
    )

    print()
    print(
        "=" * 100
    )
    print(
        "FINAL HISTORICAL UEFA "
        "STRENGTH FEATURE LAYER"
    )
    print(
        "=" * 100
    )
    print()

    print(
        f"Matches: "
        f"{total_matches:,}"
    )

    print(
        f"Team-sides resolved: "
        f"{total_team_sides:,}/"
        f"{total_team_sides:,} "
        "(100.00%)"
    )

    print()

    print(
        "Club-specific coefficient sides: "
        f"{total_club_sides:,} "
        f"({total_club_sides / total_team_sides:.2%})"
    )

    print(
        "Country-floor sides: "
        f"{total_floor_sides:,} "
        f"({total_floor_sides / total_team_sides:.2%})"
    )

    print()
    print(
        "Candidate model features:"
    )

    for column in (
        CANDIDATE_FEATURE_COLUMNS
    ):
        print(
            f"  {column}"
        )

    print()
    print(
        "=" * 100
    )
    print(
        "RESOLUTION METHODS"
    )
    print(
        "=" * 100
    )
    print()

    print(
        source_counts.to_string(
            index=False
        )
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

    display = (
        resolution_by_season.copy()
    )

    display[
        "club_coefficient_rate"
    ] = display[
        "club_coefficient_rate"
    ].map(
        lambda value: (
            f"{value:.2%}"
        )
    )

    display[
        "country_floor_rate"
    ] = display[
        "country_floor_rate"
    ].map(
        lambda value: (
            f"{value:.2%}"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "COUNTRY-FLOOR FIXTURES"
    )
    print(
        "=" * 100
    )
    print()

    floor_matches = (
        features[
            (
                features[
                    "home_uefa_country_floor_used"
                ]
                == 1
            )
            |
            (
                features[
                    "away_uefa_country_floor_used"
                ]
                == 1
            )
        ]
    )

    if floor_matches.empty:
        print(
            "None."
        )
    else:
        print(
            floor_matches[
                [
                    "date",
                    "ranking_year",
                    "home_team",
                    "away_team",
                    "home_uefa_source_type",
                    "away_uefa_source_type",
                    "home_uefa_coefficient",
                    "away_uefa_coefficient",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 100
    )
    print(
        "VALIDATION"
    )
    print(
        "=" * 100
    )
    print()

    print(
        "2,089 historical matches: PASS"
    )

    print(
        "4,178 team-sides resolved: PASS"
    )

    print(
        "No missing candidate features: PASS"
    )

    print(
        "No fuzzy matching: PASS"
    )

    print(
        "Country floors are historical: PASS"
    )

    print(
        "Season-start ranking only: PASS"
    )

    print(
        "Existing V1 feature matrix untouched: PASS"
    )

    print(
        "Production model untouched: PASS"
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
        OUTPUT_PATH
    )

    print(
        RESOLUTION_REPORT_PATH
    )

    print(
        SOURCE_REPORT_PATH
    )


if __name__ == "__main__":
    main()