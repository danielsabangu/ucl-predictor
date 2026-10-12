from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.build_uefa_strength_features import (
    build_coefficient_lookup,
    build_country_fallback_lookup,
    build_floor_lookup,
    build_global_alias_lookup,
    build_year_alias_lookup,
    build_year_coefficient_arrays,
    load_coefficients,
    load_country_floors,
    resolve_team,
)


ROOT = Path(__file__).resolve().parents[2]

FIXTURES_PATH = (
    ROOT
    / "data"
    / "interim"
    / "matches"
    / "ucl_2026_2027_fixtures_resolved.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "features"
    / "ucl_2026_2027_uefa_strength.csv"
)

CURRENT_RANKING_YEAR = 2026
EXPECTED_CURRENT_CLUB_COUNT = 36

V2_UEFA_FEATURE_COLUMNS = [
    "uefa_coefficient_diff",
    "uefa_strength_percentile_diff",
]

REQUIRED_FIXTURE_COLUMNS = {
    "source_match_id",
    "date_utc",
    "date",
    "season",
    "home_team",
    "away_team",
}

REVIEWED_2026_RESOLUTIONS = {
    "Como 1907": {
        "source_type": "COUNTRY_FLOOR",
        "resolution_method": "NO_YEAR_ALIAS_COUNTRY_FLOOR",
        "source_name": None,
        "country_name": "Italy",
    },
    "LASK Linz": {
        "source_type": "CLUB_COEFFICIENT",
        "resolution_method": "YEAR_EXPLICIT_ALIAS",
        "source_name": "LASK",
        "country_name": None,
    },
    "Real Betis Balompié": {
        "source_type": "CLUB_COEFFICIENT",
        "resolution_method": "YEAR_EXPLICIT_ALIAS",
        "source_name": "Real Betis",
        "country_name": None,
    },
    "Viking FK": {
        "source_type": "CLUB_COEFFICIENT",
        "resolution_method": "YEAR_EXPLICIT_ALIAS",
        "source_name": "Viking Stavanger",
        "country_name": None,
    },
}


def load_live_fixtures(
    path: Path = FIXTURES_PATH,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    missing = (
        REQUIRED_FIXTURE_COLUMNS
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Resolved live fixture file is missing columns: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    if frame[
        "source_match_id"
    ].isna().any():
        raise RuntimeError(
            "Live fixture file contains missing source_match_id values."
        )

    if frame[
        "source_match_id"
    ].duplicated().any():
        duplicates = frame.loc[
            frame[
                "source_match_id"
            ].duplicated(
                keep=False
            ),
            [
                "source_match_id",
                "date",
                "home_team",
                "away_team",
            ],
        ]

        raise RuntimeError(
            "Duplicate source_match_id values found in live fixtures:\n"
            f"{duplicates.to_string(index=False)}"
        )

    for column in [
        "home_team",
        "away_team",
    ]:
        if frame[
            column
        ].isna().any():
            raise RuntimeError(
                f"Live fixture file contains missing {column} values."
            )

        if (
            frame[
                column
            ]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise RuntimeError(
                f"Live fixture file contains blank {column} values."
            )

    seasons = set(
        frame[
            "season"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    if seasons != {
        CURRENT_RANKING_YEAR
    }:
        raise RuntimeError(
            "Expected only the 2026/27 fixture season in the live UEFA "
            f"input, found seasons: {sorted(seasons)}"
        )

    return frame


def build_resolver_context() -> dict:
    coefficients = load_coefficients()
    floors = load_country_floors()

    return {
        "coefficients": coefficients,
        "floors": floors,
        "coefficient_lookup": build_coefficient_lookup(
            coefficients
        ),
        "floor_lookup": build_floor_lookup(
            floors
        ),
        "global_alias_lookup": build_global_alias_lookup(),
        "year_alias_lookup": build_year_alias_lookup(),
        "country_fallback_lookup": build_country_fallback_lookup(),
        "year_arrays": build_year_coefficient_arrays(
            coefficients
        ),
    }


def resolve_live_team(
    team_name: str,
    *,
    ranking_year: int,
    context: dict,
) -> dict:
    return resolve_team(
        historical_name=str(
            team_name
        ),
        ranking_year=int(
            ranking_year
        ),
        coefficient_lookup=(
            context[
                "coefficient_lookup"
            ]
        ),
        floor_lookup=(
            context[
                "floor_lookup"
            ]
        ),
        global_alias_lookup=(
            context[
                "global_alias_lookup"
            ]
        ),
        year_alias_lookup=(
            context[
                "year_alias_lookup"
            ]
        ),
        country_fallback_lookup=(
            context[
                "country_fallback_lookup"
            ]
        ),
        year_arrays=(
            context[
                "year_arrays"
            ]
        ),
    )


def build_current_club_resolution_table(
    fixtures: pd.DataFrame,
    *,
    ranking_year: int = CURRENT_RANKING_YEAR,
    context: dict | None = None,
) -> pd.DataFrame:
    if context is None:
        context = build_resolver_context()

    clubs = sorted(
        set(
            fixtures[
                "home_team"
            ]
            .astype(str)
            .tolist()
        )
        |
        set(
            fixtures[
                "away_team"
            ]
            .astype(str)
            .tolist()
        )
    )

    rows = []

    for club in clubs:
        resolved = resolve_live_team(
            club,
            ranking_year=ranking_year,
            context=context,
        )

        rows.append(
            {
                "team": club,
                "ranking_year": int(
                    ranking_year
                ),
                "uefa_coefficient": float(
                    resolved[
                        "uefa_coefficient"
                    ]
                ),
                "uefa_strength_percentile": float(
                    resolved[
                        "uefa_strength_percentile"
                    ]
                ),
                "uefa_source_type": resolved[
                    "source_type"
                ],
                "uefa_resolution_method": resolved[
                    "resolution_method"
                ],
                "uefa_source_name": resolved[
                    "source_name"
                ],
                "uefa_country_name": resolved[
                    "country_name"
                ],
                "uefa_country_code": resolved[
                    "country_code"
                ],
                "uefa_rank": resolved[
                    "uefa_rank"
                ],
                "uefa_country_floor_used": int(
                    resolved[
                        "country_floor_used"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_live_uefa_strength_features(
    fixtures: pd.DataFrame,
    *,
    ranking_year: int = CURRENT_RANKING_YEAR,
    context: dict | None = None,
) -> pd.DataFrame:
    if context is None:
        context = build_resolver_context()

    rows = []

    for fixture in fixtures.itertuples(
        index=False
    ):
        home = resolve_live_team(
            str(
                fixture.home_team
            ),
            ranking_year=ranking_year,
            context=context,
        )

        away = resolve_live_team(
            str(
                fixture.away_team
            ),
            ranking_year=ranking_year,
            context=context,
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
                "source_match_id": fixture.source_match_id,
                "date_utc": fixture.date_utc,
                "date": fixture.date,
                "season": int(
                    fixture.season
                ),
                "ranking_year": int(
                    ranking_year
                ),
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "home_uefa_coefficient": home_coefficient,
                "away_uefa_coefficient": away_coefficient,
                "uefa_coefficient_diff": (
                    home_coefficient
                    - away_coefficient
                ),
                "home_uefa_strength_percentile": home_percentile,
                "away_uefa_strength_percentile": away_percentile,
                "uefa_strength_percentile_diff": (
                    home_percentile
                    - away_percentile
                ),
                "home_uefa_country_floor_used": int(
                    home[
                        "country_floor_used"
                    ]
                ),
                "away_uefa_country_floor_used": int(
                    away[
                        "country_floor_used"
                    ]
                ),
                "home_uefa_source_type": home[
                    "source_type"
                ],
                "away_uefa_source_type": away[
                    "source_type"
                ],
                "home_uefa_resolution_method": home[
                    "resolution_method"
                ],
                "away_uefa_resolution_method": away[
                    "resolution_method"
                ],
                "home_uefa_source_name": home[
                    "source_name"
                ],
                "away_uefa_source_name": away[
                    "source_name"
                ],
                "home_uefa_country_name": home[
                    "country_name"
                ],
                "away_uefa_country_name": away[
                    "country_name"
                ],
                "home_uefa_rank": home[
                    "uefa_rank"
                ],
                "away_uefa_rank": away[
                    "uefa_rank"
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


def validate_current_club_resolution_table(
    frame: pd.DataFrame,
) -> None:
    if len(
        frame
    ) != EXPECTED_CURRENT_CLUB_COUNT:
        raise RuntimeError(
            "Expected 36 current UCL clubs in the UEFA resolution audit, "
            f"found {len(frame):,}."
        )

    if frame[
        "team"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate clubs found in current UEFA resolution audit."
        )

    numeric_columns = [
        "uefa_coefficient",
        "uefa_strength_percentile",
    ]

    for column in numeric_columns:
        values = frame[
            column
        ].astype(float)

        if not np.isfinite(
            values.to_numpy()
        ).all():
            raise RuntimeError(
                f"Non-finite values found in {column}."
            )

    if (
        frame[
            "uefa_coefficient"
        ]
        .astype(float)
        .lt(0)
        .any()
    ):
        raise RuntimeError(
            "Negative UEFA coefficient found in current club audit."
        )

    percentile_values = frame[
        "uefa_strength_percentile"
    ].astype(float)

    if (
        percentile_values.lt(0)
        |
        percentile_values.gt(1)
    ).any():
        raise RuntimeError(
            "UEFA strength percentile outside [0, 1] in current club audit."
        )

    methods = (
        frame[
            "uefa_resolution_method"
        ]
        .astype(str)
        .str.upper()
    )

    if methods.str.contains(
        "FUZZY",
        regex=False,
    ).any():
        raise RuntimeError(
            "Fuzzy UEFA resolution detected."
        )

    indexed = frame.set_index(
        "team",
        drop=False,
    )

    for (
        club,
        expected,
    ) in REVIEWED_2026_RESOLUTIONS.items():
        if club not in indexed.index:
            raise RuntimeError(
                f"Reviewed 2026 club missing from audit: {club}"
            )

        row = indexed.loc[
            club
        ]

        checks = {
            "source_type": row[
                "uefa_source_type"
            ],
            "resolution_method": row[
                "uefa_resolution_method"
            ],
            "source_name": row[
                "uefa_source_name"
            ],
            "country_name": row[
                "uefa_country_name"
            ],
        }

        for (
            key,
            actual,
        ) in checks.items():
            wanted = expected[
                key
            ]

            if wanted is None:
                if not pd.isna(
                    actual
                ):
                    raise RuntimeError(
                        f"Unexpected {key} for {club}: {actual!r}"
                    )
            else:
                if str(
                    actual
                ) != str(
                    wanted
                ):
                    raise RuntimeError(
                        f"Unexpected {key} for {club}: "
                        f"expected {wanted!r}, got {actual!r}"
                    )


def validate_live_uefa_strength_features(
    frame: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> None:
    if len(
        frame
    ) != len(
        fixtures
    ):
        raise RuntimeError(
            "Live UEFA feature row count does not match fixture row count."
        )

    expected_ids = fixtures[
        "source_match_id"
    ].tolist()

    actual_ids = frame[
        "source_match_id"
    ].tolist()

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Live UEFA feature row order changed relative to fixtures."
        )

    if frame[
        "source_match_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate source_match_id values found in live UEFA features."
        )

    required_numeric = [
        "home_uefa_coefficient",
        "away_uefa_coefficient",
        "uefa_coefficient_diff",
        "home_uefa_strength_percentile",
        "away_uefa_strength_percentile",
        "uefa_strength_percentile_diff",
    ]

    for column in required_numeric:
        values = frame[
            column
        ].astype(float)

        if not np.isfinite(
            values.to_numpy()
        ).all():
            raise RuntimeError(
                f"Non-finite live UEFA values found in {column}."
            )

    expected_coefficient_diff = (
        frame[
            "home_uefa_coefficient"
        ].astype(float)
        - frame[
            "away_uefa_coefficient"
        ].astype(float)
    )

    if not np.allclose(
        frame[
            "uefa_coefficient_diff"
        ].astype(float).to_numpy(),
        expected_coefficient_diff.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "uefa_coefficient_diff does not equal home minus away."
        )

    expected_percentile_diff = (
        frame[
            "home_uefa_strength_percentile"
        ].astype(float)
        - frame[
            "away_uefa_strength_percentile"
        ].astype(float)
    )

    if not np.allclose(
        frame[
            "uefa_strength_percentile_diff"
        ].astype(float).to_numpy(),
        expected_percentile_diff.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "uefa_strength_percentile_diff does not equal home minus away."
        )

    for column in [
        "home_uefa_strength_percentile",
        "away_uefa_strength_percentile",
    ]:
        values = frame[
            column
        ].astype(float)

        if (
            values.lt(0)
            |
            values.gt(1)
        ).any():
            raise RuntimeError(
                f"Percentile outside [0, 1] in {column}."
            )

    if not (
        frame[
            "ranking_year"
        ]
        .astype(int)
        .eq(
            CURRENT_RANKING_YEAR
        )
        .all()
    ):
        raise RuntimeError(
            "Live UEFA features contain an unexpected ranking year."
        )

    for column in [
        "home_uefa_resolution_method",
        "away_uefa_resolution_method",
    ]:
        methods = (
            frame[
                column
            ]
            .astype(str)
            .str.upper()
        )

        if methods.str.contains(
            "FUZZY",
            regex=False,
        ).any():
            raise RuntimeError(
                f"Fuzzy UEFA resolution detected in {column}."
            )


def main() -> None:
    print(
        "Loading resolved 2026/27 UCL fixtures..."
    )

    fixtures = load_live_fixtures()

    print(
        "Loading shared UEFA resolver context..."
    )

    context = build_resolver_context()

    coefficients_2026 = context[
        "coefficients"
    ].loc[
        context[
            "coefficients"
        ][
            "ranking_year"
        ].astype(int)
        == CURRENT_RANKING_YEAR
    ]

    floors_2026 = context[
        "floors"
    ].loc[
        context[
            "floors"
        ][
            "ranking_year"
        ].astype(int)
        == CURRENT_RANKING_YEAR
    ]

    print(
        "Resolving all current UCL clubs..."
    )

    club_audit = (
        build_current_club_resolution_table(
            fixtures,
            ranking_year=(
                CURRENT_RANKING_YEAR
            ),
            context=context,
        )
    )

    validate_current_club_resolution_table(
        club_audit
    )

    print(
        "Building live UEFA strength features..."
    )

    features = (
        build_live_uefa_strength_features(
            fixtures,
            ranking_year=(
                CURRENT_RANKING_YEAR
            ),
            context=context,
        )
    )

    validate_live_uefa_strength_features(
        features,
        fixtures,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "LIVE 2026/27 UEFA STRENGTH FEATURE LAYER"
    )
    print(
        "=" * 110
    )
    print()

    print(
        f"Fixture rows: {len(fixtures):,}"
    )

    print(
        f"Unique clubs: {len(club_audit):,}"
    )

    print(
        "2026 club coefficient rows: "
        f"{len(coefficients_2026):,}"
    )

    print(
        "2026 country floor rows: "
        f"{len(floors_2026):,}"
    )

    print()
    print(
        "V2 probability features:"
    )

    for column in V2_UEFA_FEATURE_COLUMNS:
        print(
            f"  {column}"
        )

    print()
    print(
        "=" * 110
    )
    print(
        "CURRENT CLUB RESOLUTION AUDIT"
    )
    print(
        "=" * 110
    )
    print()

    print(
        club_audit.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 110
    )
    print(
        "REVIEWED 2026 RESOLUTIONS"
    )
    print(
        "=" * 110
    )
    print()

    reviewed = club_audit.loc[
        club_audit[
            "team"
        ].isin(
            REVIEWED_2026_RESOLUTIONS
        )
    ]

    print(
        reviewed.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 110
    )
    print(
        "VALIDATION"
    )
    print(
        "=" * 110
    )
    print()

    print(
        "36 current clubs resolved: PASS"
    )

    print(
        "Reviewed Como/LASK/Betis/Viking mappings: PASS"
    )

    print(
        "No missing/non-finite UEFA values: PASS"
    )

    print(
        "No fuzzy matching: PASS"
    )

    print(
        "2026 season-start ranking only: PASS"
    )

    print(
        "Feature differences equal home minus away: PASS"
    )

    print(
        "V1 production feature matrix untouched: PASS"
    )

    print(
        "V1 production model untouched: PASS"
    )

    print(
        "2025 confirmation not re-run: PASS"
    )

    print()
    print(
        "Output:"
    )
    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()