from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.live_uefa_strength import (
    CURRENT_RANKING_YEAR,
    REVIEWED_2026_RESOLUTIONS,
    V2_UEFA_FEATURE_COLUMNS,
    build_current_club_resolution_table,
    build_live_uefa_strength_features,
    build_resolver_context,
    validate_current_club_resolution_table,
    validate_live_uefa_strength_features,
)


CURRENT_2026_27_CLUBS = [
    "AEK Athens",
    "Arsenal",
    "Aston Villa",
    "Atlético Madrid",
    "Barcelona",
    "Bayern Munich",
    "Bodø/Glimt",
    "Borussia Dortmund",
    "Club Brugge",
    "Como 1907",
    "Fenerbahçe",
    "Feyenoord",
    "Galatasaray",
    "Inter Milan",
    "LASK Linz",
    "Lens",
    "Lille",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Napoli",
    "PSV Eindhoven",
    "Paris Saint-Germain",
    "Porto",
    "RB Leipzig",
    "Real Betis Balompié",
    "Real Madrid",
    "Roma",
    "Sabah FK",
    "Shakhtar Donetsk",
    "Slavia Prague",
    "Sporting CP",
    "VfB Stuttgart",
    "Viking FK",
    "Villarreal",
    "ŠK Slovan Bratislava",
]


EXPECTED_REVIEWED_VALUES = {
    "Como 1907": {
        "coefficient": 19.989,
        "percentile": 0.7590361445783133,
        "source_type": "COUNTRY_FLOOR",
        "resolution_method": "NO_YEAR_ALIAS_COUNTRY_FLOOR",
        "source_name": None,
        "country_name": "Italy",
        "country_floor_used": 1,
    },
    "LASK Linz": {
        "coefficient": 21.0,
        "percentile": 0.7686746987951807,
        "source_type": "CLUB_COEFFICIENT",
        "resolution_method": "YEAR_EXPLICIT_ALIAS",
        "source_name": "LASK",
        "country_name": None,
        "country_floor_used": 0,
    },
    "Real Betis Balompié": {
        "coefficient": 74.5,
        "percentile": 0.944578313253012,
        "source_type": "CLUB_COEFFICIENT",
        "resolution_method": "YEAR_EXPLICIT_ALIAS",
        "source_name": "Real Betis",
        "country_name": None,
        "country_floor_used": 0,
    },
    "Viking FK": {
        "coefficient": 4.5,
        "percentile": 0.41204819277108434,
        "source_type": "CLUB_COEFFICIENT",
        "resolution_method": "YEAR_EXPLICIT_ALIAS",
        "source_name": "Viking Stavanger",
        "country_name": None,
        "country_floor_used": 0,
    },
}


def build_synthetic_current_fixtures() -> pd.DataFrame:
    if len(
        CURRENT_2026_27_CLUBS
    ) != 36:
        raise RuntimeError(
            "Expected exactly 36 current clubs in test fixture."
        )

    rows = []

    for index in range(
        0,
        len(
            CURRENT_2026_27_CLUBS
        ),
        2,
    ):
        home_team = (
            CURRENT_2026_27_CLUBS[
                index
            ]
        )

        away_team = (
            CURRENT_2026_27_CLUBS[
                index + 1
            ]
        )

        rows.append(
            {
                "source_match_id": (
                    900000
                    + index
                ),
                "date_utc": (
                    "2026-10-13T19:00:00Z"
                ),
                "date": (
                    "2026-10-13"
                ),
                "season": 2026,
                "home_team": home_team,
                "away_team": away_team,
            }
        )

    return pd.DataFrame(
        rows
    )


class LiveUefaStrengthTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        cls.context = (
            build_resolver_context()
        )

        cls.fixtures = (
            build_synthetic_current_fixtures()
        )

        cls.club_audit = (
            build_current_club_resolution_table(
                cls.fixtures,
                ranking_year=(
                    CURRENT_RANKING_YEAR
                ),
                context=cls.context,
            )
        )

        cls.features = (
            build_live_uefa_strength_features(
                cls.fixtures,
                ranking_year=(
                    CURRENT_RANKING_YEAR
                ),
                context=cls.context,
            )
        )

    def test_01_v2_feature_contract_is_locked(
        self,
    ) -> None:
        self.assertEqual(
            V2_UEFA_FEATURE_COLUMNS,
            [
                "uefa_coefficient_diff",
                "uefa_strength_percentile_diff",
            ],
        )

    def test_02_all_36_current_clubs_resolve(
        self,
    ) -> None:
        validate_current_club_resolution_table(
            self.club_audit
        )

        self.assertEqual(
            len(
                self.club_audit
            ),
            36,
        )

        self.assertEqual(
            set(
                self.club_audit[
                    "team"
                ]
            ),
            set(
                CURRENT_2026_27_CLUBS
            ),
        )

    def test_03_reviewed_resolution_contract_is_locked(
        self,
    ) -> None:
        self.assertEqual(
            set(
                REVIEWED_2026_RESOLUTIONS
            ),
            {
                "Como 1907",
                "LASK Linz",
                "Real Betis Balompié",
                "Viking FK",
            },
        )

        indexed = (
            self.club_audit
            .set_index(
                "team"
            )
        )

        for (
            club,
            expected,
        ) in EXPECTED_REVIEWED_VALUES.items():
            row = indexed.loc[
                club
            ]

            self.assertEqual(
                row[
                    "uefa_source_type"
                ],
                expected[
                    "source_type"
                ],
            )

            self.assertEqual(
                row[
                    "uefa_resolution_method"
                ],
                expected[
                    "resolution_method"
                ],
            )

            self.assertEqual(
                int(
                    row[
                        "uefa_country_floor_used"
                    ]
                ),
                expected[
                    "country_floor_used"
                ],
            )

            expected_source = (
                expected[
                    "source_name"
                ]
            )

            actual_source = row[
                "uefa_source_name"
            ]

            if expected_source is None:
                self.assertTrue(
                    pd.isna(
                        actual_source
                    )
                )
            else:
                self.assertEqual(
                    actual_source,
                    expected_source,
                )

            expected_country = (
                expected[
                    "country_name"
                ]
            )

            actual_country = row[
                "uefa_country_name"
            ]

            if expected_country is None:
                self.assertTrue(
                    pd.isna(
                        actual_country
                    )
                )
            else:
                self.assertEqual(
                    actual_country,
                    expected_country,
                )

    def test_04_reviewed_numeric_values_are_locked(
        self,
    ) -> None:
        indexed = (
            self.club_audit
            .set_index(
                "team"
            )
        )

        for (
            club,
            expected,
        ) in EXPECTED_REVIEWED_VALUES.items():
            row = indexed.loc[
                club
            ]

            self.assertAlmostEqual(
                float(
                    row[
                        "uefa_coefficient"
                    ]
                ),
                expected[
                    "coefficient"
                ],
                places=9,
            )

            self.assertAlmostEqual(
                float(
                    row[
                        "uefa_strength_percentile"
                    ]
                ),
                expected[
                    "percentile"
                ],
                places=12,
            )

    def test_05_live_feature_differences_are_correct(
        self,
    ) -> None:
        validate_live_uefa_strength_features(
            self.features,
            self.fixtures,
        )

        expected_coefficient_diff = (
            self.features[
                "home_uefa_coefficient"
            ].astype(float)
            - self.features[
                "away_uefa_coefficient"
            ].astype(float)
        )

        expected_percentile_diff = (
            self.features[
                "home_uefa_strength_percentile"
            ].astype(float)
            - self.features[
                "away_uefa_strength_percentile"
            ].astype(float)
        )

        np.testing.assert_allclose(
            self.features[
                "uefa_coefficient_diff"
            ].astype(float),
            expected_coefficient_diff,
            rtol=0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            self.features[
                "uefa_strength_percentile_diff"
            ].astype(float),
            expected_percentile_diff,
            rtol=0.0,
            atol=1e-12,
        )

    def test_06_resolution_is_deterministic(
        self,
    ) -> None:
        second = (
            build_current_club_resolution_table(
                self.fixtures,
                ranking_year=(
                    CURRENT_RANKING_YEAR
                ),
                context=self.context,
            )
        )

        pd.testing.assert_frame_equal(
            self.club_audit.reset_index(
                drop=True
            ),
            second.reset_index(
                drop=True
            ),
            check_dtype=True,
            check_exact=True,
        )

    def test_07_no_fuzzy_resolution_is_used(
        self,
    ) -> None:
        methods = (
            self.club_audit[
                "uefa_resolution_method"
            ]
            .astype(str)
            .str.upper()
        )

        self.assertFalse(
            methods.str.contains(
                "FUZZY",
                regex=False,
            ).any()
        )


if __name__ == "__main__":
    unittest.main()

