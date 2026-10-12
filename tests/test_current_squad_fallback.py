from __future__ import annotations

import unittest

import pandas as pd

from src.data.build_current_squad_pool import (
    EXPECTED_FALLBACK_CLUBS,
    MIN_TRUSTED_FOOTBALL_DATA_SQUAD_SIZE,
    build_unified_pool,
    identify_fallback_clubs,
)


class CurrentSquadFallbackTests(
    unittest.TestCase
):
    def _team_snapshot(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "canonical_team": "Arsenal",
                    "squad_size": 24,
                },
                {
                    "canonical_team": "Bayern Munich",
                    "squad_size": 24,
                },
                {
                    "canonical_team": "Sabah FK",
                    "squad_size": 1,
                },
                {
                    "canonical_team": "Slavia Prague",
                    "squad_size": 6,
                },
                {
                    "canonical_team": (
                        "ŠK Slovan Bratislava"
                    ),
                    "squad_size": 0,
                },
            ]
        )

    def _football_data_pool(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "canonical_team": "Arsenal",
                    "squad_source": (
                        "football_data_current_squad"
                    ),
                    "player_name": "Arsenal Player",
                },
                {
                    "canonical_team": "Sabah FK",
                    "squad_source": (
                        "football_data_current_squad"
                    ),
                    "player_name": "Partial Sabah Player",
                },
                {
                    "canonical_team": "Slavia Prague",
                    "squad_source": (
                        "football_data_current_squad"
                    ),
                    "player_name": "Partial Slavia Player",
                },
            ]
        )

    def _fallback_pool(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "canonical_team": "Sabah FK",
                    "squad_source": (
                        "transfermarkt_current_club_fallback"
                    ),
                    "player_name": "Sabah Fallback Player",
                },
                {
                    "canonical_team": "Slavia Prague",
                    "squad_source": (
                        "transfermarkt_current_club_fallback"
                    ),
                    "player_name": "Slavia Fallback Player",
                },
                {
                    "canonical_team": (
                        "ŠK Slovan Bratislava"
                    ),
                    "squad_source": (
                        "transfermarkt_current_club_fallback"
                    ),
                    "player_name": "Slovan Fallback Player",
                },
            ]
        )

    def test_01_threshold_is_conservative(
        self,
    ):
        self.assertEqual(
            MIN_TRUSTED_FOOTBALL_DATA_SQUAD_SIZE,
            20,
        )

    def test_02_partial_squads_use_expected_fallback(
        self,
    ):
        fallback = identify_fallback_clubs(
            self._team_snapshot()
        )

        self.assertEqual(
            fallback,
            EXPECTED_FALLBACK_CLUBS,
        )

    def test_03_unexpected_partial_club_fails(
        self,
    ):
        snapshot = self._team_snapshot()

        snapshot.loc[
            snapshot[
                "canonical_team"
            ]
            ==
            "Arsenal",
            "squad_size",
        ] = 8

        with self.assertRaises(
            RuntimeError
        ):
            identify_fallback_clubs(
                snapshot
            )

    def test_04_partial_provider_rows_are_excluded(
        self,
    ):
        unified = build_unified_pool(
            football_data_pool=(
                self._football_data_pool()
            ),
            fallback_pool=(
                self._fallback_pool()
            ),
            fallback_clubs=(
                EXPECTED_FALLBACK_CLUBS
            ),
        )

        sabah_sources = set(
            unified.loc[
                unified[
                    "canonical_team"
                ]
                ==
                "Sabah FK",
                "squad_source",
            ]
        )

        slavia_sources = set(
            unified.loc[
                unified[
                    "canonical_team"
                ]
                ==
                "Slavia Prague",
                "squad_source",
            ]
        )

        self.assertEqual(
            sabah_sources,
            {
                "transfermarkt_current_club_fallback"
            },
        )

        self.assertEqual(
            slavia_sources,
            {
                "transfermarkt_current_club_fallback"
            },
        )

    def test_05_trusted_provider_club_remains_primary(
        self,
    ):
        unified = build_unified_pool(
            football_data_pool=(
                self._football_data_pool()
            ),
            fallback_pool=(
                self._fallback_pool()
            ),
            fallback_clubs=(
                EXPECTED_FALLBACK_CLUBS
            ),
        )

        arsenal = unified[
            unified[
                "canonical_team"
            ]
            ==
            "Arsenal"
        ]

        self.assertEqual(
            len(arsenal),
            1,
        )

        self.assertEqual(
            arsenal.iloc[0][
                "squad_source"
            ],
            "football_data_current_squad",
        )

    def test_06_no_club_has_mixed_current_sources(
        self,
    ):
        unified = build_unified_pool(
            football_data_pool=(
                self._football_data_pool()
            ),
            fallback_pool=(
                self._fallback_pool()
            ),
            fallback_clubs=(
                EXPECTED_FALLBACK_CLUBS
            ),
        )

        source_counts = (
            unified.groupby(
                "canonical_team"
            )[
                "squad_source"
            ]
            .nunique()
        )

        self.assertTrue(
            (
                source_counts
                <=
                1
            ).all()
        )


if __name__ == "__main__":
    unittest.main()