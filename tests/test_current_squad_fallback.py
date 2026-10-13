from __future__ import annotations

import unittest

import pandas as pd

from src.data.build_current_squad_pool import (
    EXPECTED_FALLBACK_CLUBS,
    EXPECTED_FOOTBALL_DATA_SQUAD_CLUBS,
    MIN_TRUSTED_FOOTBALL_DATA_SQUAD_SIZE,
    build_unified_pool,
    identify_fallback_clubs,
    refresh_trusted_provider_rows,
)


class CurrentSquadFallbackTests(unittest.TestCase):
    """Safeguards for the 2026/27 current-squad provider-source policy."""

    def _team_snapshot(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"canonical_team": "Arsenal", "squad_size": 24},
                {"canonical_team": "Bayern Munich", "squad_size": 24},
                {"canonical_team": "Sabah FK", "squad_size": 28},
                {"canonical_team": "Slavia Prague", "squad_size": 29},
                {"canonical_team": "ŠK Slovan Bratislava", "squad_size": 0},
            ]
        )

    def _football_data_pool(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "canonical_team": "Arsenal",
                    "squad_source": "football_data_current_squad",
                    "player_name": "Arsenal Player",
                },
                {
                    "canonical_team": "Sabah FK",
                    "squad_source": "football_data_current_squad",
                    "player_name": "Sabah Player",
                },
                {
                    "canonical_team": "Slavia Prague",
                    "squad_source": "football_data_current_squad",
                    "player_name": "Slavia Player",
                },
                {
                    "canonical_team": "ŠK Slovan Bratislava",
                    "squad_source": "football_data_current_squad",
                    "player_name": "Incomplete Slovan Player",
                },
            ]
        )

    def _fallback_pool(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "canonical_team": "ŠK Slovan Bratislava",
                    "squad_source": "transfermarkt_current_club_fallback",
                    "player_name": "Slovan Fallback Player",
                }
            ]
        )

    def _unified_pool(self) -> pd.DataFrame:
        return build_unified_pool(
            football_data_pool=self._football_data_pool(),
            fallback_pool=self._fallback_pool(),
            fallback_clubs=EXPECTED_FALLBACK_CLUBS,
        )

    def test_01_threshold_and_provider_counts_are_locked(self):
        self.assertEqual(MIN_TRUSTED_FOOTBALL_DATA_SQUAD_SIZE, 20)
        self.assertEqual(EXPECTED_FOOTBALL_DATA_SQUAD_CLUBS, 35)
        self.assertEqual(
            EXPECTED_FALLBACK_CLUBS,
            {"ŠK Slovan Bratislava"},
        )

    def test_02_only_slovan_requires_fallback(self):
        fallback = identify_fallback_clubs(self._team_snapshot())
        self.assertEqual(fallback, {"ŠK Slovan Bratislava"})

    def test_03_unexpected_partial_club_fails(self):
        snapshot = self._team_snapshot()
        snapshot.loc[
            snapshot["canonical_team"] == "Arsenal", "squad_size"
        ] = 8

        with self.assertRaisesRegex(
            RuntimeError,
            "Unexpected football-data incomplete-squad set",
        ):
            identify_fallback_clubs(snapshot)

    def test_04_partial_slovan_provider_rows_are_excluded(self):
        unified = self._unified_pool()
        slovan = unified.loc[
            unified["canonical_team"] == "ŠK Slovan Bratislava"
        ]

        self.assertEqual(len(slovan), 1)
        self.assertEqual(
            set(slovan["squad_source"]),
            {"transfermarkt_current_club_fallback"},
        )
        self.assertNotIn(
            "Incomplete Slovan Player", set(unified["player_name"])
        )

    def test_05_trusted_provider_clubs_remain_primary(self):
        unified = self._unified_pool()

        for club in ("Arsenal", "Sabah FK", "Slavia Prague"):
            with self.subTest(club=club):
                club_rows = unified.loc[
                    unified["canonical_team"] == club
                ]
                self.assertEqual(len(club_rows), 1)
                self.assertEqual(
                    set(club_rows["squad_source"]),
                    {"football_data_current_squad"},
                )

    def test_06_no_club_has_mixed_current_sources(self):
        unified = self._unified_pool()
        self.assertEqual(unified["canonical_team"].nunique(), 4)
        self.assertTrue(
            unified.groupby("canonical_team")["squad_source"]
            .nunique()
            .eq(1)
            .all()
        )

    def test_07_missing_expected_fallback_requires_review(self):
        snapshot = self._team_snapshot()
        snapshot.loc[
            snapshot["canonical_team"] == "ŠK Slovan Bratislava",
            "squad_size",
        ] = 25

        with self.assertRaisesRegex(
            RuntimeError,
            "Unexpected football-data incomplete-squad set",
        ):
            identify_fallback_clubs(snapshot)

    def test_08_unexpected_mixed_source_is_rejected(self):
        # A bad fallback feed must not quietly add a second current
        # roster source for a trusted football-data club.
        bad_fallback_pool = pd.concat(
            [
                self._fallback_pool(),
                pd.DataFrame(
                    [
                        {
                            "canonical_team": "Sabah FK",
                            "squad_source": "transfermarkt_current_club_fallback",
                            "player_name": "Unwanted Sabah Fallback",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "multiple current squad source types",
        ):
            build_unified_pool(
                football_data_pool=self._football_data_pool(),
                fallback_pool=bad_fallback_pool,
                fallback_clubs=EXPECTED_FALLBACK_CLUBS,
            )

    def test_09_stale_mapping_recovery_uses_fresh_provider_rosters(self):
        # Sabah/Slavia had new provider rosters (28/29 players), but the
        # older valuation-resolution artifact had no rows for them.
        snapshot = pd.DataFrame(
            [
                {"canonical_team": "Sabah FK", "squad_size": 28},
                {"canonical_team": "Slavia Prague", "squad_size": 29},
                {"canonical_team": "ŠK Slovan Bratislava", "squad_size": 0},
            ]
        )
        current_team_map = pd.DataFrame(
            [
                {"canonical_team": "Sabah FK", "football_data_team_id": 10233},
                {"canonical_team": "Slavia Prague", "football_data_team_id": 930},
            ]
        )
        club_map = pd.DataFrame(
            [
                {"canonical_team": "Sabah FK", "transfermarkt_club_id": 101},
                {"canonical_team": "Slavia Prague", "transfermarkt_club_id": 62},
            ]
        )
        raw_rosters = pd.DataFrame(
            [
                {
                    "canonical_team": team,
                    "football_data_team_id": team_id,
                    "football_data_player_id": 100000 + index,
                    "player_name": f"{team} Player {index}",
                    "date_of_birth": "2000-01-01",
                    "position": "Midfield",
                    "nationality": "Unknown",
                }
                for team, team_id, size in (
                    ("Sabah FK", 10233, 28),
                    ("Slavia Prague", 930, 29),
                )
                for index in range(size)
            ]
        )
        old_resolved = pd.DataFrame(
            [
                {
                    "canonical_team": "Arsenal",
                    "football_data_team_id": 57,
                    "football_data_player_id": 999,
                    "player_name": "Existing Arsenal Player",
                    "transfermarkt_club_id": 11,
                    "transfermarkt_player_id": 9999,
                    "usable_market_value_eur": 20_000_000.0,
                    "mapping_status": "MATCHED",
                    "valuation_source": "EXISTING",
                }
            ]
        )

        refreshed = refresh_trusted_provider_rows(
            resolved_players=old_resolved,
            current_squad_players=raw_rosters,
            team_snapshot=snapshot,
            current_team_map=current_team_map,
            club_map=club_map,
            fallback_clubs=EXPECTED_FALLBACK_CLUBS,
        )

        self.assertEqual(len(refreshed), 58)
        self.assertEqual(
            refreshed["canonical_team"].value_counts().to_dict(),
            {"Sabah FK": 28, "Slavia Prague": 29, "Arsenal": 1},
        )

        recovered = refreshed.loc[
            refreshed["canonical_team"].isin(
                ("Sabah FK", "Slavia Prague")
            )
        ]
        self.assertTrue(recovered["transfermarkt_player_id"].isna().all())
        self.assertTrue(recovered["usable_market_value_eur"].isna().all())
        self.assertEqual(
            set(recovered["mapping_status"]),
            {"FOOTBALL_DATA_CURRENT_UNMAPPED"},
        )

        existing = refreshed.loc[
            refreshed["canonical_team"] == "Arsenal"
        ].iloc[0]
        self.assertEqual(existing["transfermarkt_player_id"], 9999)
        self.assertEqual(existing["usable_market_value_eur"], 20_000_000.0)

    def test_10_incomplete_fresh_roster_cannot_be_recovered_as_trusted(self):
        # A provider snapshot claiming 28 players must not allow a
        # newly collected file with only one Sabah player through.
        snapshot = pd.DataFrame(
            [
                {"canonical_team": "Sabah FK", "squad_size": 28},
                {"canonical_team": "ŠK Slovan Bratislava", "squad_size": 0},
            ]
        )
        raw_rosters = pd.DataFrame(
            [
                {
                    "canonical_team": "Sabah FK",
                    "football_data_player_id": 111,
                    "player_name": "Only Player",
                }
            ]
        )
        current_team_map = pd.DataFrame(
            [{"canonical_team": "Sabah FK", "football_data_team_id": 10233}]
        )
        club_map = pd.DataFrame(
            [{"canonical_team": "Sabah FK", "transfermarkt_club_id": 101}]
        )
        old_resolved = pd.DataFrame(columns=["canonical_team"])

        with self.assertRaisesRegex(
            RuntimeError,
            "Fresh football-data squad rows are too small",
        ):
            refresh_trusted_provider_rows(
                resolved_players=old_resolved,
                current_squad_players=raw_rosters,
                team_snapshot=snapshot,
                current_team_map=current_team_map,
                club_map=club_map,
                fallback_clubs=EXPECTED_FALLBACK_CLUBS,
            )


if __name__ == "__main__":
    unittest.main()
