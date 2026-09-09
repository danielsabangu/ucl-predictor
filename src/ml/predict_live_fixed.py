from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import domestic_form
from src.features import elo
from src.features import team_form

from src.ml import predict_live as base


# ============================================================
# TIMEZONE HELPERS
# ============================================================


def to_naive_utc(
    value,
) -> pd.Timestamp:
    """
    Convert a timestamp to UTC and then remove timezone info.

    Historical domestic_form data uses tz-naive timestamps.
    Live fixtures arrive as tz-aware UTC timestamps.

    We standardize live UCL timestamps to naive UTC immediately
    before the domestic feature stage so comparisons are valid.

    This changes timestamp representation only.
    It does NOT change the represented instant in time.
    """

    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is not None:

        timestamp = (
            timestamp
            .tz_convert(
                "UTC"
            )
            .tz_localize(
                None
            )
        )

    return timestamp


def normalize_frame_date_to_naive_utc(
    dataframe: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    """
    Convert an entire match-date column to naive UTC.
    """

    output = dataframe.copy()

    parsed = pd.to_datetime(
        output[
            date_column
        ],
        errors="raise",
        utc=True,
    )

    output[
        date_column
    ] = parsed.dt.tz_convert(
        None
    )

    return output


# ============================================================
# CORRECTED FIXTURE FEATURE BUILDER
# ============================================================


def build_fixture_features(
    *,
    fixture: pd.Series,
    base_replay: pd.DataFrame,
    date_column: str,
    production_features: list[str],
    families: dict[str, list[str]],
    domestic_histories,
    domestic_dates,
    squad_state: pd.DataFrame,
    squad_difference_mapping: dict[str, str],
) -> tuple[
    dict,
    dict,
]:
    """
    Build the exact frozen 290-feature row for one future match.

    Important:
    - Elo logic is unchanged.
    - UCL-form logic is unchanged.
    - Domestic logic is unchanged.
    - Squad logic is unchanged.
    - Production feature order is unchanged.

    The only compatibility adjustment is converting the UCL
    timestamps to naive UTC before domestic_form receives them.
    """

    home_team = str(
        fixture[
            "home_team"
        ]
    )

    away_team = str(
        fixture[
            "away_team"
        ]
    )

    fixture_date = to_naive_utc(
        fixture[
            "date_utc"
        ]
    )

    # --------------------------------------------------------
    # BUILD ONE FUTURE PLACEHOLDER
    # --------------------------------------------------------

    future = (
        base.build_future_placeholder(
            base=base_replay,
            date_column=date_column,
            fixture=fixture,
        )
    )

    combined = pd.concat(
        [
            base_replay.copy(),
            future,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # CRITICAL TIMEZONE COMPATIBILITY FIX
    #
    # domestic_form histories are historically tz-naive.
    # Therefore the live UCL side must use the same convention
    # before any bisect/date-window comparisons occur.
    # --------------------------------------------------------

    combined = (
        normalize_frame_date_to_naive_utc(
            dataframe=combined,
            date_column=date_column,
        )
    )

    combined = combined.sort_values(
        by=[
            date_column,
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # IDENTIFY FUTURE ROW
    # --------------------------------------------------------

    future_mask = (
        (
            combined[
                date_column
            ]
            ==
            fixture_date
        )
        &
        (
            combined[
                "home_team"
            ].astype(
                str
            )
            ==
            home_team
        )
        &
        (
            combined[
                "away_team"
            ].astype(
                str
            )
            ==
            away_team
        )
    )

    future_count = int(
        future_mask.sum()
    )

    if future_count != 1:

        raise RuntimeError(
            "Could not uniquely identify future fixture row.\n"
            f"Fixture: {home_team} vs {away_team}\n"
            f"Kickoff: {fixture_date}\n"
            f"Matches found: {future_count}"
        )

    # --------------------------------------------------------
    # ELO
    # --------------------------------------------------------

    elo_frame, _ = (
        elo.build_elo_features(
            combined.copy()
        )
    )

    # --------------------------------------------------------
    # UCL FORM
    # --------------------------------------------------------

    form_frame, _ = (
        team_form.build_form_features(
            elo_frame.copy()
        )
    )

    # Defensive normalization:
    # make absolutely sure team_form did not restore an
    # aware timestamp representation.
    form_frame = (
        normalize_frame_date_to_naive_utc(
            dataframe=form_frame,
            date_column=date_column,
        )
    )

    # --------------------------------------------------------
    # DOMESTIC FORM
    #
    # Both sides are now tz-naive, matching the historical
    # domestic pipeline's expected timestamp convention.
    # --------------------------------------------------------

    domestic_frame = (
        domestic_form.build_domestic_form_features(
            form_frame.copy(),
            domestic_histories,
            domestic_dates,
        )
    )

    domestic_dates_column = pd.to_datetime(
        domestic_frame[
            date_column
        ],
        errors="raise",
    )

    final_mask = (
        (
            domestic_dates_column
            ==
            fixture_date
        )
        &
        (
            domestic_frame[
                "home_team"
            ].astype(
                str
            )
            ==
            home_team
        )
        &
        (
            domestic_frame[
                "away_team"
            ].astype(
                str
            )
            ==
            away_team
        )
    )

    final_count = int(
        final_mask.sum()
    )

    if final_count != 1:

        raise RuntimeError(
            "Could not recover unique final "
            "pre-match feature row.\n"
            f"Fixture: {home_team} vs {away_team}\n"
            f"Kickoff: {fixture_date}\n"
            f"Rows found: {final_count}"
        )

    generated = domestic_frame.loc[
        final_mask
    ].iloc[
        0
    ]

    # --------------------------------------------------------
    # INITIALIZE EXACT FROZEN FEATURE SET
    # --------------------------------------------------------

    feature_row = {
        feature: np.nan
        for feature in production_features
    }

    # --------------------------------------------------------
    # COPY ELO / UCL FORM / DOMESTIC
    # --------------------------------------------------------

    for family in [
        "elo",
        "ucl_form",
        "domestic",
    ]:

        for feature in families[
            family
        ]:

            if feature not in generated.index:

                raise RuntimeError(
                    "Generated fixture row is missing "
                    f"{family} production feature:\n"
                    f"{feature}"
                )

            feature_row[
                feature
            ] = base.safe_numeric(
                generated[
                    feature
                ]
            )

    # --------------------------------------------------------
    # SAFE LIVE SQUAD FEATURES
    # --------------------------------------------------------

    base.attach_safe_squad_features(
        feature_row=feature_row,
        home_team=home_team,
        away_team=away_team,
        squad_state=squad_state,
        squad_production_features=families[
            "squad"
        ],
        difference_mapping=squad_difference_mapping,
    )

    # --------------------------------------------------------
    # FEATURE AVAILABILITY
    # --------------------------------------------------------

    availability = {}

    for family, features in families.items():

        values = [
            feature_row[
                feature
            ]
            for feature in features
        ]

        available_count = int(
            pd.Series(
                values
            ).notna().sum()
        )

        availability[
            family
        ] = {
            "available": (
                available_count
            ),

            "total": (
                len(
                    features
                )
            ),

            "rate": (
                available_count
                /
                len(
                    features
                )
            ),
        }

    total_available = int(
        pd.Series(
            [
                feature_row[
                    feature
                ]
                for feature
                in
                production_features
            ]
        ).notna().sum()
    )

    availability[
        "overall"
    ] = {
        "available": (
            total_available
        ),

        "total": (
            len(
                production_features
            )
        ),

        "rate": (
            total_available
            /
            len(
                production_features
            )
        ),
    }

    return (
        feature_row,
        availability,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """
    Apply the corrected live fixture builder to the existing
    production prediction pipeline.

    Everything else continues to come from predict_live.py.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - LIVE PREDICTION TIMEZONE FIX"
    )

    print(
        "=" * 75
    )

    print(
        "\nApplying domestic-form timestamp compatibility:"
    )

    print(
        "  tz-aware live UCL timestamps"
    )

    print(
        "  -> UTC"
    )

    print(
        "  -> tz-naive historical convention"
    )

    print(
        "\nNo feature formulas are being changed."
    )

    print(
        "No production model settings are being changed."
    )

    print(
        "No historical data is being modified."
    )

    # Replace only the function used internally by
    # base.predict_live().
    base.build_fixture_features = (
        build_fixture_features
    )

    base.predict_live()


if __name__ == "__main__":
    main()