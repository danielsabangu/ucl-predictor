from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.build_uefa_coefficients import (
    END_YEAR,
    START_YEAR,
    clean_string,
    load_table,
    ranking_method,
    ranking_url,
    to_float,
)


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "data"
    / "research"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "uefa_country_floors_2010_2026.csv"
)


def is_blank(
    value,
) -> bool:
    if pd.isna(
        value
    ):
        return True

    text = str(
        value
    ).strip()

    return (
        not text
        or text.lower() == "nan"
    )


def is_numeric_rank(
    value,
) -> bool:
    number = to_float(
        value
    )

    if number is None:
        return False

    return (
        number >= 1
        and float(
            number
        ).is_integer()
    )


def looks_like_separator(
    value,
) -> bool:
    text = clean_string(
        value
    )

    if text is None:
        return True

    stripped = (
        text
        .replace(
            "-",
            "",
        )
        .replace(
            "–",
            "",
        )
        .replace(
            "—",
            "",
        )
        .strip()
    )

    return not stripped


def parse_method4_country_floors(
    table: pd.DataFrame,
    ranking_year: int,
) -> list[dict]:
    """
    Pre-2018 table layout:

        rank
        club/country name
        country code
        five seasonal columns
        ranking total

    Country-floor rows have:

        - no numeric club rank
        - a country name in column 1
        - no three-letter club country code
        - a numeric total coefficient
    """

    if len(
        table.columns
    ) < 9:
        raise RuntimeError(
            "Unexpected method4 table shape "
            f"for {ranking_year}: "
            f"{table.shape}"
        )

    rows = []

    for raw in table.values.tolist():
        if len(
            raw
        ) < 9:
            continue

        rank_raw = (
            raw[0]
        )

        country_name_raw = (
            raw[1]
        )

        country_code_raw = (
            raw[2]
        )

        if is_numeric_rank(
            rank_raw
        ):
            continue

        country_name = (
            clean_string(
                country_name_raw
            )
        )

        if country_name is None:
            continue

        if looks_like_separator(
            country_name
        ):
            continue

        country_code_text = (
            clean_string(
                country_code_raw
            )
        )

        # Normal club rows have a 3-character
        # association code in this column.
        if (
            country_code_text is not None
            and len(
                country_code_text
            )
            == 3
            and country_code_text.isalpha()
        ):
            continue

        floor_coefficient = (
            to_float(
                raw[-1]
            )
        )

        if floor_coefficient is None:
            continue

        yearly_values = [
            to_float(
                value
            )
            for value
            in raw[3:-1]
        ]

        numeric_year_values = [
            value
            for value
            in yearly_values
            if value is not None
        ]

        # Country rows should contain multiple
        # historical yearly values.
        if len(
            numeric_year_values
        ) < 3:
            continue

        rows.append(
            {
                "ranking_year": (
                    ranking_year
                ),
                "season_start": (
                    ranking_year
                ),
                "method": (
                    "method4"
                ),
                "country_name": (
                    country_name
                ),
                "country_floor_coefficient": (
                    float(
                        floor_coefficient
                    )
                ),
                "year_1_floor": (
                    yearly_values[0]
                    if len(
                        yearly_values
                    ) > 0
                    else None
                ),
                "year_2_floor": (
                    yearly_values[1]
                    if len(
                        yearly_values
                    ) > 1
                    else None
                ),
                "year_3_floor": (
                    yearly_values[2]
                    if len(
                        yearly_values
                    ) > 2
                    else None
                ),
                "year_4_floor": (
                    yearly_values[3]
                    if len(
                        yearly_values
                    ) > 3
                    else None
                ),
                "year_5_floor": (
                    yearly_values[4]
                    if len(
                        yearly_values
                    ) > 4
                    else None
                ),
                "source_url": (
                    ranking_url(
                        ranking_year
                    )
                ),
            }
        )

    return rows


def parse_method5_country_floors(
    table: pd.DataFrame,
    ranking_year: int,
) -> list[dict]:
    """
    2018+ table layout:

        rank
        movement / blank
        club/country name
        association code
        five yearly club-point columns
        Total Points
        Country Part

    Country-floor rows have:

        - no numeric club rank
        - country name in the club-name column
        - no association code
        - Country Part containing the floor

    Zero is allowed.

    Example:
        Kosovo's 2018 country floor is genuinely
        0.000 in the source.
    """

    if len(
        table.columns
    ) < 10:
        raise RuntimeError(
            "Unexpected method5 table shape "
            f"for {ranking_year}: "
            f"{table.shape}"
        )

    rows = []

    for raw in table.values.tolist():
        if len(
            raw
        ) < 10:
            continue

        rank_raw = (
            raw[0]
        )

        if is_numeric_rank(
            rank_raw
        ):
            continue

        country_name = (
            clean_string(
                raw[2]
            )
        )

        if country_name is None:
            continue

        if looks_like_separator(
            country_name
        ):
            continue

        country_code_text = (
            clean_string(
                raw[3]
            )
        )

        # A normal club row has an association
        # code such as Eng, Esp, Ger, etc.
        if (
            country_code_text is not None
            and len(
                country_code_text
            )
            == 3
            and country_code_text.isalpha()
        ):
            continue

        total_points = (
            to_float(
                raw[-2]
            )
        )

        country_part = (
            to_float(
                raw[-1]
            )
        )

        if country_part is None:
            continue

        # Country-floor rows should not contain
        # a club-specific Total Points value.
        if total_points is not None:
            continue

        yearly_values = [
            to_float(
                value
            )
            for value
            in raw[
                4:-2
            ]
        ]

        rows.append(
            {
                "ranking_year": (
                    ranking_year
                ),
                "season_start": (
                    ranking_year
                ),
                "method": (
                    "method5"
                ),
                "country_name": (
                    country_name
                ),
                "country_floor_coefficient": (
                    float(
                        country_part
                    )
                ),
                "year_1_floor": (
                    yearly_values[0]
                    if len(
                        yearly_values
                    ) > 0
                    else None
                ),
                "year_2_floor": (
                    yearly_values[1]
                    if len(
                        yearly_values
                    ) > 1
                    else None
                ),
                "year_3_floor": (
                    yearly_values[2]
                    if len(
                        yearly_values
                    ) > 2
                    else None
                ),
                "year_4_floor": (
                    yearly_values[3]
                    if len(
                        yearly_values
                    ) > 3
                    else None
                ),
                "year_5_floor": (
                    yearly_values[4]
                    if len(
                        yearly_values
                    ) > 4
                    else None
                ),
                "source_url": (
                    ranking_url(
                        ranking_year
                    )
                ),
            }
        )

    return rows


def validate_country_floors(
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        raise RuntimeError(
            "Country-floor dataset is empty."
        )

    expected_years = set(
        range(
            START_YEAR,
            END_YEAR + 1,
        )
    )

    actual_years = set(
        frame[
            "ranking_year"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    missing_years = (
        expected_years
        - actual_years
    )

    if missing_years:
        raise RuntimeError(
            "Missing country-floor ranking years: "
            f"{sorted(missing_years)}"
        )

    duplicate_mask = (
        frame.duplicated(
            subset=[
                "ranking_year",
                "country_name",
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
                    "country_floor_coefficient",
                ],
            ]
            .sort_values(
                [
                    "ranking_year",
                    "country_name",
                ]
            )
        )

        raise RuntimeError(
            "Duplicate country-floor records:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if frame[
        "country_floor_coefficient"
    ].isna().any():
        invalid = (
            frame[
                frame[
                    "country_floor_coefficient"
                ].isna()
            ]
        )

        raise RuntimeError(
            "Missing country-floor coefficient:\n"
            f"{invalid.to_string(index=False)}"
        )

    # --------------------------------------------------------
    # ZERO IS VALID.
    #
    # Kosovo's 2018 five-year country ranking was
    # genuinely zero, so its minimum club coefficient
    # was also zero.
    #
    # We reject only negative values.
    # --------------------------------------------------------

    negative_mask = (
        frame[
            "country_floor_coefficient"
        ]
        < 0
    )

    if negative_mask.any():
        invalid = (
            frame.loc[
                negative_mask
            ]
        )

        raise RuntimeError(
            "Negative country-floor value found:\n"
            f"{invalid.to_string(index=False)}"
        )

    counts = (
        frame.groupby(
            "ranking_year"
        )[
            "country_name"
        ]
        .count()
    )

    # UEFA has approximately 50+ associations
    # across this period. Anything dramatically
    # below 40 strongly suggests parser failure.
    suspicious = (
        counts[
            counts < 40
        ]
    )

    if not suspicious.empty:
        raise RuntimeError(
            "Suspiciously few country-floor rows "
            "were parsed:\n"
            f"{suspicious.to_string()}"
        )

    bad_patterns = [
        "Ã",
        "Â",
        "â€",
        "�",
    ]

    for pattern in bad_patterns:
        bad = (
            frame[
                "country_name"
            ]
            .astype(
                str
            )
            .str.contains(
                pattern,
                regex=False,
                na=False,
            )
        )

        if bad.any():
            examples = (
                frame.loc[
                    bad,
                    "country_name",
                ]
                .head(
                    10
                )
                .tolist()
            )

            raise RuntimeError(
                "Possible encoding corruption "
                "in country names: "
                f"{examples}"
            )


def show_sample(
    frame: pd.DataFrame,
    ranking_year: int,
) -> None:
    subset = (
        frame[
            frame[
                "ranking_year"
            ]
            == ranking_year
        ]
        .sort_values(
            [
                "country_floor_coefficient",
                "country_name",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    print()
    print(
        f"Ranking {ranking_year} "
        f"country-floor sample:"
    )

    print()

    print(
        subset[
            [
                "country_name",
                "country_floor_coefficient",
            ]
        ]
        .head(
            12
        )
        .to_string(
            index=False
        )
    )


def print_zero_floor_records(
    frame: pd.DataFrame,
) -> None:
    zero_rows = (
        frame[
            frame[
                "country_floor_coefficient"
            ]
            == 0
        ]
        .copy()
    )

    print()
    print(
        "=" * 100
    )
    print(
        "ZERO COUNTRY-FLOOR VALUES"
    )
    print(
        "=" * 100
    )
    print()

    if zero_rows.empty:
        print(
            "None."
        )

        return

    print(
        "Zero is historically valid and "
        "is not treated as missing data."
    )

    print()

    print(
        zero_rows[
            [
                "ranking_year",
                "country_name",
                "country_floor_coefficient",
            ]
        ]
        .sort_values(
            [
                "ranking_year",
                "country_name",
            ]
        )
        .to_string(
            index=False
        )
    )


def print_target_country_checks(
    frame: pd.DataFrame,
) -> None:
    print()
    print(
        "=" * 100
    )
    print(
        "TARGET COUNTRY CHECKS"
    )
    print(
        "=" * 100
    )
    print()

    target_checks = {
        2010: [
            "Slovakia",
            "Turkey",
        ],
        2011: [
            "Romania",
        ],
        2012: [
            "Spain",
        ],
        2013: [
            "Spain",
        ],
        2014: [
            "France",
        ],
        2016: [
            "England",
            "Russia",
        ],
        2017: [
            "Germany",
        ],
        2023: [
            "England",
            "France",
        ],
        2024: [
            "England",
            "France",
            "Germany",
            "Italy",
            "Spain",
        ],
        2025: [
            "Cyprus",
        ],
    }

    for (
        ranking_year,
        countries,
    ) in target_checks.items():

        year_frame = (
            frame[
                frame[
                    "ranking_year"
                ]
                == ranking_year
            ]
        )

        for country in countries:
            result = (
                year_frame[
                    year_frame[
                        "country_name"
                    ]
                    == country
                ]
            )

            if result.empty:
                print(
                    f"  {ranking_year} "
                    f"{country}: "
                    "NOT FOUND"
                )

                continue

            floor_value = float(
                result.iloc[
                    0
                ][
                    "country_floor_coefficient"
                ]
            )

            print(
                f"  {ranking_year} "
                f"{country}: "
                f"FOUND "
                f"(floor={floor_value:.3f})"
            )


def main() -> None:
    print(
        "Building historical UEFA "
        "country-floor dataset..."
    )

    print()

    all_rows = []

    for ranking_year in range(
        START_YEAR,
        END_YEAR + 1,
    ):
        method = (
            ranking_method(
                ranking_year
            )
        )

        print(
            f"Parsing country floors "
            f"{ranking_year} "
            f"({method})..."
        )

        table = (
            load_table(
                ranking_year
            )
        )

        if method == "method4":
            rows = (
                parse_method4_country_floors(
                    table,
                    ranking_year,
                )
            )

        else:
            rows = (
                parse_method5_country_floors(
                    table,
                    ranking_year,
                )
            )

        print(
            f"  Parsed country floors: "
            f"{len(rows)}"
        )

        if not rows:
            raise RuntimeError(
                "No country-floor rows parsed "
                f"for {ranking_year}."
            )

        all_rows.extend(
            rows
        )

    frame = pd.DataFrame(
        all_rows
    )

    frame = frame[
        [
            "ranking_year",
            "season_start",
            "method",
            "country_name",
            "country_floor_coefficient",
            "year_1_floor",
            "year_2_floor",
            "year_3_floor",
            "year_4_floor",
            "year_5_floor",
            "source_url",
        ]
    ]

    frame = (
        frame.sort_values(
            [
                "ranking_year",
                "country_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validate_country_floors(
        frame
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    counts = (
        frame.groupby(
            "ranking_year"
        )
        .agg(
            countries=(
                "country_name",
                "count",
            ),
            highest_floor=(
                "country_floor_coefficient",
                "max",
            ),
            lowest_floor=(
                "country_floor_coefficient",
                "min",
            ),
            zero_floor_count=(
                "country_floor_coefficient",
                lambda values: int(
                    (
                        values
                        == 0
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    print()
    print(
        "=" * 100
    )
    print(
        "UEFA COUNTRY FLOOR DATASET"
    )
    print(
        "=" * 100
    )
    print()

    print(
        counts.to_string(
            index=False
        )
    )

    print()

    print(
        f"Total country-year records: "
        f"{len(frame):,}"
    )

    print(
        f"Unique country names: "
        f"{frame['country_name'].nunique():,}"
    )

    print(
        f"Ranking years: "
        f"{frame['ranking_year'].nunique()}"
    )

    print(
        "Negative floor values: "
        f"{int((frame['country_floor_coefficient'] < 0).sum())}"
    )

    print(
        "Zero floor values: "
        f"{int((frame['country_floor_coefficient'] == 0).sum())}"
    )

    print_zero_floor_records(
        frame
    )

    for sample_year in [
        2010,
        2016,
        2018,
        2024,
        2025,
    ]:
        show_sample(
            frame,
            sample_year,
        )

    print_target_country_checks(
        frame
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
        "All ranking years present: PASS"
    )

    print(
        "No duplicate country/year rows: PASS"
    )

    print(
        "No missing country-floor values: PASS"
    )

    print(
        "No negative country-floor values: PASS"
    )

    print(
        "Zero country-floor values permitted: PASS"
    )

    print(
        "Minimum country-count sanity check: PASS"
    )

    print(
        "Encoding check: PASS"
    )

    print()
    print(
        "=" * 100
    )
    print(
        "OUTPUT"
    )
    print(
        "=" * 100
    )
    print()

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()