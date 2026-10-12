from __future__ import annotations

from io import StringIO
from pathlib import Path
import re

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]

START_YEAR = 2010
END_YEAR = 2026

TIMEOUT_SECONDS = 30

BASE_URL = (
    "https://kassiesa.net/uefa/data"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "research"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "uefa_club_coefficients_2010_2026.csv"
)


def ranking_method(
    ranking_year: int,
) -> str:
    if ranking_year <= 2017:
        return "method4"

    return "method5"


def ranking_url(
    ranking_year: int,
) -> str:
    method = ranking_method(
        ranking_year
    )

    return (
        f"{BASE_URL}/"
        f"{method}/"
        f"trank{ranking_year}.html"
    )


def fetch_html(
    ranking_year: int,
) -> str:
    url = ranking_url(
        ranking_year
    )

    response = requests.get(
        url,
        timeout=TIMEOUT_SECONDS,
        headers={
            "User-Agent": (
                "ucl-predictor-research/1.0"
            ),
        },
    )

    response.raise_for_status()

    # Kassiesa's pages contain UTF-8 club names.
    # requests may otherwise infer Latin-1 and produce
    # strings such as "Bayern MÃ¼nchen".
    response.encoding = "utf-8"

    return response.text


def flatten_column(
    column,
) -> str:
    if isinstance(
        column,
        tuple,
    ):
        parts = []

        for item in column:
            value = str(
                item
            ).strip()

            if not value:
                continue

            if value.lower().startswith(
                "unnamed:"
            ):
                continue

            if value not in parts:
                parts.append(
                    value
                )

        return " ".join(
            parts
        )

    value = str(
        column
    ).strip()

    if value.lower().startswith(
        "unnamed:"
    ):
        return ""

    return value


def load_table(
    ranking_year: int,
) -> pd.DataFrame:
    html = fetch_html(
        ranking_year
    )

    tables = pd.read_html(
        StringIO(
            html
        )
    )

    if not tables:
        raise RuntimeError(
            f"No tables found for {ranking_year}."
        )

    table = max(
        tables,
        key=lambda frame: (
            len(frame),
            len(frame.columns),
        ),
    )

    table = (
        table
        .copy()
        .reset_index(
            drop=True
        )
    )

    table.columns = [
        flatten_column(
            column
        )
        for column in table.columns
    ]

    return table


def clean_string(
    value,
) -> str | None:
    if pd.isna(
        value
    ):
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    if text.lower() == "nan":
        return None

    return text


def is_country_code(
    value,
) -> bool:
    text = clean_string(
        value
    )

    if text is None:
        return False

    return bool(
        re.fullmatch(
            r"[A-Za-z]{3}",
            text,
        )
    )


def to_float(
    value,
) -> float | None:
    if pd.isna(
        value
    ):
        return None

    try:
        number = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if pd.isna(
        number
    ):
        return None

    return number


def parse_method4(
    table: pd.DataFrame,
    ranking_year: int,
) -> list[dict]:
    """
    method4 layout:

    rank
    club
    country
    five yearly coefficient columns
    ranking total

    Country fallback rows are also present in the table.
    They have no 3-letter country code, so we exclude them.
    """

    rows = []

    if len(
        table.columns
    ) < 9:
        raise RuntimeError(
            f"Unexpected method4 shape for "
            f"{ranking_year}: "
            f"{table.shape}"
        )

    values = table.values.tolist()

    last_rank = None

    for raw in values:
        rank_raw = (
            raw[0]
            if len(raw) > 0
            else None
        )

        club_raw = (
            raw[1]
            if len(raw) > 1
            else None
        )

        country_raw = (
            raw[2]
            if len(raw) > 2
            else None
        )

        if not is_country_code(
            country_raw
        ):
            continue

        club = clean_string(
            club_raw
        )

        country = clean_string(
            country_raw
        )

        if (
            club is None
            or country is None
        ):
            continue

        numeric_rank = to_float(
            rank_raw
        )

        if numeric_rank is not None:
            last_rank = int(
                numeric_rank
            )

        if last_rank is None:
            continue

        coefficient = to_float(
            raw[-1]
        )

        if coefficient is None:
            continue

        yearly_values = [
            to_float(
                value
            )
            for value in raw[3:-1]
        ]

        rows.append(
            {
                "ranking_year": (
                    ranking_year
                ),
                "season_start": (
                    ranking_year
                ),
                "method": "method4",
                "uefa_rank": (
                    last_rank
                ),
                "club_source": club,
                "country_code": (
                    country
                ),
                "uefa_coefficient": (
                    coefficient
                ),
                "country_part": None,
                "year_1_points": (
                    yearly_values[0]
                    if len(yearly_values) > 0
                    else None
                ),
                "year_2_points": (
                    yearly_values[1]
                    if len(yearly_values) > 1
                    else None
                ),
                "year_3_points": (
                    yearly_values[2]
                    if len(yearly_values) > 2
                    else None
                ),
                "year_4_points": (
                    yearly_values[3]
                    if len(yearly_values) > 3
                    else None
                ),
                "year_5_points": (
                    yearly_values[4]
                    if len(yearly_values) > 4
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


def parse_method5(
    table: pd.DataFrame,
    ranking_year: int,
) -> list[dict]:
    """
    method5 layout:

    rank
    movement/blank indicator
    club
    country
    five yearly club-point columns
    Total Points
    Country Part

    Tied clubs often have a blank rank cell.
    In those cases we carry forward the preceding
    displayed rank.

    Country fallback rows are excluded because
    their country-code field is not a valid
    three-letter country code.
    """

    rows = []

    if len(
        table.columns
    ) < 10:
        raise RuntimeError(
            f"Unexpected method5 shape for "
            f"{ranking_year}: "
            f"{table.shape}"
        )

    values = table.values.tolist()

    last_rank = None

    for raw in values:
        # Current method5 table has:
        # 0 rank
        # 1 movement / blank marker
        # 2 club
        # 3 country
        #
        # We still guard against a shifted layout below.

        club_index = None
        country_index = None

        candidate_pairs = [
            (2, 3),
            (1, 2),
        ]

        for possible_club, possible_country in candidate_pairs:
            if (
                len(raw)
                > possible_country
                and is_country_code(
                    raw[
                        possible_country
                    ]
                )
            ):
                club_index = (
                    possible_club
                )

                country_index = (
                    possible_country
                )

                break

        if (
            club_index is None
            or country_index is None
        ):
            continue

        club = clean_string(
            raw[
                club_index
            ]
        )

        country = clean_string(
            raw[
                country_index
            ]
        )

        if (
            club is None
            or country is None
        ):
            continue

        numeric_rank = to_float(
            raw[0]
        )

        if numeric_rank is not None:
            last_rank = int(
                numeric_rank
            )

        if last_rank is None:
            continue

        # Final two fields are:
        # Total Points
        # Country Part
        coefficient = to_float(
            raw[-2]
        )

        country_part = to_float(
            raw[-1]
        )

        if coefficient is None:
            continue

        yearly_values = [
            to_float(
                value
            )
            for value
            in raw[
                country_index + 1:
                -2
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
                "method": "method5",
                "uefa_rank": (
                    last_rank
                ),
                "club_source": club,
                "country_code": (
                    country
                ),
                "uefa_coefficient": (
                    coefficient
                ),
                "country_part": (
                    country_part
                ),
                "year_1_points": (
                    yearly_values[0]
                    if len(yearly_values) > 0
                    else None
                ),
                "year_2_points": (
                    yearly_values[1]
                    if len(yearly_values) > 1
                    else None
                ),
                "year_3_points": (
                    yearly_values[2]
                    if len(yearly_values) > 2
                    else None
                ),
                "year_4_points": (
                    yearly_values[3]
                    if len(yearly_values) > 3
                    else None
                ),
                "year_5_points": (
                    yearly_values[4]
                    if len(yearly_values) > 4
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


def add_rank_percentile(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    frame[
        "clubs_in_ranking"
    ] = (
        frame.groupby(
            "ranking_year"
        )[
            "club_source"
        ]
        .transform(
            "count"
        )
        .astype(
            int
        )
    )

    denominator = (
        frame[
            "clubs_in_ranking"
        ]
        - 1
    )

    frame[
        "uefa_rank_percentile"
    ] = (
        (
            frame[
                "uefa_rank"
            ]
            - 1
        )
        / denominator.where(
            denominator > 0,
            1,
        )
    )

    frame[
        "uefa_rank_percentile"
    ] = (
        frame[
            "uefa_rank_percentile"
        ]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    return frame


def validate_frame(
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        raise RuntimeError(
            "UEFA coefficient dataset is empty."
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
            "Missing ranking years: "
            f"{sorted(missing_years)}"
        )

    duplicate_keys = (
        frame.duplicated(
            subset=[
                "ranking_year",
                "club_source",
                "country_code",
            ],
            keep=False,
        )
    )

    if duplicate_keys.any():
        duplicates = (
            frame.loc[
                duplicate_keys,
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
            "Duplicate club/year records found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if frame[
        "uefa_coefficient"
    ].isna().any():
        raise RuntimeError(
            "Missing UEFA coefficient values."
        )

    if (
        frame[
            "uefa_coefficient"
        ]
        < 0
    ).any():
        raise RuntimeError(
            "Negative UEFA coefficient found."
        )

    if (
        frame[
            "uefa_rank"
        ]
        < 1
    ).any():
        raise RuntimeError(
            "Invalid UEFA rank found."
        )

    bad_encoding_patterns = [
        "Ã",
        "Â",
        "â€",
        "�",
    ]

    corrupt_mask = pd.Series(
        False,
        index=frame.index,
    )

    for pattern in bad_encoding_patterns:
        corrupt_mask |= (
            frame[
                "club_source"
            ]
            .astype(str)
            .str.contains(
                pattern,
                regex=False,
                na=False,
            )
        )

    if corrupt_mask.any():
        examples = (
            frame.loc[
                corrupt_mask,
                "club_source",
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            "Possible mojibake remains in club names: "
            f"{examples}"
        )


def main() -> None:
    all_rows = []

    print(
        "Building historical UEFA "
        "club coefficient dataset..."
    )
    print()

    for ranking_year in range(
        START_YEAR,
        END_YEAR + 1,
    ):
        method = ranking_method(
            ranking_year
        )

        print(
            f"Parsing ranking "
            f"{ranking_year} "
            f"({method})..."
        )

        table = load_table(
            ranking_year
        )

        if method == "method4":
            rows = parse_method4(
                table,
                ranking_year,
            )
        else:
            rows = parse_method5(
                table,
                ranking_year,
            )

        if not rows:
            raise RuntimeError(
                f"No club rows parsed "
                f"for {ranking_year}."
            )

        print(
            f"  Parsed clubs: "
            f"{len(rows)}"
        )

        all_rows.extend(
            rows
        )

    frame = pd.DataFrame(
        all_rows
    )

    frame = add_rank_percentile(
        frame
    )

    frame = frame[
        [
            "ranking_year",
            "season_start",
            "method",
            "uefa_rank",
            "uefa_rank_percentile",
            "clubs_in_ranking",
            "club_source",
            "country_code",
            "uefa_coefficient",
            "country_part",
            "year_1_points",
            "year_2_points",
            "year_3_points",
            "year_4_points",
            "year_5_points",
            "source_url",
        ]
    ]

    frame = frame.sort_values(
        [
            "ranking_year",
            "uefa_rank",
            "club_source",
        ]
    ).reset_index(
        drop=True
    )

    validate_frame(
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

    print()
    print(
        "=" * 100
    )
    print(
        "UEFA COEFFICIENT DATASET"
    )
    print(
        "=" * 100
    )
    print()

    yearly = (
        frame.groupby(
            "ranking_year"
        )
        .agg(
            clubs=(
                "club_source",
                "count",
            ),
            highest_coefficient=(
                "uefa_coefficient",
                "max",
            ),
            lowest_coefficient=(
                "uefa_coefficient",
                "min",
            ),
        )
        .reset_index()
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    print()
    print(
        f"Total club-year records: "
        f"{len(frame):,}"
    )

    print(
        f"Unique source club names: "
        f"{frame['club_source'].nunique():,}"
    )

    print(
        f"Ranking years: "
        f"{frame['ranking_year'].nunique()}"
    )

    print()
    print(
        "Encoding checks:"
    )

    checks = [
        "Bayern München",
        "Atlético Madrid",
        "Fenerbahçe",
    ]

    for club in checks:
        found = (
            frame[
                "club_source"
            ]
            == club
        ).any()

        print(
            f"  {club}: "
            f"{'FOUND' if found else 'NOT FOUND'}"
        )

    print()
    print(
        "2025 top 10:"
    )

    print(
        frame[
            frame[
                "ranking_year"
            ]
            == 2025
        ][
            [
                "uefa_rank",
                "club_source",
                "country_code",
                "uefa_coefficient",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
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