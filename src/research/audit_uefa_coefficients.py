from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]

START_SEASON = 2010
END_SEASON = 2026

TIMEOUT_SECONDS = 30

BASE_URL = (
    "https://kassiesa.net/uefa/data"
)


def ranking_method(
    ranking_year: int,
) -> str:
    """
    Historical Kassiesa storage follows the UEFA
    coefficient-method transition.

    Rankings through 2017 use method4.
    Rankings from 2018 onward use method5.
    """

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

        return " | ".join(
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


def table_score(
    frame: pd.DataFrame,
) -> tuple[int, int]:
    """
    Prefer large tables with many rows.

    The actual club-ranking table should be much
    larger than navigation/layout tables.
    """

    return (
        len(frame),
        len(frame.columns),
    )


def fetch_ranking_tables(
    ranking_year: int,
) -> tuple[
    str,
    list[pd.DataFrame],
]:
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

    tables = pd.read_html(
        StringIO(
            response.text
        )
    )

    return (
        url,
        tables,
    )


def select_ranking_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    if not tables:
        raise ValueError(
            "No HTML tables found."
        )

    candidates = [
        frame
        for frame in tables
        if len(frame) >= 20
    ]

    if not candidates:
        candidates = tables

    selected = max(
        candidates,
        key=table_score,
    )

    selected = (
        selected
        .copy()
        .reset_index(
            drop=True
        )
    )

    selected.columns = [
        flatten_column(
            column
        )
        for column in selected.columns
    ]

    return selected


def numeric_first_column_count(
    frame: pd.DataFrame,
) -> int:
    if frame.empty:
        return 0

    first_column = (
        frame.iloc[
            :,
            0,
        ]
    )

    numeric = pd.to_numeric(
        first_column,
        errors="coerce",
    )

    return int(
        numeric.notna().sum()
    )


def first_nonempty_rows(
    frame: pd.DataFrame,
    count: int = 3,
) -> list[str]:
    if frame.empty:
        return []

    sample = (
        frame
        .dropna(
            how="all"
        )
        .head(
            count
        )
    )

    return [
        " | ".join(
            str(value)
            for value in row
            if (
                pd.notna(value)
                and str(value).strip()
            )
        )
        for row in sample.values.tolist()
    ]


def main() -> None:
    audit_rows = []

    print(
        "Auditing historical UEFA "
        "club coefficient rankings..."
    )

    print()

    for season_start in range(
        START_SEASON,
        END_SEASON + 1,
    ):
        ranking_year = (
            season_start
        )

        method = ranking_method(
            ranking_year
        )

        season_label = (
            f"{season_start}/"
            f"{str(season_start + 1)[-2:]}"
        )

        print(
            f"Checking season "
            f"{season_label} "
            f"using ranking "
            f"{ranking_year} "
            f"({method})..."
        )

        url = ranking_url(
            ranking_year
        )

        try:
            (
                _,
                tables,
            ) = fetch_ranking_tables(
                ranking_year
            )

            selected = (
                select_ranking_table(
                    tables
                )
            )

            column_names = [
                str(column)
                for column
                in selected.columns
            ]

            samples = (
                first_nonempty_rows(
                    selected
                )
            )

            audit_rows.append(
                {
                    "season_start": (
                        season_start
                    ),
                    "season_label": (
                        season_label
                    ),
                    "ranking_year": (
                        ranking_year
                    ),
                    "method": method,
                    "available": True,
                    "http_url": url,
                    "table_count": len(
                        tables
                    ),
                    "selected_rows": len(
                        selected
                    ),
                    "selected_columns": len(
                        selected.columns
                    ),
                    "numeric_rank_rows": (
                        numeric_first_column_count(
                            selected
                        )
                    ),
                    "column_names": (
                        " || ".join(
                            column_names
                        )
                    ),
                    "sample_row_1": (
                        samples[0]
                        if len(samples) > 0
                        else None
                    ),
                    "sample_row_2": (
                        samples[1]
                        if len(samples) > 1
                        else None
                    ),
                    "sample_row_3": (
                        samples[2]
                        if len(samples) > 2
                        else None
                    ),
                    "error": None,
                }
            )

        except Exception as exc:
            audit_rows.append(
                {
                    "season_start": (
                        season_start
                    ),
                    "season_label": (
                        season_label
                    ),
                    "ranking_year": (
                        ranking_year
                    ),
                    "method": method,
                    "available": False,
                    "http_url": url,
                    "table_count": None,
                    "selected_rows": None,
                    "selected_columns": None,
                    "numeric_rank_rows": None,
                    "column_names": None,
                    "sample_row_1": None,
                    "sample_row_2": None,
                    "sample_row_3": None,
                    "error": str(
                        exc
                    ),
                }
            )

    audit = pd.DataFrame(
        audit_rows
    )

    output_dir = (
        ROOT
        / "reports"
        / "research"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "uefa_coefficient_coverage.csv"
    )

    audit.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "UEFA CLUB COEFFICIENT "
        "HISTORICAL COVERAGE"
    )
    print(
        "=" * 110
    )
    print()

    display_columns = [
        "season_label",
        "ranking_year",
        "method",
        "available",
        "table_count",
        "selected_rows",
        "selected_columns",
        "numeric_rank_rows",
    ]

    print(
        audit[
            display_columns
        ].to_string(
            index=False
        )
    )

    available = audit[
        audit[
            "available"
        ]
        == True
    ]

    print()
    print(
        "=" * 110
    )
    print(
        "SUMMARY"
    )
    print(
        "=" * 110
    )
    print()

    print(
        f"Available rankings: "
        f"{len(available)}/"
        f"{len(audit)}"
    )

    if not available.empty:
        print(
            "Minimum selected table rows: "
            f"{int(available['selected_rows'].min())}"
        )

        print(
            "Maximum selected table rows: "
            f"{int(available['selected_rows'].max())}"
        )

        print(
            "Minimum numeric rank rows: "
            f"{int(available['numeric_rank_rows'].min())}"
        )

        print(
            "Maximum numeric rank rows: "
            f"{int(available['numeric_rank_rows'].max())}"
        )

    failures = audit[
        audit[
            "available"
        ]
        == False
    ]

    if not failures.empty:
        print()
        print(
            "FAILED YEARS:"
        )

        for _, row in failures.iterrows():
            print(
                f"  {row['ranking_year']}: "
                f"{row['error']}"
            )

    # Show the two sides of the methodology
    # transition so we can verify parser shape.
    print()
    print(
        "=" * 110
    )
    print(
        "PARSER SAMPLES"
    )
    print(
        "=" * 110
    )

    for sample_year in [
        2017,
        2018,
        2025,
    ]:
        sample = audit[
            audit[
                "ranking_year"
            ]
            == sample_year
        ]

        if sample.empty:
            continue

        row = sample.iloc[0]

        print()
        print(
            f"Ranking {sample_year}"
        )

        print(
            f"Method: "
            f"{row['method']}"
        )

        print(
            "Columns:"
        )

        print(
            row[
                "column_names"
            ]
        )

        print(
            "First rows:"
        )

        for key in [
            "sample_row_1",
            "sample_row_2",
            "sample_row_3",
        ]:
            value = row[
                key
            ]

            if pd.notna(
                value
            ):
                print(
                    f"  {value}"
                )

    print()
    print(
        "=" * 110
    )
    print(
        "OUTPUT"
    )
    print(
        "=" * 110
    )
    print()

    print(
        f"Saved coverage report:"
    )
    print(
        output_path
    )


if __name__ == "__main__":
    main()