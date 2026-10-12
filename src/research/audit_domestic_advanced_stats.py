from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "matches"
    / "domestic"
    / "github_football_datasets"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)

FILE_AUDIT_PATH = (
    REPORT_DIR
    / "domestic_advanced_stats_file_audit.csv"
)

COLUMN_AUDIT_PATH = (
    REPORT_DIR
    / "domestic_advanced_stats_column_audit.csv"
)

ANOMALY_PATH = (
    REPORT_DIR
    / "domestic_advanced_stats_anomalies.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "domestic_advanced_stats_audit_summary.json"
)


# ============================================================
# CONTRACT
# ============================================================


EXPECTED_SEASONS = [
    f"{year}-{str(year + 1)[-2:]}"
    for year in range(
        2010,
        2026,
    )
]

EXPECTED_LEAGUES = [
    "premier-league",
    "la-liga",
    "bundesliga",
    "serie-a",
    "ligue-1",
]

STAT_COLUMNS = [
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

CORE_PERFORMANCE_COLUMNS = [
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
]

DISCIPLINE_COLUMNS = [
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
]


# Conservative plausibility screens.
# These are for flagging only, not automatic deletion.
PLAUSIBILITY_MAX = {
    "HS": 60,
    "AS": 60,
    "HST": 30,
    "AST": 30,
    "HF": 50,
    "AF": 50,
    "HC": 30,
    "AC": 30,
    "HY": 15,
    "AY": 15,
    "HR": 5,
    "AR": 5,
}


# ============================================================
# HELPERS
# ============================================================


def expected_files() -> list[
    tuple[str, str, Path]
]:
    rows = []

    for season in EXPECTED_SEASONS:

        for league in EXPECTED_LEAGUES:

            path = (
                RAW_ROOT
                / season
                / f"{league}.csv"
            )

            rows.append(
                (
                    season,
                    league,
                    path,
                )
            )

    return rows


def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# AUDIT ONE FILE
# ============================================================


def audit_file(
    *,
    season_label: str,
    league: str,
    path: Path,
) -> tuple[
    dict,
    list[dict],
    list[dict],
]:
    file_row = {
        "season_label": season_label,
        "league": league,
        "path": str(
            path.relative_to(
                ROOT
            )
        ),
        "exists": path.exists(),
    }

    column_rows = []

    anomaly_rows = []

    if not path.exists():

        file_row.update(
            {
                "rows": 0,
                "columns": 0,
                "all_stat_columns_present": False,
                "core_stats_complete": False,
            }
        )

        return (
            file_row,
            column_rows,
            anomaly_rows,
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    file_row[
        "rows"
    ] = int(
        len(
            dataframe
        )
    )

    file_row[
        "columns"
    ] = int(
        len(
            dataframe.columns
        )
    )

    missing_stats = [
        column
        for column in STAT_COLUMNS
        if column
        not in dataframe.columns
    ]

    file_row[
        "missing_stat_columns"
    ] = " | ".join(
        missing_stats
    )

    file_row[
        "all_stat_columns_present"
    ] = (
        len(
            missing_stats
        )
        ==
        0
    )

    core_complete = True

    for column in STAT_COLUMNS:

        if column not in dataframe.columns:

            column_rows.append(
                {
                    "season_label": season_label,
                    "league": league,
                    "column": column,
                    "present": False,
                    "rows": int(
                        len(
                            dataframe
                        )
                    ),
                    "numeric_rows": 0,
                    "non_null_rows": 0,
                    "non_numeric_rows": 0,
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "p99": np.nan,
                    "negative_rows": 0,
                    "above_plausibility_rows": 0,
                }
            )

            if column in CORE_PERFORMANCE_COLUMNS:

                core_complete = False

            continue

        raw = dataframe[
            column
        ]

        numeric = safe_numeric(
            raw
        )

        non_null_rows = int(
            raw.notna().sum()
        )

        numeric_rows = int(
            numeric.notna().sum()
        )

        non_numeric_rows = int(
            (
                raw.notna()
                &
                numeric.isna()
            ).sum()
        )

        negative_rows = int(
            (
                numeric
                <
                0
            ).sum()
        )

        maximum = (
            PLAUSIBILITY_MAX[
                column
            ]
        )

        above_rows = int(
            (
                numeric
                >
                maximum
            ).sum()
        )

        if (
            column
            in CORE_PERFORMANCE_COLUMNS
            and
            (
                numeric_rows
                !=
                len(
                    dataframe
                )
            )
        ):

            core_complete = False

        column_rows.append(
            {
                "season_label": season_label,
                "league": league,
                "column": column,
                "present": True,
                "rows": int(
                    len(
                        dataframe
                    )
                ),
                "numeric_rows": (
                    numeric_rows
                ),
                "non_null_rows": (
                    non_null_rows
                ),
                "non_numeric_rows": (
                    non_numeric_rows
                ),
                "min": (
                    float(
                        numeric.min()
                    )
                    if numeric.notna().any()
                    else np.nan
                ),
                "max": (
                    float(
                        numeric.max()
                    )
                    if numeric.notna().any()
                    else np.nan
                ),
                "mean": (
                    float(
                        numeric.mean()
                    )
                    if numeric.notna().any()
                    else np.nan
                ),
                "median": (
                    float(
                        numeric.median()
                    )
                    if numeric.notna().any()
                    else np.nan
                ),
                "p99": (
                    float(
                        numeric.quantile(
                            0.99
                        )
                    )
                    if numeric.notna().any()
                    else np.nan
                ),
                "negative_rows": (
                    negative_rows
                ),
                "above_plausibility_rows": (
                    above_rows
                ),
            }
        )

        anomaly_mask = (
            numeric.lt(
                0
            )
            |
            numeric.gt(
                maximum
            )
            |
            (
                raw.notna()
                &
                numeric.isna()
            )
        )

        if anomaly_mask.any():

            anomaly_indices = (
                dataframe.index[
                    anomaly_mask
                ]
            )

            for index in anomaly_indices:

                source_row = (
                    dataframe.loc[
                        index
                    ]
                )

                anomaly_rows.append(
                    {
                        "season_label": (
                            season_label
                        ),
                        "league": (
                            league
                        ),
                        "file": str(
                            path.relative_to(
                                ROOT
                            )
                        ),
                        "row_index": int(
                            index
                        ),
                        "date": (
                            source_row.get(
                                "Date"
                            )
                        ),
                        "home_team": (
                            source_row.get(
                                "HomeTeam"
                            )
                        ),
                        "away_team": (
                            source_row.get(
                                "AwayTeam"
                            )
                        ),
                        "column": (
                            column
                        ),
                        "raw_value": (
                            raw.loc[
                                index
                            ]
                        ),
                        "numeric_value": (
                            numeric.loc[
                                index
                            ]
                        ),
                        "plausibility_max": (
                            maximum
                        ),
                        "reason": (
                            "negative"
                            if (
                                pd.notna(
                                    numeric.loc[
                                        index
                                    ]
                                )
                                and
                                numeric.loc[
                                    index
                                ]
                                <
                                0
                            )
                            else (
                                "above_plausibility"
                                if (
                                    pd.notna(
                                        numeric.loc[
                                            index
                                        ]
                                    )
                                    and
                                    numeric.loc[
                                        index
                                    ]
                                    >
                                    maximum
                                )
                                else
                                "non_numeric"
                            )
                        ),
                    }
                )

    file_row[
        "core_stats_complete"
    ] = bool(
        core_complete
    )

    return (
        file_row,
        column_rows,
        anomaly_rows,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=" * 110
    )

    print(
        "UCL PREDICTOR V2 - CANDIDATE C"
    )

    print(
        "DOMESTIC ADVANCED PERFORMANCE STATS AUDIT"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Research-only audit."
    )

    print(
        "No model is trained."
    )

    print(
        "No raw data is modified."
    )

    print(
        "Plausibility thresholds only FLAG values; "
        "nothing is automatically deleted."
    )

    print()

    files = (
        expected_files()
    )

    print(
        f"Expected files: "
        f"{len(files)}"
    )

    file_rows = []

    column_rows = []

    anomaly_rows = []

    for (
        season,
        league,
        path,
    ) in files:

        print(
            f"Auditing "
            f"{season} / {league}..."
        )

        (
            file_row,
            file_columns,
            file_anomalies,
        ) = audit_file(
            season_label=season,
            league=league,
            path=path,
        )

        file_rows.append(
            file_row
        )

        column_rows.extend(
            file_columns
        )

        anomaly_rows.extend(
            file_anomalies
        )

    file_audit = pd.DataFrame(
        file_rows
    )

    column_audit = pd.DataFrame(
        column_rows
    )

    anomalies = pd.DataFrame(
        anomaly_rows
    )

    existing_files = int(
        file_audit[
            "exists"
        ].sum()
    )

    complete_stat_files = int(
        (
            file_audit[
                "exists"
            ]
            &
            file_audit[
                "all_stat_columns_present"
            ]
        ).sum()
    )

    complete_core_files = int(
        (
            file_audit[
                "exists"
            ]
            &
            file_audit[
                "core_stats_complete"
            ]
        ).sum()
    )

    total_rows = int(
        file_audit[
            "rows"
        ].sum()
    )

    missing_files = file_audit[
        ~file_audit[
            "exists"
        ]
    ]

    incomplete_stats = file_audit[
        file_audit[
            "exists"
        ]
        &
        ~file_audit[
            "all_stat_columns_present"
        ]
    ]

    incomplete_core = file_audit[
        file_audit[
            "exists"
        ]
        &
        ~file_audit[
            "core_stats_complete"
        ]
    ]

    print()
    print(
        "=" * 110
    )

    print(
        "OVERALL FILE COVERAGE"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Expected files: "
        f"{len(files)}"
    )

    print(
        f"Existing files: "
        f"{existing_files}"
    )

    print(
        f"Files with all 12 stat columns: "
        f"{complete_stat_files}"
    )

    print(
        f"Files with complete core "
        f"shots/SOT/corners data: "
        f"{complete_core_files}"
    )

    print(
        f"Total domestic match rows audited: "
        f"{total_rows:,}"
    )

    print()

    if missing_files.empty:

        print(
            "Missing expected files: NONE"
        )

    else:

        print(
            "Missing expected files:"
        )

        print(
            missing_files[
                [
                    "season_label",
                    "league",
                    "path",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 110
    )

    print(
        "CORE PERFORMANCE COLUMN COVERAGE"
    )

    print(
        "=" * 110
    )

    print()

    for column in (
        CORE_PERFORMANCE_COLUMNS
    ):

        subset = column_audit[
            column_audit[
                "column"
            ]
            ==
            column
        ]

        if subset.empty:

            print(
                f"{column}: NO DATA"
            )

            continue

        total_expected = int(
            subset[
                "rows"
            ].sum()
        )

        total_numeric = int(
            subset[
                "numeric_rows"
            ].sum()
        )

        rate = (
            total_numeric
            /
            total_expected
            if total_expected
            else 0.0
        )

        print(
            f"{column:<4} "
            f"{total_numeric:>7,}/"
            f"{total_expected:<7,} "
            f"({rate * 100:6.2f}%)"
        )

    print()
    print(
        "=" * 110
    )

    print(
        "PLAUSIBILITY / DATA QUALITY FLAGS"
    )

    print(
        "=" * 110
    )

    print()

    if anomalies.empty:

        print(
            "No flagged values."
        )

    else:

        print(
            f"Flagged cells: "
            f"{len(anomalies):,}"
        )

        print()

        reason_counts = (
            anomalies[
                "reason"
            ]
            .value_counts()
        )

        print(
            "By reason:"
        )

        print(
            reason_counts.to_string()
        )

        print()

        column_counts = (
            anomalies[
                "column"
            ]
            .value_counts()
        )

        print(
            "By column:"
        )

        print(
            column_counts.to_string()
        )

        print()

        print(
            "First 50 flagged cells:"
        )

        print(
            anomalies.head(
                50
            ).to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 110
    )

    print(
        "FILES WITH INCOMPLETE CORE DATA"
    )

    print(
        "=" * 110
    )

    print()

    if incomplete_core.empty:

        print(
            "NONE"
        )

    else:

        print(
            incomplete_core[
                [
                    "season_label",
                    "league",
                    "rows",
                    "missing_stat_columns",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 110
    )

    print(
        "FILES WITH INCOMPLETE FULL STAT SCHEMA"
    )

    print(
        "=" * 110
    )

    print()

    if incomplete_stats.empty:

        print(
            "NONE"
        )

    else:

        print(
            incomplete_stats[
                [
                    "season_label",
                    "league",
                    "rows",
                    "missing_stat_columns",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_audit.to_csv(
        FILE_AUDIT_PATH,
        index=False,
    )

    column_audit.to_csv(
        COLUMN_AUDIT_PATH,
        index=False,
    )

    anomalies.to_csv(
        ANOMALY_PATH,
        index=False,
    )

    summary = {
        "experiment": (
            "candidate_c_domestic_advanced_stats_audit"
        ),
        "status": (
            "research_only"
        ),
        "expected_files": int(
            len(
                files
            )
        ),
        "existing_files": (
            existing_files
        ),
        "files_with_all_stat_columns": (
            complete_stat_files
        ),
        "files_with_complete_core_stats": (
            complete_core_files
        ),
        "total_match_rows": (
            total_rows
        ),
        "core_performance_columns": (
            CORE_PERFORMANCE_COLUMNS
        ),
        "discipline_columns": (
            DISCIPLINE_COLUMNS
        ),
        "flagged_cells": int(
            len(
                anomalies
            )
        ),
        "raw_data_modified": False,
        "production_modified": False,
        "plausibility_policy": (
            "flag_only_no_automatic_correction"
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
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
        "Raw files modified: NO"
    )

    print(
        "Production feature matrix modified: NO"
    )

    print(
        "Production model modified: NO"
    )

    print(
        "Plausibility flags automatically corrected: NO"
    )

    print()

    print(
        "=" * 110
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 110
    )

    print()

    print(
        FILE_AUDIT_PATH
    )

    print(
        COLUMN_AUDIT_PATH
    )

    print(
        ANOMALY_PATH
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()