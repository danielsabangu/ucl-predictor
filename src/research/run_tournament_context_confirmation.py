from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research import (
    run_domestic_advanced_common_coverage_backtest as c1,
)
from src.research import (
    run_tournament_context_ablation as e1,
)


# ============================================================
# PATHS
# ============================================================


ROOT = Path(__file__).resolve().parents[2]


ABLATION_METADATA_PATH = (
    ROOT
    / "reports"
    / "research"
    / "tournament_context_ablation_summary.json"
)


REPORT_DIR = (
    ROOT
    / "reports"
    / "research"
)


METRICS_OUTPUT_PATH = (
    REPORT_DIR
    / "tournament_context_2025_confirmation_metrics.csv"
)


SUMMARY_OUTPUT_PATH = (
    REPORT_DIR
    / "tournament_context_2025_confirmation_summary.json"
)


# ============================================================
# LOCKED POLICY
# ============================================================


LOCKED_SCENARIO = "ALL_CONTEXT"


CONFIRMATION_SEASON = 2025


EXPECTED_CONTROL_FEATURES = 290


EXPECTED_CANDIDATE_FEATURES = 298


EXPECTED_CONFIRMATION_MATCHES = 189


EXPECTED_TRAINING_ROWS = 1656


# ============================================================
# CONFIRMATION POLICY
# ============================================================
#
# Primary probability confirmation:
#
#   1. Log loss must improve versus identical-row CONTROL.
#   2. Brier score must not worsen.
#
# Accuracy, F1, draw recall and ECE are diagnostics.
#
# No feature selection or tuning occurs after opening 2025.
# ============================================================


def load_locked_ablation_metadata() -> dict:
    if not ABLATION_METADATA_PATH.exists():

        raise FileNotFoundError(
            "Candidate E development metadata not found:\n"
            f"{ABLATION_METADATA_PATH}\n\n"
            "Run first:\n"
            "python -m "
            "src.research.run_tournament_context_ablation"
        )

    with ABLATION_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    selected = metadata.get(
        "selected_scenario"
    )

    if (
        selected
        !=
        LOCKED_SCENARIO
    ):

        raise RuntimeError(
            "Candidate E locked scenario mismatch.\n"
            f"Expected: {LOCKED_SCENARIO}\n"
            f"Development metadata says: {selected}"
        )

    if bool(
        metadata.get(
            "confirmation_2025_used",
            False,
        )
    ):

        raise RuntimeError(
            "Development metadata says the 2025 "
            "confirmation was already used."
        )

    scenarios = metadata.get(
        "scenarios",
        {}
    )

    locked_metadata = scenarios.get(
        LOCKED_SCENARIO
    )

    if not isinstance(
        locked_metadata,
        dict,
    ):

        raise RuntimeError(
            "Locked Candidate E scenario is missing "
            "from development metadata."
        )

    if not bool(
        locked_metadata.get(
            "passes_signal_screen",
            False,
        )
    ):

        raise RuntimeError(
            "Locked Candidate E scenario did not pass "
            "the development signal screen."
        )

    return metadata


def main() -> None:
    print(
        "=" * 110
    )

    print(
        "UCL PREDICTOR V2 - CANDIDATE E"
    )

    print(
        "ONE-TIME 2025/26 TOURNAMENT CONTEXT CONFIRMATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Locked development scenario: "
        f"{LOCKED_SCENARIO}"
    )

    print(
        "No further Candidate E feature selection."
    )

    print(
        "No hyperparameter tuning."
    )

    print(
        "Common-coverage training population: "
        "2012-2024."
    )

    print(
        "Confirmation season: 2025/26."
    )

    print(
        "Production remains unchanged."
    )

    # --------------------------------------------------------
    # VERIFY DEVELOPMENT LOCK
    # --------------------------------------------------------

    development_metadata = (
        load_locked_ablation_metadata()
    )

    print()

    print(
        "Development lock verified."
    )

    # --------------------------------------------------------
    # LOAD IDENTICAL DATA PIPELINE
    # --------------------------------------------------------

    base, control_features = (
        e1.load_base_data()
    )

    context = (
        e1.load_context_features()
    )

    dataframe = (
        e1.merge_context_features(
            base=base,
            context=context,
        )
    )

    eligible = (
        e1.build_eligible_dataset(
            dataframe
        )
    )

    # --------------------------------------------------------
    # LOCK FEATURES
    # --------------------------------------------------------

    locked_additions = list(
        e1.ALL_CONTEXT_FEATURES
    )

    if (
        e1.SCENARIOS[
            LOCKED_SCENARIO
        ]
        !=
        locked_additions
    ):

        raise RuntimeError(
            "ALL_CONTEXT feature definition changed "
            "since development selection."
        )

    if len(
        control_features
    ) != EXPECTED_CONTROL_FEATURES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_CONTROL_FEATURES} "
            "frozen V1 features, found "
            f"{len(control_features)}."
        )

    collisions = [
        feature
        for feature
        in locked_additions
        if feature
        in control_features
    ]

    if collisions:

        raise RuntimeError(
            "Candidate E feature collision detected: "
            f"{collisions}"
        )

    candidate_features = [
        *control_features,
        *locked_additions,
    ]

    if len(
        candidate_features
    ) != EXPECTED_CANDIDATE_FEATURES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_CANDIDATE_FEATURES} "
            "locked Candidate E features, found "
            f"{len(candidate_features)}."
        )

    if len(
        set(
            candidate_features
        )
    ) != len(
        candidate_features
    ):

        raise RuntimeError(
            "Duplicate feature found in locked "
            "Candidate E feature set."
        )

    # --------------------------------------------------------
    # ONE-TIME SPLIT
    # --------------------------------------------------------

    train = eligible[
        eligible[
            "season"
        ]
        <
        CONFIRMATION_SEASON
    ].copy()

    confirmation = eligible[
        eligible[
            "season"
        ]
        ==
        CONFIRMATION_SEASON
    ].copy()

    if len(
        train
    ) != EXPECTED_TRAINING_ROWS:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_TRAINING_ROWS:,} "
            "2012-2024 training rows, found "
            f"{len(train):,}."
        )

    if len(
        confirmation
    ) != EXPECTED_CONFIRMATION_MATCHES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_CONFIRMATION_MATCHES} "
            "2025/26 confirmation rows, found "
            f"{len(confirmation)}."
        )

    if (
        train[
            "date"
        ].max()
        >=
        confirmation[
            "date"
        ].min()
    ):

        raise RuntimeError(
            "Chronological overlap detected between "
            "training and 2025 confirmation."
        )

    if (
        confirmation[
            "tournament_context_available"
        ]
        .ne(
            1
        )
        .any()
    ):

        raise RuntimeError(
            "2025 confirmation does not have "
            "complete Candidate E context."
        )

    if (
        confirmation[
            locked_additions
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "2025 Candidate E features contain "
            "missing values."
        )

    print()

    print(
        "=" * 110
    )

    print(
        "LOCKED CONFIRMATION SPLIT"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Training seasons: "
        f"{train['season'].min()}-"
        f"{train['season'].max()}"
    )

    print(
        f"Confirmation rows: "
        f"{len(confirmation):,}"
    )

    print(
        f"Confirmation season: "
        f"{CONFIRMATION_SEASON}"
    )

    print()

    print(
        f"CONTROL features: "
        f"{len(control_features)}"
    )

    print(
        f"{LOCKED_SCENARIO} features: "
        f"{len(candidate_features)}"
    )

    # --------------------------------------------------------
    # TRAIN CONTROL
    # --------------------------------------------------------

    print()

    print(
        "Training identical-row CONTROL..."
    )

    control = (
        c1.fit_and_evaluate(
            train=train,
            validation=confirmation,
            features=control_features,
        )
    )

    # --------------------------------------------------------
    # TRAIN LOCKED CANDIDATE
    # --------------------------------------------------------

    print(
        f"Training locked "
        f"{LOCKED_SCENARIO}..."
    )

    candidate = (
        c1.fit_and_evaluate(
            train=train,
            validation=confirmation,
            features=candidate_features,
        )
    )

    # --------------------------------------------------------
    # TARGET IDENTITY
    # --------------------------------------------------------

    if not np.array_equal(
        np.asarray(
            control[
                "y_true"
            ]
        ),
        np.asarray(
            candidate[
                "y_true"
            ]
        ),
    ):

        raise RuntimeError(
            "CONTROL and Candidate E confirmation "
            "targets differ."
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    rows = []

    for metric in c1.METRICS:

        control_value = (
            e1.get_metric(
                control,
                metric,
            )
        )

        candidate_value = (
            e1.get_metric(
                candidate,
                metric,
            )
        )

        improvement = (
            c1.improvement(
                metric=metric,
                control=control_value,
                candidate=candidate_value,
            )
        )

        rows.append(
            {
                "metric": (
                    metric
                ),
                "better_direction": (
                    "lower"
                    if metric
                    in c1.LOWER_IS_BETTER
                    else
                    "higher"
                ),
                "control": float(
                    control_value
                ),
                "candidate": float(
                    candidate_value
                ),
                "candidate_improvement": float(
                    improvement
                ),
            }
        )

    metrics = pd.DataFrame(
        rows
    )

    log_loss_row = (
        metrics[
            metrics[
                "metric"
            ]
            ==
            "log_loss"
        ]
        .iloc[
            0
        ]
    )

    brier_row = (
        metrics[
            metrics[
                "metric"
            ]
            ==
            "brier_score"
        ]
        .iloc[
            0
        ]
    )

    log_loss_improved = bool(
        float(
            log_loss_row[
                "candidate_improvement"
            ]
        )
        >
        0.0
    )

    brier_not_worse = bool(
        float(
            brier_row[
                "candidate_improvement"
            ]
        )
        >=
        -1e-12
    )

    confirmation_pass = bool(
        log_loss_improved
        and
        brier_not_worse
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
    )

    summary = {
        "experiment": (
            "candidate_e_tournament_context_"
            "2025_confirmation"
        ),
        "status": (
            "complete"
        ),
        "comparison_type": (
            "one_time_common_coverage_confirmation"
        ),
        "locked_scenario": (
            LOCKED_SCENARIO
        ),
        "locked_added_features": (
            locked_additions
        ),
        "control_feature_count": int(
            len(
                control_features
            )
        ),
        "candidate_feature_count": int(
            len(
                candidate_features
            )
        ),
        "training_first_season": int(
            train[
                "season"
            ].min()
        ),
        "training_last_season": int(
            train[
                "season"
            ].max()
        ),
        "training_rows": int(
            len(
                train
            )
        ),
        "confirmation_season": int(
            CONFIRMATION_SEASON
        ),
        "confirmation_rows": int(
            len(
                confirmation
            )
        ),
        "confirmation_policy": {
            "log_loss_must_improve": True,
            "brier_must_not_worsen": True,
            "secondary_metrics_do_not_override": True,
        },
        "confirmation_results": {
            "log_loss_improved": (
                log_loss_improved
            ),
            "brier_not_worse": (
                brier_not_worse
            ),
            "passes_confirmation": (
                confirmation_pass
            ),
        },
        "metrics": {
            row[
                "metric"
            ]: {
                "control": float(
                    row[
                        "control"
                    ]
                ),
                "candidate": float(
                    row[
                        "candidate"
                    ]
                ),
                "candidate_improvement": float(
                    row[
                        "candidate_improvement"
                    ]
                ),
            }
            for row
            in rows
        },
        "development_selected_scenario": (
            development_metadata.get(
                "selected_scenario"
            )
        ),
        "additional_feature_selection_after_development": False,
        "hyperparameters_tuned": False,
        "raw_transfermarkt_aggregate_used": False,
        "current_match_score_used_as_candidate_feature": False,
        "fuzzy_matching_used": False,
        "production_modified": False,
    }

    with SUMMARY_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()

    print(
        "=" * 110
    )

    print(
        "2025/26 CONFIRMATION METRICS"
    )

    print(
        "=" * 110
    )

    display = metrics.copy()

    for column in [
        "control",
        "candidate",
        "candidate_improvement",
    ]:

        display[
            column
        ] = display[
            column
        ].map(
            lambda value: (
                f"{value:+.6f}"
                if column
                ==
                "candidate_improvement"
                else
                f"{value:.6f}"
            )
        )

    print()

    print(
        display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    print()

    print(
        "=" * 110
    )

    print(
        "CANDIDATE E CONFIRMATION DECISION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Log Loss improvement: "
        f"{float(log_loss_row['candidate_improvement']):+.6f}"
    )

    print(
        f"Brier improvement: "
        f"{float(brier_row['candidate_improvement']):+.6f}"
    )

    print()

    if confirmation_pass:

        print(
            "Decision: "
            "CONFIRM_TOURNAMENT_CONTEXT_SIGNAL"
        )

        print()

        print(
            "Locked ALL_CONTEXT passed the one-time "
            "2025 probability confirmation."
        )

        print(
            "Candidate E may proceed to the final "
            "V2 combination experiment."
        )

    else:

        print(
            "Decision: "
            "REJECT_TOURNAMENT_CONTEXT_AFTER_CONFIRMATION"
        )

        print()

        print(
            "Locked ALL_CONTEXT did not satisfy the "
            "pre-registered 2025 probability confirmation."
        )

        print(
            "Do not test alternative Candidate E subsets "
            "on 2025."
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

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
        "Development-selected scenario unchanged: PASS"
    )

    print(
        "CONTROL and candidate use identical rows: PASS"
    )

    print(
        "Training restricted to common coverage 2012-2024: PASS"
    )

    print(
        "2025 opened exactly once for locked Candidate E: PASS"
    )

    print(
        "No post-2025 feature selection performed: PASS"
    )

    print(
        "Raw Transfermarkt aggregate used: NO"
    )

    print(
        "Current-match score used as Candidate E feature: NO"
    )

    print(
        "Hyperparameters tuned: NO"
    )

    print(
        "Production modified: NO"
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
        METRICS_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()