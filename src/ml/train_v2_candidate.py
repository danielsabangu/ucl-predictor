from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_METADATA_DIR,
    TRAINED_MODELS_DIR,
)
from src.ml import train_production_model as v1


ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_UEFA_FILE = (
    ROOT
    / "data"
    / "research"
    / "ucl_uefa_strength_features.csv"
)

CONFIRMATION_SUMMARY_FILE = (
    ROOT
    / "reports"
    / "research"
    / "uefa_diffs_2025_confirmation_summary.json"
)

V2_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "ucl_predictor_v2_candidate.joblib"
)

V2_METADATA_FILE = (
    MODEL_METADATA_DIR
    / "ucl_predictor_v2_candidate.json"
)

EXPECTED_ROWS = 2089

EXPECTED_V1_FEATURES = 290

EXPECTED_V2_FEATURES = 292

MODEL_VERSION = "ucl_predictor_v2_candidate"

SELECTED_UEFA_FEATURES = [
    "uefa_coefficient_diff",
    "uefa_strength_percentile_diff",
]

IDENTITY_COLUMNS = [
    "match_id",
    "date",
    "season",
    "home_team",
    "away_team",
]


def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"{description} is empty:\n{path}"
        )


def sha256_file(
    path: Path,
) -> str:
    require_file(
        path,
        str(path),
    )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def verify_locked_confirmation() -> dict:
    require_file(
        CONFIRMATION_SUMMARY_FILE,
        "Locked UEFA confirmation summary",
    )

    with CONFIRMATION_SUMMARY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    expected_exact = {
        "experiment": (
            "uefa_diffs_2025_confirmation"
        ),
        "status": (
            "research_only"
        ),
        "selection_source": (
            "2018_2024_rolling_ablation"
        ),
        "comparison_season": 2025,
        "comparison_season_already_consumed": True,
        "production_modified": False,
        "control_feature_count": (
            EXPECTED_V1_FEATURES
        ),
        "candidate_feature_count": (
            EXPECTED_V2_FEATURES
        ),
        "no_further_uefa_selection_from_result": True,
    }

    for (
        key,
        expected,
    ) in expected_exact.items():
        actual = summary.get(
            key
        )

        if actual != expected:
            raise RuntimeError(
                "Locked UEFA confirmation metadata "
                f"does not match for {key!r}: "
                f"expected {expected!r}, "
                f"found {actual!r}."
            )

    selected = summary.get(
        "selected_uefa_features"
    )

    if selected != SELECTED_UEFA_FEATURES:
        raise RuntimeError(
            "Locked UEFA feature selection changed. "
            f"Expected {SELECTED_UEFA_FEATURES}, "
            f"found {selected}."
        )

    control_metrics = summary.get(
        "control_metrics"
    )

    candidate_metrics = summary.get(
        "candidate_metrics"
    )

    if not isinstance(
        control_metrics,
        dict,
    ):
        raise RuntimeError(
            "Control metrics missing from locked confirmation."
        )

    if not isinstance(
        candidate_metrics,
        dict,
    ):
        raise RuntimeError(
            "Candidate metrics missing from locked confirmation."
        )

    for metric in [
        "log_loss",
        "brier_score",
    ]:
        if metric not in control_metrics:
            raise RuntimeError(
                f"Control confirmation missing {metric}."
            )

        if metric not in candidate_metrics:
            raise RuntimeError(
                f"Candidate confirmation missing {metric}."
            )

        control_value = float(
            control_metrics[
                metric
            ]
        )

        candidate_value = float(
            candidate_metrics[
                metric
            ]
        )

        if not np.isfinite(
            control_value
        ):
            raise RuntimeError(
                f"Non-finite control {metric}."
            )

        if not np.isfinite(
            candidate_value
        ):
            raise RuntimeError(
                f"Non-finite candidate {metric}."
            )

        if candidate_value >= control_value:
            raise RuntimeError(
                "Locked UEFA candidate no longer improves "
                f"{metric}: control={control_value}, "
                f"candidate={candidate_value}."
            )

    return summary


def load_historical_uefa_features() -> pd.DataFrame:
    require_file(
        HISTORICAL_UEFA_FILE,
        "Historical UEFA feature layer",
    )

    frame = pd.read_csv(
        HISTORICAL_UEFA_FILE,
        low_memory=False,
    )

    required = set(
        IDENTITY_COLUMNS
        + SELECTED_UEFA_FEATURES
    )

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Historical UEFA feature layer "
            f"is missing columns: {sorted(missing)}"
        )

    if len(
        frame
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} historical "
            f"UEFA rows, found {len(frame):,}."
        )

    if frame[
        "match_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate match IDs found in "
            "historical UEFA feature layer."
        )

    frame = frame.copy()

    frame[
        "date"
    ] = pd.to_datetime(
        frame[
            "date"
        ],
        errors="raise",
    )

    frame[
        "season"
    ] = pd.to_numeric(
        frame[
            "season"
        ],
        errors="raise",
    ).astype(
        int
    )

    for column in (
        SELECTED_UEFA_FEATURES
    ):
        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="raise",
        )

        values = (
            frame[
                column
            ]
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                "Historical UEFA feature "
                f"{column} contains non-finite values."
            )

    return frame


def merge_v2_features(
    base: pd.DataFrame,
    uefa: pd.DataFrame,
) -> pd.DataFrame:
    for feature in (
        SELECTED_UEFA_FEATURES
    ):
        if feature in base.columns:
            raise RuntimeError(
                "V2 UEFA feature already exists in "
                f"the frozen V1 matrix: {feature}"
            )

    right = uefa[
        IDENTITY_COLUMNS
        + SELECTED_UEFA_FEATURES
    ].copy()

    right = right.rename(
        columns={
            "date": "uefa_date",
            "season": "uefa_season",
            "home_team": "uefa_home_team",
            "away_team": "uefa_away_team",
        }
    )

    merged = base.merge(
        right,
        on="match_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(
        merged
    ) != len(
        base
    ):
        raise RuntimeError(
            "Historical row count changed "
            "while merging UEFA features."
        )

    if merged[
        "match_id"
    ].tolist() != base[
        "match_id"
    ].tolist():
        raise RuntimeError(
            "Historical match order changed "
            "while merging UEFA features."
        )

    required_right = [
        "uefa_date",
        "uefa_season",
        "uefa_home_team",
        "uefa_away_team",
        *SELECTED_UEFA_FEATURES,
    ]

    missing_counts = (
        merged[
            required_right
        ]
        .isna()
        .sum()
    )

    if (
        missing_counts
        > 0
    ).any():
        raise RuntimeError(
            "Missing UEFA merge values:\n"
            f"{missing_counts.to_string()}"
        )

    base_dates = pd.to_datetime(
        merged[
            "date"
        ],
        errors="raise",
    ).dt.normalize()

    uefa_dates = pd.to_datetime(
        merged[
            "uefa_date"
        ],
        errors="raise",
    ).dt.normalize()

    if not base_dates.equals(
        uefa_dates
    ):
        raise RuntimeError(
            "Historical date identity mismatch "
            "between V1 and UEFA data."
        )

    if not (
        merged[
            "season"
        ]
        .astype(int)
        .eq(
            merged[
                "uefa_season"
            ].astype(int)
        )
        .all()
    ):
        raise RuntimeError(
            "Historical season identity mismatch "
            "between V1 and UEFA data."
        )

    if not (
        merged[
            "home_team"
        ]
        .astype(str)
        .eq(
            merged[
                "uefa_home_team"
            ].astype(str)
        )
        .all()
    ):
        raise RuntimeError(
            "Historical home-team identity mismatch "
            "between V1 and UEFA data."
        )

    if not (
        merged[
            "away_team"
        ]
        .astype(str)
        .eq(
            merged[
                "uefa_away_team"
            ].astype(str)
        )
        .all()
    ):
        raise RuntimeError(
            "Historical away-team identity mismatch "
            "between V1 and UEFA data."
        )

    merged = merged.drop(
        columns=[
            "uefa_date",
            "uefa_season",
            "uefa_home_team",
            "uefa_away_team",
        ]
    )

    return merged


def validate_v2_feature_contract(
    *,
    control_features: list[str],
    candidate_features: list[str],
    dataframe: pd.DataFrame,
) -> None:
    if len(
        control_features
    ) != EXPECTED_V1_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_V1_FEATURES} "
            "frozen V1 features, found "
            f"{len(control_features)}."
        )

    if len(
        candidate_features
    ) != EXPECTED_V2_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_V2_FEATURES} "
            "V2 candidate features, found "
            f"{len(candidate_features)}."
        )

    if len(
        candidate_features
    ) != len(
        set(
            candidate_features
        )
    ):
        raise RuntimeError(
            "Duplicate V2 feature names detected."
        )

    if candidate_features[
        :EXPECTED_V1_FEATURES
    ] != control_features:
        raise RuntimeError(
            "Frozen V1 feature order changed "
            "inside the V2 candidate."
        )

    if candidate_features[
        EXPECTED_V1_FEATURES:
    ] != SELECTED_UEFA_FEATURES:
        raise RuntimeError(
            "The final two V2 features are not "
            "the locked UEFA differences."
        )

    if len(
        dataframe
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} V2 "
            f"historical rows, found "
            f"{len(dataframe):,}."
        )

    missing = (
        dataframe[
            candidate_features
        ]
        .isna()
        .sum()
    )

    uefa_missing = missing[
        SELECTED_UEFA_FEATURES
    ]

    if (
        uefa_missing
        > 0
    ).any():
        raise RuntimeError(
            "Locked UEFA features contain "
            "missing historical values:\n"
            f"{uefa_missing.to_string()}"
        )

    for column in (
        SELECTED_UEFA_FEATURES
    ):
        values = (
            dataframe[
                column
            ]
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"Non-finite values in {column}."
            )


def validate_locked_model_spec(
    model,
) -> None:
    if "imputer" not in (
        model.named_steps
    ):
        raise RuntimeError(
            "Locked model is missing imputer step."
        )

    if "model" not in (
        model.named_steps
    ):
        raise RuntimeError(
            "Locked model is missing CatBoost step."
        )

    imputer = model.named_steps[
        "imputer"
    ]

    if imputer.strategy != "median":
        raise RuntimeError(
            "Locked imputer strategy changed."
        )

    if not bool(
        imputer.add_indicator
    ):
        raise RuntimeError(
            "Locked imputer missing-indicator "
            "policy changed."
        )

    estimator = model.named_steps[
        "model"
    ]

    if (
        estimator.__class__.__name__
        !=
        "CatBoostClassifier"
    ):
        raise RuntimeError(
            "Locked estimator is no longer "
            "CatBoostClassifier."
        )

    params = estimator.get_params()

    expected = {
        "iterations": 700,
        "learning_rate": 0.025,
        "depth": 5,
        "loss_function": "MultiClass",
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "allow_writing_files": False,
    }

    for (
        key,
        wanted,
    ) in expected.items():
        actual = params.get(
            key
        )

        if actual != wanted:
            raise RuntimeError(
                "Locked CatBoost specification "
                f"changed for {key}: "
                f"expected {wanted!r}, "
                f"found {actual!r}."
            )


def validate_probability_integrity(
    model,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    sample = dataframe[
        feature_columns
    ].head(
        32
    )

    probabilities = model.predict_proba(
        sample
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.shape != (
        len(
            sample
        ),
        3,
    ):
        raise RuntimeError(
            "Unexpected V2 probability shape: "
            f"{probabilities.shape}"
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "V2 candidate produced "
            "non-finite probabilities."
        )

    if (
        probabilities
        < 0
    ).any():
        raise RuntimeError(
            "V2 candidate produced "
            "negative probabilities."
        )

    if (
        probabilities
        > 1
    ).any():
        raise RuntimeError(
            "V2 candidate produced "
            "probabilities above 1."
        )

    sums = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        sums,
        1.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "V2 candidate probabilities "
            "do not sum to one."
        )


def prepare_v2_training_data() -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
    dict,
]:
    confirmation = (
        verify_locked_confirmation()
    )

    v1.verify_final_evaluation()

    manifest = (
        v1.load_feature_manifest()
    )

    master_features = manifest[
        "feature_columns"
    ]

    (
        control_features,
        transfer_features,
    ) = (
        v1.build_production_features(
            master_features
        )
    )

    base = (
        v1.load_training_data(
            master_features
        )
    )

    uefa = (
        load_historical_uefa_features()
    )

    dataframe = (
        merge_v2_features(
            base,
            uefa,
        )
    )

    candidate_features = [
        *control_features,
        *SELECTED_UEFA_FEATURES,
    ]

    validate_v2_feature_contract(
        control_features=(
            control_features
        ),
        candidate_features=(
            candidate_features
        ),
        dataframe=dataframe,
    )

    return (
        dataframe,
        master_features,
        control_features,
        transfer_features,
        confirmation,
    )


def train_v2_candidate(
    *,
    validate_only: bool,
) -> None:
    print(
        "=" * 80
    )
    print(
        "UCL PREDICTOR V2 - CANDIDATE TRAINING"
    )
    print(
        "=" * 80
    )
    print()

    print(
        "Policy:"
    )
    print(
        "  V1 production remains untouched."
    )
    print(
        "  2025/26 confirmation is NOT re-run."
    )
    print(
        "  No feature selection occurs here."
    )
    print(
        "  No hyperparameter tuning occurs here."
    )
    print()

    (
        dataframe,
        master_features,
        control_features,
        transfer_features,
        confirmation,
    ) = prepare_v2_training_data()

    candidate_features = [
        *control_features,
        *SELECTED_UEFA_FEATURES,
    ]

    model = (
        v1.build_production_model()
    )

    validate_locked_model_spec(
        model
    )

    print(
        "=" * 80
    )
    print(
        "LOCKED V2 CONTRACT"
    )
    print(
        "=" * 80
    )
    print()

    print(
        f"Historical rows: "
        f"{len(dataframe):,}"
    )

    print(
        f"Training seasons: "
        f"{dataframe['season'].min()} "
        f"to "
        f"{dataframe['season'].max()}"
    )

    print(
        f"V1 master features: "
        f"{len(master_features):,}"
    )

    print(
        f"Transfer features excluded: "
        f"{len(transfer_features):,}"
    )

    print(
        f"Frozen V1 probability features: "
        f"{len(control_features):,}"
    )

    print(
        f"Locked UEFA additions: "
        f"{len(SELECTED_UEFA_FEATURES):,}"
    )

    print(
        f"V2 candidate features: "
        f"{len(candidate_features):,}"
    )

    print()

    for feature in (
        SELECTED_UEFA_FEATURES
    ):
        print(
            f"  + {feature}"
        )

    availability = (
        1.0
        -
        dataframe[
            candidate_features
        ]
        .isna()
        .mean()
        .mean()
    )

    print()
    print(
        "Overall 292-feature historical "
        f"availability: "
        f"{availability * 100:.2f}%"
    )

    control_metrics = confirmation[
        "control_metrics"
    ]

    candidate_metrics = confirmation[
        "candidate_metrics"
    ]

    print()
    print(
        "Locked 2025 historical confirmation "
        "(read only):"
    )

    print(
        "  Log loss: "
        f"{float(control_metrics['log_loss']):.6f}"
        " -> "
        f"{float(candidate_metrics['log_loss']):.6f}"
    )

    print(
        "  Brier:    "
        f"{float(control_metrics['brier_score']):.6f}"
        " -> "
        f"{float(candidate_metrics['brier_score']):.6f}"
    )

    print()

    print(
        "Exact frozen CatBoost specification: PASS"
    )
    print(
        "290 V1 feature order preserved: PASS"
    )
    print(
        "Exactly two locked UEFA additions: PASS"
    )
    print(
        "2,089 historical rows aligned: PASS"
    )
    print(
        "2025 confirmation metadata locked: PASS"
    )
    print(
        "No new 2025 evaluation performed: PASS"
    )

    if validate_only:
        print()
        print(
            "=" * 80
        )
        print(
            "VALIDATION-ONLY COMPLETE"
        )
        print(
            "=" * 80
        )
        print()

        print(
            "Model trained: NO"
        )
        print(
            "V2 artifact written: NO"
        )
        print(
            "V1 artifact modified: NO"
        )

        return

    require_file(
        v1.PRODUCTION_MODEL_FILE,
        "Frozen V1 production model",
    )

    require_file(
        v1.PRODUCTION_METADATA_FILE,
        "Frozen V1 production metadata",
    )

    v1_model_hash_before = (
        sha256_file(
            v1.PRODUCTION_MODEL_FILE
        )
    )

    v1_metadata_hash_before = (
        sha256_file(
            v1.PRODUCTION_METADATA_FILE
        )
    )

    X = dataframe[
        candidate_features
    ]

    y = dataframe[
        "target"
    ]

    print()
    print(
        "Training separate 292-feature "
        "V2 candidate..."
    )

    model.fit(
        X,
        y,
    )

    classes = list(
        model.classes_
    )

    if classes != [
        0,
        1,
        2,
    ]:
        raise RuntimeError(
            "Unexpected V2 model classes: "
            f"{classes}"
        )

    validate_probability_integrity(
        model,
        dataframe,
        candidate_features,
    )

    TRAINED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        V2_MODEL_FILE,
    )

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "model_version": MODEL_VERSION,
        "status": (
            "candidate_not_promoted"
        ),
        "model_role": (
            "2026/27 shadow H/D/A probability candidate"
        ),
        "model": (
            "CatBoostClassifier"
        ),
        "training_rows": int(
            len(
                dataframe
            )
        ),
        "training_first_season": int(
            dataframe[
                "season"
            ].min()
        ),
        "training_last_season": int(
            dataframe[
                "season"
            ].max()
        ),
        "training_first_date": (
            dataframe[
                "date"
            ]
            .min()
            .date()
            .isoformat()
        ),
        "training_last_date": (
            dataframe[
                "date"
            ]
            .max()
            .date()
            .isoformat()
        ),
        "v1_feature_count": int(
            len(
                control_features
            )
        ),
        "uefa_feature_count": int(
            len(
                SELECTED_UEFA_FEATURES
            )
        ),
        "candidate_feature_count": int(
            len(
                candidate_features
            )
        ),
        "feature_columns": (
            candidate_features
        ),
        "selected_uefa_features": (
            SELECTED_UEFA_FEATURES
        ),
        "excluded_transfer_feature_count": int(
            len(
                transfer_features
            )
        ),
        "model_parameters": {
            "iterations": 700,
            "learning_rate": 0.025,
            "depth": 5,
            "loss_function": (
                "MultiClass"
            ),
            "l2_leaf_reg": 5.0,
            "random_seed": 42,
        },
        "locked_confirmation": {
            "source": (
                str(
                    CONFIRMATION_SUMMARY_FILE
                    .relative_to(
                        ROOT
                    )
                )
            ),
            "comparison_season": 2025,
            "comparison_season_already_consumed": True,
            "selection_source": (
                "2018_2024_rolling_ablation"
            ),
            "control_metrics": (
                confirmation[
                    "control_metrics"
                ]
            ),
            "candidate_metrics": (
                confirmation[
                    "candidate_metrics"
                ]
            ),
            "control_top_label_ece_10bin": (
                confirmation[
                    "control_top_label_ece_10bin"
                ]
            ),
            "candidate_top_label_ece_10bin": (
                confirmation[
                    "candidate_top_label_ece_10bin"
                ]
            ),
        },
        "locked_decisions": {
            "v1_feature_order_preserved": True,
            "uefa_subset_locked": True,
            "further_2025_tuning": False,
            "transfer_features_excluded": True,
            "probability_calibration": False,
            "ensemble": False,
            "dedicated_draw_layer": False,
            "production_promoted": False,
        },
        "important_note": (
            "This is a separate V2 shadow candidate. "
            "It uses the frozen V1 CatBoost specification "
            "with exactly two previously selected UEFA "
            "difference features. The 2025/26 confirmation "
            "was not re-run and was not used for further "
            "feature or hyperparameter tuning."
        ),
    }

    with V2_METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    v1_model_hash_after = (
        sha256_file(
            v1.PRODUCTION_MODEL_FILE
        )
    )

    v1_metadata_hash_after = (
        sha256_file(
            v1.PRODUCTION_METADATA_FILE
        )
    )

    if (
        v1_model_hash_after
        !=
        v1_model_hash_before
    ):
        raise RuntimeError(
            "Frozen V1 production model changed "
            "during V2 training."
        )

    if (
        v1_metadata_hash_after
        !=
        v1_metadata_hash_before
    ):
        raise RuntimeError(
            "Frozen V1 production metadata changed "
            "during V2 training."
        )

    print()
    print(
        "=" * 80
    )
    print(
        "V2 CANDIDATE TRAINED SUCCESSFULLY"
    )
    print(
        "=" * 80
    )
    print()

    print(
        "Model version:"
    )
    print(
        f"  {MODEL_VERSION}"
    )

    print()
    print(
        "Feature count:"
    )
    print(
        f"  {len(candidate_features)}"
    )

    print()
    print(
        "Model artifact:"
    )
    print(
        f"  {V2_MODEL_FILE}"
    )

    print()
    print(
        "Metadata artifact:"
    )
    print(
        f"  {V2_METADATA_FILE}"
    )

    print()
    print(
        "Probability sanity check: PASS"
    )
    print(
        "V1 model hash unchanged: PASS"
    )
    print(
        "V1 metadata hash unchanged: PASS"
    )
    print(
        "Production promoted: NO"
    )
    print(
        "2025 confirmation re-run: NO"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train or validate the separate "
            "292-feature UCL Predictor V2 candidate."
        )
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the complete locked V2 contract "
            "without training or writing model artifacts."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_v2_candidate(
        validate_only=(
            args.validate_only
        )
    )


if __name__ == "__main__":
    main()