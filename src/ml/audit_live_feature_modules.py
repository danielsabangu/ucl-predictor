from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from src.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    TRAINED_MODELS_DIR,
)


# ============================================================
# SETTINGS
# ============================================================

LOCAL_TIMEZONE = ZoneInfo(
    "Africa/Johannesburg"
)

EXPECTED_PRODUCTION_FEATURES = 290

EXPECTED_PRODUCTION_FAMILIES = {
    "elo": 6,
    "ucl_form": 81,
    "domestic": 102,
    "squad": 101,
}


# ============================================================
# PATHS
# ============================================================

FEATURES_DIR = (
    BASE_DIR
    / "src"
    / "features"
)

PRODUCTION_MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "production_catboost.joblib"
)

MASTER_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_master.csv"
)

ELO_MATCH_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_elo.csv"
)

FORM_MATCH_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_form.csv"
)

DOMESTIC_MATCH_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_domestic_form.csv"
)

SQUAD_MATCH_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_matches_with_squad_strength.csv"
)

ELO_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_elo_latest.csv"
)

FORM_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_form_latest.csv"
)

DOMESTIC_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_domestic_form_latest.csv"
)

SQUAD_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "ucl_team_squad_strength_latest.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "models"
    / "metadata"
    / "live_feature_module_audit.json"
)


# ============================================================
# CURRENT NEW CLUBS
# ============================================================

CURRENT_NEW_CLUBS = {
    "Como 1907",
    "Fenerbahçe",
    "LASK Linz",
    "Real Betis Balompié",
    "Sabah FK",
    "Viking FK",
}


# ============================================================
# FILE HELPERS
# ============================================================


def require_file(
    path: Path,
    description: str,
) -> None:
    """
    Require one local artifact.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}"
        )


def load_header(
    path: Path,
) -> list[str]:
    """
    Read CSV header only.
    """

    require_file(
        path,
        "Required CSV",
    )

    dataframe = pd.read_csv(
        path,
        nrows=0,
    )

    return list(
        dataframe.columns
    )


# ============================================================
# PRODUCTION FEATURE SCHEMA
# ============================================================


def load_production_features() -> list[str]:
    """
    Recover the exact 290 input columns used by the frozen
    production model.

    This is the authoritative source for determining whether
    a historical pipeline-stage column is actually a model
    feature.
    """

    require_file(
        PRODUCTION_MODEL_FILE,
        "Frozen production model",
    )

    model = joblib.load(
        PRODUCTION_MODEL_FILE
    )

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is None:

        named_steps = getattr(
            model,
            "named_steps",
            None,
        )

        if named_steps is not None:

            for step in named_steps.values():

                candidate = getattr(
                    step,
                    "feature_names_in_",
                    None,
                )

                if candidate is not None:

                    feature_names = (
                        candidate
                    )

                    break

    if feature_names is None:

        raise RuntimeError(
            "Could not recover production feature names "
            "from production_catboost.joblib."
        )

    features = [
        str(
            feature
        )
        for feature in feature_names
    ]

    if (
        len(
            features
        )
        !=
        EXPECTED_PRODUCTION_FEATURES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PRODUCTION_FEATURES} "
            f"production features, found "
            f"{len(features)}."
        )

    if (
        len(
            set(
                features
            )
        )
        !=
        len(
            features
        )
    ):

        raise RuntimeError(
            "Duplicate production feature names found."
        )

    return features


# ============================================================
# FEATURE FAMILY RECONSTRUCTION
# ============================================================


def reconstruct_feature_families(
    production_features: list[str],
) -> tuple[
    dict[str, list[str]],
    dict[str, list[str]],
]:
    """
    Recover:

        1. every column generated at each historical stage
        2. only the columns from that stage actually used by
           the frozen production model

    The distinction matters because historical stage outputs
    can contain operational/state columns that were never
    model inputs.

    Example:
        Elo stage generated 8 new columns,
        but production uses only 6 of them.
    """

    master_columns = set(
        load_header(
            MASTER_FILE
        )
    )

    elo_columns = set(
        load_header(
            ELO_MATCH_FILE
        )
    )

    form_columns = set(
        load_header(
            FORM_MATCH_FILE
        )
    )

    domestic_columns = set(
        load_header(
            DOMESTIC_MATCH_FILE
        )
    )

    squad_columns = set(
        load_header(
            SQUAD_MATCH_FILE
        )
    )

    generated = {
        "elo": (
            elo_columns
            -
            master_columns
        ),

        "ucl_form": (
            form_columns
            -
            elo_columns
        ),

        "domestic": (
            domestic_columns
            -
            form_columns
        ),

        "squad": (
            squad_columns
            -
            domestic_columns
        ),
    }

    production_set = set(
        production_features
    )

    production_families = {
        family: sorted(
            columns
            &
            production_set
        )
        for (
            family,
            columns
        ) in generated.items()
    }

    raw_generated_families = {
        family: sorted(
            columns
        )
        for (
            family,
            columns
        ) in generated.items()
    }

    return (
        raw_generated_families,
        production_families,
    )


# ============================================================
# FAMILY VERIFICATION
# ============================================================


def verify_family_counts(
    *,
    raw_generated_families: dict[
        str,
        list[str],
    ],
    production_families: dict[
        str,
        list[str],
    ],
    production_features: list[str],
) -> None:
    """
    Verify only the model-selected columns against the frozen
    architecture.

    Raw stage-generated counts are displayed for diagnostic
    purposes but are not required to equal production counts.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "HISTORICAL FEATURE FAMILY VERIFICATION"
    )

    print(
        "=" * 75
    )

    production_total = 0

    for family in [
        "elo",
        "ucl_form",
        "domestic",
        "squad",
    ]:

        raw_count = len(
            raw_generated_families[
                family
            ]
        )

        production_count = len(
            production_families[
                family
            ]
        )

        expected = (
            EXPECTED_PRODUCTION_FAMILIES[
                family
            ]
        )

        production_total += (
            production_count
        )

        print(
            f"{family:15s}: "
            f"stage_generated={raw_count:3d}  "
            f"production_used={production_count:3d}"
        )

        if (
            production_count
            !=
            expected
        ):

            raise RuntimeError(
                f"{family} production family expected "
                f"{expected} features but "
                f"{production_count} were reconstructed."
            )

    print(
        f"\nProduction feature total: "
        f"{production_total}"
    )

    if (
        production_total
        !=
        EXPECTED_PRODUCTION_FEATURES
    ):

        raise RuntimeError(
            f"Expected 290 production features, "
            f"found {production_total}."
        )

    classified = set()

    for features in (
        production_families.values()
    ):

        classified.update(
            features
        )

    production_set = set(
        production_features
    )

    unclassified = (
        production_set
        -
        classified
    )

    print(
        f"Unclassified production features: "
        f"{len(unclassified)}"
    )

    if unclassified:

        raise RuntimeError(
            "Some frozen production features were not "
            "classified into the historical feature "
            "pipeline:\n"
            f"{sorted(unclassified)}"
        )

    print(
        "\nFrozen production feature families verified ✓"
    )


# ============================================================
# SHOW NON-MODEL STAGE COLUMNS
# ============================================================


def print_non_model_stage_columns(
    *,
    raw_generated_families: dict[
        str,
        list[str],
    ],
    production_families: dict[
        str,
        list[str],
    ],
) -> None:
    """
    Show stage-generated columns that are not inputs to the
    frozen model.

    This explains cases such as the 8-vs-6 Elo difference.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "STAGE COLUMNS NOT USED BY PRODUCTION MODEL"
    )

    print(
        "=" * 75
    )

    for family in [
        "elo",
        "ucl_form",
        "domestic",
        "squad",
    ]:

        raw = set(
            raw_generated_families[
                family
            ]
        )

        used = set(
            production_families[
                family
            ]
        )

        excluded = sorted(
            raw
            -
            used
        )

        print(
            f"\n{family.upper()}"
        )

        print(
            f"Generated but not model inputs: "
            f"{len(excluded)}"
        )

        if excluded:

            for column in excluded:

                print(
                    f"  {column}"
                )

        else:

            print(
                "  None"
            )


# ============================================================
# TEAM COLUMN DETECTION
# ============================================================


def detect_team_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """
    Detect team-name column in latest-state tables.
    """

    candidates = [
        "team",
        "canonical_team",
        "team_name",
        "club",
        "club_name",
    ]

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    return None


# ============================================================
# LATEST-STATE TABLE AUDIT
# ============================================================


def inspect_latest_table(
    *,
    label: str,
    path: Path,
) -> dict:
    """
    Inspect one historical latest-state artifact.
    """

    require_file(
        path,
        f"{label} latest-state table",
    )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    team_column = detect_team_column(
        dataframe
    )

    if team_column is None:

        teams = []

    else:

        teams = sorted(
            dataframe[
                team_column
            ]
            .dropna()
            .astype(
                str
            )
            .unique()
            .tolist()
        )

    available_set = set(
        teams
    )

    new_clubs_present = sorted(
        CURRENT_NEW_CLUBS
        &
        available_set
    )

    new_clubs_missing = sorted(
        CURRENT_NEW_CLUBS
        -
        available_set
    )

    return {
        "label": (
            label
        ),

        "path": str(
            path
        ),

        "rows": int(
            len(
                dataframe
            )
        ),

        "columns": list(
            dataframe.columns
        ),

        "column_count": int(
            len(
                dataframe.columns
            )
        ),

        "team_column": (
            team_column
        ),

        "unique_teams": int(
            len(
                teams
            )
        ),

        "new_clubs_present": (
            new_clubs_present
        ),

        "new_clubs_missing": (
            new_clubs_missing
        ),
    }


# ============================================================
# PYTHON SOURCE MODULE AUDIT
# ============================================================


def parse_python_module(
    path: Path,
) -> dict:
    """
    Inspect Python source using AST.

    Source is not imported or executed.
    """

    source = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    try:

        tree = ast.parse(
            source
        )

    except SyntaxError as error:

        return {
            "path": str(
                path
            ),

            "relative_path": str(
                path.relative_to(
                    BASE_DIR
                )
            ),

            "syntax_error": str(
                error
            ),

            "functions": [],

            "classes": [],

            "imports": [],

            "keyword_hits": [],
        }

    functions = []

    classes = []

    imports = []

    for node in ast.walk(
        tree
    ):

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            functions.append(
                node.name
            )

        elif isinstance(
            node,
            ast.ClassDef,
        ):

            classes.append(
                node.name
            )

        elif isinstance(
            node,
            ast.Import,
        ):

            for alias in node.names:

                imports.append(
                    alias.name
                )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):

            module_name = (
                node.module
                or
                ""
            )

            imported_names = ",".join(
                alias.name
                for alias in node.names
            )

            imports.append(
                f"{module_name}:{imported_names}"
            )

    searchable = (
        path.name.lower()
        +
        "\n"
        +
        source.lower()
    )

    keywords = [
        "elo",
        "rolling",
        "form",
        "domestic",
        "squad",
        "lineup",
        "appearance",
        "core_xi",
        "continuity",
        "rest",
        "days_since",
        "valuation",
        "player_history",
        "recent",
        "window",
    ]

    keyword_hits = [
        keyword
        for keyword in keywords
        if keyword in searchable
    ]

    return {
        "path": str(
            path
        ),

        "relative_path": str(
            path.relative_to(
                BASE_DIR
            )
        ),

        "syntax_error": None,

        "functions": sorted(
            set(
                functions
            )
        ),

        "classes": sorted(
            set(
                classes
            )
        ),

        "imports": sorted(
            set(
                imports
            )
        ),

        "keyword_hits": (
            keyword_hits
        ),
    }


def inspect_feature_modules() -> list[dict]:
    """
    Discover feature source modules likely relevant to live
    feature generation.
    """

    if not FEATURES_DIR.exists():

        raise FileNotFoundError(
            "Feature source directory not found:\n"
            f"{FEATURES_DIR}"
        )

    results = []

    for path in sorted(
        FEATURES_DIR.rglob(
            "*.py"
        )
    ):

        if (
            "__pycache__"
            in
            path.parts
        ):

            continue

        result = parse_python_module(
            path
        )

        if result[
            "keyword_hits"
        ]:

            results.append(
                result
            )

    return results


# ============================================================
# PRINT LATEST-STATE SCHEMAS
# ============================================================


def print_latest_tables(
    latest_tables: list[dict],
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LATEST-STATE TABLE SCHEMAS"
    )

    print(
        "=" * 75
    )

    for table in latest_tables:

        print(
            f"\n{table['label'].upper()}"
        )

        print(
            "-" * 75
        )

        print(
            f"Rows: "
            f"{table['rows']}"
        )

        print(
            f"Columns: "
            f"{table['column_count']}"
        )

        print(
            f"Team column: "
            f"{table['team_column']}"
        )

        print(
            f"Unique teams: "
            f"{table['unique_teams']}"
        )

        print(
            "New 2026/27 clubs present: "
            f"{table['new_clubs_present']}"
        )

        print(
            "New 2026/27 clubs missing: "
            f"{table['new_clubs_missing']}"
        )

        print(
            "\nColumn names:"
        )

        for column in table[
            "columns"
        ]:

            print(
                f"  {column}"
            )


# ============================================================
# PRINT FEATURE MODULE DISCOVERY
# ============================================================


def print_feature_modules(
    modules: list[dict],
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "EXISTING FEATURE MODULE DISCOVERY"
    )

    print(
        "=" * 75
    )

    print(
        f"\nRelevant feature modules found: "
        f"{len(modules)}"
    )

    for module in modules:

        print(
            "\n"
            + "-" * 75
        )

        print(
            module[
                "relative_path"
            ]
        )

        print(
            "-" * 75
        )

        print(
            "Keyword hits: "
            +
            ", ".join(
                module[
                    "keyword_hits"
                ]
            )
        )

        if module[
            "syntax_error"
        ]:

            print(
                "Syntax error while inspecting: "
                f"{module['syntax_error']}"
            )

            continue

        print(
            "\nFunctions:"
        )

        if module[
            "functions"
        ]:

            for function in module[
                "functions"
            ]:

                print(
                    f"  {function}"
                )

        else:

            print(
                "  None"
            )

        print(
            "\nClasses:"
        )

        if module[
            "classes"
        ]:

            for class_name in module[
                "classes"
            ]:

                print(
                    f"  {class_name}"
                )

        else:

            print(
                "  None"
            )


# ============================================================
# SAVE AUDIT
# ============================================================


def save_output(
    *,
    production_features: list[str],
    raw_generated_families: dict[
        str,
        list[str],
    ],
    production_families: dict[
        str,
        list[str],
    ],
    latest_tables: list[dict],
    modules: list[dict],
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    non_model_columns = {
        family: sorted(
            set(
                raw_generated_families[
                    family
                ]
            )
            -
            set(
                production_families[
                    family
                ]
            )
        )
        for family in raw_generated_families
    }

    payload = {
        "created_at_local": (
            datetime.now(
                tz=LOCAL_TIMEZONE
            ).isoformat()
        ),

        "timezone": (
            "Africa/Johannesburg"
        ),

        "production_feature_count": int(
            len(
                production_features
            )
        ),

        "expected_production_family_counts": (
            EXPECTED_PRODUCTION_FAMILIES
        ),

        "raw_stage_generated_counts": {
            family: int(
                len(
                    features
                )
            )
            for (
                family,
                features
            ) in (
                raw_generated_families.items()
            )
        },

        "production_family_counts": {
            family: int(
                len(
                    features
                )
            )
            for (
                family,
                features
            ) in (
                production_families.items()
            )
        },

        "raw_stage_generated_features": (
            raw_generated_families
        ),

        "production_features_by_family": (
            production_families
        ),

        "stage_columns_not_used_by_model": (
            non_model_columns
        ),

        "latest_state_tables": (
            latest_tables
        ),

        "feature_modules": (
            modules
        ),

        "rules": {
            "production_model_is_authoritative_for_feature_selection": True,

            "stage_generated_columns_are_not_assumed_to_be_model_features": True,

            "source_files_executed": False,

            "source_files_modified": False,

            "training_data_modified": False,

            "production_model_modified": False,

            "purpose": (
                "Identify exact historical feature logic "
                "for reuse in the live 2026/27 pipeline."
            ),
        },
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================


def audit_live_feature_modules() -> None:
    """
    Audit exact production feature-family membership,
    historical latest-state schemas and reusable source
    modules.

    Nothing is modified.
    """

    print(
        "=" * 75
    )

    print(
        "UCL PREDICTOR - LIVE FEATURE MODULE AUDIT"
    )

    print(
        "=" * 75
    )

    production_features = (
        load_production_features()
    )

    print(
        f"\nFrozen production features: "
        f"{len(production_features)}"
    )

    (
        raw_generated_families,
        production_families,
    ) = reconstruct_feature_families(
        production_features
    )

    verify_family_counts(
        raw_generated_families=(
            raw_generated_families
        ),
        production_families=(
            production_families
        ),
        production_features=(
            production_features
        ),
    )

    print_non_model_stage_columns(
        raw_generated_families=(
            raw_generated_families
        ),
        production_families=(
            production_families
        ),
    )

    latest_tables = [
        inspect_latest_table(
            label="Elo",
            path=ELO_LATEST_FILE,
        ),

        inspect_latest_table(
            label="UCL form",
            path=FORM_LATEST_FILE,
        ),

        inspect_latest_table(
            label="Domestic form",
            path=DOMESTIC_LATEST_FILE,
        ),

        inspect_latest_table(
            label="Squad strength",
            path=SQUAD_LATEST_FILE,
        ),
    ]

    modules = (
        inspect_feature_modules()
    )

    print_latest_tables(
        latest_tables
    )

    print_feature_modules(
        modules
    )

    save_output(
        production_features=(
            production_features
        ),
        raw_generated_families=(
            raw_generated_families
        ),
        production_families=(
            production_families
        ),
        latest_tables=(
            latest_tables
        ),
        modules=(
            modules
        ),
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LIVE FEATURE MODULE AUDIT COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        "\nAudit metadata:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Stage-generated column counts and production "
        "feature counts are intentionally distinguished."
    )

    print(
        "No source files were executed or modified."
    )

    print(
        "No historical data or production model was changed."
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    audit_live_feature_modules()