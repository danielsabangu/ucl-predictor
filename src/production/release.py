from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

PUBLIC_DIR = ROOT / "data" / "app"
PUBLIC_PREDICTIONS = PUBLIC_DIR / "current_predictions.csv"
PUBLIC_STATUS = PUBLIC_DIR / "pipeline_status.json"
PUBLIC_METADATA = PUBLIC_DIR / "snapshot_metadata.json"

HISTORY_DIR = PUBLIC_DIR / "history"
HISTORY_PREDICTIONS = HISTORY_DIR / "predictions.jsonl"
HISTORY_RESULTS = HISTORY_DIR / "results.jsonl"
HISTORY_EVALUATIONS = HISTORY_DIR / "evaluations.jsonl"
HISTORY_SUMMARY = HISTORY_DIR / "summary.json"

CREST_MANIFEST = ROOT / "assets" / "crests" / "manifest.json"

RELEASE_SCRIPT = ROOT / "src" / "production" / "release.py"

PUBLIC_SNAPSHOT_FILES = (
    PUBLIC_PREDICTIONS,
    PUBLIC_STATUS,
    PUBLIC_METADATA,
)

HISTORY_FILES = (
    HISTORY_PREDICTIONS,
    HISTORY_RESULTS,
    HISTORY_EVALUATIONS,
    HISTORY_SUMMARY,
)

PUBLIC_FILES = (
    *PUBLIC_SNAPSHOT_FILES,
    *HISTORY_FILES,
)

STAGE_CANDIDATES = (
    RELEASE_SCRIPT,
    PUBLIC_PREDICTIONS,
    PUBLIC_STATUS,
    PUBLIC_METADATA,
    HISTORY_PREDICTIONS,
    HISTORY_RESULTS,
    HISTORY_EVALUATIONS,
    HISTORY_SUMMARY,
    CREST_MANIFEST,
)

ALLOWED_STAGED_PATHS = {
    "src/production/release.py",
    "data/app/current_predictions.csv",
    "data/app/pipeline_status.json",
    "data/app/snapshot_metadata.json",
    "data/app/history/predictions.jsonl",
    "data/app/history/results.jsonl",
    "data/app/history/evaluations.jsonl",
    "data/app/history/summary.json",
    "assets/crests/manifest.json",
}

FORBIDDEN_PUBLIC_FRAGMENTS = (
    ".env",
    "/home/",
    "data/raw/",
    "data/interim/",
    "data/predictions/",
    "models/trained/",
    "football_data_api_key",
    "api_football_key",
    "thesportsdb_api_key",
    "secret_key",
    "access_token",
    "bearer ",
)


class ReleaseError(RuntimeError):
    """Raised when a production release safety check fails."""


def banner(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def run_command(
    command: list[str],
    *,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    display = " ".join(command)
    print(f"\n$ {display}")

    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=capture,
        env=env,
    )

    if capture:
        if result.stdout:
            print(result.stdout.rstrip())

        if result.stderr and result.returncode != 0:
            print(result.stderr.rstrip(), file=sys.stderr)

    if result.returncode != 0:
        raise ReleaseError(
            f"Command failed with exit code {result.returncode}: {display}"
        )

    return result


def run_python_module(module: str) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            module,
        ]
    )


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise ReleaseError(
            f"Git command failed: git {' '.join(args)}\n"
            f"{result.stderr.strip()}"
        )

    return result.stdout.strip()


def verify_git_repository() -> None:
    try:
        inside = git_output(
            "rev-parse",
            "--is-inside-work-tree",
        )
    except ReleaseError as exc:
        raise ReleaseError(
            "This command must be run inside the UCL Predictor Git repository."
        ) from exc

    if inside.lower() != "true":
        raise ReleaseError(
            "Current directory is not a Git working tree."
        )

    top_level = Path(
        git_output(
            "rev-parse",
            "--show-toplevel",
        )
    ).resolve()

    if top_level != ROOT.resolve():
        raise ReleaseError(
            "Git repository root does not match the UCL Predictor project root.\n"
            f"Expected: {ROOT}\n"
            f"Found:    {top_level}"
        )


def verify_no_existing_staged_changes() -> None:
    staged = git_output(
        "diff",
        "--cached",
        "--name-only",
    )

    if staged:
        raise ReleaseError(
            "There are already staged Git changes.\n\n"
            "Commit, unstage, or deal with them before running the release "
            "workflow so they cannot accidentally be mixed into the prediction "
            "refresh commit.\n\n"
            f"Currently staged:\n{staged}"
        )


def print_unstaged_warning() -> None:
    status = git_output(
        "status",
        "--short",
    )

    if not status:
        print("Git working tree is clean.")
        return

    print(
        "Existing unstaged/untracked files detected. "
        "They will NOT be staged automatically unless they are one of the "
        "approved release artifacts."
    )


def verify_current_public_snapshot_is_committed() -> None:
    """
    Protect prediction-history integrity.

    The snapshot imported into the immutable prediction ledger must already
    be the committed public snapshot, not an uncommitted local release
    candidate from a failed or abandoned run.
    """

    snapshot_paths = [
        relative_path(path)
        for path in PUBLIC_SNAPSHOT_FILES
    ]

    for snapshot_path in PUBLIC_SNAPSHOT_FILES:
        if not snapshot_path.exists():
            raise ReleaseError(
                "Cannot preserve the current public prediction snapshot "
                f"because this file is missing: {relative_path(snapshot_path)}"
            )

    dirty = git_output(
        "status",
        "--porcelain",
        "--",
        *snapshot_paths,
    )

    if dirty:
        raise ReleaseError(
            "The existing public prediction snapshot has local Git changes.\n\n"
            "Refusing to record it as a published prediction because it may "
            "be an uncommitted release candidate.\n\n"
            f"Dirty public snapshot files:\n{dirty}"
        )

    for snapshot_path in snapshot_paths:
        tracked = git_output(
            "ls-files",
            "--",
            snapshot_path,
        )

        if not tracked:
            raise ReleaseError(
                "The current public prediction snapshot is not tracked by Git: "
                f"{snapshot_path}"
            )

    print(
        "Current public prediction snapshot is committed and safe "
        "to preserve in prediction history."
    )


def parse_utc(value: str) -> datetime:
    text = str(value).strip()

    if not text:
        raise ReleaseError(
            "Encountered an empty UTC timestamp."
        )

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ReleaseError(
            f"Could not parse UTC timestamp: {value}"
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc,
        )

    return parsed.astimezone(timezone.utc)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)
    except FileNotFoundError as exc:
        raise ReleaseError(
            f"Required release file does not exist: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ReleaseError(
            f"Invalid JSON file: {path}\n{exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ReleaseError(
            f"Expected a JSON object in: {path}"
        )

    return payload


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sanitize_repo_paths(value: Any) -> Any:
    """
    Convert absolute paths rooted inside this repository to repo-relative paths.

    This is primarily used for the public crest manifest so a local Windows
    path such as F:/ucl-predictor/... can never be published.
    """

    if isinstance(value, dict):
        return {
            key: sanitize_repo_paths(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            sanitize_repo_paths(item)
            for item in value
        ]

    if not isinstance(value, str):
        return value

    normalized = value.replace(
        "\\",
        "/",
    )

    root_normalized = ROOT.resolve().as_posix().rstrip("/")

    if normalized.lower().startswith(
        root_normalized.lower() + "/"
    ):
        return normalized[
            len(root_normalized) + 1:
        ]

    return value


def sanitize_crest_manifest() -> None:
    if not CREST_MANIFEST.exists():
        print(
            "Crest manifest not found. "
            "Deployment tests will verify whether crests are required."
        )
        return

    payload = load_json(
        CREST_MANIFEST,
    )

    provenance_removed = (
        payload.pop(
            "source_team_snapshot",
            None,
        )
        is not None
    )

    sanitized = sanitize_repo_paths(
        payload,
    )

    changed = (
        provenance_removed
        or
        sanitized != payload
    )

    if not changed:
        print(
            "Crest manifest contains no private provenance "
            "or repository-local absolute paths."
        )
        return

    with CREST_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            sanitized,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    if provenance_removed:
        print(
            "Removed private source provenance from "
            "assets/crests/manifest.json."
        )

    print(
        "Crest manifest sanitized for public deployment."
    )


def validate_public_files_exist() -> None:
    missing = [
        relative_path(path)
        for path in PUBLIC_FILES
        if not path.exists()
    ]

    if missing:
        raise ReleaseError(
            "Public snapshot packaging did not produce all required files:\n"
            + "\n".join(
                f"  - {item}"
                for item in missing
            )
        )


def validate_public_predictions() -> tuple[int, list[dict[str, str]]]:
    with PUBLIC_PREDICTIONS.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ReleaseError(
            "Public prediction snapshot contains zero fixtures."
        )

    required_columns = {
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_probability",
        "draw_probability",
        "away_probability",
        "predicted_result",
        "predicted_selection",
        "confidence_percent",
        "predicted_scoreline",
        "overall_feature_availability",
    }

    columns = set(rows[0].keys())

    missing_columns = sorted(
        required_columns - columns
    )

    if missing_columns:
        raise ReleaseError(
            "Public prediction snapshot is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    now = datetime.now(
        timezone.utc,
    )

    future_count = 0

    for index, row in enumerate(
        rows,
        start=1,
    ):
        home_team = row.get(
            "home_team",
            "",
        ).strip()

        away_team = row.get(
            "away_team",
            "",
        ).strip()

        if not home_team or not away_team:
            raise ReleaseError(
                f"Fixture row {index} has a missing team name."
            )

        kickoff = parse_utc(
            row["kickoff_utc"]
        )

        if kickoff > now:
            future_count += 1

        try:
            home_probability = float(
                row["home_probability"]
            )
            draw_probability = float(
                row["draw_probability"]
            )
            away_probability = float(
                row["away_probability"]
            )
        except ValueError as exc:
            raise ReleaseError(
                f"Fixture row {index} contains invalid probabilities."
            ) from exc

        probabilities = (
            home_probability,
            draw_probability,
            away_probability,
        )

        if any(
            probability < 0.0 or probability > 1.0
            for probability in probabilities
        ):
            raise ReleaseError(
                f"Fixture row {index} contains a probability outside 0..1."
            )

        total = sum(
            probabilities
        )

        if abs(
            total - 1.0
        ) > 1e-6:
            raise ReleaseError(
                f"Fixture row {index} probabilities sum to {total:.10f}, "
                "not 1.0."
            )

    if future_count == 0:
        raise ReleaseError(
            "The packaged snapshot contains no upcoming fixtures. "
            "Refusing to publish an expired matchday."
        )

    print(
        f"Public prediction snapshot: {len(rows)} fixtures, "
        f"{future_count} currently upcoming."
    )

    return len(rows), rows


def validate_public_status() -> dict[str, Any]:
    status = load_json(
        PUBLIC_STATUS,
    )

    if status.get(
        "status"
    ) != "SUCCESS":
        raise ReleaseError(
            "Public pipeline status is not SUCCESS."
        )

    if status.get(
        "production_ready"
    ) is not True:
        raise ReleaseError(
            "Public pipeline status is not production-ready."
        )

    if status.get(
        "public_snapshot"
    ) is not True:
        raise ReleaseError(
            "Packaged status is not marked as a public snapshot."
        )

    finished_at = status.get(
        "finished_at_utc"
    )

    if not finished_at:
        raise ReleaseError(
            "Public pipeline status has no finished_at_utc value."
        )

    finished_time = parse_utc(
        str(finished_at)
    )

    now = datetime.now(
        timezone.utc,
    )

    age_hours = (
        now - finished_time
    ).total_seconds() / 3600.0

    if age_hours < -0.25:
        raise ReleaseError(
            "Pipeline completion time is unexpectedly in the future."
        )

    if age_hours > 6.0:
        raise ReleaseError(
            f"The newly packaged production run is already "
            f"{age_hours:.2f} hours old. "
            "Refusing to publish a stale release."
        )

    print(
        f"Pipeline status: SUCCESS, production-ready, "
        f"age {max(age_hours, 0.0):.2f}h."
    )

    return status


def validate_public_metadata(
    fixture_count: int,
) -> dict[str, Any]:
    metadata = load_json(
        PUBLIC_METADATA,
    )

    metadata_fixture_count = metadata.get(
        "fixture_count"
    )

    try:
        metadata_fixture_count = int(
            metadata_fixture_count
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ReleaseError(
            "snapshot_metadata.json has an invalid fixture_count."
        ) from exc

    if metadata_fixture_count != fixture_count:
        raise ReleaseError(
            "Public metadata fixture count does not match the prediction CSV.\n"
            f"Metadata: {metadata_fixture_count}\n"
            f"CSV:      {fixture_count}"
        )

    model_version = str(
        metadata.get(
            "model_version",
            "",
        )
    ).strip()

    if not model_version:
        raise ReleaseError(
            "Public snapshot metadata has no model_version."
        )

    print(
        f"Snapshot metadata: model {model_version}, "
        f"{metadata_fixture_count} fixtures."
    )

    return metadata


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ReleaseError(
            f"Required JSONL file does not exist: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ReleaseError(
                    f"Invalid JSONL record at {path}:{line_number}"
                ) from exc

            if not isinstance(payload, dict):
                raise ReleaseError(
                    f"Expected JSON object at {path}:{line_number}"
                )

            records.append(payload)

    return records


def validate_history_artifacts() -> None:
    """
    Rebuild the expected evaluation state from the immutable prediction and
    result ledgers and verify the public Evidence Vault artifacts match it.
    """

    from src.history.archive import load_records
    from src.history.evaluation import (
        build_evaluations,
        build_summary,
    )
    from src.history.results_archive import load_results

    predictions = load_records(
        HISTORY_PREDICTIONS
    )

    if not predictions:
        raise ReleaseError(
            "Prediction history contains zero archived predictions."
        )

    results = load_results(
        HISTORY_RESULTS
    )

    prediction_ids = [
        record.prediction_id
        for record in predictions
    ]

    if len(prediction_ids) != len(set(prediction_ids)):
        raise ReleaseError(
            "Prediction history contains duplicate prediction IDs."
        )

    prediction_fixture_ids = {
        record.fixture_id
        for record in predictions
    }

    for result in results:
        if result.fixture_id not in prediction_fixture_ids:
            raise ReleaseError(
                "Result history contains a fixture that has no archived "
                f"prediction: {result.fixture_id}"
            )

    expected_evaluations = build_evaluations(
        predictions_path=HISTORY_PREDICTIONS,
        results_path=HISTORY_RESULTS,
    )

    expected_evaluation_payloads = [
        evaluation.to_dict()
        for evaluation in expected_evaluations
    ]

    stored_evaluations = load_jsonl(
        HISTORY_EVALUATIONS
    )

    if stored_evaluations != expected_evaluation_payloads:
        raise ReleaseError(
            "Stored Evidence Vault evaluations do not match the immutable "
            "prediction/result ledgers. Re-run src.history.evaluation."
        )

    expected_summary = build_summary(
        expected_evaluations
    )

    stored_summary = load_json(
        HISTORY_SUMMARY
    )

    if stored_summary != expected_summary:
        raise ReleaseError(
            "Stored Evidence Vault summary does not match the evaluation "
            "engine output. Re-run src.history.evaluation."
        )

    print(
        "Evidence Vault validated: "
        f"{len(predictions)} prediction revisions, "
        f"{len(results)} verified results, "
        f"{len(expected_evaluations)} evaluations."
    )


def scan_public_files_for_private_content() -> None:
    files_to_scan = [
        *PUBLIC_FILES,
    ]

    if CREST_MANIFEST.exists():
        files_to_scan.append(
            CREST_MANIFEST
        )

    root_windows = str(
        ROOT.resolve()
    ).lower()

    root_forward = ROOT.resolve().as_posix().lower()

    forbidden = [
        item.lower()
        for item in FORBIDDEN_PUBLIC_FRAGMENTS
    ]

    forbidden.extend(
        [
            root_windows,
            root_forward,
        ]
    )

    findings: list[str] = []

    for path in files_to_scan:
        if not path.exists():
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        normalized = text.replace(
            "\\",
            "/",
        ).lower()

        for fragment in forbidden:
            normalized_fragment = fragment.replace(
                "\\",
                "/",
            ).lower()

            if (
                normalized_fragment
                and normalized_fragment in normalized
            ):
                findings.append(
                    f"{relative_path(path)} -> {fragment}"
                )

    if findings:
        raise ReleaseError(
            "Private/local content was detected in publishable files:\n"
            + "\n".join(
                f"  - {finding}"
                for finding in findings
            )
        )

    print(
        "Public artifact privacy scan passed."
    )


def run_deployment_tests() -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ]
    )


def stage_release_files() -> None:
    existing = [
        relative_path(path)
        for path in STAGE_CANDIDATES
        if path.exists()
    ]

    if not existing:
        raise ReleaseError(
            "No release artifacts were available to stage."
        )

    run_command(
        [
            "git",
            "add",
            "--",
            *existing,
        ]
    )


def validate_staged_files() -> list[str]:
    staged_text = git_output(
        "diff",
        "--cached",
        "--name-only",
    )

    staged = [
        line.strip().replace(
            "\\",
            "/",
        )
        for line in staged_text.splitlines()
        if line.strip()
    ]

    unexpected = [
        path
        for path in staged
        if path not in ALLOWED_STAGED_PATHS
    ]

    if unexpected:
        subprocess.run(
            [
                "git",
                "reset",
                "--",
                *staged,
            ],
            cwd=ROOT,
            check=False,
        )

        raise ReleaseError(
            "Unexpected files were staged. "
            "The release staging area has been reset.\n\n"
            + "\n".join(
                f"  - {path}"
                for path in unexpected
            )
        )

    return staged


def validate_staged_patch() -> None:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--check",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        if result.stdout:
            print(
                result.stdout.rstrip()
            )

        if result.stderr:
            print(
                result.stderr.rstrip(),
                file=sys.stderr,
            )

        raise ReleaseError(
            "git diff --cached --check found problems."
        )

    print(
        "Staged Git patch check passed."
    )


def print_release_summary(
    staged: list[str],
) -> None:
    banner(
        "RELEASE READY"
    )

    if staged:
        print(
            "The following files are staged:"
        )

        for path in staged:
            print(
                f"  - {path}"
            )
    else:
        print(
            "The refreshed public snapshot is identical to the "
            "currently committed version."
        )

    print()

    branch = git_output(
        "branch",
        "--show-current",
    )

    commit_date = datetime.now().astimezone().date().isoformat()

    if staged:
        print(
            "Everything passed. Run these two commands:"
        )
        print()
        print(
            f'git commit -m "Refresh UCL predictions {commit_date}"'
        )
        print(
            f"git push origin {branch or 'main'}"
        )
        print()
        print(
            "Streamlit Community Cloud should then redeploy "
            "from the updated GitHub branch automatically."
        )
    else:
        print(
            "There is nothing new to commit or push."
        )

    print()

    status = git_output(
        "status",
        "--short",
    )

    if status:
        print(
            "Current Git status:"
        )
        print(
            status
        )
    else:
        print(
            "Git working tree is clean."
        )


def execute_release(
    *,
    stage: bool,
) -> None:
    banner(
        "UCL PREDICTOR - PRODUCTION RELEASE"
    )

    print(
        f"Project root: {ROOT}"
    )
    print(
        f"Python: {sys.executable}"
    )

    banner(
        "STEP 1/10 - PREFLIGHT"
    )

    verify_git_repository()
    verify_no_existing_staged_changes()
    print_unstaged_warning()

    verify_current_public_snapshot_is_committed()

    banner(
        "STEP 2/10 - PRESERVE CURRENT PUBLIC PREDICTIONS"
    )

    run_python_module(
        "src.history.snapshot_import"
    )

    banner(
        "STEP 3/10 - REFRESH PRODUCTION PIPELINE"
    )

    run_python_module(
        "src.production.run"
    )

    banner(
        "STEP 4/10 - RESOLVE EVIDENCE VAULT"
    )

    run_python_module(
        "src.history.resolver"
    )

    run_python_module(
        "src.history.evaluation"
    )

    banner(
        "STEP 5/10 - PACKAGE NEW PUBLIC SNAPSHOT"
    )

    run_python_module(
        "src.production.package_app_snapshot"
    )

    banner(
        "STEP 6/10 - SANITIZE PUBLIC METADATA"
    )

    sanitize_crest_manifest()

    banner(
        "STEP 7/10 - VALIDATE PUBLIC RELEASE"
    )

    validate_public_files_exist()

    fixture_count, _ = validate_public_predictions()

    validate_public_status()

    validate_public_metadata(
        fixture_count,
    )

    validate_history_artifacts()

    scan_public_files_for_private_content()

    banner(
        "STEP 8/10 - RUN DEPLOYMENT TESTS"
    )

    run_deployment_tests()

    banner(
        "STEP 9/10 - GIT RELEASE SAFETY"
    )

    if stage:
        stage_release_files()

        staged = validate_staged_files()

        validate_staged_patch()
    else:
        print(
            "Staging skipped because --no-stage was supplied."
        )
        staged = []

    banner(
        "STEP 10/10 - FINAL SUMMARY"
    )

    print_release_summary(
        staged,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve the committed prediction snapshot, refresh the "
            "complete UCL Predictor production pipeline, resolve and "
            "evaluate prediction history, package the sanitized Streamlit "
            "snapshot, validate it, run deployment tests, and safely stage "
            "the public release."
        )
    )

    parser.add_argument(
        "--no-stage",
        action="store_true",
        help=(
            "Run the complete refresh and validation workflow "
            "without staging deployment files in Git."
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        execute_release(
            stage=not args.no_stage,
        )
    except KeyboardInterrupt:
        print(
            "\nRelease cancelled by user.",
            file=sys.stderr,
        )
        return 130
    except ReleaseError as exc:
        banner(
            "RELEASE BLOCKED"
        )
        print(
            str(exc),
            file=sys.stderr,
        )
        print()
        print(
            "Nothing has been committed or pushed."
        )
        return 1
    except Exception as exc:
        banner(
            "UNEXPECTED RELEASE ERROR"
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print()
        print(
            "Nothing has been committed or pushed."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )