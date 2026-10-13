from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from src.config import BASE_DIR
from src.history.evaluation import (
    PredictionEvaluation,
    build_evaluations,
    latest_revision_per_fixture,
)
from src.production.club_crests import crest_html
from src.ui.formatters import LOCAL_TZ, escape


HISTORY_DIR = BASE_DIR / "data" / "app" / "history"
PREDICTION_HISTORY_FILE = HISTORY_DIR / "predictions.jsonl"
RESULT_HISTORY_FILE = HISTORY_DIR / "results.jsonl"
HISTORICAL_RECEIPTS_FILE = HISTORY_DIR / "historical_receipts.json"

PROVENANCE_SYSTEM = "SYSTEM_ARCHIVED_FORWARD_TEST"
PROVENANCE_HISTORICAL = "VERIFIED_HISTORICAL_PRE_KICKOFF_RECEIPT"


@dataclass(frozen=True)
class VaultRecord:
    record_id: str
    provenance: str
    home_team: str
    away_team: str
    kickoff_utc: str
    predicted_result: str
    predicted_selection: str
    actual_result: str
    home_probability: float
    draw_probability: float
    away_probability: float
    predicted_scoreline: str | None
    predicted_scoreline_probability: float | None
    actual_scoreline: str
    result_pick_correct: bool
    exact_score_correct: bool | None
    evidence_score: float | None
    evidence_label: str | None
    verified_at_utc: str | None
    proof_status: str
    proof_source_summary: str
    receipt_asset: str | None = None
    receipt_transaction_id: str | None = None
    revision: int | None = None
    model_version: str | None = None
    multiclass_brier: float | None = None
    log_loss: float | None = None
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None
    overall_features_available: int | None = None
    overall_feature_availability: float | None = None


@dataclass(frozen=True)
class EvidenceVaultData:
    available: bool
    records: tuple[VaultRecord, ...]
    summary: dict
    message: str | None = None


def _parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def local_datetime_label(value: str) -> str:
    return (
        _parse_utc(value)
        .astimezone(LOCAL_TZ)
        .strftime("%d %b %Y // %H:%M SAST")
        .upper()
    )


def format_lead_time(minutes: int) -> str:
    minutes = max(0, int(minutes))
    days, remainder = divmod(minutes, 1440)
    hours, mins = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins or not parts:
        parts.append(f"{mins}m")
    return " ".join(parts)


def published_lead_minutes(record: VaultRecord) -> int | None:
    if record.verified_at_utc is None:
        return None
    kickoff = _parse_utc(record.kickoff_utc)
    verified = _parse_utc(record.verified_at_utc)
    return max(0, int((kickoff - verified).total_seconds() // 60))


def result_probability(record: VaultRecord) -> float:
    if record.predicted_result == "H":
        return float(record.home_probability)
    if record.predicted_result == "D":
        return float(record.draw_probability)
    return float(record.away_probability)


def evidence_percent(record: VaultRecord) -> float | None:
    if record.evidence_score is None:
        return None
    value = float(record.evidence_score)
    if abs(value) <= 1.0:
        value *= 100.0
    return max(0.0, min(100.0, value))


def provenance_label(record: VaultRecord) -> str:
    if record.provenance == PROVENANCE_HISTORICAL:
        return "VERIFIED HISTORICAL RECEIPT"
    return "SYSTEM-ARCHIVED FORWARD TEST"


def proof_label(record: VaultRecord) -> str:
    if record.provenance == PROVENANCE_HISTORICAL:
        return "VERIFIED PRE-KICKOFF"
    return "SYSTEM-LOCKED PRE-KICKOFF"


def _from_evaluation(evaluation: PredictionEvaluation) -> VaultRecord:
    if evaluation.predicted_result == "H":
        selection = evaluation.home_team
    elif evaluation.predicted_result == "A":
        selection = evaluation.away_team
    else:
        selection = "Draw"

    return VaultRecord(
        record_id=evaluation.prediction_id,
        provenance=PROVENANCE_SYSTEM,
        home_team=evaluation.home_team,
        away_team=evaluation.away_team,
        kickoff_utc=evaluation.kickoff_utc,
        predicted_result=evaluation.predicted_result,
        predicted_selection=selection,
        actual_result=evaluation.actual_result,
        home_probability=float(evaluation.home_probability),
        draw_probability=float(evaluation.draw_probability),
        away_probability=float(evaluation.away_probability),
        predicted_scoreline=evaluation.predicted_scoreline,
        predicted_scoreline_probability=None,
        actual_scoreline=evaluation.actual_scoreline,
        result_pick_correct=bool(evaluation.result_pick_correct),
        exact_score_correct=evaluation.exact_score_correct,
        evidence_score=evaluation.evidence_score,
        evidence_label=evaluation.evidence_label,
        verified_at_utc=evaluation.published_at_utc,
        proof_status="SYSTEM_ARCHIVED_PRE_KICKOFF",
        proof_source_summary=(
            "Immutable live forward-test archive. "
            "The history schema requires publication before kickoff."
        ),
        revision=evaluation.revision,
        model_version=evaluation.model_version,
        multiclass_brier=evaluation.multiclass_brier,
        log_loss=evaluation.log_loss,
        overall_feature_availability=evaluation.overall_feature_availability,
    )


def _load_historical_records(
    path: str | Path = HISTORICAL_RECEIPTS_FILE,
) -> list[VaultRecord]:
    path = Path(path)
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("historical_receipts.json must contain a records list.")

    records: list[VaultRecord] = []
    for item in raw_records:
        if not isinstance(item, dict):
            raise ValueError("Historical receipt records must be JSON objects.")

        record = VaultRecord(
            record_id=str(item["record_id"]),
            provenance=str(item["provenance"]),
            home_team=str(item["home_team"]),
            away_team=str(item["away_team"]),
            kickoff_utc=str(item["kickoff_utc"]),
            predicted_result=str(item["predicted_result"]),
            predicted_selection=str(item["predicted_selection"]),
            actual_result=str(item["actual_result"]),
            home_probability=float(item["home_probability"]),
            draw_probability=float(item["draw_probability"]),
            away_probability=float(item["away_probability"]),
            predicted_scoreline=(
                str(item["predicted_scoreline"])
                if item.get("predicted_scoreline") is not None else None
            ),
            predicted_scoreline_probability=(
                float(item["predicted_scoreline_probability"])
                if item.get("predicted_scoreline_probability") is not None else None
            ),
            actual_scoreline=str(item["actual_scoreline"]),
            result_pick_correct=bool(item["result_pick_correct"]),
            exact_score_correct=(
                bool(item["exact_score_correct"])
                if item.get("exact_score_correct") is not None else None
            ),
            evidence_score=(
                float(item["evidence_score"])
                if item.get("evidence_score") is not None else None
            ),
            evidence_label=(
                str(item["evidence_label"])
                if item.get("evidence_label") is not None else None
            ),
            verified_at_utc=(
                str(item["earliest_verified_receipt_utc"])
                if item.get("earliest_verified_receipt_utc") else None
            ),
            proof_status=str(item["proof_status"]),
            proof_source_summary=str(item["proof_source_summary"]),
            receipt_asset=(
                str(item["receipt_asset"]) if item.get("receipt_asset") else None
            ),
            receipt_transaction_id=(
                str(item["receipt_transaction_id"])
                if item.get("receipt_transaction_id") else None
            ),
            expected_home_goals=(
                float(item["expected_home_goals"])
                if item.get("expected_home_goals") is not None else None
            ),
            expected_away_goals=(
                float(item["expected_away_goals"])
                if item.get("expected_away_goals") is not None else None
            ),
            overall_features_available=(
                int(item["overall_features_available"])
                if item.get("overall_features_available") is not None else None
            ),
            overall_feature_availability=(
                float(item["overall_feature_availability"])
                if item.get("overall_feature_availability") is not None else None
            ),
        )

        if (
            record.proof_status == "VERIFIED_PRE_KICKOFF"
            and record.verified_at_utc is not None
            and _parse_utc(record.verified_at_utc) >= _parse_utc(record.kickoff_utc)
        ):
            raise ValueError(
                "Historical pre-kickoff proof timestamp must be before kickoff: "
                f"{record.record_id}"
            )
        records.append(record)
    return records


def _build_combined_summary(records: list[VaultRecord]) -> dict:
    resolved = len(records)
    correct = sum(record.result_pick_correct for record in records)
    exact_eligible = [
        record for record in records if record.exact_score_correct is not None
    ]
    exact_correct = sum(bool(record.exact_score_correct) for record in exact_eligible)
    historical_count = sum(
        record.provenance == PROVENANCE_HISTORICAL for record in records
    )
    system_count = sum(
        record.provenance == PROVENANCE_SYSTEM for record in records
    )
    return {
        "resolved_fixture_count": resolved,
        "correct_result_picks": correct,
        "exact_score_eligible": len(exact_eligible),
        "exact_scores_correct": exact_correct,
        "historical_receipt_count": historical_count,
        "system_archive_count": system_count,
        "sample_warning": (
            "INSUFFICIENT_SAMPLE_FOR_RELIABLE_PERFORMANCE_CLAIMS"
            if resolved < 30 else None
        ),
    }


def _fixture_identity(record: VaultRecord) -> tuple[str, str, str]:
    return (
        _parse_utc(record.kickoff_utc).isoformat(),
        " ".join(record.home_team.casefold().split()),
        " ".join(record.away_team.casefold().split()),
    )


def load_evidence_vault(
    *,
    predictions_path: str | Path = PREDICTION_HISTORY_FILE,
    results_path: str | Path = RESULT_HISTORY_FILE,
    historical_receipts_path: str | Path = HISTORICAL_RECEIPTS_FILE,
) -> EvidenceVaultData:
    records: list[VaultRecord] = []
    predictions_path = Path(predictions_path)
    results_path = Path(results_path)

    if predictions_path.exists() and results_path.exists():
        evaluations = build_evaluations(
            predictions_path=predictions_path,
            results_path=results_path,
        )
        for evaluation in latest_revision_per_fixture(evaluations):
            records.append(_from_evaluation(evaluation))

    system_fixture_keys = {
        _fixture_identity(record)
        for record in records
    }

    for historical_record in _load_historical_records(
        historical_receipts_path
    ):
        if _fixture_identity(historical_record) in system_fixture_keys:
            continue

        records.append(historical_record)

    if not records:
        return EvidenceVaultData(
            available=False,
            records=(),
            summary={},
            message="No completed forecast history is available yet.",
        )

    records = sorted(
        records,
        key=lambda item: (item.kickoff_utc, item.record_id),
        reverse=True,
    )
    return EvidenceVaultData(
        available=True,
        records=tuple(records),
        summary=_build_combined_summary(records),
        message=None,
    )


def filter_evaluations(
    records: Iterable[VaultRecord],
    mode: str,
) -> list[VaultRecord]:
    normalized = str(mode).strip().lower()
    items = list(records)
    if normalized == "correct":
        return [item for item in items if item.result_pick_correct]
    if normalized == "missed":
        return [item for item in items if not item.result_pick_correct]
    return items


def find_evaluation(
    records: Iterable[VaultRecord],
    record_id: str | None,
) -> VaultRecord | None:
    if not record_id:
        return None
    for item in records:
        if item.record_id == record_id:
            return item
    return None


def receipt_asset_path(record: VaultRecord) -> Path | None:
    if not record.receipt_asset:
        return None
    path = BASE_DIR / record.receipt_asset
    return path if path.exists() else None


def _actual_selection(record: VaultRecord) -> str:
    if record.actual_result == "H":
        return record.home_team
    if record.actual_result == "A":
        return record.away_team
    return "Draw"


def build_vault_card_html(record: VaultRecord) -> str:
    outcome_class = "correct" if record.result_pick_correct else "missed"
    outcome_text = (
        "RESULT PICK CORRECT" if record.result_pick_correct else "RESULT PICK MISSED"
    )
    result_chance = result_probability(record) * 100.0
    projected = record.predicted_scoreline or "—"
    lead = published_lead_minutes(record)
    lead_text = f"{format_lead_time(lead)} EARLY" if lead is not None else "PRE-KICKOFF"

    return f"""
    <div class="vault-record-card {outcome_class}">
        <div class="vault-record-topline">
            <span>{escape(local_datetime_label(record.kickoff_utc))}</span>
            <span class="vault-outcome-badge {outcome_class}">{outcome_text}</span>
        </div>
        <div class="vault-provenance">{escape(provenance_label(record))}</div>
        <div class="vault-record-fixture">
            <div class="vault-record-team">
                {crest_html(record.home_team, side="home")}
                <span>{escape(record.home_team)}</span>
            </div>
            <div class="vault-record-score">{escape(record.actual_scoreline)}</div>
            <div class="vault-record-team">
                {crest_html(record.away_team, side="away")}
                <span>{escape(record.away_team)}</span>
            </div>
        </div>
        <div class="vault-record-grid">
            <div>
                <span class="vault-record-label">PREDICTED</span>
                <strong>{escape(record.predicted_selection)}</strong>
                <small>{result_chance:.1f}% RESULT CHANCE</small>
            </div>
            <div>
                <span class="vault-record-label">PROJECTED SCORE</span>
                <strong>{escape(projected)}</strong>
                <small>BEFORE KICKOFF</small>
            </div>
            <div>
                <span class="vault-record-label">ACTUAL</span>
                <strong>{escape(_actual_selection(record))}</strong>
                <small>FINAL RESULT</small>
            </div>
        </div>
        <div class="vault-proof-line">
            ● {escape(proof_label(record))} // {escape(lead_text)}
        </div>
    </div>
    """


def build_vault_detail_html(record: VaultRecord) -> str:
    outcome_class = "correct" if record.result_pick_correct else "missed"
    outcome_text = (
        "RESULT PICK CORRECT" if record.result_pick_correct else "RESULT PICK MISSED"
    )
    result_chance = result_probability(record) * 100.0
    evidence = evidence_percent(record)
    evidence_text = f"{evidence:.0f}%" if evidence is not None else "N/A"
    projected = record.predicted_scoreline or "—"
    exact_result = "YES" if record.exact_score_correct else "NO"
    lead = published_lead_minutes(record)
    lead_text = format_lead_time(lead) if lead is not None else "PRE-KICKOFF"
    verified_label = (
        "EARLIEST VERIFIED RECEIPT"
        if record.provenance == PROVENANCE_HISTORICAL
        else "FORECAST PUBLISHED"
    )
    verified_value = (
        escape(local_datetime_label(record.verified_at_utc))
        if record.verified_at_utc else "PRE-KICKOFF VERIFIED"
    )

    return f"""
    <div class="vault-detail">
        <div class="vault-detail-head">
            <div>
                <div class="vault-detail-kicker">ARCHIVED FORECAST RECORD</div>
                <div class="vault-detail-title">
                    {escape(record.home_team)} VS {escape(record.away_team)}
                </div>
                <div class="vault-provenance vault-provenance-detail">
                    {escape(provenance_label(record))}
                </div>
            </div>
            <div class="vault-outcome-badge {outcome_class}">{outcome_text}</div>
        </div>

        <div class="vault-final-score">
            <div>
                {crest_html(record.home_team, side="home")}
                <span>{escape(record.home_team)}</span>
            </div>
            <strong>{escape(record.actual_scoreline)}</strong>
            <div>
                {crest_html(record.away_team, side="away")}
                <span>{escape(record.away_team)}</span>
            </div>
        </div>

        <div class="vault-compare-grid">
            <div class="vault-compare-card">
                <span>PREDICTED WINNER</span>
                <strong>{escape(record.predicted_selection)}</strong>
                <small>{result_chance:.1f}% RESULT CHANCE</small>
            </div>
            <div class="vault-compare-card">
                <span>PROJECTED SCORE</span>
                <strong>{escape(projected)}</strong>
                <small>EXACT SCORE HIT: {exact_result}</small>
            </div>
            <div class="vault-compare-card">
                <span>ACTUAL WINNER</span>
                <strong>{escape(_actual_selection(record))}</strong>
                <small>VERIFIED FINAL RESULT</small>
            </div>
            <div class="vault-compare-card">
                <span>EVIDENCE STRENGTH</span>
                <strong>{evidence_text}</strong>
                <small>AT FORECAST TIME</small>
            </div>
        </div>

        <div class="vault-proof-panel">
            <div class="vault-proof-title">✓ {escape(proof_label(record))}</div>
            <div class="vault-proof-grid">
                <div>
                    <span>{verified_label}</span>
                    <strong>{verified_value}</strong>
                </div>
                <div>
                    <span>KICKOFF</span>
                    <strong>{escape(local_datetime_label(record.kickoff_utc))}</strong>
                </div>
                <div>
                    <span>VERIFIED LEAD</span>
                    <strong>{escape(lead_text)} BEFORE KICKOFF</strong>
                </div>
            </div>
            <div class="vault-proof-source">
                {escape(record.proof_source_summary)}
            </div>
        </div>
    </div>
    """
