from __future__ import annotations

import json

from src.ui.evidence_vault import (
    PROVENANCE_HISTORICAL,
    PROVENANCE_SYSTEM,
    VaultRecord,
    filter_evaluations,
    format_lead_time,
    load_evidence_vault,
    proof_label,
    published_lead_minutes,
)


def historical_record() -> VaultRecord:
    return VaultRecord(
        record_id="hist_test",
        provenance=PROVENANCE_HISTORICAL,
        home_team="Napoli",
        away_team="Arsenal",
        kickoff_utc="2026-09-09T19:00:00Z",
        predicted_result="A",
        predicted_selection="Arsenal",
        actual_result="A",
        home_probability=0.3001,
        draw_probability=0.3008,
        away_probability=0.3991,
        predicted_scoreline="0-1",
        predicted_scoreline_probability=0.1434,
        actual_scoreline="0-1",
        result_pick_correct=True,
        exact_score_correct=True,
        evidence_score=0.6911,
        evidence_label="MODERATE",
        verified_at_utc="2026-09-09T15:45:00Z",
        proof_status="VERIFIED_PRE_KICKOFF",
        proof_source_summary="Private creator proof retained.",
        receipt_asset="assets/history/receipts/test.png",
        receipt_transaction_id="202609092100-NAP-ARS",
    )


def test_verified_historical_lead_time():
    record = historical_record()

    assert published_lead_minutes(record) == 195
    assert format_lead_time(195) == "3h 15m"
    assert proof_label(record) == "VERIFIED PRE-KICKOFF"


def test_historical_filtering():
    correct = historical_record()

    missed = VaultRecord(
        **{
            **correct.__dict__,
            "record_id": "hist_missed",
            "result_pick_correct": False,
        }
    )

    assert filter_evaluations([correct, missed], "correct") == [correct]
    assert filter_evaluations([correct, missed], "missed") == [missed]


def test_historical_manifest_is_loaded(tmp_path):
    historical_path = tmp_path / "historical_receipts.json"

    payload = {
        "schema_version": 1,
        "records": [
            {
                "record_id": "hist_fixture",
                "provenance": PROVENANCE_HISTORICAL,
                "home_team": "Napoli",
                "away_team": "Arsenal",
                "kickoff_utc": "2026-09-09T19:00:00Z",
                "earliest_verified_receipt_utc": "2026-09-09T15:45:00Z",
                "proof_status": "VERIFIED_PRE_KICKOFF",
                "proof_source_summary": "Receipt timestamp + social archive.",
                "receipt_transaction_id": "202609092100-NAP-ARS",
                "receipt_asset": "assets/history/receipts/example.png",
                "predicted_result": "A",
                "predicted_selection": "Arsenal",
                "home_probability": 0.3001,
                "draw_probability": 0.3008,
                "away_probability": 0.3991,
                "predicted_scoreline": "0-1",
                "predicted_scoreline_probability": 0.1434,
                "evidence_score": 0.6911,
                "evidence_label": "MODERATE",
                "expected_home_goals": 0.94,
                "expected_away_goals": 1.12,
                "overall_features_available": 187,
                "overall_feature_availability": 0.6448,
                "actual_result": "A",
                "actual_scoreline": "0-1",
                "result_pick_correct": True,
                "exact_score_correct": True,
            }
        ],
    }

    historical_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    vault = load_evidence_vault(
        predictions_path=tmp_path / "no_predictions.jsonl",
        results_path=tmp_path / "no_results.jsonl",
        historical_receipts_path=historical_path,
    )

    assert vault.available is True
    assert len(vault.records) == 1
    assert vault.records[0].provenance == PROVENANCE_HISTORICAL
    assert vault.summary["historical_receipt_count"] == 1
    assert vault.summary["system_archive_count"] == 0


def test_system_and_historical_provenance_constants_are_distinct():
    assert PROVENANCE_SYSTEM != PROVENANCE_HISTORICAL
