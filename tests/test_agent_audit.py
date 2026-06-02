import json

from src.agent_audit import (
    annotate_mispricing_scores,
    capture_rejections,
    record_model_decisions,
    record_model_predictions,
    write_scan_audit,
)
from src.database import TradeDatabase


def _make_pick(
    symbol: str,
    *,
    strategy: str = "PCS",
    score: float = 0.5,
    premium: float = 0.8,
    quantity: int = 1,
    expiry: str = "2026-06-19",
    short_put: float = 500.0,
    long_put: float = 495.0,
    short_call: float | None = None,
    long_call: float | None = None,
) -> dict:
    pick = {
        "symbol": symbol,
        "strategy": strategy,
        "expiry": expiry,
        "premium": premium,
        "quantity": quantity,
        "score": score,
        "prob_win": 0.72,
        "roi": 0.18,
        "ranking_reason": f"score={score:.6f}, dte=17, otm=4.20%",
        "ranking_context": {"dte": 17, "short_leg_otm_pct": 0.042},
        "large_loss_prob": 0.08,
        "stop_loss_prob": 0.11,
        "model_id": "champion_v1",
        "model_type": "linear_least_squares_v001",
        "model_version": "champion_v1",
        "features": {"dte": 17, "option_entry_price": 1.25},
        "score_components": {"expected_pnl": score, "prob_win": 0.72},
    }
    if strategy == "PCS":
        pick["short_put"] = short_put
        pick["long_put"] = long_put
    else:
        pick["short_call"] = short_call if short_call is not None else 520.0
        pick["long_call"] = long_call if long_call is not None else 525.0
    return pick


def test_capture_rejections_marks_missing_candidates_once():
    before = [_make_pick("SPY", score=0.9), _make_pick("QQQ", score=0.7)]
    after = [_make_pick("SPY", score=0.9)]
    rejected = []

    capture_rejections(
        before,
        after,
        "Portfolio gamma risk",
        lambda pick: f"{pick['symbol']} stress cap exceeded",
        rejected,
    )
    capture_rejections(
        before,
        after,
        "Portfolio gamma risk",
        lambda pick: f"{pick['symbol']} stress cap exceeded",
        rejected,
    )

    assert len(rejected) == 1
    assert rejected[0]["symbol"] == "QQQ"
    assert rejected[0]["filtered_stage"] == "Portfolio gamma risk"
    assert rejected[0]["reject_reason"] == "QQQ stress cap exceeded"
    assert rejected[0]["mispricing_score"] == 0.7


def test_annotate_mispricing_scores_sets_default_basis():
    picks = annotate_mispricing_scores(
        [
            {"symbol": "SPY", "score": 0.81234},
            {"symbol": "QQQ", "mispricing_score": 0.61234, "mispricing_score_basis": "custom basis"},
        ]
    )

    assert picks[0]["mispricing_score"] == 0.8123
    assert "expected utility" in picks[0]["mispricing_score_basis"]
    assert picks[1]["mispricing_score"] == 0.6123
    assert picks[1]["mispricing_score_basis"] == "custom basis"


def test_write_scan_audit_filters_rejections_by_selected_floor(tmp_path):
    path = tmp_path / "model_candidates.json"
    selected = [_make_pick("SPY", score=0.75, quantity=3)]
    rejected = [
        {**_make_pick("QQQ", score=0.81), "filtered_stage": "gamma", "reject_reason": "too much stress"},
        {**_make_pick("IWM", score=0.72), "filtered_stage": "gamma", "reject_reason": "too much stress"},
    ]

    write_scan_audit(selected, rejected, path=str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selected"][0]["symbol"] == "SPY"
    assert payload["selected"][0]["total_credit"] == 240.0
    assert payload["selected"][0]["ranking_reason"] == selected[0]["ranking_reason"]
    assert len(payload["rejected"]) == 1
    assert payload["rejected"][0]["symbol"] == "QQQ"
    assert payload["rejected"][0]["stop_loss_prob"] == 0.11


def test_write_scan_audit_falls_back_to_all_rejections_and_caps_count(tmp_path):
    path = tmp_path / "model_candidates.json"
    selected = [_make_pick("SPY", score=0.95)]
    rejected = [
        {**_make_pick("QQQ", score=0.40), "filtered_stage": "gamma", "reject_reason": "risk"},
        {**_make_pick("IWM", score=0.60), "filtered_stage": "gamma", "reject_reason": "risk"},
        {**_make_pick("DIA", score=0.50), "filtered_stage": "gamma", "reject_reason": "risk"},
    ]

    write_scan_audit(selected, rejected, path=str(path), max_rejected=2)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [row["symbol"] for row in payload["rejected"]] == ["IWM", "DIA"]


def test_record_model_predictions_skips_existing_ids(tmp_path):
    db = TradeDatabase(db_path=str(tmp_path / "trades.db"))
    picks = [_make_pick("SPY"), _make_pick("QQQ")]

    record_model_predictions(db, picks)
    first_ids = [pick["model_prediction_id"] for pick in picks]
    record_model_predictions(db, picks)

    predictions = db.get_model_predictions(limit=10)
    assert len(predictions) == 2
    assert [pick["model_prediction_id"] for pick in picks] == first_ids


def test_record_model_decisions_backfills_predictions_and_writes_selected_and_rejected(tmp_path):
    db = TradeDatabase(db_path=str(tmp_path / "trades.db"))
    selected = [_make_pick("SPY", score=0.82, quantity=2)]
    rejected = [
        {
            **_make_pick("QQQ", strategy="CCS", score=0.79, short_call=520.0, long_call=525.0),
            "filtered_stage": "Portfolio gamma risk",
            "reject_reason": "stress cap exceeded",
        }
    ]

    record_model_decisions(db, selected, rejected)

    predictions = sorted(db.get_model_predictions(limit=10), key=lambda row: row["id"])
    decisions = sorted(db.get_model_decisions(limit=10), key=lambda row: row["id"])

    assert len(predictions) == 2
    assert len(decisions) == 2
    assert selected[0]["model_prediction_id"] is not None
    assert selected[0]["model_decision_id"] is not None
    assert rejected[0]["model_prediction_id"] is not None
    assert rejected[0]["model_decision_id"] is not None
    assert decisions[0]["decision"] == "SELECTED"
    assert decisions[0]["selected_rank"] == 1
    assert decisions[0]["quantity"] == 2
    assert decisions[1]["decision"] == "REJECTED"
    assert decisions[1]["risk_gate"] == "Portfolio gamma risk"
    assert decisions[1]["reject_reason"] == "stress cap exceeded"
