"""Headline Arena 前瞻检验模块测试（纯离线，无网络）。

覆盖 forwardtest 包四大能力：
  1) translator：宏观主题关键词 -> 资产/方向/置信度 翻译；
  2) ledger：预测在结果出现前锁定、结算后字段不可篡改；
  3) scorecard：HA 官方分 / Brier / 校准分桶（与回测独立的统计）；
  4) client/runner：HTTP 请求构造与 dry_run 编排（mock 传输层）。
"""
import io
import json
import os
import sys
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from forwardtest import runner  # noqa: E402
from forwardtest.client import HAError, HeadlineArenaClient  # noqa: E402
from forwardtest.ledger import ForwardLedger, LedgerError  # noqa: E402
from forwardtest.scorecard import (  # noqa: E402
    aggregate_stats,
    brier_for_record,
    calibration_rows,
    ha_directional_score,
    render_markdown,
)
from forwardtest.translator import (  # noqa: E402
    factor_to_theme,
    parse_theme,
    view_to_prediction,
)

# ---------------------------------------------------------------- translator #


class TestTranslator:
    def test_rates_dovish_maps_to_zn_bullish(self):
        views = parse_theme("降息预期升温，流动性宽松", assets=["ZN"])
        assert views and views[0]["asset"] == "ZN"
        assert views[0]["direction"] == "bullish"
        assert 0.34 <= views[0]["confidence"] <= 0.95

    def test_rates_hawkish_maps_to_zn_bearish(self):
        views = parse_theme("美联储鹰派，利率上行，收益率上行", assets=["ZN"])
        assert views and views[0]["asset"] == "ZN"
        assert views[0]["direction"] == "bearish"

    def test_risk_off_maps_gold_bullish(self):
        views = parse_theme("避险情绪升温，衰退担忧加剧", assets=["GC"])
        assert views and views[0]["asset"] == "GC"
        assert views[0]["direction"] == "bullish"

    def test_asset_alias_detected_without_direction_word(self):
        views = parse_theme("关注美债走势", assets=["ZN"])
        assert views and views[0]["asset"] == "ZN"

    def test_no_signal_returns_empty(self):
        assert parse_theme("今日天气不错", assets=["GC"]) == []

    def test_conflicting_signals_net_to_neutral(self):
        # 同一资产多空措辞同时出现：净信号为 0 -> neutral
        views = parse_theme("看多黄金，黄金走弱", assets=["GC"])
        assert views and views[0]["direction"] == "neutral"

    def test_clear_gold_bearish_phrase(self):
        # 明确空头短语不被裸词"黄金"子串污染
        views = parse_theme("金价下跌", assets=["GC"])
        assert views and views[0]["direction"] == "bearish"

    def test_view_to_prediction_cleans_and_validates(self):
        payload = view_to_prediction(
            {"asset": "GC", "direction": " Bullish ", "confidence": 0.999,
             "reasoning": "避险"})
        assert payload["direction"] == "bullish"
        assert float(payload["confidence"]) <= 0.95  # 钳制上限
        assert "prompt" not in payload

        challenge = {"asset": "ES", "question": "q"}
        try:
            view_to_prediction({"asset": "GC", "direction": "bullish",
                                "confidence": 0.6}, challenge)
            assert False, "资产不一致应抛 ValueError"
        except ValueError:
            pass

    def test_factor_to_theme_macro_keyword_detection(self):
        text = factor_to_theme("rate_factor", "利率下行周期的动量因子")
        assert text is not None and "利率下行" in text
        assert factor_to_theme("mom", "20日动量纯价量因子") is None


# -------------------------------------------------------------------- ledger #


def _base_record(**kw):
    rec = {"asset": "GC", "direction": "bullish", "confidence": 0.7,
           "reasoning": "避险测试", "mode": "live",
           "challenge_id": "ch-1", "deadline": "2026-09-10T20:00:00Z"}
    rec.update(kw)
    return rec


class TestLedger:
    def test_log_then_settle_roundtrip(self, tmp_path):
        lg = ForwardLedger(str(tmp_path / "ledger.jsonl"))
        uid = lg.log_prediction(_base_record())
        assert lg.find(uid=uid) is not None
        assert lg.stats()["total"] == 1 and lg.stats()["pending"] == 1

        lg.settle(uid, {"status": "resolved", "result": "bullish",
                        "open_price": 2400.0, "close_price": 2415.0,
                        "resolved_at": "2026-09-11T20:00:00Z",
                        "is_correct": True, "score": 85.0})
        settled = lg.settled()
        assert len(settled) == 1
        assert settled[0]["settled"]["is_correct"] is True
        assert lg.stats()["pending"] == 0

    def test_double_settle_rejected(self, tmp_path):
        lg = ForwardLedger(str(tmp_path / "ledger.jsonl"))
        uid = lg.log_prediction(_base_record())
        lg.settle(uid, {"status": "resolved", "result": "bearish",
                        "is_correct": False, "score": 15.0})
        try:
            lg.settle(uid, {"status": "resolved", "result": "bullish"})
            assert False, "重复结算应抛 LedgerError"
        except LedgerError:
            pass

    def test_prediction_fields_frozen_after_settle(self, tmp_path):
        lg = ForwardLedger(str(tmp_path / "ledger.jsonl"))
        uid = lg.log_prediction(_base_record())
        lg.settle(uid, {"status": "resolved", "result": "bullish"})
        rec = lg.find(uid=uid)
        # 结算回填不改写预测字段，且结果与原始预测一致（证据链完整）
        assert rec["direction"] == "bullish"
        assert rec["confidence"] == 0.7
        assert rec["settled"]["result"] == "bullish"

    def test_duplicate_uid_rejected(self, tmp_path):
        lg = ForwardLedger(str(tmp_path / "ledger.jsonl"))
        rec = _base_record(uid="dup-1")
        lg.log_prediction(rec)
        try:
            lg.log_prediction(rec)
            assert False, "重复 uid 应抛 LedgerError"
        except LedgerError:
            pass

    def test_dry_run_default_mode(self, tmp_path):
        lg = ForwardLedger(str(tmp_path / "ledger.jsonl"))
        lg.log_prediction({"asset": "CL", "direction": "bearish", "confidence": 0.6})
        assert lg.records[0]["mode"] == "dry_run"


# ----------------------------------------------------------------- scorecard #


class TestScorecard:
    def test_ha_formula(self):
        assert ha_directional_score(True, 0.75) == 87.5
        assert ha_directional_score(False, 0.75) == 12.5

    def test_brier_bullish_correct_conf_05(self):
        rec = _base_record(confidence=0.5)
        rec["probabilities"] = {"bullish": 0.5, "bearish": 0.25, "neutral": 0.25}
        rec["settled"] = {"result": "bullish"}
        assert abs(brier_for_record(rec) - 0.375) < 1e-9

    def test_aggregate_stats_on_single_correct(self):
        rec = _base_record(confidence=0.6)
        rec["probabilities"] = {"bullish": 0.6, "bearish": 0.2, "neutral": 0.2}
        rec["settled"] = {"result": "bullish"}
        agg = aggregate_stats([rec])
        assert agg["n"] == 1 and agg["accuracy"] == 1.0
        assert abs(agg["brier"] - 0.24) < 1e-9
        assert abs(agg["mean_ha_score"] - 80.0) < 1e-9

    def test_dry_run_excluded_from_score(self):
        rec = _base_record(confidence=0.8)
        rec["mode"] = "dry_run"
        rec["settled"] = {"result": "bullish"}
        assert aggregate_stats([rec])["n"] == 0

    def test_calibration_bucket(self):
        rec = _base_record(confidence=0.7)
        rec["settled"] = {"result": "bullish"}
        rows = calibration_rows([rec])
        row = next(r for r in rows if r["n"] == 1)
        assert row["mean_forecast"] == 0.7 and row["hit_rate"] == 1.0

    def test_render_markdown_empty(self):
        md = render_markdown([])
        assert "前瞻检验评分卡" in md and "暂无已结算" in md


# ----------------------------------------------------------------- client #


class _FakeResp:
    def __init__(self, payload, status=200):
        if not isinstance(payload, (bytes, str)):
            payload = json.dumps(payload)
        self._data = payload.encode("utf-8") if isinstance(payload, str) else payload
        self.status = status

    def read(self):
        return self._data


def _capture_urlopen():
    """返回 (patcher, captured)。captured 收集每次调用的 Request。"""
    captured = []

    def _fake(request, timeout):
        captured.append(request)
        path = request.full_url
        if "auth/token" in path:
            return _FakeResp({"access_token": "tok-123", "expires_in": 3600})
        if "predict" in path:
            return _FakeResp({"prediction_id": "p-1", "counts_for_score": True})
        if "results" in path:
            return _FakeResp({"status": "open"})
        if "prediction-scopes" in path and request.get_method() == "GET":
            return _FakeResp(["GC", "ES", "CL"])
        return _FakeResp({})

    return mock.patch("forwardtest.client._urlopen", side_effect=_fake), captured


class TestClientRequestBuild:
    def test_token_payload_and_bearer_header(self, tmp_path):
        with mock.patch("forwardtest.client._urlopen",
                        side_effect=lambda req, timeout: _FakeResp(
                            {"access_token": "tok-abc"})) as fake:
            c = HeadlineArenaClient(agent_id="ag-1", client_secret="sec-1",
                                    creds_dir=str(tmp_path))
            tok = c.access_token()
            assert tok == "tok-abc"
            req = fake.call_args[0][0]
            body = json.loads(req.data.decode("utf-8"))
            assert body["grant_type"] == "client_credentials"
            assert body["agent_id"] == "ag-1"
            assert body["client_secret"] == "sec-1"

    def test_predict_payload_and_auth_headers(self, tmp_path):
        patcher, captured = _capture_urlopen()
        with patcher:
            c = HeadlineArenaClient(agent_id="ag-1", client_secret="sec-1",
                                    creds_dir=str(tmp_path))
            c._token, c._token_exp = "tok-x", 1e18  # 跳过真实鉴权
            c.submit_prediction("ch-9", "bullish", 0.72, reasoning="降息预期")
        req = captured[0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["direction"] == "bullish"
        assert abs(body["confidence"] - 0.72) < 1e-9
        assert body["reasoning"] == "降息预期"
        assert len(body["prompt_hash"]) == 64  # SHA-256 锁定推理
        assert req.get_header("Authorization") == "Bearer tok-x"
        # urllib 会把 header 名 capitalize（X-Agent-Id -> X-agent-id），按小写比较
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        assert headers_lower.get("x-agent-id") == "ag-1"

    def test_open_challenges_public_path(self, tmp_path):
        patcher, captured = _capture_urlopen()
        with patcher:
            c = HeadlineArenaClient(agent_id="ag-1", client_secret="sec-1",
                                    creds_dir=str(tmp_path))
            challenges = c.open_challenges()
        assert challenges == []
        assert "eval/challenges?status=open" in captured[0].full_url

    def test_http_403_raises_haerror_with_detail(self, tmp_path):
        err = urllib.error.HTTPError(
            "https://x", 403, "Forbidden", {},
            io.BytesIO(b'{"detail": "Missing required scope: credits:stake"}'))
        with mock.patch("forwardtest.client._urlopen", side_effect=err):
            c = HeadlineArenaClient(agent_id="ag-1", client_secret="sec-1",
                                    creds_dir=str(tmp_path))
            try:
                c.open_challenges()
                assert False, "403 应抛 HAError"
            except HAError as e:
                assert e.status == 403
                assert "credits:stake" in str(e)


# ------------------------------------------------------------------- runner #


class _FakeHA:
    agent_id = "ag-1"

    def __init__(self, results):
        self._results = results

    def challenge_results(self, challenge_id):
        return self._results.get(challenge_id, {"status": "open"})


class TestRunner:
    def test_dry_run_offline_full_flow(self, tmp_path):
        lg = ForwardLedger(str(tmp_path / "ledger.jsonl"))
        summary = runner.run_forward(
            {"dry_run": True, "assets": ["GC", "CL", "ZN"]},
            theme="避险升温，降息预期增强，油价上行", ledger=lg)
        assert summary["mode"] == "dry_run"
        assert summary["submitted"] >= 1
        assert summary["dry_run"] == summary["submitted"]
        # 全部为离线占位挑战（无网络环境也应可跑）
        assert all(r.get("challenge_id", "").startswith("demo-")
                   for r in lg.records)
        assert all(r["mode"] == "dry_run" for r in lg.records)

    def test_no_signal_skips(self, tmp_path):
        lg = ForwardLedger(str(tmp_path / "ledger.jsonl"))
        summary = runner.run_forward(
            {"assets": ["GC"]}, theme="", factor_desc="20日动量纯价量因子",
            ledger=lg)
        assert summary.get("reason") == "no_signal"
        assert lg.stats()["total"] == 0

    def test_live_without_credentials_degrades_to_dry_run(self, tmp_path):
        for var in ("HA_AGENT_ID", "HA_CLIENT_SECRET"):
            os.environ.pop(var, None)
        lg = ForwardLedger(str(tmp_path / "ledger.jsonl"))
        summary = runner.run_forward(
            {"assets": ["GC"]}, theme="避险情绪升温", live=True, ledger=lg)
        assert summary["mode"] == "dry_run"
        assert any("凭据" in w for w in summary["warnings"])

    def test_resolve_ledger_accepts_dir_or_file(self, tmp_path):
        as_dir = runner.resolve_ledger({"ledger_dir": str(tmp_path)})
        assert as_dir.path.name == "ledger.jsonl"
        assert as_dir.path.parent == tmp_path
        as_file = runner.resolve_ledger({"ledger_dir": str(tmp_path / "custom.jsonl")})
        assert str(as_file.path) == str(tmp_path / "custom.jsonl")

    def test_settle_backfills_results(self, tmp_path):
        lg = ForwardLedger(str(tmp_path / "ledger.jsonl"))
        lg.log_prediction(_base_record(mode="live"))
        fake = _FakeHA({
            "ch-1": {"status": "resolved", "result": "bullish",
                     "open_price": 2400.0, "close_price": 2420.0,
                     "resolved_at": "2026-09-11T20:00:00Z",
                     "predictions": [
                         {"agent_id": "other", "direction": "bearish",
                          "is_correct": False, "score": 20.0},
                         {"agent_id": "ag-1", "direction": "bullish",
                          "is_correct": True, "score": 85.0},
                     ]},
        })
        out = runner.settle_ledger(ledger=lg, client=fake)
        assert out["settled"] == 1
        settled = lg.settled()[0]
        assert settled["settled"]["is_correct"] is True
        assert settled["settled"]["score"] == 85.0
