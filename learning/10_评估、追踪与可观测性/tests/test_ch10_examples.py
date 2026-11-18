"""第 10 章测试：轨迹追踪、评估器、成本归因。"""

import json

import pytest

import ch10_cost_report as cost_demo
import ch10_evaluators as eval_demo
import ch10_run_records as records_demo
from ch10_evaluators import EvalCase, EvalSuite
from ch10_run_records import CallRecord, RunRecord, TracedLLM
from common.mock_llm import MockLLM


# ---------- 示例 01：轨迹追踪 ----------
def test_run_records_main_end_to_end():
    records_demo.main()


def test_traced_llm_is_transparent_proxy():
    inner = MockLLM(scripted=["回答A", "回答B"])
    traced = TracedLLM(inner, agent_name="测试")
    r1 = traced.chat([{"role": "user", "content": "问题一"}])
    r2 = traced.chat([{"role": "user", "content": "问题二"}])
    assert r1.content == "回答A" and r2.content == "回答B"   # 代理返回与内部一致
    assert len(traced.records) == 2
    assert traced.records[0].seq == 1


def test_pseudo_tokens_sum_matches_records():
    traced = TracedLLM(MockLLM(handler=lambda m, t: {"content": "abcd"}))
    traced.chat([{"role": "user", "content": "12345678"}])
    assert traced.total_pseudo_tokens == 8 + 4


def test_injectable_clock_no_real_wait():
    ticks = iter([0.0, 1.0])  # 模拟耗时 1 秒
    traced = TracedLLM(MockLLM(scripted=["ok"]))
    traced._clock = lambda: next(ticks)
    traced.chat([{"role": "user", "content": "q"}])
    assert traced.records[0].duration_ms == pytest.approx(1000.0, rel=1e-6)


def test_runrecord_jsonl_roundtrip():
    traced = TracedLLM(MockLLM(scripted=["答案"]), agent_name="测试")
    traced.chat([{"role": "user", "content": "问题"}])
    run = RunRecord(run_id="run_9", task="任务", traced_agents=[traced])
    lines = [json.loads(l) for l in run.to_jsonl().splitlines()]
    assert lines[0]["run_id"] == "run_9"
    assert lines[0]["agent"] == "测试"
    assert lines[0]["output_chars"] == 2


def test_call_record_preview_truncates_at_construction():
    # 截断发生在 TracedLLM 构造记录时；to_dict 只是原样序列化
    inner = MockLLM(scripted=["y" * 80])
    traced = TracedLLM(inner, agent_name="t")
    traced.chat([{"role": "user", "content": "x" * 100}])
    record = traced.records[0]
    assert len(record.prompt_preview) == 50
    assert len(record.output_preview) == 50
    assert record.to_dict()["prompt_preview"] == record.prompt_preview  # 序列化不二次加工


# ---------- 示例 02：评估器 ----------
def test_evaluators_main_end_to_end():
    eval_demo.main()


def test_eval_suite_catches_missing_and_forbidden():
    cases = [EvalCase(name="c", question="q", must_contain=["价格"],
                      must_not_contain=["免费"], pattern=r"价格.*\d+元")]
    suite = EvalSuite(cases, lambda q: "价格是9.9元")
    result = suite.evaluate_one(cases[0])
    assert result.passed

    bad = EvalSuite(cases, lambda q: "完全免费，超值")
    result_bad = bad.evaluate_one(cases[0])
    assert not result_bad.passed
    assert any("缺少关键信息「价格」" in f for f in result_bad.failures)
    assert any("出现禁用内容「免费」" in f for f in result_bad.failures)


def test_eval_pattern_failure():
    cases = [EvalCase(name="fmt", question="q", pattern=r"^\d+元$")]
    suite = EvalSuite(cases, lambda q: "大约九元")   # 不匹配
    assert not suite.evaluate_one(cases[0]).passed


def test_pass_rate_computation():
    cases = [EvalCase(name="a", question="q", must_contain=["x"]),
             EvalCase(name="b", question="q", must_contain=["x"]),
             EvalCase(name="c", question="q", must_contain=["不存在的词"])]
    suite = EvalSuite(cases, lambda q: "x")
    results, rate = suite.run()
    assert rate == pytest.approx(2 / 3)
    assert not results[2].passed


# ---------- 示例 03：成本归因 ----------
def test_cost_report_main_end_to_end():
    cost_demo.main()


def test_cost_report_ranks_agents_by_usage():
    cheap = TracedLLM(MockLLM(scripted=["x"]), agent_name="便宜Agent")
    cheap.chat([{"role": "user", "content": "一"}])
    expensive = TracedLLM(MockLLM(scripted=["y" * 100]), agent_name="贵Agent")
    expensive.chat([{"role": "user", "content": "二" * 200}])
    run = RunRecord(run_id="r", task="t", traced_agents=[cheap, expensive])
    report = cost_demo.cost_report([run])
    lines = report.splitlines()
    expensive_line = next(l for l in lines if "贵Agent" in l)
    cheap_line = next(l for l in lines if "便宜Agent" in l)
    assert lines.index(expensive_line) < lines.index(cheap_line)  # 按消耗降序
    assert "总计" in report


def test_before_after_optimization_shows_savings():
    before = cost_demo.simulate_orchestrator_run("b", long_context=True)
    after = cost_demo.simulate_orchestrator_run("a", long_context=False)
    assert after.total_pseudo_tokens < before.total_pseudo_tokens * 0.5
