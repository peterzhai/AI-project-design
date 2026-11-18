"""第 8 章测试：任务分解（schema+结构检查）、计划执行器、动态重规划。"""

import pytest

import ch08_plan_executor as executor_demo
import ch08_replanning as replan_demo
import ch08_task_decomposition as plan_demo
from ch08_task_decomposition import Plan, PlanStep, make_plan
from common.mock_llm import MockLLM
from common.toolbox import ToolBox
from common.validators import ValidationError


# ---------- 示例 01：任务分解 ----------
def test_task_decomposition_main_end_to_end():
    plan_demo.main()


def test_make_plan_valid_and_with_tools():
    data = {
        "goal": "g", "steps": [
            {"id": 1, "description": "第一步", "tool": "f", "arguments": {"x": 1}},
            {"id": 2, "description": "第二步", "depends_on": [1]},
        ]
    }
    plan = make_plan(data)
    assert plan.steps[0].tool == "f" and plan.steps[0].arguments == {"x": 1}
    assert plan.steps[1].depends_on == (1,)


def test_make_plan_rejects_cycle_dup_and_missing_dep():
    cyclic = {"goal": "g", "steps": [
        {"id": 1, "description": "步骤甲", "depends_on": [2]},
        {"id": 2, "description": "步骤乙", "depends_on": [1]},
    ]}
    with pytest.raises(ValidationError, match="存在环"):
        make_plan(cyclic)

    dup = {"goal": "g", "steps": [{"id": 1, "description": "步骤甲"}, {"id": 1, "description": "步骤乙"}]}
    with pytest.raises(ValidationError, match="id 重复"):
        make_plan(dup)

    missing = {"goal": "g", "steps": [{"id": 1, "description": "步骤甲", "depends_on": [7]}]}
    with pytest.raises(ValidationError, match="不存在的步骤 id"):
        make_plan(missing)


def test_plan_with_llm_repairs_bad_json():
    flaky = MockLLM(scripted=[
        {"content": "这不是JSON"},
        {"content": '{"goal": "g", "steps": [{"id": 1, "description": "唯一步骤"}]}'},
    ])
    plan = plan_demo.make_plan_with_llm(flaky, "目标")
    assert len(plan.steps) == 1
    assert flaky.call_count == 2


def test_self_referencing_dependency_is_cycle():
    self_loop = {"goal": "g", "steps": [{"id": 1, "description": "自指步骤", "depends_on": [1]}]}
    with pytest.raises(ValidationError, match="存在环"):
        make_plan(self_loop)


# ---------- 示例 02：计划执行器 ----------
def test_plan_executor_main_end_to_end():
    executor_demo.main()


def test_executor_skips_on_failed_dependency():
    audit: list[str] = []
    toolbox = executor_demo.build_toolbox(audit, venue_fail_times=99)
    plan = Plan(goal="g", steps=(
        PlanStep(id=1, description="预订", tool="call_venue", arguments={"venue": "X"}),
        PlanStep(id=2, description="通知", depends_on=(1,), tool="send_notice",
                 arguments={"content": "c"}),
    ))
    results = executor_demo.PlanExecutor(toolbox).execute(plan)
    assert [r.status for r in results] == ["failed", "skipped"]
    assert audit == []


def test_executor_retry_records_attempts():
    audit: list[str] = []
    toolbox = executor_demo.build_toolbox(audit, venue_fail_times=1)
    plan = Plan(goal="g", steps=(
        PlanStep(id=1, description="预订", tool="call_venue", arguments={"venue": "Y"}),
    ))
    results = executor_demo.PlanExecutor(toolbox).execute(plan)
    assert results[0].status == "done" and results[0].attempts == 2


def test_toolbox_try_execute_returns_flag():
    audit: list[str] = []
    toolbox = executor_demo.build_toolbox(audit)
    ok, text = toolbox.try_execute("call_venue", {"venue": "Z"})
    assert ok is True and "已预订" in text
    ok2, text2 = toolbox.try_execute("call_venue", {"venue": 123})  # 类型错
    assert ok2 is False and "参数校验失败" in text2
    ok3, text3 = toolbox.try_execute("nope", {})
    assert ok3 is False and "未注册" in text3


# ---------- 示例 03：动态重规划 ----------
def test_replanning_main_end_to_end():
    replan_demo.main()


def test_replanning_switches_to_train():
    toolbox = replan_demo.build_toolbox()
    planner = MockLLM(scripted=[{"content": __import__("json").dumps(replan_demo.PLAN_B, ensure_ascii=False)}])
    final_plan, results, replans = replan_demo.run_with_replanning(
        planner, toolbox, make_plan(replan_demo.PLAN_A))
    assert replans == 1
    assert final_plan.steps[0].tool == "search_train"
    assert all(r.status == "done" for r in results)


def test_replanning_budget_exhaustion_is_honest():
    toolbox = replan_demo.build_toolbox()
    stubborn = MockLLM(handler=lambda m, t: {"content": __import__("json").dumps(replan_demo.PLAN_A, ensure_ascii=False)})
    _, results, replans = replan_demo.run_with_replanning(
        stubborn, toolbox, make_plan(replan_demo.PLAN_A), max_replans=2)
    assert replans == 2
    assert any(r.status != "done" for r in results)


def test_replan_context_includes_failure_observation():
    from ch08_task_decomposition import make_plan as mp
    plan = mp(replan_demo.PLAN_A)
    results = [executor_demo.StepResult(step_id=1, status="failed",
                                        output="工具「search_flight」执行失败: 无直飞航班")]
    context = replan_demo.replan_context(plan, results)
    assert "无直飞航班" in context and "重新规划" in context
