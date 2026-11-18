"""示例 03：动态重规划 —— 计划失效时，把失败观察喂回 Planner 生成新计划。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/08_规划与任务分解/code/ch08_replanning.py

静态计划的死穴：计划 A 假设"有直飞航班"，执行到一半发现没有 —— 整个计划作废。
动态重规划：执行失败后，把「原计划 + 失败观察」反馈给 Planner 生成计划 B。
对比实验：同一个失败场景，静态计划失败收场；重规划后成功。
"""

import json
import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from ch08_plan_executor import PlanExecutor, StepResult, render_report  # noqa: E402
from ch08_task_decomposition import Plan, make_plan, make_plan_with_llm  # noqa: E402
from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402
from common.toolbox import ToolBox  # noqa: E402


def search_flight(origin: str, destination: str, direct_only: bool = False) -> str:
    """搜索航班；direct_only 为 True 时只看直飞。

    约定：业务失败（无直飞）抛异常而非返回文本 —— 只有异常才会被
    ToolBox 防线 3 捕获并让执行器判定步骤 failed。"""
    if direct_only:
        raise ValueError("无直飞航班（北京->上海每日 0 班直飞）")
    return "CA1501 08:00（经停济南）"


def search_train(depart: str, arrive: str) -> str:
    """搜索火车班次。"""
    return "G1 07:00 北京南->上海虹桥（4.5 小时）"


def book(text: str) -> str:
    """预订出行票务。"""
    return f"已预订: {text}"


def build_toolbox() -> ToolBox:
    return ToolBox().register(search_flight).register(search_train).register(book)


# 静态计划：假设有直飞（这是"拍脑袋"的计划）
PLAN_A = {
    "goal": "安排北京到上海的出行",
    "steps": [
        {"id": 1, "description": "查直飞航班", "tool": "search_flight",
         "arguments": {"origin": "北京", "destination": "上海", "direct_only": True}},
        {"id": 2, "description": "预订航班", "depends_on": [1], "tool": "book",
         "arguments": {"text": "直飞航班"}},
    ],
}

# 重规划后的计划：改走高铁
PLAN_B = {
    "goal": "安排北京到上海的出行（改道高铁）",
    "steps": [
        {"id": 1, "description": "查高铁班次", "tool": "search_train",
         "arguments": {"depart": "北京", "arrive": "上海"}},
        {"id": 2, "description": "预订高铁票", "depends_on": [1], "tool": "book",
         "arguments": {"text": "G1 高铁票"}},
    ],
}


def replan_context(original: Plan, results: list[StepResult]) -> str:
    """构造重规划提示词：原计划 + 每步执行结果 + 失败原因。"""
    lines = ["原计划执行失败，请重新规划。", "", f"原目标: {original.goal}", "执行结果:"]
    for r in results:
        lines.append(f"  步骤{r.step_id}: {r.status} — {r.output}")
    lines += ["", "要求：避开失败路径（无直飞航班），换一种可行方案；只输出计划 JSON。"]
    return "\n".join(lines)


def run_with_replanning(planner_llm, toolbox: ToolBox, plan: Plan,
                        max_replans: int = 2) -> tuple[Plan, list[StepResult], int]:
    """执行 -> 失败则重规划 -> 再执行，直到成功或耗尽重规划次数。"""
    current = plan
    replans = 0
    while True:
        results = PlanExecutor(toolbox).execute(current)
        if all(r.status == "done" for r in results):
            return current, results, replans
        if replans >= max_replans:
            return current, results, replans
        # 失败观察喂回 Planner，生成新计划
        prompt = replan_context(current, results)
        current = make_plan_with_llm(planner_llm, prompt)
        replans += 1


def main() -> None:
    banner("示例 03：动态重规划 vs 静态计划")
    toolbox = build_toolbox()

    # 对照 1：静态计划 —— 一步失败，满盘皆输
    section("对照 1：静态计划（假设有直飞，现实没有）")
    static_results = PlanExecutor(toolbox).execute(make_plan(PLAN_A))
    print(render_report(static_results))
    verify("步骤 1 failed（无直飞）", static_results[0].status == "failed")
    verify("步骤 2 skipped（失败传播）", static_results[1].status == "skipped")
    verify("任务最终未完成", any(r.status != "done" for r in static_results))

    # 对照 2：动态重规划 —— 失败观察喂回，换道高铁
    section("对照 2：重规划（Planner 看到「无直飞」后改道高铁）")
    planner = MockLLM(scripted=[
        {"content": json.dumps(PLAN_B, ensure_ascii=False)},  # 重规划请求的唯一响应
    ])
    final_plan, final_results, replans = run_with_replanning(
        planner, toolbox, make_plan(PLAN_A))
    print(f"重规划 {replans} 次，新计划:")
    print(final_plan.render())
    print(render_report(final_results))
    verify("重规划 1 次后全部步骤 done",
           replans == 1 and all(r.status == "done" for r in final_results))
    verify("新计划改用 search_train", final_plan.steps[0].tool == "search_train")
    verify("Planner 收到了失败观察", "无直飞航班" in planner.calls[0]["messages"][1]["content"])

    # 对照 3：重规划预算耗尽 -> 诚实报告失败（而不是死循环）
    section("对照 3：重规划预算耗尽 -> 显式失败")
    stubborn_planner = MockLLM(handler=lambda m, t: {
        "content": json.dumps(PLAN_A, ensure_ascii=False),  # 每次都返回同样的坏计划
    })
    _, stuck_results, stuck_replans = run_with_replanning(
        stubborn_planner, toolbox, make_plan(PLAN_A), max_replans=2)
    verify("预算内重规划 2 次后停手", stuck_replans == 2)
    verify("最终状态诚实标记为未完成", any(r.status != "done" for r in stuck_results))

    finish("ch08_replanning")


if __name__ == "__main__":
    main()
