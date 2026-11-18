"""示例 03：成本归因 —— 每一步花了多少伪 token，账单要能摊到人头。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/10_评估、追踪与可观测性/code/ch10_cost_report.py

可观测性的最后一环：把追踪记录变成**成本报表** ——
按 Agent 分摊伪 token 消耗，找出最贵的环节；对比优化前后的账单，让优化有数字证据。
第 11 章的预算护栏将直接消费这里的数字。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from ch10_run_records import RunRecord, TracedLLM  # noqa: E402
from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402


def cost_report(runs: list[RunRecord]) -> str:
    """按 Agent 分摊的成本报表（伪 token）。"""
    usage: dict[str, int] = {}
    for run in runs:
        for agent in run.traced_agents:
            usage[agent.agent_name] = usage.get(agent.agent_name, 0) + agent.total_pseudo_tokens
    total = sum(usage.values()) or 1
    lines = ["成本报表（伪 token，1 字符 ≈ 1 伪 token）", "-" * 46]
    for name, tokens in sorted(usage.items(), key=lambda kv: kv[1], reverse=True):
        bar = "█" * int(tokens / total * 30)
        lines.append(f"  {name:<12} {tokens:>6}  {tokens / total:>5.1%}  {bar}")
    lines.append("-" * 46)
    lines.append(f"  {'总计':<12} {total:>6}")
    return "\n".join(lines)


def simulate_orchestrator_run(run_id: str, long_context: bool) -> RunRecord:
    """模拟一次三 Agent 协作（Planner + 两个 Worker），可开关长上下文。"""
    padding = "背景资料：" + "历史市场数据很长很长。" * (20 if long_context else 2)

    planner_inner = MockLLM(handler=lambda m, t: {
        "content": '{"subtasks": [{"worker": "调研A", "instruction": "查A市场"},'
                   '{"worker": "调研B", "instruction": "查B市场"}]}'})
    planner = TracedLLM(planner_inner, agent_name="Planner")
    planner.chat([{"role": "user", "content": f"{padding}请拆解任务：调研A/B市场"}])

    worker_a_inner = MockLLM(scripted=["A市场：偏好小型化产品。"])
    worker_a = TracedLLM(worker_a_inner, agent_name="调研A")
    worker_a.chat([{"role": "user", "content": f"{padding}调研A市场偏好"}])

    worker_b_inner = MockLLM(scripted=["B市场：偏好性价比产品。"])
    worker_b = TracedLLM(worker_b_inner, agent_name="调研B")
    worker_b.chat([{"role": "user", "content": f"{padding}调研B市场偏好"}])

    return RunRecord(run_id=run_id, task="市场调研",
                     traced_agents=[planner, worker_a, worker_b])


def main() -> None:
    banner("示例 03：成本归因报表")

    # 场景 1：单次运行的成本结构
    section("场景 1：一次协作运行的成本归因")
    run1 = simulate_orchestrator_run("run_A", long_context=True)
    print(cost_report([run1]))
    planner_tokens = run1.traced_agents[0].total_pseudo_tokens
    worker_a_tokens = run1.traced_agents[1].total_pseudo_tokens
    verify("每个 Agent 都有独立的消耗记录", planner_tokens > 0 and worker_a_tokens > 0)
    verify("Planner 持长上下文，是最贵环节",
           planner_tokens > worker_a_tokens,
           f"Planner {planner_tokens} vs 调研A {worker_a_tokens}")

    # 场景 2：优化前后对比（压缩上下文 = 直接省钱）
    section("场景 2：优化前后账单对比（上下文瘦身）")
    run_before = simulate_orchestrator_run("before", long_context=True)
    run_after = simulate_orchestrator_run("after", long_context=False)
    print(cost_report([run_before, run_after]))
    before_total = run_before.total_pseudo_tokens
    after_total = run_after.total_pseudo_tokens
    print(f"\n优化前 {before_total} 伪 token -> 优化后 {after_total} 伪 token，"
          f"节省 {(1 - after_total / before_total):.0%}")

    verify("优化后总消耗下降", after_total < before_total)
    verify("节省比例显著（长上下文是成本大头）",
           after_total < before_total * 0.5,
           f"{after_total} vs {before_total}")

    # 场景 3：把账单喂给下一章的护栏
    section("场景 3：成本数据是护栏的输入")
    budget = 500
    ok = run_after.total_pseudo_tokens <= budget
    verify("优化后的运行在预算内（护栏放行）", ok, f"{after_total} <= {budget}")
    verify("优化前的运行超预算（护栏会拦）", before_total > budget)
    print("-> 第 11 章将用同样的数字在'超预算时拒绝下一次调用'。")

    finish("ch10_cost_report")


if __name__ == "__main__":
    main()
