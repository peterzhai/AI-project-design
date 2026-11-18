"""示例 02：计划执行器 —— 按依赖顺序执行，失败重试，前置失败则跳过。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/08_规划与任务分解/code/ch08_plan_executor.py

执行器的三条纪律：
1. 依赖顺序：depends_on 的步骤全部 done 才执行（否则 skipped）；
2. 单步重试：瞬时失败（工具抽风）重试 1 次，仍失败标记 failed；
3. 失败传播：前置 failed 的步骤自动 skipped，不浪费调用。
每步的 status/attempts/output 全部落入执行报告 —— 可审计、可回放。
"""

import sys
from dataclasses import dataclass
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from ch08_task_decomposition import Plan, PlanStep  # noqa: E402
from common.check import banner, finish, section, verify  # noqa: E402
from common.toolbox import ToolBox  # noqa: E402


@dataclass
class StepResult:
    """一个步骤的执行结果。"""

    step_id: int
    status: str        # done / failed / skipped
    output: str
    attempts: int = 1


def _deps_all_done(results: dict[int, StepResult], step: PlanStep) -> bool:
    """前置依赖是否全部成功。"""
    return all(results.get(dep) is not None and results[dep].status == "done"
               for dep in step.depends_on)


class PlanExecutor:
    """按计划顺序执行步骤；失败重试；前置失败自动跳过。"""

    def __init__(self, toolbox: ToolBox, max_attempts: int = 2):
        self.toolbox = toolbox
        self.max_attempts = max_attempts

    def execute(self, plan: Plan) -> list[StepResult]:
        results: dict[int, StepResult] = {}
        ordered: list[StepResult] = []
        for step in plan.steps:
            if not _deps_all_done(results, step):
                result = StepResult(step_id=step.id, status="skipped",
                                    output="前置步骤未成功，跳过")
            else:
                result = self._run_step(step)
            results[step.id] = result
            ordered.append(result)
        return ordered

    def _run_step(self, step: PlanStep) -> StepResult:
        """执行单步：调用工具，失败重试至 max_attempts 次。"""
        observation = ""
        for attempt in range(1, self.max_attempts + 1):
            ok, observation = self.toolbox.try_execute(step.tool, step.arguments)
            if ok:
                return StepResult(step_id=step.id, status="done",
                                  output=observation, attempts=attempt)
        return StepResult(step_id=step.id, status="failed",
                          output=observation, attempts=self.max_attempts)


def render_report(results: list[StepResult]) -> str:
    """执行报告：每步状态一目了然。"""
    lines = ["执行报告:"]
    for r in results:
        mark = {"done": "✅", "failed": "❌", "skipped": "⏭️"}[r.status]
        lines.append(f"  {mark} 步骤{r.step_id}: {r.status}"
                     f"（尝试 {r.attempts} 次）{r.output[:40]}")
    return "\n".join(lines)


# ---------------------------------------------------------------
# 工具定义（带审计与可配置的瞬时故障）。
# ---------------------------------------------------------------
def build_toolbox(audit: list[str], venue_fail_times: int = 0) -> ToolBox:
    """构造执行工具箱：venue_fail_times 控制预订工具前 N 次抛错。"""
    state = {"venue_calls": 0}

    def call_venue(venue: str) -> str:
        """预订场地（前 venue_fail_times 次模拟预订系统超时）。"""
        state["venue_calls"] += 1
        if state["venue_calls"] <= venue_fail_times:
            raise ConnectionError("预订系统超时")
        audit.append(f"预订{venue}")
        return f"{venue} 已预订"

    def send_notice(content: str) -> str:
        """向全员发送通知。"""
        audit.append("发送通知")
        return f"通知已发送: {content}"

    return ToolBox().register(call_venue).register(send_notice)


def main() -> None:
    banner("示例 02：计划执行器")

    # 场景 1：线性计划全部成功
    section("场景 1：全部成功")
    audit: list[str] = []
    toolbox = build_toolbox(audit)
    plan = Plan(goal="组织团建", steps=(
        PlanStep(id=1, description="预订A场地", tool="call_venue", arguments={"venue": "A"}),
        PlanStep(id=2, description="发送通知", depends_on=(1,),
                 tool="send_notice", arguments={"content": "团建定在下周六"}),
    ))
    results = PlanExecutor(toolbox).execute(plan)
    print(render_report(results))
    verify("两步都 done", [r.status for r in results] == ["done", "done"])
    verify("步骤 2 拿到依赖产物后执行（审计顺序正确）", audit == ["预订A", "发送通知"])

    # 场景 2：瞬时失败 -> 重试成功
    section("场景 2：预订系统抖动一次 -> 重试成功（attempts=2）")
    audit2: list[str] = []
    toolbox2 = build_toolbox(audit2, venue_fail_times=1)
    plan2 = Plan(goal="组织团建", steps=(
        PlanStep(id=1, description="预订B场地", tool="call_venue", arguments={"venue": "B"}),
    ))
    results2 = PlanExecutor(toolbox2).execute(plan2)
    print(render_report(results2))
    verify("步骤 1 重试后 done", results2[0].status == "done")
    verify("attempts 记录为 2", results2[0].attempts == 2, f"实际 {results2[0].attempts}")

    # 场景 3：彻底失败 -> 前置失败 -> 后续跳过
    section("场景 3：预订彻底失败 -> 通知步骤自动跳过")
    audit3: list[str] = []
    toolbox3 = build_toolbox(audit3, venue_fail_times=99)  # 永远失败
    plan3 = Plan(goal="组织团建", steps=(
        PlanStep(id=1, description="预订C场地", tool="call_venue", arguments={"venue": "C"}),
        PlanStep(id=2, description="发送通知", depends_on=(1,),
                 tool="send_notice", arguments={"content": "团建取消通知"}),
    ))
    results3 = PlanExecutor(toolbox3).execute(plan3)
    print(render_report(results3))
    verify("步骤 1 failed（重试耗尽）", results3[0].status == "failed")
    verify("步骤 2 skipped（不浪费调用）", results3[1].status == "skipped")
    verify("审计里没有通知（失败传播正确）", audit3 == [])

    # 场景 4：依赖多个前序步骤（部分失败即跳过）
    section("场景 4：多依赖步骤（依赖 failed -> skipped）")
    audit4: list[str] = []
    toolbox4 = build_toolbox(audit4, venue_fail_times=99)
    plan4 = Plan(goal="组织团建", steps=(
        PlanStep(id=1, description="预订D场地", tool="call_venue", arguments={"venue": "D"}),
        PlanStep(id=2, description="预订E场地", tool="call_venue", arguments={"venue": "E"}),
        PlanStep(id=3, description="发送通知", depends_on=(1, 2),
                 tool="send_notice", arguments={"content": "两个场地都定了"}),
    ))
    results4 = PlanExecutor(toolbox4).execute(plan4)
    print(render_report(results4))
    verify("多依赖一步失败即跳过", [r.status for r in results4] == ["failed", "failed", "skipped"])

    finish("ch08_plan_executor")


if __name__ == "__main__":
    main()
