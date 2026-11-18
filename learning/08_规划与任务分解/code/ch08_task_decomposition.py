"""示例 01：任务分解 —— 让 LLM 先"列计划"，而不是一头扎进去硬做。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/08_规划与任务分解/code/ch08_task_decomposition.py

ReAct（第 5 章）是"走一步看一步"；规划（Planning）是"先画路线图再出发"：
1. Planner（LLM）把目标拆成有序步骤，每步声明依赖（depends_on）；
2. 计划先过 schema 校验 + 结构合法性检查（id 唯一、依赖存在、**无环**）；
3. 校验过的计划才能交给执行器（示例 02）。
计划是结构化输出（第 3 章），所以本章直接复用 validators 与 JSON 修复闭环。
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.json_utils import JsonParseError, parse_llm_json  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402
from common.validators import ValidationError, validate  # noqa: E402

# 计划的输出契约：goal + 有序步骤（id、描述、可选依赖）。与第 3 章同一思路。
PLAN_SCHEMA = {
    "type": "object",
    "required": ["goal", "steps"],
    "additionalProperties": False,
    "properties": {
        "goal": {"type": "string", "minLength": 1},
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "description"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer", "minimum": 1},
                    "description": {"type": "string", "minLength": 2},
                    "depends_on": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                    "tool": {"type": "string", "minLength": 1},
                    "arguments": {"type": "object"},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class PlanStep:
    """计划中的一个步骤。tool/arguments 可选：填了才能被执行器直接执行。"""

    id: int
    description: str
    depends_on: tuple[int, ...] = ()
    tool: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    """一份完整的执行计划。"""

    goal: str
    steps: tuple[PlanStep, ...]

    def render(self) -> str:
        lines = [f"目标: {self.goal}"]
        for step in self.steps:
            deps = f"（依赖: {list(step.depends_on)}）" if step.depends_on else ""
            tool = f" [工具: {step.tool}]" if step.tool else ""
            lines.append(f"  {step.id}. {step.description}{deps}{tool}")
        return "\n".join(lines)


def _find_cycle(steps: tuple[PlanStep, ...]) -> Optional[list[int]]:
    """检测依赖环（有环 = 死锁，永远执行不了）。返回环路径或 None。"""
    by_id = {s.id: s for s in steps}
    state: dict[int, int] = {}  # 0=未访问 1=访问中 2=完成
    path: list[int] = []
    cycle: list[int] = []

    def visit(step_id: int) -> bool:
        state[step_id] = 1
        path.append(step_id)
        for dep in by_id[step_id].depends_on:
            if dep not in by_id:
                continue  # 引用不存在的依赖由 make_plan 单独报错
            if state.get(dep) == 1:               # 回到访问中 -> 找到环
                cycle.extend(path[path.index(dep):])
                return True
            if state.get(dep, 0) == 0 and visit(dep):
                return True
        state[step_id] = 2
        path.pop()
        return False

    for step in steps:
        if state.get(step.id, 0) == 0 and visit(step.id):
            return cycle
    return None


def make_plan(data: dict) -> Plan:
    """校验并构建 Plan：schema 校验 + 三条结构合法性检查。"""
    errors = validate(data, PLAN_SCHEMA)
    if errors:
        raise ValidationError(errors)
    steps = tuple(
        PlanStep(id=s["id"], description=s["description"],
                 depends_on=tuple(s.get("depends_on", [])),
                 tool=s.get("tool", ""),
                 arguments=s.get("arguments", {}))
        for s in data["steps"]
    )
    # 检查 1：id 必须唯一（重复 id 会让依赖指向产生歧义）
    ids = [s.id for s in steps]
    if len(ids) != len(set(ids)):
        raise ValidationError([f"步骤 id 重复: {sorted(ids)}"])
    # 检查 2：依赖必须指向已存在的步骤
    missing = [dep for s in steps for dep in s.depends_on if dep not in set(ids)]
    if missing:
        raise ValidationError([f"依赖引用了不存在的步骤 id: {sorted(set(missing))}"])
    # 检查 3：依赖图必须无环（有环 = 死锁）
    cycle = _find_cycle(steps)
    if cycle:
        raise ValidationError([f"依赖存在环，无法执行: {' -> '.join(map(str, cycle))}"])
    return Plan(goal=data["goal"], steps=steps)


PLANNER_PROMPT = (
    "你是任务规划器。把用户目标拆解为可执行步骤，只输出 JSON：\n"
    '{"goal": "...", "steps": [{"id": 1, "description": "...", "depends_on": []}]}\n'
    "要求：步骤 id 从 1 递增；有前置关系的步骤用 depends_on 声明；依赖不能成环。"
)


def make_plan_with_llm(llm, goal: str, max_attempts: int = 2) -> Plan:
    """让 LLM 分解目标；解析/校验失败走第 3 章的修复闭环（简化版：一次重试）。"""
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": goal},
    ]
    for attempt in range(1, max_attempts + 1):
        response = llm.chat(messages)
        try:
            data = parse_llm_json(response.content)
            return make_plan(data)  # type: ignore[arg-type]
        except (JsonParseError, ValidationError) as exc:
            if attempt == max_attempts:
                raise
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",
                             "content": f"计划不合法：{exc}\n请修正后重新只输出合法 JSON。"})
    raise RuntimeError("不可达")


def main() -> None:
    banner("示例 01：任务分解")

    # 场景 1：正常分解（步骤可带工具调用，供执行器直接执行）
    section("场景 1：「组织一次部门团建」的分解")
    planner = MockLLM(scripted=[
        {"content": json.dumps({
            "goal": "组织一次部门团建",
            "steps": [
                {"id": 1, "description": "调研候选场地与预算"},
                {"id": 2, "description": "发起内部投票选出场地", "depends_on": [1]},
                {"id": 3, "description": "预订A场地并确认档期", "depends_on": [2],
                 "tool": "call_venue", "arguments": {"venue": "A"}},
                {"id": 4, "description": "向全员发送通知", "depends_on": [3]},
            ],
        }, ensure_ascii=False)},
    ])
    plan = make_plan_with_llm(planner, "组织一次部门团建")
    print(plan.render())
    verify("计划含 4 个步骤", len(plan.steps) == 4)
    verify("依赖链呈线性 1->2->3->4",
           plan.steps[1].depends_on == (1,) and plan.steps[3].depends_on == (3,))
    verify("步骤 3 携带可执行的工具调用",
           plan.steps[2].tool == "call_venue" and plan.steps[2].arguments == {"venue": "A"})
    verify("无环、依赖合法（通过全部结构检查）", True)
    verify("Planner 被调用 1 次（一次通过）", planner.call_count == 1)

    # 场景 2：计划有环 -> 结构检查拦下
    section("场景 2：依赖成环的计划被当场拒绝")
    cyclic = {"goal": "搬起石头砸自己的脚", "steps": [
        {"id": 1, "description": "先拿起石头", "depends_on": [2]},
        {"id": 2, "description": "但手要空着才能砸", "depends_on": [1]},
    ]}
    # 注意：except 变量只在块内有效（Python 会在块结束时删除），断言必须写在块内。
    try:
        make_plan(cyclic)
        raise AssertionError("场景 2 失败：应当检出环")
    except ValidationError as exc:
        print(f"按预期拒绝: {exc.errors[0]}")
        verify("环被检出", "存在环" in exc.errors[0])

    # 场景 3：id 重复 / 依赖不存在
    section("场景 3：其他结构非法的计划")
    dup_ids = {"goal": "g", "steps": [
        {"id": 1, "description": "步骤甲"}, {"id": 1, "description": "步骤乙"},
    ]}
    try:
        make_plan(dup_ids)
        raise AssertionError("场景 3a 失败：应当拒绝重复 id")
    except ValidationError as exc:
        verify("重复 id 被拒绝", "id 重复" in exc.errors[0])

    bad_dep = {"goal": "g", "steps": [{"id": 1, "description": "步骤甲", "depends_on": [9]}]}
    try:
        make_plan(bad_dep)
        raise AssertionError("场景 3b 失败：应当拒绝不存在的依赖")
    except ValidationError as exc:
        verify("不存在的依赖被拒绝", "不存在的步骤 id" in exc.errors[0])

    # 场景 4：模型输出坏 JSON -> 修复闭环
    section("场景 4：Planner 第一次输出坏 JSON -> 喂回修复")
    flaky_planner = MockLLM(scripted=[
        {"content": "计划如下：{goal: 没有引号，不是JSON}"},   # 坏 JSON
        {"content": json.dumps({
            "goal": "g2", "steps": [{"id": 1, "description": "修复后的步骤"}]
        }, ensure_ascii=False)},
    ])
    plan2 = make_plan_with_llm(flaky_planner, "g2")
    verify("修复闭环后拿到合法计划", len(plan2.steps) == 1 and flaky_planner.call_count == 2)

    finish("ch08_task_decomposition")


if __name__ == "__main__":
    main()
