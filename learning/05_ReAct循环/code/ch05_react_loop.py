"""示例 01：ReAct —— 把"思考"显式化：Thought -> Action -> Observation 循环。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/05_ReAct循环/code/ch05_react_loop.py

与第 1/4 章的区别：
- 之前的循环只记录"行动"与"观察"，模型的思考是黑盒；
- ReAct 让模型在每次行动前**先说出思考**（assistant 消息的 content 字段），
  每一步变成可读的三段式 —— 这就是"推理 + 行动"（Reasoning + Acting）。

本示例实现一个带轨迹记录的 ReActAgent，后续章节将复用它的 Trace 结构。
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM, ToolCall  # noqa: E402
from common.toolbox import ToolBox  # noqa: E402


# ---------------------------------------------------------------
# 轨迹数据结构：每一步都是一个可打印的三段式。
# ---------------------------------------------------------------
@dataclass
class Step:
    """ReAct 的一步：思考 -> 行动 -> 观察（终止步则是 思考 -> 最终回答）。"""

    turn: int
    thought: str                       # 模型的显式思考（assistant 消息的 content）
    action: Optional[ToolCall] = None  # 行动（None 表示本轮直接回答）
    observation: Optional[str] = None  # 工具观察结果
    final_answer: Optional[str] = None # 终止步的最终回答

    def render(self) -> str:
        lines = [f"第 {self.turn} 步", f"  思考: {self.thought or '（无）'}"]
        if self.action is not None:
            lines.append(f"  行动: {self.action}")
            lines.append(f"  观察: {self.observation}")
        if self.final_answer is not None:
            lines.append(f"  最终回答: {self.final_answer}")
        return "\n".join(lines)


@dataclass
class Trace:
    """一次完整运行的轨迹：任务 + 逐步记录 + 终止原因。"""

    task: str
    steps: list[Step] = field(default_factory=list)
    termination: str = ""  # 为什么停（完成/保险丝/其他条件）

    def render(self) -> str:
        header = f"任务: {self.task}\n终止原因: {self.termination}\n" + "-" * 40
        return header + "\n" + "\n".join(step.render() for step in self.steps)


# ---------------------------------------------------------------
# ReActAgent：显式三段式循环。
# ---------------------------------------------------------------
class ReActAgent:
    """每轮先记录思考，再行动、观察；模型不再调工具时输出最终回答。"""

    def __init__(self, llm, toolbox: ToolBox, max_turns: int = 8):
        self.llm = llm
        self.toolbox = toolbox
        self.max_turns = max_turns

    def run(self, task: str) -> Trace:
        trace = Trace(task=task)
        messages: list[dict] = [{"role": "user", "content": task}]
        for turn in range(1, self.max_turns + 1):
            response = self.llm.chat(messages, tools=self.toolbox.schemas())
            if not response.tool_calls:
                # 终止：模型给出最终回答
                trace.steps.append(Step(turn=turn, thought=response.content,
                                        final_answer=response.content))
                trace.termination = f"完成：模型在第 {turn} 步给出最终回答"
                messages.append({"role": "assistant", "content": response.content})
                return trace
            # 行动：执行响应中的全部工具调用（观察合并为一段文本）
            messages.append({
                "role": "assistant",
                "content": response.content,   # ReAct 灵魂：思考显式进入上下文
                "tool_calls": response.tool_calls,
            })
            observations: list[str] = []
            for call in response.tool_calls:
                observations.append(self.toolbox.execute(call.name, call.arguments))
            observation_text = "\n".join(observations)
            messages.append({"role": "tool", "tool_call_id": response.tool_calls[0].id,
                             "name": response.tool_calls[0].name, "content": observation_text})
            trace.steps.append(Step(turn=turn, thought=response.content,
                                    action=response.tool_calls[0], observation=observation_text))
        trace.termination = f"保险丝：达到最大轮数 {self.max_turns}，强制停止"
        return trace


# ---------------------------------------------------------------
# 演示
# ---------------------------------------------------------------
def get_weather(city: str) -> str:
    """查询城市天气。"""
    fake_db = {"北京": "晴，5~12°C"}
    return fake_db.get(city, f"{city}：多云")


def search_train(depart: str, arrive: str) -> str:
    """查询两地之间的火车班次。"""
    return f"{depart}->{arrive}: G101 08:00 / G103 10:30"


def main() -> None:
    banner("示例 01：ReAct 三段式循环")

    toolbox = ToolBox().register(get_weather).register(search_train)
    llm = MockLLM(scripted=[
        {"content": "用户想去北京游玩，我需要先确认北京的天气。",
         "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "天气不错。接下来查一下出发去北京的火车。",
         "tool_calls": [{"name": "search_train", "arguments": {"depart": "天津", "arrive": "北京"}}]},
        {"content": "信息齐了：北京晴，G101 08:00 出发时间合适，建议乘坐。"},
    ])

    agent = ReActAgent(llm=llm, toolbox=toolbox)
    trace = agent.run("帮我从天津去北京玩，查下天气和火车，给个建议")
    print(trace.render())

    section("自检")
    verify("共 3 步（2 次行动 + 1 次最终回答）", len(trace.steps) == 3, f"实际 {len(trace.steps)}")
    verify("每一步都有显式思考", all(s.thought for s in trace.steps))
    verify("行动步都有观察", all(s.observation for s in trace.steps if s.action))
    verify("思考被写进上下文（第 2 次调用可见第 1 步思考）",
           "先确认北京的天气" in llm.calls[1]["messages"][1]["content"])
    verify("最终回答基于两次观察", "G101" in trace.steps[-1].final_answer and "晴" in trace.steps[-1].final_answer)
    verify("终止原因记录为完成", "完成" in trace.termination)

    section("ReAct 的价值：轨迹可读，黑盒变白盒")
    for step in trace.steps:
        print(f"  turn{step.turn}: 思考[{step.thought[:20]}…] " +
              (f"行动[{step.action.name}]" if step.action else "→ 最终回答"))

    finish("ch05_react_loop")


if __name__ == "__main__":
    main()
