"""示例 03：终止条件设计 —— Agent 怎么"知道该停"？

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/05_ReAct循环/code/ch05_termination.py

三种终止：
1. 正常完成：模型给出最终回答（不再调工具）；
2. 重复调用检测：同一工具 + 同一参数累计出现 N 次 -> 疑似死循环，提前止损；
3. 保险丝：max_turns 兜底（第 1 章就有的设计，这里是最后的防线）。

可插拔设计：终止条件是"函数（轨迹）-> 终止原因或 None"，
新条件随时插入，不影响循环主体 —— 开闭原则。
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from ch05_react_loop import ReActAgent, Step, Trace, ToolBox  # noqa: E402
from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402

# 终止条件签名：读轨迹，返回终止原因字符串（None = 继续）。
StopCondition = Callable[[Trace], Optional[str]]


def repeated_call_detector(times: int = 2) -> StopCondition:
    """重复调用检测：同一 (工具名, 参数) 累计出现 times 次 -> 终止。

    参数必须 canonical 化（键排序）后比较，避免 {'a':1,'b':2} 与 {'b':2,'a':1}
    被误判为不同调用。
    """
    def check(trace: Trace) -> Optional[str]:
        counts: Counter = Counter()
        for step in trace.steps:
            if step.action is None:
                continue
            key = (step.action.name,
                   json.dumps(step.action.arguments, sort_keys=True, ensure_ascii=False))
            counts[key] += 1
            if counts[key] >= times:
                return f"重复调用检测：{key[0]}({key[1]}) 已累计 {counts[key]} 次，疑似死循环"
        return None

    return check


class GuardedAgent(ReActAgent):
    """带可插拔终止条件的 ReActAgent。"""

    def __init__(self, llm, toolbox: ToolBox, max_turns: int = 8):
        super().__init__(llm, toolbox, max_turns)
        self.stop_conditions: list[StopCondition] = []

    def add_stop_condition(self, condition: StopCondition) -> "GuardedAgent":
        self.stop_conditions.append(condition)
        return self

    def run(self, task: str) -> Trace:
        trace = Trace(task=task)
        messages: list[dict] = [{"role": "user", "content": task}]
        for turn in range(1, self.max_turns + 1):
            response = self.llm.chat(messages, tools=self.toolbox.schemas())
            if not response.tool_calls:
                trace.steps.append(Step(turn=turn, thought=response.content,
                                        final_answer=response.content))
                trace.termination = f"完成：第 {turn} 步给出最终回答"
                return trace
            messages.append({"role": "assistant", "content": response.content,
                             "tool_calls": response.tool_calls})
            # 与 ReActAgent 保持一致：执行响应中的全部工具调用，观察合并
            observations = [self.toolbox.execute(c.name, c.arguments)
                            for c in response.tool_calls]
            observation_text = "\n".join(observations)
            messages.append({"role": "tool", "tool_call_id": response.tool_calls[0].id,
                             "name": response.tool_calls[0].name, "content": observation_text})
            trace.steps.append(Step(turn=turn, thought=response.content,
                                    action=response.tool_calls[0], observation=observation_text))
            # 可插拔终止检查：任一条件命中即止损
            for condition in self.stop_conditions:
                reason = condition(trace)
                if reason:
                    trace.termination = reason
                    return trace
        trace.termination = f"保险丝：达到最大轮数 {self.max_turns}，强制停止"
        return trace


def get_weather(city: str) -> str:
    """查询城市天气。"""
    fake_db = {"北京": "晴，5~12°C"}
    return fake_db.get(city, f"{city}：多云")


def main() -> None:
    banner("示例 03：终止条件设计")

    toolbox = ToolBox().register(get_weather)

    # 场景 1：正常完成
    section("场景 1：正常完成（模型查完天气给出回答）")
    llm1 = MockLLM(scripted=[
        {"content": "先查天气。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "北京晴，5~12°C，穿厚外套合适。"},
    ])
    agent1 = GuardedAgent(llm1, toolbox).add_stop_condition(repeated_call_detector(times=3))
    trace1 = agent1.run("北京穿什么")
    print(f"终止原因: {trace1.termination}")
    verify("正常完成终止", trace1.termination.startswith("完成") and trace1.steps[-1].final_answer)

    # 场景 2：重复调用检测止损
    section("场景 2：模型反复查同一城市 -> 重复检测提前止损")
    llm2 = MockLLM(handler=lambda m, t: {
        "content": "我再查一次确认。",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}],
    })
    agent2 = GuardedAgent(llm2, toolbox, max_turns=10).add_stop_condition(repeated_call_detector(times=3))
    trace2 = agent2.run("北京天气到底怎么样")
    print(f"终止原因: {trace2.termination}")
    verify("在第 3 次重复时止损（而非烧满 10 轮）",
           len(trace2.steps) == 3 and "重复调用检测" in trace2.termination,
           f"实际步数 {len(trace2.steps)}，原因 {trace2.termination}")

    # 场景 3：参数不同的调用不算重复
    section("场景 3：参数不同 -> 不触发重复检测（合法的多次查询）")
    llm3 = MockLLM(scripted=[
        {"content": "查北京。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "再查上海。", "tool_calls": [{"name": "get_weather", "arguments": {"city": "上海"}}]},
        {"content": "两地都查完了：北京晴，上海小雨。"},
    ])
    agent3 = GuardedAgent(llm3, toolbox).add_stop_condition(repeated_call_detector(times=2))
    trace3 = agent3.run("北京和上海天气")
    print(f"终止原因: {trace3.termination}")
    verify("参数不同不误杀", trace3.termination.startswith("完成") and len(trace3.steps) == 3)

    # 场景 4：保险丝兜底
    section("场景 4：永不满足任何条件 -> max_turns 保险丝兜底")
    llm4 = MockLLM(handler=lambda m, t: {
        "content": "查点别的。",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": f"城市{len(m)}"}},  # 参数每次不同
                       ],
    })
    agent4 = GuardedAgent(llm4, toolbox, max_turns=4)
    trace4 = agent4.run("随便查查")
    print(f"终止原因: {trace4.termination}")
    verify("保险丝在第 4 步兜底", len(trace4.steps) == 4 and "保险丝" in trace4.termination)

    finish("ch05_termination")


if __name__ == "__main__":
    main()
