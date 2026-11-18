"""示例 02：轨迹持久化与断点续跑 —— Trace 序列化为 JSON，恢复现场继续跑。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/05_ReAct循环/code/ch05_trace_replay.py

为什么要持久化轨迹？
- 调试：出问题的运行可以逐帧回看（第 10 章可观测性的基础）；
- 审计：合规场景要求"每一步都留痕"；
- 恢复：进程崩了/超时了，从轨迹断点继续，而不是从头烧钱。
"""

import json
import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM, ToolCall  # noqa: E402
from ch05_react_loop import ReActAgent, Step, Trace, ToolBox  # noqa: E402


def get_weather(city: str) -> str:
    """查询城市天气。"""
    fake_db = {"北京": "晴，5~12°C"}
    return fake_db.get(city, f"{city}：多云")


# ---------------------------------------------------------------
# Trace <-> JSON 双向转换（纯函数式，便于测试）。
# ---------------------------------------------------------------
def trace_to_json(trace: Trace) -> str:
    """序列化：ToolCall 转为 {name, arguments, id}。"""
    payload = {
        "task": trace.task,
        "termination": trace.termination,
        "steps": [
            {
                "turn": step.turn,
                "thought": step.thought,
                "action": ({"name": step.action.name,
                            "arguments": step.action.arguments,
                            "id": step.action.id} if step.action else None),
                "observation": step.observation,
                "final_answer": step.final_answer,
            }
            for step in trace.steps
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def trace_from_json(text: str) -> Trace:
    """反序列化：重建 Step 与 ToolCall。"""
    payload = json.loads(text)
    steps = [
        Step(
            turn=s["turn"],
            thought=s["thought"],
            action=(ToolCall(name=s["action"]["name"],
                             arguments=s["action"]["arguments"],
                             id=s["action"]["id"]) if s["action"] else None),
            observation=s["observation"],
            final_answer=s["final_answer"],
        )
        for s in payload["steps"]
    ]
    return Trace(task=payload["task"], steps=steps, termination=payload["termination"])


# ---------------------------------------------------------------
# 断点续跑：从已有轨迹重建消息上下文，交给 LLM 继续推理。
# ---------------------------------------------------------------
def messages_from_trace(trace: Trace) -> list[dict]:
    """把轨迹还原成 LLM 消息序列（模型看到的上下文与中断前一致）。"""
    messages: list[dict] = [{"role": "user", "content": trace.task}]
    for step in trace.steps:
        if step.action is not None:
            messages.append({"role": "assistant", "content": step.thought,
                             "tool_calls": [step.action]})
            messages.append({"role": "tool", "tool_call_id": step.action.id,
                             "name": step.action.name, "content": step.observation})
        elif step.final_answer is not None:
            messages.append({"role": "assistant", "content": step.final_answer})
    return messages


def continue_from_trace(llm, toolbox, trace: Trace, toolbox_results: dict) -> Trace:
    """从轨迹断点继续运行：工具结果由调用方注入（模拟环境恢复）。

    toolbox_results: {工具名: 观察文本} —— 真实场景中应重新执行工具，
    这里用注入表保证离线确定性。
    """
    messages = messages_from_trace(trace)
    resumed = Trace(task=trace.task)
    for turn in range(1, 6):
        response = llm.chat(messages, tools=toolbox.schemas())
        if not response.tool_calls:
            resumed.steps.append(Step(turn=turn, thought=response.content,
                                      final_answer=response.content))
            resumed.termination = f"续跑完成：第 {turn} 步给出最终回答"
            return resumed
        messages.append({"role": "assistant", "content": response.content,
                         "tool_calls": response.tool_calls})
        call = response.tool_calls[0]
        observation = toolbox_results.get(call.name, f"（续跑注入）{call.name} 无结果")
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "name": call.name, "content": observation})
        resumed.steps.append(Step(turn=turn, thought=response.content,
                                  action=call, observation=observation))
    resumed.termination = "续跑保险丝触发"
    return resumed


def main() -> None:
    banner("示例 02：轨迹持久化与断点续跑")

    toolbox = ToolBox().register(get_weather)
    llm = MockLLM(scripted=[
        {"content": "先查北京天气。",
         "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "拿到天气了，可以给最终建议了。"},
    ])
    trace = ReActAgent(llm=llm, toolbox=toolbox).run("去北京玩穿什么")

    # 实验 1：序列化 -> 文件级 JSON
    section("实验 1：轨迹序列化为 JSON")
    text = trace_to_json(trace)
    print(text)
    verify("JSON 可解析且包含任务", json.loads(text)["task"] == "去北京玩穿什么")
    verify("工具调用被完整序列化", json.loads(text)["steps"][0]["action"]["name"] == "get_weather")

    # 实验 2：反序列化 -> 重建 Trace，与原轨迹等价
    section("实验 2：反序列化恢复，与原轨迹等价")
    restored = trace_from_json(text)
    verify("任务一致", restored.task == trace.task)
    verify("步数一致", len(restored.steps) == len(trace.steps))
    verify("行动一致（含 id）", restored.steps[0].action == trace.steps[0].action)
    verify("观察一致", restored.steps[0].observation == trace.steps[0].observation)
    verify("渲染文本一致", restored.render() == trace.render())

    # 实验 3：断点续跑 —— 中断后从轨迹恢复现场
    section("实验 3：断点续跑（模拟进程中断后恢复）")
    llm2 = MockLLM(scripted=[
        {"content": "（续跑）查到北京晴、5~12°C，建议穿厚外套。"},
    ])
    resumed = continue_from_trace(
        llm2, toolbox, restored,
        toolbox_results={"get_weather": "晴，5~12°C"},  # 注入环境结果（离线确定性）
    )
    print(resumed.render())
    verify("续跑 1 步即给出最终回答", len(resumed.steps) == 1 and resumed.steps[0].final_answer)
    verify("续跑上下文包含历史思考与观察",
           "先查北京天气" in llm2.calls[0]["messages"][1]["content"]
           and "晴" in llm2.calls[0]["messages"][2]["content"])
    verify("最终回答接续历史", "厚外套" in resumed.steps[0].final_answer)

    finish("ch05_trace_replay")


if __name__ == "__main__":
    main()
