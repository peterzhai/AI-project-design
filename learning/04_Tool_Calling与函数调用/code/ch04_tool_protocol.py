"""示例 03：完整 Tool-Calling 协议 —— 消息到底长什么样？

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/04_Tool_Calling与函数调用/code/ch04_tool_protocol.py

真实线上的消息往返（OpenAI tools 协议，逐字段还原）：
  1. 应用发 messages + tools（工具描述清单）给模型；
  2. 模型回 assistant 消息，带 tool_calls: [{id, type, function: {name, arguments}}]
     —— 注意 arguments 在线协议是 JSON **字符串**，不是 dict；
  3. 应用执行工具，回 role="tool" 消息，tool_call_id 必须对应；
  4. 模型看到全部观察，给最终回答。
本示例把这 4 步的消息逐条打印，并用 Mock 断言每一步的形状。
"""

import json
import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402
from common.toolbox import ToolBox  # noqa: E402


def get_weather(city: str) -> str:
    """查询城市天气。"""
    fake_db = {"北京": "晴，5~12°C"}
    return fake_db.get(city, f"{city}：多云")


def get_time(city: str) -> str:
    """查询城市当前时间。"""
    return f"{city} 当前 10:00"


def pretty(messages: list[dict]) -> None:
    """打印整段对话（协议视图）。"""
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        if "tool_calls" in m:
            calls = json.dumps(m["tool_calls"], ensure_ascii=False)
            print(f"  [{i}] {role}: (content={m.get('content')!r}, tool_calls={calls})")
        elif m.get("role") == "tool":
            print(f"  [{i}] tool(tool_call_id={m['tool_call_id']}, name={m['name']}): {m['content']!r}")
        else:
            print(f"  [{i}] {role}: {m['content']!r}")


def run_protocol(llm, toolbox: ToolBox, task: str) -> tuple[str, list[dict]]:
    """标准 Tool-Calling 协议一轮往返，返回 (最终回答, 完整消息列表)。"""
    # 第 1 步：应用 -> 模型：带上工具清单
    messages: list[dict] = [{"role": "user", "content": task}]
    response = llm.chat(messages, tools=toolbox.schemas())

    if response.tool_calls:
        # 第 2 步：模型 -> 应用：assistant 消息带 tool_calls（协议原样组装）
        assistant_msg = {
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),  # 线上是 JSON 字符串！
                    },
                }
                for call in response.tool_calls
            ],
        }
        messages.append(assistant_msg)

        # 第 3 步：应用执行工具 -> 回 role="tool" 消息，tool_call_id 一一对应
        for call in response.tool_calls:
            result = toolbox.execute(call.name, call.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})

        # 第 4 步：再次调用模型（带着全部观察）拿最终回答
        final = llm.chat(messages, tools=toolbox.schemas())
        messages.append({"role": "assistant", "content": final.content})
        return final.content, messages

    messages.append({"role": "assistant", "content": response.content})
    return response.content, messages


def main() -> None:
    banner("示例 03：完整 Tool-Calling 协议")

    toolbox = ToolBox().register(get_weather).register(get_time)
    llm = MockLLM(scripted=[
        {"tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "北京今天晴，5~12°C，适合户外活动。"},
    ])

    answer, messages = run_protocol(llm, toolbox, "北京天气怎么样？")
    print("\n协议消息全貌：")
    pretty(messages)

    section("协议自检")
    verify("工具清单随请求发出（含 schema）",
           llm.calls[0]["tools"] == toolbox.schemas() and
           llm.calls[0]["tools"][0]["function"]["name"] == "get_weather")
    verify("模型自动生成了 tool_call_id", messages[1]["tool_calls"][0]["id"] == "call_1")
    verify("线协议 arguments 是 JSON 字符串",
           messages[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "北京"}')
    verify("tool 消息正确回传 tool_call_id",
           messages[2] == {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "晴，5~12°C"})
    verify("第二轮模型带完整观察上下文", len(llm.calls[1]["messages"]) == 3)
    verify("最终回答基于观察结果", "晴" in answer)

    # 并行工具调用：一个响应里同时要两个工具
    section("进阶：并行工具调用（一个响应两个 tool_calls）")
    llm2 = MockLLM(scripted=[
        {"tool_calls": [
            {"name": "get_weather", "arguments": {"city": "北京"}},
            {"name": "get_time", "arguments": {"city": "上海"}},
        ]},
        {"content": "北京晴；上海现在 10:00。"},
    ])
    answer2, messages2 = run_protocol(llm2, toolbox, "北京天气和上海时间")
    print("\n并行调用消息全貌：")
    pretty(messages2)
    verify("两个工具调用各自有独立 id", [c["id"] for c in messages2[1]["tool_calls"]] == ["call_1", "call_2"])
    verify("两条 tool 消息各自对应 tool_call_id",
           [m["tool_call_id"] for m in messages2[2:4]] == ["call_1", "call_2"])
    verify("最终回答同时用上两个观察", "晴" in answer2 and "10:00" in answer2)

    finish("ch04_tool_protocol")


if __name__ == "__main__":
    main()
