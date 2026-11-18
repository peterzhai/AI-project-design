"""示例 01：十几行代码看清 Agent 的本质 —— LLM + 工具 + 循环。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/01_导论与心智模型/code/ch01_hello_agent.py

看完本示例你只需要记住一个公式：
    Agent = LLM（大脑）+ Tools（手脚）+ Loop（心跳）
"""

import sys
from pathlib import Path

# 让脚本无论从哪个工作目录运行，都能 import 到 learning/common。
LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402

# ---------------------------------------------------------------
# 第 1 部分：工具 = 普通_python函数，Agent 的"手脚"。
# ---------------------------------------------------------------
FAKE_WEATHER_DB = {"北京": "晴，5~12°C，西北风3级", "上海": "小雨，14~18°C"}


def get_weather(city: str) -> str:
    """查询城市天气（演示用假数据，不联网）。"""
    return FAKE_WEATHER_DB.get(city, f"{city}：多云，气温适中")


TOOLS = {"get_weather": get_weather}


# ---------------------------------------------------------------
# 第 2 部分：最小 Agent 循环 —— 本示例的核心，请逐行读。
# ---------------------------------------------------------------
def run_minimal_agent(llm, tools: dict, task: str, max_turns: int = 5) -> str:
    """最小 Agent 循环。

    参数:
        llm:      任何提供 chat(messages, tools) -> ChatResponse 的客户端
        tools:    {工具名: 函数} 字典
        task:     用户的自然语言任务
        max_turns: 最大思考-行动轮数（防止无限循环的保险丝）
    """
    messages = [{"role": "user", "content": task}]
    for _ in range(max_turns):
        response = llm.chat(messages, tools=list(tools))
        if not response.tool_calls:               # 模型不再请求工具 => 已给出最终答案
            return response.content
        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": response.tool_calls,    # 记录模型的"行动决定"
        })
        for call in response.tool_calls:
            result = tools[call.name](**call.arguments)   # 真正执行工具
            messages.append({
                "role": "tool",
                "name": call.name,
                "content": str(result),           # 工具结果作为"观察"回传给模型
            })
    raise RuntimeError(f"超过最大轮数 {max_turns}：模型可能陷入了工具调用循环")


# ---------------------------------------------------------------
# 第 3 部分：三组对照实验 + 自检
# ---------------------------------------------------------------
def main() -> None:
    banner("示例 01：最小 Agent —— LLM + 工具 + 循环")

    # 实验 1：带工具的任务（两轮：调工具 -> 给最终答案）
    section("实验 1：带工具的任务「查北京天气并给穿衣建议」")
    llm = MockLLM(scripted=[
        {"tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "北京今天晴，5~12°C，昼夜温差大，建议穿件厚外套。"},
    ])
    answer = run_minimal_agent(llm, TOOLS, "查一下北京今天的天气，并给穿衣建议")
    print(f"最终答案: {answer}")
    verify("答案包含穿衣建议（提到外套）", "外套" in answer)
    verify("LLM 共被调用 2 次（1 次决定调工具，1 次总结）", llm.call_count == 2, f"实际 {llm.call_count}")
    verify(
        "第 2 次调用时模型已看到工具返回的天气",
        "晴" in llm.calls[1]["messages"][-1]["content"],
    )

    # 实验 2：不需要工具的任务（零轮工具调用，直接回答）
    section("实验 2：无需工具的任务「用一句话自我介绍」")
    llm2 = MockLLM(scripted=[{"content": "我是一个能使用工具完成任务的助手。"}])
    answer2 = run_minimal_agent(llm2, TOOLS, "用一句话自我介绍")
    print(f"最终答案: {answer2}")
    verify("无需工具时 1 次调用即结束", llm2.call_count == 1, f"实际 {llm2.call_count}")

    # 实验 3：保险丝——模型若永远只调工具，循环会被 max_turns 截断
    section("实验 3：保险丝「模型反复调工具时 max_turns 兜底」")
    looping_llm = MockLLM(scripted=[
        {"tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]}
        for _ in range(10)  # 预设 10 次工具调用，远超 max_turns
    ])
    try:
        run_minimal_agent(looping_llm, TOOLS, "查天气", max_turns=3)
        raise AssertionError("实验 3 失败：max_turns=3 时应当抛出 RuntimeError")
    except RuntimeError as exc:
        print(f"按预期被保险丝截断: {exc}")
    verify("保险丝生效（抛出 RuntimeError 而非死循环）", True)

    finish("ch01_hello_agent")


if __name__ == "__main__":
    main()
