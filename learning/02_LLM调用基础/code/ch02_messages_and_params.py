"""示例 01：LLM 调用的第一课 —— 消息角色与参数。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/02_LLM调用基础/code/ch02_messages_and_params.py

三个要点：
1. 对话是"消息列表"：system 定身份、user 提问、assistant 作答、tool 给观察。
2. 模型的行为由 system 提示词塑造 —— 同一句输入，换 system 得到完全不同的输出。
3. temperature / max_tokens 等参数随调用透传，是控制成本与稳定性的旋钮。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM, messages_text  # noqa: E402


# ---------------------------------------------------------------
# 一个"有身份感"的假 LLM：读 system 提示词决定行为（模拟真实模型）。
# ---------------------------------------------------------------
def make_translator_llm() -> MockLLM:
    def handler(messages, tools):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        text = messages[-1]["content"]
        if "法语" in system:
            return {"content": f"Bonjour ! （法语回应: {text}）"}
        return {"content": f"（原样回应: {text}）"}

    return MockLLM(handler=handler)


def main() -> None:
    banner("示例 01：消息角色与参数")

    # 实验 1：同一句输入，system 提示词不同 -> 行为完全不同
    section("实验 1：system 提示词塑造模型行为")
    llm = make_translator_llm()
    with_system = llm.chat([
        {"role": "system", "content": "你是一个法语翻译，任何输入都用法语回复。"},
        {"role": "user", "content": "你好"},
    ])
    without_system = llm.chat([{"role": "user", "content": "你好"}])
    print(f"带 system: {with_system.content}")
    print(f"不带 system: {without_system.content}")
    verify("法语 system 让模型切到法语", with_system.content.startswith("Bonjour"))
    verify("无 system 时原样回应", without_system.content.startswith("（原样"))

    # 实验 2：参数透传 —— temperature / max_tokens 随调用记录，可审计
    section("实验 2：调用参数透传（temperature / max_tokens）")
    llm2 = MockLLM(scripted=["无论参数如何，Mock 都是确定性输出。"])
    llm2.chat(
        [{"role": "user", "content": "讲个笑话"}],
        temperature=0.2,   # 低温度：更确定，适合抽取类任务
        max_tokens=100,    # 上限：控制成本
    )
    recorded = llm2.calls[0]["params"]
    print(f"客户端记录到的参数: {recorded}")
    verify("参数被完整透传并记录", recorded == {"temperature": 0.2, "max_tokens": 100})
    print("（真实模型中 temperature=0.2 与 0.9 的输出差异极大；Mock 固定输出便于测试）")

    # 实验 3：多轮对话的上下文组装 —— 每次调用都把历史带上
    section("实验 3：多轮对话 = 不断增长的消息列表")
    llm3 = make_translator_llm()
    conversation = [
        {"role": "system", "content": "你是一个法语翻译，任何输入都用法语回复。"},
        {"role": "user", "content": "你好"},
    ]
    reply1 = llm3.chat(conversation)                    # 第 1 轮
    conversation.append({"role": "assistant", "content": reply1.content})
    conversation.append({"role": "user", "content": "谢谢"})
    reply2 = llm3.chat(conversation)                    # 第 2 轮：带着全部历史
    print(f"第 1 轮: {reply1.content}")
    print(f"第 2 轮: {reply2.content}")
    verify("第 2 次调用的上下文包含 4 条消息", len(llm3.calls[1]["messages"]) == 4,
           f"实际 {len(llm3.calls[1]['messages'])}")
    verify("历史中的 assistant 回答被带上", "Bonjour" in llm3.calls[1]["messages"][2]["content"])

    print("\n完整对话渲染：")
    print(messages_text(llm3.calls[1]["messages"]))

    finish("ch02_messages_and_params")


if __name__ == "__main__":
    main()
