"""示例 01：滑动窗口 —— 对话越长越贵，窗口外的消息不能直接丢。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/06_记忆系统（短期_长期_摘要）/code/ch06_sliding_window.py

问题：上下文每轮都在增长 -> token 费用线性上涨，最终撑爆窗口。
朴素解法：只保留最近 N 条 —— 但会丢掉 system 提示词（身份）和早期关键信息。
本示例实现 ContextWindow：
1. system 消息永远保留（身份不能丢）；
2. 按消息条数滑动窗口；
3. 内置"字符数近似 token"预算（教程约定：1 字符 ≈ 1 伪 token，便于离线确定性）。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402


def approx_tokens(messages: list[dict]) -> int:
    """估算消息列表的 token 数（教程约定：1 字符 ≈ 1 伪 token）。"""
    return sum(len(str(m.get("content", ""))) for m in messages)


class ContextWindow:
    """滑动窗口记忆：system 永驻 + 最近 max_messages 条 + 可选预算控制。"""

    def __init__(self, max_messages: int = 6, max_tokens: int = 10_000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.system: str = ""           # system 提示词单独保存，永不进滑动区
        self._history: list[dict] = []  # 除 system 外的全部对话

    def set_system(self, content: str) -> None:
        self.system = content

    def add(self, message: dict) -> None:
        if message.get("role") == "system":
            self.set_system(str(message.get("content", "")))
            return
        self._history.append(dict(message))

    def render(self) -> list[dict]:
        """输出受窗口约束的上下文：system + 最近 max_messages 条。"""
        windowed = self._history[-self.max_messages:]
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(windowed)
        return messages

    def evicted(self) -> list[dict]:
        """被窗口挤出、暂未进入长期记忆的消息（第 2 个示例会接住它们）。"""
        return self._history[:-self.max_messages] if len(self._history) > self.max_messages else []

    def stats(self) -> str:
        return f"历史 {len(self._history)} 条 / 窗口内 {len(self.render())} 条 / 约 {approx_tokens(self.render())} 伪token"


def main() -> None:
    banner("示例 01：滑动窗口记忆")

    memory = ContextWindow(max_messages=4)
    memory.set_system("你是音乐推荐助手，回答保持简短。")
    # 模拟 8 轮对话（只加 user/assistant）
    for i in range(1, 8):
        memory.add({"role": "user", "content": f"推荐一首第{i}类风格的歌曲"})
        memory.add({"role": "assistant", "content": f"推荐第{i}号歌曲《旋律{i}》"})

    section("窗口效果")
    context = memory.render()
    roles = [m["role"] for m in context]
    print(f"渲染出的上下文角色序列: {roles}")
    print(f"统计: {memory.stats()}")

    verify("system 永驻窗口头部", roles[0] == "system" and "音乐推荐助手" in context[0]["content"])
    verify("窗口只保留最近 4 条历史", len(context) == 5, f"实际 {len(context)}")
    verify("最早的消息已被挤出", all("第1类" not in m.get("content", "") for m in context))
    verify("最近一轮完整保留", context[-1]["content"] == "推荐第7号歌曲《旋律7》")
    verify("被挤出消息可追溯（不计丢失）", len(memory.evicted()) == 10, f"实际 {len(memory.evicted())}")

    section("反面对照：丢掉 system 的朴素窗口")
    naive = memory._history[-4:]  # 朴素做法：直接切最近 N 条
    verify("朴素窗口丢失 system（模型会'忘记身份'）",
           all(m["role"] != "system" for m in naive))
    print("-> 这就是为什么 system 必须单独保存、永驻头部。")

    finish("ch06_sliding_window")


if __name__ == "__main__":
    main()
