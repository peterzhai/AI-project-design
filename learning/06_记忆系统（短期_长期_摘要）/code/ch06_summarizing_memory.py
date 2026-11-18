"""示例 03：摘要压缩 —— 窗口外的旧对话，压缩成一段摘要再回注上下文。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/06_记忆系统（短期_长期_摘要）/code/ch06_summarizing_memory.py

三层记忆架构（本章最终形态）：
    system（身份）+ 历史摘要（压缩记忆）+ 最近 N 条（滑动窗口原文）
本示例实现 SummarizingMemory：缓冲区超出窗口时，把最老的几条交给 LLM 压缩，
摘要累积保存、随上下文注入 —— 用几次便宜的摘要调用，换回窗口外的关键信息。
对比实验采用同一口径（都含 system），长对话下收益一目了然。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from ch06_sliding_window import approx_tokens  # noqa: E402
from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402


class SummarizingMemory:
    """system 永驻 + 摘要累积 + 最近 max_recent 条原文。"""

    def __init__(self, llm, max_recent: int = 4, system: str = ""):
        self.llm = llm
        self.max_recent = max_recent
        self.system = system
        self.summary: str = ""          # 累积的历史摘要（多批按时间顺序拼接）
        self._buffer: list[dict] = []   # 尚未压缩的近期消息
        self.compress_count = 0         # 统计：摘要调用了多少次

    def add(self, message: dict) -> None:
        if message.get("role") == "system":
            self.system = str(message.get("content", ""))
            return
        self._buffer.append(dict(message))
        while len(self._buffer) > self.max_recent:
            self._consolidate_oldest()

    def _consolidate_oldest(self) -> None:
        """把最老的 2 条（仅剩 1 条超窗时压 1 条）交给 LLM 压缩，摘要累积。"""
        batch = self._buffer[:2] if len(self._buffer) - self.max_recent >= 1 else self._buffer[:1]
        self._buffer = self._buffer[len(batch):]
        prompt = (
            "把以下对话压缩成一段简洁摘要，保留关键事实（作品名/决定/约束），不要客套话：\n"
            + "\n".join(f"[{m['role']}] {m['content']}" for m in batch)
        )
        response = self.llm.chat([{"role": "user", "content": prompt}])
        self.compress_count += 1
        # 摘要累积：旧摘要在前，新摘要在后（时间顺序）
        self.summary = (self.summary + " " + response.content).strip() if self.summary else response.content

    def render(self) -> list[dict]:
        """输出压缩后的上下文：system（含摘要段）+ 最近原文。"""
        system_content = self.system
        if self.summary:
            system_content = (system_content + "\n\n【历史对话摘要】\n" + self.summary).strip()
        messages: list[dict] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.extend(self._buffer)
        return messages


def make_summarizer_llm() -> MockLLM:
    """假摘要模型：从压缩请求里提取每条消息的前 10 字符作为要点（确定性）。"""

    def handler(messages, tools):
        prompt = messages[0]["content"]
        lines = [line for line in prompt.splitlines() if line.startswith("[")]
        points = []
        for line in lines:
            text = line.split("] ", 1)[1]
            points.append(text[:10] + ("…" if len(text) > 10 else ""))
        return {"content": "；".join(points)}

    return MockLLM(handler=handler)


def build_long_conversation() -> list[dict]:
    """10 轮较长对话（20 条消息），模拟真实的音乐推荐历史。"""
    conversation: list[dict] = []
    for i in range(1, 11):
        conversation.append({"role": "user",
                             "content": f"我想听第{i}类风格的歌曲，最好旋律舒缓，适合晚上下班后放松心情"})
        conversation.append({"role": "assistant",
                             "content": f"为你推荐《旋律{i}》，非常契合第{i}类风格，希望你喜欢"})
    return conversation


def main() -> None:
    banner("示例 03：摘要压缩记忆")

    summarizer = make_summarizer_llm()
    memory = SummarizingMemory(llm=summarizer, max_recent=4, system="你是音乐推荐助手。")
    conversation = build_long_conversation()
    for m in conversation:
        memory.add(m)

    section("压缩效果（同口径对比：两侧都含 system）")
    context = memory.render()
    # 公平口径：完整方案 = 纯身份 system + 全部历史；压缩方案 = system(含摘要) + 最近 4 条
    full_context = [{"role": "system", "content": "你是音乐推荐助手。"}, *conversation]
    full_tokens = approx_tokens(full_context)
    compressed_tokens = approx_tokens(context)
    print(f"完整方案: {len(full_context)} 条 ≈ {full_tokens} 伪token")
    print(f"压缩方案: {len(context)} 条 ≈ {compressed_tokens} 伪token（摘要调用 {memory.compress_count} 次）")
    print("\n压缩后的 system 内容（前 220 字符）:")
    print("  " + context[0]["content"][:220] + "…")

    verify("system 含身份 + 累积摘要段", "音乐推荐助手" in context[0]["content"] and "【历史对话摘要】" in context[0]["content"])
    verify("摘要保留了早期作品名（旋律1/旋律2）",
           "旋律1" in context[0]["content"] and "旋律2" in context[0]["content"])
    verify("最近 4 条原文逐字保留",
           [m["content"] for m in context[1:]] == [m["content"] for m in conversation[-4:]])
    verify("长对话下压缩显著更省", compressed_tokens < full_tokens * 0.6,
           f"{compressed_tokens} vs {full_tokens}")
    verify("每 2 条消息一次摘要（16 条被压缩 -> 8 次）",
           memory.compress_count == 8, f"实际 {memory.compress_count}")

    section("摘要确实进入了模型的下一次调用")
    chat_llm = MockLLM(scripted=[{"content": "结合你听过的旋律1到旋律8，第11类推荐《旋律11》。"}])
    chat_llm.chat(memory.render())
    verify("模型上下文包含摘要（能'记得'早期对话）",
           "旋律1" in chat_llm.calls[0]["messages"][0]["content"])

    finish("ch06_summarizing_memory")


if __name__ == "__main__":
    main()
