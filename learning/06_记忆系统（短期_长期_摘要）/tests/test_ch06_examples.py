"""第 6 章测试：滑动窗口、长期记忆、摘要压缩。"""

import pytest

import ch06_long_term_memory as ltm_demo
import ch06_sliding_window as window_demo
import ch06_summarizing_memory as summary_demo
from ch06_sliding_window import ContextWindow, approx_tokens
from ch06_summarizing_memory import SummarizingMemory, make_summarizer_llm
from common.mock_llm import MockLLM


# ---------- 示例 01：滑动窗口 ----------
def test_sliding_window_main_end_to_end():
    window_demo.main()


def test_window_keeps_system_and_latest():
    memory = ContextWindow(max_messages=4)
    memory.set_system("你是助手A。")
    for i in range(3):
        memory.add({"role": "user", "content": f"q{i}"})
        memory.add({"role": "assistant", "content": f"a{i}"})
    context = memory.render()
    assert [m["role"] for m in context] == ["system", "user", "assistant", "user", "assistant"]
    assert context[1]["content"] == "q1"     # q0/a0 被挤出
    assert context[-1]["content"] == "a2"


def test_system_message_via_add_is_redirected():
    memory = ContextWindow(max_messages=2)
    memory.add({"role": "system", "content": "通过add设置的system"})
    memory.add({"role": "user", "content": "q1"})
    memory.add({"role": "assistant", "content": "a1"})
    memory.add({"role": "user", "content": "q2"})
    context = memory.render()
    assert context[0]["content"] == "通过add设置的system"  # system 永驻
    assert len(context) == 3


def test_evicted_messages_are_recoverable():
    memory = ContextWindow(max_messages=2)
    memory.add({"role": "user", "content": "早期重要信息"})
    memory.add({"role": "assistant", "content": "ok"})
    memory.add({"role": "user", "content": "新问题"})
    assert memory.evicted()[0]["content"] == "早期重要信息"


def test_approx_tokens_counts_chars():
    messages = [{"role": "user", "content": "abcd"}, {"role": "assistant", "content": "ef"}]
    assert approx_tokens(messages) == 6


# ---------- 示例 02：长期记忆 ----------
def test_long_term_memory_main_end_to_end():
    ltm_demo.main()


def test_same_key_overwrites_keeps_single_item():
    memory = ltm_demo.LongTermMemory()
    memory.remember("地址", "朝阳区")
    memory.remember("地址", "海淀区")
    assert len([i for i in memory.items if i.key == "地址"]) == 1
    assert memory.items[0].content == "海淀区"


def test_search_ranks_most_relevant_first():
    memory = ltm_demo.LongTermMemory()
    memory.remember("饮食禁忌", "用户对花生过敏，所有推荐必须避开花生")
    memory.remember("口味偏好", "用户喜欢清淡口味，讨厌香菜")
    memory.remember("完全无关", "xyz123")
    hits = memory.search("推荐晚餐菜谱")
    assert hits[0].key == "饮食禁忌"
    assert all(h.key != "完全无关" for h in hits)   # 零重合的不返回


def test_recall_block_injectable_as_system_text():
    memory = ltm_demo.LongTermMemory()
    memory.remember("饮食禁忌", "用户对花生过敏")
    block = memory.recall_for("推荐含花生的菜")  # query 需与条目有字符重合才会命中
    assert block.startswith("以下是关于该用户的已知记忆")
    assert "花生" in block
    empty = ltm_demo.LongTermMemory().recall_for("任何问题")
    assert empty == ""  # 无记忆/零重合时空串（不注入）


# ---------- 示例 03：摘要压缩 ----------
def test_summarizing_memory_main_end_to_end():
    summary_demo.main()


def test_summarizer_compression_saves_tokens():
    summarizer = make_summarizer_llm()
    memory = SummarizingMemory(llm=summarizer, max_recent=4, system="你是音乐推荐助手。")
    conversation = summary_demo.build_long_conversation()
    for m in conversation:
        memory.add(m)
    context = memory.render()
    full = [{"role": "system", "content": "你是音乐推荐助手。"}, *conversation]
    assert approx_tokens(context) < approx_tokens(full) * 0.6
    assert memory.compress_count == 8
    assert "旋律1" in context[0]["content"]  # 早期信息进入摘要


def test_buffer_never_exceeds_window():
    summarizer = make_summarizer_llm()
    memory = SummarizingMemory(llm=summarizer, max_recent=4, system="s")
    for i in range(30):
        memory.add({"role": "user", "content": f"消息{i}，内容足够长以模拟真实对话"})
    assert len(memory._buffer) <= 4
    assert len(memory.render()) == 5  # system + 4


def test_summary_accumulates_in_order():
    summarizer = make_summarizer_llm()
    memory = SummarizingMemory(llm=summarizer, max_recent=2, system="s")
    for m in [{"role": "user", "content": "第一条AAA"},
              {"role": "assistant", "content": "回复A"},
              {"role": "user", "content": "第二条BBB"},
              {"role": "assistant", "content": "回复B"},
              {"role": "user", "content": "第三条CCC"},
              {"role": "assistant", "content": "回复C"},
              {"role": "user", "content": "第四条DDD"},
              {"role": "assistant", "content": "回复D"}]:
        memory.add(m)
    # 早期消息按时间顺序累积进摘要；最近两条仍在原文窗口（6 条被压缩 = 3 批）
    assert memory.compress_count == 3
    assert memory.summary.index("第一条AAA") < memory.summary.index("第二条BBB")
    assert "第四条DDD" not in memory.summary
    assert memory.render()[1]["content"] == "第四条DDD"
