"""第 2 轮公共层扩展的契约测试：流式、SSE 解析、重试器。"""

import json

import pytest

from common.llm_client import LLMError, OpenAICompatClient, parse_sse_lines
from common.mock_llm import MockLLM, StreamChunk
from common.retry import RetryExhaustedError, is_retryable, with_retries


# ---------- MockLLM.stream ----------
def test_stream_concatenation_equals_chat_content():
    text = "北京今天晴，5~12°C，建议穿件厚外套。"
    scripted_llm = MockLLM(scripted=[text, text])  # 同一脚本分别供 stream 与 chat
    chunks = list(scripted_llm.stream([{"role": "user", "content": "q"}], chunk_size=4))
    joined = "".join(c.content for c in chunks)
    assert joined == text
    assert chunks[-1].finish_reason == "stop"
    assert all(c.finish_reason is None for c in chunks[:-1])
    assert len(chunks) == (len(text) + 3) // 4  # 4 字符一块，最后一块可不足


def test_stream_records_call_snapshot():
    llm = MockLLM(scripted=["ok"])
    list(llm.stream([{"role": "user", "content": "hi"}], temperature=0.3))
    assert llm.call_count == 1
    assert llm.calls[0]["params"] == {"temperature": 0.3}


def test_stream_tool_call_yields_single_empty_chunk():
    llm = MockLLM(scripted=[{"tool_calls": [{"name": "f", "arguments": {}}]}])
    chunks = list(llm.stream([{"role": "user", "content": "q"}]))
    assert len(chunks) == 1
    assert chunks[0] == StreamChunk(content="", finish_reason="tool_calls")


def test_chat_records_passthrough_params():
    llm = MockLLM(scripted=["ok"])
    llm.chat([{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=50)
    assert llm.calls[0]["params"] == {"temperature": 0.1, "max_tokens": 50}


# ---------- parse_sse_lines（纯函数，不联网） ----------
SAMPLE_SSE = """: 这是注释行，应被跳过

data: {"choices": [{"delta": {"content": "北京"}}]}

data: {"choices": [{"delta": {"content": "今天晴"}}]}

data: [DONE]

data: {"choices": [{"delta": {"content": "这段不应出现"}}]}
"""


def test_parse_sse_lines_skips_comments_and_stops_at_done():
    events = parse_sse_lines(SAMPLE_SSE)
    contents = [e["choices"][0]["delta"]["content"] for e in events]
    assert contents == ["北京", "今天晴"]


def test_parse_sse_lines_empty_and_bad_input():
    assert parse_sse_lines("") == []
    with pytest.raises(json.JSONDecodeError):
        parse_sse_lines("data: {broken json}")


# ---------- is_retryable 分类 ----------
def test_is_retryable_classification():
    assert is_retryable(LLMError("conn refused")) is True          # 无状态码=网络级
    assert is_retryable(LLMError("限流", status_code=429)) is True
    assert is_retryable(LLMError("网关坏", status_code=503)) is True
    assert is_retryable(LLMError("密钥错", status_code=401)) is False
    assert is_retryable(LLMError("参数错", status_code=400)) is False
    assert is_retryable(RuntimeError("普通错误")) is True           # 无 status_code 属性


# ---------- with_retries 重试器 ----------
def test_with_retries_succeeds_after_two_failures_with_backoff():
    attempts = []
    sleeps: list[float] = []

    def flaky():
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise LLMError("限流", status_code=429)
        return "成功"

    result = with_retries(flaky, max_retries=3, base_delay=0.5, sleep_fn=sleeps.append)
    assert result == "成功"
    assert len(attempts) == 3
    assert sleeps == [0.5, 1.0]  # 指数退避：0.5 -> 1.0（可注入时钟，无真实等待）


def test_with_retries_exhausted_raises_with_cause():
    def always_fail():
        raise LLMError("一直限流", status_code=429)

    with pytest.raises(RetryExhaustedError, match="4 次尝试") as info:
        with_retries(always_fail, max_retries=3, base_delay=0.5, sleep_fn=lambda s: None)
    assert isinstance(info.value.__cause__, LLMError)


def test_with_retries_non_retryable_raises_immediately():
    sleeps: list[float] = []

    def forbidden():
        raise LLMError("密钥错", status_code=401)

    with pytest.raises(LLMError, match="密钥错"):
        with_retries(forbidden, max_retries=5, base_delay=0.5, sleep_fn=sleeps.append)
    assert sleeps == []  # 不可重试错误：一次都没等待


def test_with_retries_no_failure_calls_once():
    assert with_retries(lambda: 42, max_retries=3, sleep_fn=lambda s: None) == 42
