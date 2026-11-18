"""第 2 章测试：消息角色/参数、错误重试、流式与 SSE。"""

import pytest

import ch02_errors_retry as errors_demo
import ch02_messages_and_params as params_demo
import ch02_streaming as stream_demo
from common.llm_client import LLMError, parse_sse_lines
from common.mock_llm import MockLLM
from common.retry import RetryExhaustedError, with_retries


# ---------- 示例 01：消息角色与参数 ----------
def test_messages_main_end_to_end():
    params_demo.main()


def test_system_prompt_changes_behavior():
    llm = params_demo.make_translator_llm()
    french = llm.chat([
        {"role": "system", "content": "你是一个法语翻译，任何输入都用法语回复。"},
        {"role": "user", "content": "你好"},
    ])
    plain = llm.chat([{"role": "user", "content": "你好"}])
    assert french.content.startswith("Bonjour")
    assert plain.content.startswith("（原样")


def test_multi_turn_context_grows():
    llm = params_demo.make_translator_llm()
    conversation = [{"role": "system", "content": "你是一个法语翻译，任何输入都用法语回复。"},
                    {"role": "user", "content": "你好"}]
    reply1 = llm.chat(conversation)
    conversation.append({"role": "assistant", "content": reply1.content})
    conversation.append({"role": "user", "content": "谢谢"})
    llm.chat(conversation)
    messages = llm.calls[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "Bonjour" in messages[2]["content"]  # 历史回答完整保留


# ---------- 示例 02：错误分类与重试 ----------
def test_retry_main_end_to_end():
    errors_demo.main()


def _flaky(fail_times, status_code):
    return errors_demo.FlakyClient(
        fail_times=fail_times, status_code=status_code,
        fallback=MockLLM(scripted=["ok"]),
    )


def test_flaky_then_retry_success_with_backoff():
    flaky = _flaky(fail_times=2, status_code=429)
    sleeps: list[float] = []
    answer = with_retries(
        lambda: flaky.chat([{"role": "user", "content": "q"}]),
        max_retries=3, base_delay=0.5, sleep_fn=sleeps.append,
    )
    assert answer.content == "ok"
    assert flaky.calls == 3
    assert sleeps == [0.5, 1.0]


def test_exhausted_budget_raises_after_max_attempts():
    flaky = _flaky(fail_times=99, status_code=429)
    with pytest.raises(RetryExhaustedError):
        with_retries(lambda: flaky.chat([]), max_retries=2, base_delay=0.25, sleep_fn=lambda s: None)
    assert flaky.calls == 3  # 1 + max_retries


def test_auth_error_not_retried():
    flaky = _flaky(fail_times=99, status_code=401)
    with pytest.raises(LLMError, match="401"):
        with_retries(lambda: flaky.chat([]), max_retries=5, sleep_fn=lambda s: None)
    assert flaky.calls == 1


# ---------- 示例 03：流式与 SSE ----------
def test_streaming_main_end_to_end():
    stream_demo.main()


def test_stream_concat_equals_chat_with_same_handler_llm():
    llm = MockLLM(handler=lambda m, t: {"content": "abcdefg"})
    joined = "".join(c.content for c in llm.stream([{"role": "user", "content": "q"}], chunk_size=2))
    full = llm.chat([{"role": "user", "content": "q"}])
    assert joined == full.content == "abcdefg"
    assert len(joined) == 7 and len(joined) // 2 + 1 >= 4


def test_sample_sse_in_ch02_parses_correctly():
    events = parse_sse_lines(stream_demo.SAMPLE_SSE)
    deltas = [e["choices"][0]["delta"]["content"] for e in events]
    assert deltas == ["北京", "今天", "晴，5~12°C"]
