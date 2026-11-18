"""公共层基础设施的测试：MockLLM 与统一客户端的行为契约。

这一层是全教程的地基 —— 它的行为必须先被钉死，后续章节才可信。
"""

import json

import pytest

from common import llm_client, mock_llm
from common.check import check_all
from common.llm_client import LLMError, OpenAICompatClient
from common.mock_llm import ChatResponse, MockLLM, MockLLMError, ToolCall, messages_text


# ---------- MockLLM：脚本模式 ----------
def test_scripted_content_replay_in_order():
    llm = MockLLM(scripted=["第一条", "第二条"])
    r1 = llm.chat([{"role": "user", "content": "hi"}])
    r2 = llm.chat([{"role": "user", "content": "again"}])
    check_all("脚本按顺序回放", [
        ("r1.content", r1.content == "第一条"),
        ("r2.content", r2.content == "第二条"),
        ("finish_reason=stop", r1.finish_reason == "stop"),
        ("调用次数=2", llm.call_count == 2),
    ])


def test_scripted_tool_calls_and_finish_reason():
    llm = MockLLM(scripted=[
        {"tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
    ])
    resp = llm.chat([{"role": "user", "content": "天气如何"}])
    check_all("工具调用响应解析", [
        ("有1个tool_call", len(resp.tool_calls) == 1),
        ("name", resp.tool_calls[0].name == "get_weather"),
        ("arguments", resp.tool_calls[0].arguments == {"city": "北京"}),
        ("finish_reason=tool_calls", resp.finish_reason == "tool_calls"),
        ("has_tool_calls", resp.has_tool_calls is True),
    ])


def test_script_exhausted_raises_with_clear_message():
    llm = MockLLM(scripted=["唯一一条"])
    llm.chat([{"role": "user", "content": "1"}])
    with pytest.raises(MockLLMError, match="脚本耗尽"):
        llm.chat([{"role": "user", "content": "2"}])


def test_plain_string_script_is_wrapped():
    llm = MockLLM(scripted=["纯文本"])
    resp = llm.chat([{"role": "user", "content": "x"}])
    assert resp == ChatResponse(content="纯文本", tool_calls=(), finish_reason="stop", model="mock")


def test_scripted_and_handler_are_mutually_exclusive():
    with pytest.raises(MockLLMError, match="二选一"):
        MockLLM(scripted=["a"], handler=lambda m, t: {})
    with pytest.raises(MockLLMError, match="至少需要"):
        MockLLM()


# ---------- MockLLM：handler 模式 ----------
def test_handler_routes_by_last_user_text():
    def handler(messages, tools):
        if "诗" in messages[-1]["content"]:
            return {"content": "床前明月光"}
        return {"tool_calls": [{"name": "search", "arguments": {"q": "天气"}}]}

    llm = MockLLM(handler=handler)
    poem = llm.chat([{"role": "user", "content": "写首诗"}])
    tool = llm.chat([{"role": "user", "content": "今天天气怎么样"}])
    check_all("handler 按输入路由", [
        ("诗->文本", poem.content == "床前明月光" and not poem.has_tool_calls),
        ("天气->工具", tool.tool_calls[0].name == "search"),
    ])


def test_calls_snapshot_records_history_and_is_isolated():
    llm = MockLLM(handler=lambda m, t: {"content": "ok"})
    messages = [{"role": "user", "content": "v1"}]
    llm.chat(messages)
    messages[0]["content"] = "被改掉了吗"  # 原地修改不应影响已记录的快照
    llm.chat([{"role": "user", "content": "v2"}])
    check_all("快照隔离", [
        ("第1次快照保持v1", llm.calls[0]["messages"][0]["content"] == "v1"),
        ("第2次快照是v2", llm.calls[1]["messages"][0]["content"] == "v2"),
        ("last_user_text", llm.last_user_text() == "[user] v2"),
    ])


def test_reset_script_reuses_llm():
    llm = MockLLM(scripted=["第一次"])
    llm.chat([{"role": "user", "content": "1"}])
    llm.reset_script(["第二次"])
    resp = llm.chat([{"role": "user", "content": "2"}])
    assert resp.content == "第二次"


# ---------- 消息渲染 ----------
def test_messages_text_renders_all_roles():
    text = messages_text([
        {"role": "user", "content": "北京天气"},
        {"role": "assistant", "tool_calls": [ToolCall(name="get_weather", arguments={"city": "北京"})]},
        {"role": "tool", "name": "get_weather", "content": "晴"},
    ])
    check_all("渲染包含全部角色", [
        ("user", "[user] 北京天气" in text),
        ("assistant工具调用", "调用工具: get_weather" in text),
        ("tool结果", "get_weather] 晴" in text),
    ])


# ---------- 统一客户端工厂 ----------
def test_factory_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("USE_REAL_LLM", raising=False)
    client = llm_client.make_client()
    assert isinstance(client, MockLLM)


def test_factory_returns_injected_mock(monkeypatch):
    monkeypatch.delenv("USE_REAL_LLM", raising=False)
    mine = MockLLM(scripted=["注入的"])
    assert llm_client.make_client(mine) is mine


def test_real_mode_requires_api_key(monkeypatch):
    monkeypatch.setenv("USE_REAL_LLM", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        llm_client.make_client()


# ---------- 真实客户端的协议解析（不联网，只测解析） ----------
OPENAI_STYLE_PAYLOAD = {
    "model": "gpt-4o-mini",
    "choices": [{
        "finish_reason": "tool_calls",
        "message": {
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "北京"})},
            }],
        },
    }],
}


def test_parse_openai_style_response():
    client = OpenAICompatClient(api_key="dummy")
    resp = client._parse(OPENAI_STYLE_PAYLOAD)
    check_all("OpenAI 格式解析", [
        ("tool_calls数量", len(resp.tool_calls) == 1),
        ("name", resp.tool_calls[0].name == "get_weather"),
        ("arguments是dict", resp.tool_calls[0].arguments == {"city": "北京"}),
        ("finish_reason", resp.finish_reason == "tool_calls"),
        ("model", resp.model == "gpt-4o-mini"),
    ])


def test_parse_invalid_tool_json_raises():
    client = OpenAICompatClient(api_key="dummy")
    bad = json.loads(json.dumps(OPENAI_STYLE_PAYLOAD))  # 深拷贝后篡改
    bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "这不是JSON"
    with pytest.raises(LLMError, match="合法 JSON"):
        client._parse(bad)


def test_parse_empty_choices_raises():
    client = OpenAICompatClient(api_key="dummy")
    with pytest.raises(LLMError, match="choices"):
        client._parse({"choices": []})
