"""第 4 章测试：schema 生成、参数校验防线、Tool-Calling 协议。"""

import json

import pytest

import ch04_param_validation as validation_demo
import ch04_schema_generation as schema_demo
import ch04_tool_protocol as protocol_demo
from common.mock_llm import MockLLM
from common.toolbox import ToolBox, function_to_schema


# ---------- 示例 01：schema 自动生成 ----------
def test_schema_generation_main_end_to_end():
    schema_demo.main()


def test_generated_schema_equals_handwritten():
    assert function_to_schema(schema_demo.search_flight) == schema_demo.HAND_WRITTEN_SCHEMA


# ---------- 示例 02：参数校验三层防线 ----------
def test_param_validation_main_end_to_end():
    validation_demo.main()


def test_dirty_params_never_execute_function():
    box, audit = validation_demo.build_toolbox()
    box.execute("cancel_order", {"order_id": 12345})  # 类型错
    assert audit == []                                 # 没有真实执行
    box.execute("cancel_order", {"order_id": "A1001"})
    assert audit == ["cancel_order(A1001, 用户取消)"]  # 合法才执行


def test_all_three_defenses_return_observations():
    box, _ = validation_demo.build_toolbox()
    assert "未注册的工具" in box.execute("nope", {})
    assert "参数校验失败" in box.execute("refund", {"amount": "abc"})
    assert "执行失败" in box.execute("refund", {"amount": -1})


# ---------- 示例 03：协议消息组装 ----------
def test_tool_protocol_main_end_to_end():
    protocol_demo.main()


def _protocol_llm():
    return MockLLM(scripted=[
        {"tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        {"content": "最终回答"},
    ])


def test_protocol_message_shapes():
    llm = _protocol_llm()
    box = ToolBox().register(protocol_demo.get_weather)
    answer, messages = protocol_demo.run_protocol(llm, box, "北京天气？")
    assert answer == "最终回答"
    # assistant 消息的 tool_calls 是协议原样结构
    call = messages[1]["tool_calls"][0]
    assert call == {"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}
    assert isinstance(call["function"]["arguments"], str)  # 线协议是 JSON 字符串
    json.loads(call["function"]["arguments"])  # 且必须是合法 JSON
    # tool 回传消息
    assert messages[2]["tool_call_id"] == "call_1"
    assert messages[2]["role"] == "tool"
    # 工具清单两次请求都携带
    assert llm.calls[0]["tools"] == box.schemas()
    assert llm.calls[1]["tools"] == box.schemas()


def test_parallel_tool_calls_have_matching_ids():
    llm = MockLLM(scripted=[
        {"tool_calls": [
            {"name": "get_weather", "arguments": {"city": "北京"}},
            {"name": "get_time", "arguments": {"city": "上海"}},
        ]},
        {"content": "综合回答"},
    ])
    box = ToolBox().register(protocol_demo.get_weather).register(protocol_demo.get_time)
    _, messages = protocol_demo.run_protocol(llm, box, "多任务")
    ids = [c["id"] for c in messages[1]["tool_calls"]]
    tool_ids = [m["tool_call_id"] for m in messages[2:4]]
    assert ids == tool_ids == ["call_1", "call_2"]


def test_no_tool_needed_single_call():
    llm = MockLLM(handler=lambda m, t: {"content": "直接回答"})
    box = ToolBox().register(protocol_demo.get_weather)
    answer, messages = protocol_demo.run_protocol(llm, box, "你好")
    assert answer == "直接回答"
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert llm.call_count == 1
