"""第 3 章测试：提示词模式、JSON 提取与修复闭环、schema 校验闭环。"""

import pytest

import ch03_json_extraction as json_demo
import ch03_prompt_patterns as prompt_demo
import ch03_schema_validation as schema_demo
from common.json_utils import JsonParseError, parse_llm_json
from common.mock_llm import MockLLM
from common.validators import ValidationError, validate


# ---------- 示例 01：提示词四板斧 ----------
def test_prompt_patterns_main_end_to_end():
    prompt_demo.main()


def test_structured_prompt_triggers_json_output():
    llm = prompt_demo.make_extraction_llm()
    structured = llm.chat([
        {"role": "system", "content": prompt_demo.STRUCTURED_PROMPT},
        {"role": "user", "content": f"<message>{prompt_demo.CUSTOMER_MESSAGE}</message>"},
    ])
    data = parse_llm_json(structured.content)
    assert data == {"product": "扫地机器人", "issue_type": "卡住/障碍", "urgency": "高"}


def test_naive_prompt_yields_prose():
    llm = prompt_demo.make_extraction_llm()
    prose = llm.chat([
        {"role": "system", "content": prompt_demo.NAIVE_PROMPT},
        {"role": "user", "content": prompt_demo.CUSTOMER_MESSAGE},
    ])
    with pytest.raises(JsonParseError):
        parse_llm_json(prose.content)


# ---------- 示例 02：JSON 提取与修复循环 ----------
def test_json_extraction_main_end_to_end():
    json_demo.main()


def test_ask_for_json_repairs_after_bad_output():
    llm = MockLLM(scripted=[
        '{"a": 1,}',          # 尾逗号
        '{"a": 1, "b": "ok"}',  # 修复
    ])
    result = json_demo.ask_for_json(llm, "只输出JSON", "任意", max_attempts=3)
    assert result == {"a": 1, "b": "ok"}
    assert llm.call_count == 2
    feedback = llm.calls[1]["messages"][-1]["content"]
    assert "无法解析" in feedback      # 错误信息被喂回
    assert '{"a": 1,}' in llm.calls[1]["messages"][2]["content"]  # 坏输出也在上下文里


def test_ask_for_json_exhausts_budget():
    llm = MockLLM(handler=lambda m, t: {"content": "不会输出JSON"})
    with pytest.raises(JsonParseError, match="重试 2 次后仍无法解析"):
        json_demo.ask_for_json(llm, "任务", "输入", max_attempts=2)
    assert llm.call_count == 2


# ---------- 示例 03：schema 校验闭环 ----------
def test_schema_validation_main_end_to_end():
    schema_demo.main()


def test_ask_for_valid_json_repairs_enum_violation():
    llm = MockLLM(scripted=[
        '{"product": "音箱", "issue_type": "没声音", "urgency": "爆表"}',
        '{"product": "音箱", "issue_type": "没声音", "urgency": "中"}',
    ])
    ticket = schema_demo.ask_for_valid_json(llm, "任务", "输入", schema_demo.TICKET_SCHEMA)
    assert ticket["urgency"] == "中"
    assert llm.call_count == 2
    assert "枚举" in llm.calls[1]["messages"][-1]["content"]


def test_ask_for_valid_json_syntax_then_semantics_then_success():
    llm = MockLLM(scripted=[
        '{"a": 1,}',                                       # 语法错
        '{"product": "灯", "urgency": "高"}',               # 语义错（缺 issue_type）
        '{"product": "台灯", "issue_type": "不亮", "urgency": "高"}',  # 通过
    ])
    ticket = schema_demo.ask_for_valid_json(llm, "任务", "输入", schema_demo.TICKET_SCHEMA)
    assert ticket == {"product": "台灯", "issue_type": "不亮", "urgency": "高"}
    assert llm.call_count == 3


def test_ask_for_valid_json_raises_validation_error_when_hopeless():
    llm = MockLLM(handler=lambda m, t: {"content": '{"urgency": "爆表"}'})  # 永远缺字段
    with pytest.raises(ValidationError):
        schema_demo.ask_for_valid_json(llm, "任务", "输入", schema_demo.TICKET_SCHEMA, max_attempts=2)


def test_ticket_schema_rejects_semantic_violations():
    errors = validate({"product": "扫地机器人", "issue_type": "卡住", "urgency": "超级紧急"},
                      schema_demo.TICKET_SCHEMA)
    assert any("$.urgency" in e for e in errors)
