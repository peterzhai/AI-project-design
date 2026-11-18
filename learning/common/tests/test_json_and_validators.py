"""JSON 提取与 schema 校验器的契约测试。"""

import pytest

from common.json_utils import JsonParseError, extract_json_object, parse_llm_json, strip_code_fence
from common.validators import ValidationError, require_valid, validate


# ---------- strip_code_fence ----------
def test_strip_code_fence_variants():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'  # 无围栏原样返回


# ---------- extract_json_object ----------
def test_extract_json_object_from_mixed_text():
    text = '好的！结果如下：\n{"name": "张三", "age": 30}\n希望对你有帮助。'
    assert extract_json_object(text) == '{"name": "张三", "age": 30}'


def test_extract_json_object_without_braces_returns_original():
    assert extract_json_object("没有花括号的文本") == "没有花括号的文本"


# ---------- parse_llm_json ----------
def test_parse_llm_json_full_pipeline():
    assert parse_llm_json('```json\n{"ok": true}\n```') == {"ok": True}
    assert parse_llm_json('前置说明 {"x": [1, 2]} 后置说明') == {"x": [1, 2]}


def test_parse_llm_json_failure_carries_raw_text():
    with pytest.raises(JsonParseError) as info:
        parse_llm_json('{"name": "张三",}')  # 尾逗号：LLM 常见错误
    assert info.value.raw_text == '{"name": "张三",}'


# ---------- validate：类型矩阵 ----------
def test_validate_type_matrix():
    cases = [
        ({"type": "object"}, {}, True),
        ({"type": "object"}, [], False),
        ({"type": "array"}, [1], True),
        ({"type": "string"}, "hi", True),
        ({"type": "string"}, 1, False),
        ({"type": "boolean"}, True, True),
        ({"type": "boolean"}, 1, False),          # bool 不是 number/integer
        ({"type": "integer"}, 3, True),
        ({"type": "integer"}, 3.5, False),
        ({"type": "integer"}, True, False),       # bool 是 int 子类，必须排除
        ({"type": "number"}, 3.5, True),
        ({"type": "number"}, "3", False),
        ({"type": "null"}, None, True),
    ]
    for schema, instance, ok in cases:
        errors = validate(instance, schema)
        assert (len(errors) == 0) is ok, f"schema={schema}, instance={instance!r}, errors={errors}"


# ---------- validate：对象与必填 ----------
SCHEMA = {
    "type": "object",
    "required": ["product", "urgency"],
    "additionalProperties": False,
    "properties": {
        "product": {"type": "string", "minLength": 2},
        "urgency": {"type": "string", "enum": ["低", "中", "高"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
    },
}


def test_validate_pass_case():
    assert validate({"product": "扫地机器人", "urgency": "高", "score": 9}, SCHEMA) == []


def test_validate_missing_and_extra_fields():
    errors = validate({"urgency": "高"}, SCHEMA)  # 缺 product
    assert any("缺少必填字段「product」" in e for e in errors)
    errors2 = validate({"product": "机器人", "urgency": "高", "extra": 1}, SCHEMA)
    assert any("未定义的字段「extra」" in e for e in errors2)


def test_validate_enum_and_range_and_length():
    errors = validate({"product": "机", "urgency": "超级紧急", "score": 11}, SCHEMA)
    assert any("不在允许的枚举" in e for e in errors)
    assert any("长度少于 2" in e for e in errors)
    assert any("超过上限 10" in e for e in errors)


def test_validate_nested_objects_and_arrays():
    schema = {
        "type": "object",
        "required": ["tags"],
        "properties": {
            "tags": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {"type": "string"},
            },
            "owner": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
        },
    }
    assert validate({"tags": ["a"], "owner": {"name": "李四"}}, schema) == []
    errors = validate({"tags": [], "owner": {}}, schema)
    assert any("数组长度少于 1" in e for e in errors)
    assert any("缺少必填字段「name」" in e and "$.owner" in e for e in errors)  # 路径嵌套
    errors2 = validate({"tags": [1]}, schema)
    assert any("$.tags[0]: 期望类型 string" in e for e in errors2)  # 数组元素路径


def test_require_valid_raises_with_all_errors():
    with pytest.raises(ValidationError) as info:
        require_valid({"product": "机", "urgency": "超级紧急"}, SCHEMA)  # 还缺 score? 不缺，score 非必填
    assert len(info.value.errors) == 2
