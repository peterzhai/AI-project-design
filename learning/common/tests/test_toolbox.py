"""ToolBox 2.0 契约测试：schema 自动生成、三层防线。"""

import pytest

from common.toolbox import ToolBox, function_to_schema


def get_weather(city: str, detail: bool = False) -> str:
    """查询城市天气。

    detail 为 True 时返回更多信息。
    """
    return f"{city}晴"


# ---------- function_to_schema ----------
def test_schema_from_signature_and_docstring():
    schema = function_to_schema(get_weather)
    fn = schema["function"]
    assert schema["type"] == "function"
    assert fn["name"] == "get_weather"
    assert fn["description"] == "查询城市天气。"  # docstring 第一段
    assert fn["parameters"]["properties"] == {"city": {"type": "string"}, "detail": {"type": "boolean"}}
    assert fn["parameters"]["required"] == ["city"]  # 带默认值的参数非必填


def test_schema_matches_handwritten():
    handwritten = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "detail": {"type": "boolean"}},
                "required": ["city"],
            },
        },
    }
    assert function_to_schema(get_weather) == handwritten  # 自动生成 == 手写


def test_type_mapping_matrix():
    def f(a: int, b: float, c: str, d: bool) -> str:
        """四个基本类型。"""
        return "ok"

    props = function_to_schema(f)["function"]["parameters"]["properties"]
    assert props == {
        "a": {"type": "integer"},
        "b": {"type": "number"},
        "c": {"type": "string"},
        "d": {"type": "boolean"},
    }


def test_untyped_or_unsupported_params_rejected():
    def no_annotation(x):
        """无类型标注。"""
        return "ok"

    def unsupported(x: list[int]):
        """不支持的类型。"""
        return "ok"

    with pytest.raises(TypeError, match="缺少类型标注"):
        function_to_schema(no_annotation)
    with pytest.raises(TypeError, match="不受支持"):
        function_to_schema(unsupported)


def test_missing_docstring_rejected():
    def secret(x: str):
        return "ok"

    with pytest.raises(ValueError, match="docstring"):
        function_to_schema(secret)


# ---------- ToolBox 三层防线 ----------
def build_box() -> tuple[ToolBox, list[int]]:
    calls: list[int] = []  # 副作用计数器：证明校验失败时函数没被执行

    def add(a: int, b: int) -> str:
        """计算两数之和。"""
        calls.append(1)
        return f"{a + b}"

    box = ToolBox()
    box.register(add)
    return box, calls


def test_defense_1_unknown_tool():
    box, calls = build_box()
    observation = box.execute("send_email", {})
    assert "未注册的工具" in observation
    assert calls == []  # 函数没被执行


def test_defense_2_invalid_params_never_reach_function():
    box, calls = build_box()
    observation = box.execute("add", {"a": "不是数字", "b": 1})
    assert "参数校验失败" in observation
    assert "$.a: 期望类型 integer" in observation
    assert calls == []  # 关键：脏参数没有打到真实函数

    observation2 = box.execute("add", {"b": 1})  # 缺必填
    assert "缺少必填字段「a」" in observation2
    assert calls == []


def test_defense_3_runtime_exception_is_observation():
    box, calls = build_box()
    calls.clear()  # 校验通过的调用才会执行

    def boom(x: int) -> str:
        """故意抛错。"""
        raise ValueError("内部爆炸")

    box.register(boom)
    observation = box.execute("boom", {"x": 1})
    assert "执行失败" in observation and "内部爆炸" in observation


def test_happy_path_and_chained_registration():
    box, calls = build_box()
    result = box.execute("add", {"a": 3, "b": 5})
    assert result == "8"
    assert calls == [1]
    assert box.names == ["add"]


def test_duplicate_registration_rejected():
    box, _ = build_box()
    def add_again(a: int, b: int) -> str:
        """重复名字。"""
        return "0"

    with pytest.raises(ValueError, match="重复注册"):
        box.register(add_again, name="add")


def test_schemas_passed_to_llm_shape():
    box, _ = build_box()
    schemas = box.schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["parameters"]["required"] == ["a", "b"]
