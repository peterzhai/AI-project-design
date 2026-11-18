"""迷你 JSON Schema 校验器（纯标准库，零依赖）。

为什么不用 pydantic/jsonschema？—— 教程依赖最小化红线。本实现覆盖
结构化输出最常用的关键字子集，足够支撑全教程：

    type / required / properties / enum / items
    minLength / maxLength / minimum / maximum / minItems / maxItems
    additionalProperties（布尔）

错误消息格式统一为 "$.路径: 说明"，可直接拼进提示词喂回模型修复。
"""

from __future__ import annotations

from typing import Any


class ValidationError(AssertionError):
    """schema 校验失败。errors 为人类可读错误列表。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    # 注意 bool 是 int 的子类，integer/number 校验必须显式排除 bool。
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


def validate(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """校验 instance 是否满足 schema，返回错误消息列表（空列表 = 通过）。"""
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None:
        check = _TYPE_CHECKS.get(expected_type)
        if check is None:
            errors.append(f"{path}: 未知的 schema 类型「{expected_type}」")
            return errors  # 类型未知无法继续下钻
        if not check(instance):
            errors.append(f"{path}: 期望类型 {expected_type}，实际为 {type(instance).__name__}")
            return errors  # 类型不符，下钻无意义

    # ---- 对象类关键字 ----
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: 缺少必填字段「{key}」")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                errors.extend(validate(value, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: 出现 schema 未定义的字段「{key}」")
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errors.append(f"{path}: 字段数少于 {schema['minProperties']}")

    # ---- 数组类关键字 ----
    elif isinstance(instance, list):
        items_schema = schema.get("items")
        if items_schema is not None:
            for i, item in enumerate(instance):
                errors.extend(validate(item, items_schema, f"{path}[{i}]"))
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: 数组长度少于 {schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: 数组长度超过 {schema['maxItems']}")

    # ---- 字符串类关键字 ----
    elif isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: 长度少于 {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: 长度超过 {schema['maxLength']}")

    # ---- 数值类关键字 ----
    elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: 小于下限 {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: 超过上限 {schema['maximum']}")

    # ---- 枚举（适用于所有类型）----
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: 值 {instance!r} 不在允许的枚举 {schema['enum']} 中")

    return errors


def require_valid(instance: Any, schema: dict) -> None:
    """校验失败时抛 ValidationError（错误列表可直接喂回模型修复）。"""
    errors = validate(instance, schema)
    if errors:
        raise ValidationError(errors)
