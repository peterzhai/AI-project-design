"""ToolBox 2.0：带 JSON Schema 描述、参数校验与结构化错误回传的工具注册表。

相比第 1 章的教学版 ToolBox（ch01_agent_anatomy.py），正规军有三件武器：
1. schemas()：从函数签名 + docstring 自动生成 OpenAI tools 格式描述
   （"单一事实来源"：写一次函数，描述与实现永不脱节）；
2. 参数校验：模型给的参数先过 schema 校验再执行，脏参数到不了真实函数；
3. 结构化错误回传：未知工具/校验失败/执行异常都变成可读观察，Agent 不崩溃。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Optional

from common.validators import validate


def function_to_schema(fn: Callable, description: Optional[str] = None) -> dict:
    """从函数签名 + docstring 生成 OpenAI tools 格式的函数描述。

    规则：
    - 参数必须带类型标注，支持 int -> integer / float -> number /
      str -> string / bool -> boolean；
    - 带默认值的参数视为可选，其余为 required；
    - docstring 第一段作为 description。
    """
    signature = inspect.signature(fn)
    type_map = {int: "integer", float: "number", str: "string", bool: "boolean"}
    properties: dict = {}
    required: list[str] = []
    for param_name, param in signature.parameters.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"函数 {fn.__name__} 的参数「{param_name}」缺少类型标注，无法生成 schema")
        if annotation not in type_map:
            raise TypeError(
                f"函数 {fn.__name__} 的参数「{param_name}」类型 {annotation} 不受支持"
                f"（支持: int/float/str/bool）"
            )
        properties[param_name] = {"type": type_map[annotation]}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    doc = inspect.getdoc(fn) or ""
    first_paragraph = doc.split("\n\n")[0].strip()
    if not description and not first_paragraph:
        raise ValueError(f"函数 {fn.__name__} 缺少 docstring（模型需要描述才能正确调用）")

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description or first_paragraph,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


@dataclass(frozen=True)
class Tool:
    """一个已注册工具：实现 + 自动生成的 schema。"""

    fn: Callable
    schema: dict


class ToolBox:
    """工具注册与执行（生产版）。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ---- 注册 ----
    def register(self, fn: Callable, *, name: Optional[str] = None,
                 description: Optional[str] = None) -> "ToolBox":
        """注册一个带类型标注与 docstring 的函数；返回自身以支持链式注册。"""
        tool_name = name or fn.__name__
        if tool_name in self._tools:
            raise ValueError(f"工具「{tool_name}」重复注册")
        self._tools[tool_name] = Tool(fn=fn, schema=function_to_schema(fn, description))
        return self

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ---- 给 LLM 看的工具清单（OpenAI tools 格式） ----
    def schemas(self) -> list[dict]:
        return [tool.schema for tool in self._tools.values()]

    # ---- 执行：三层防线 ----
    def try_execute(self, name: str, arguments: dict) -> tuple[bool, str]:
        """执行并返回 (是否成功, 观察文本)。供程序化判定（如计划执行器重试）。"""
        # 防线 1：工具存在性
        if name not in self._tools:
            return False, f"错误：未注册的工具「{name}」，可用工具: {self.names}"
        tool = self._tools[name]
        # 防线 2：参数 schema 校验（脏参数到不了真实函数）
        param_errors = validate(arguments, tool.schema["function"]["parameters"])
        if param_errors:
            return (False, "错误：参数校验失败，请修正后重试。\n- "
                    + "\n- ".join(param_errors))
        # 防线 3：执行异常
        try:
            return True, str(tool.fn(**arguments))
        except Exception as exc:
            return False, f"工具「{name}」执行失败: {exc}"

    def execute(self, name: str, arguments: dict) -> str:
        """按名字执行工具，任何失败都返回可读的观察文本（绝不向上抛异常）。"""
        return self.try_execute(name, arguments)[1]

    def get(self, name: str) -> Tool:
        return self._tools[name]
