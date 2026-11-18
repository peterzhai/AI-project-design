"""示例 01：工具描述自动生成 —— 手写 JSON Schema 是苦力活，让代码替你写。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/04_Tool_Calling与函数调用/code/ch04_schema_generation.py

核心思想（单一事实来源）：
工具 = 带类型标注 + docstring 的普通函数。schema 由 common/toolbox.function_to_schema
自动生成 —— 实现改了，描述自动跟上，永不脱节。
"""

import json
import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.toolbox import function_to_schema  # noqa: E402


# ---------------------------------------------------------------
# 工具就是普通函数：类型标注 + docstring 是给模型看的"说明书原料"。
# ---------------------------------------------------------------
def search_flight(origin: str, destination: str, direct_only: bool = False) -> str:
    """搜索两地之间的航班。

    origin 与 destination 为三字码；direct_only 为 True 时只看直飞。
    """
    return f"{origin}->{destination} 共 3 个航班"


def convert_currency(amount: float, rate: float) -> str:
    """按汇率换算金额。"""
    return f"{amount * rate:.2f}"


# ---------------------------------------------------------------
# 对照：同一个工具，手写 schema（传统做法）vs 自动生成（本章做法）。
# ---------------------------------------------------------------
HAND_WRITTEN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_flight",
        "description": "搜索两地之间的航班。",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "direct_only": {"type": "boolean"},
            },
            "required": ["origin", "destination"],
        },
    },
}


def main() -> None:
    banner("示例 01：工具描述自动生成")

    section("实验 1：手写 schema（苦力活，还容易和实现脱节）")
    print(json.dumps(HAND_WRITTEN_SCHEMA, ensure_ascii=False, indent=2))
    verify("手写 schema 是 OpenAI tools 标准形状",
           HAND_WRITTEN_SCHEMA["function"]["parameters"]["required"] == ["origin", "destination"])

    section("实验 2：自动生成 —— 一行搞定，且与手写版逐字段一致")
    generated = function_to_schema(search_flight)
    print(json.dumps(generated, ensure_ascii=False, indent=2))
    verify("自动生成的 schema 与手写版完全一致", generated == HAND_WRITTEN_SCHEMA)

    section("实验 3：类型映射与必填规则")
    schema2 = function_to_schema(convert_currency)
    params = schema2["function"]["parameters"]
    print(f"convert_currency 参数映射: {params['properties']}，必填: {params['required']}")
    verify("float -> number", params["properties"]["amount"] == {"type": "number"})
    verify("两个参数都无默认值 -> 都必填", params["required"] == ["amount", "rate"])
    verify("docstring 第一段成为描述", schema2["function"]["description"] == "按汇率换算金额。")

    section("实验 4：纪律检查 —— 缺类型标注 / 缺 docstring 直接拒绝注册")
    def bad_no_annotation(x):
        return x

    def bad_no_doc(x: str):
        return x

    for fn, why in ((bad_no_annotation, "缺类型标注"), (bad_no_doc, "缺 docstring")):
        try:
            function_to_schema(fn)
            raise AssertionError(f"实验 4 失败：{why} 的函数应当被拒绝")
        except (TypeError, ValueError) as exc:
            print(f"  {why} -> 按预期拒绝: {exc}")
    verify("两种劣质工具都被前置拦截", True)

    finish("ch04_schema_generation")


if __name__ == "__main__":
    main()
