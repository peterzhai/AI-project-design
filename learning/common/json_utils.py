"""从 LLM 输出中提取 JSON 的工具集。

LLM 即使被要求"只输出 JSON"，也经常夹带解释文字或 Markdown 代码围栏，
例如：

    好的，以下是抽取结果：
    ```json
    {"name": "张三"}
    ```

本模块的职责：把这类文本稳稳地变成 dict，失败时抛出带上下文的明确异常。
"""

from __future__ import annotations

import json


class JsonParseError(ValueError):
    """无法从 LLM 输出中解析出 JSON。携带原始文本便于回炉重试。"""

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


def strip_code_fence(text: str) -> str:
    """剥掉 Markdown 代码围栏（```json ... ``` 或 ``` ... ```）。

    没有围栏时原样返回；有围栏时返回围栏内部内容。
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # 第一行形如 ```json 或 ```；最后一行是 ```
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
        return stripped.strip("`").strip()
    return stripped


def extract_json_object(text: str) -> str:
    """从混合文本中定位第一个 '{' 到最后一个 '}' 之间的内容（含括号）。

    兼容"解释文字 + JSON + 尾部注释"的输出；不含 '{' 时返回原文（让
    json.loads 抛出更标准的错误）。
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_llm_json(text: str) -> object:
    """把 LLM 输出解析为 Python 对象。

    流程：剥围栏 -> 定位 JSON 对象 -> json.loads。
    失败抛 JsonParseError（携带原文，供修复循环把错误喂回模型）。
    """
    candidate = extract_json_object(strip_code_fence(text))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JsonParseError(
            f"JSON 解析失败（位置 {exc.pos} 附近）: {exc.msg}；候选文本: {candidate[:200]!r}",
            raw_text=text,
        ) from exc
