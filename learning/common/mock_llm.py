"""MockLLM：离线假 LLM，整个教程的地基。

设计目标：
1. 完全离线：不联网、不需要 API Key，响应完全确定，便于测试与复现。
2. 两种用法：
   - scripted（脚本模式）：按顺序回放预设响应，最常见。
   - handler（函数模式）：传入 (messages, tools) -> dict 的函数，按输入动态决定输出。
3. 可断言：calls 属性记录每次调用的完整输入快照（含透传参数），供测试检查。
4. 流式支持：stream() 把响应文本切片逐块吐出，拼接结果与 chat().content 逐字一致。

响应格式（dict 或直接给字符串）：
    {"content": "回答文本"}
    {"tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]}
    字符串会被自动包装成 {"content": 字符串}。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Sequence

# handler 的类型签名：接收 (messages, tools)，返回一个响应 dict。
Handler = Callable[[list[dict], Optional[list[dict]]], dict]


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用请求（等价于 OpenAI 返回的 tool_calls，此处已解析为简单结构）。"""

    name: str            # 要调用的工具（函数）名
    arguments: dict      # 调用参数，已是 dict（不是 JSON 字符串）
    id: str = ""         # 调用 ID（协议回传 tool_call_id 时必需；Mock 会自动补齐）

    def __str__(self) -> str:
        return f"{self.name}({json.dumps(self.arguments, ensure_ascii=False)})"


@dataclass(frozen=True)
class ChatResponse:
    """统一的一次对话返回结果。真实客户端与 Mock 客户端都返回这个结构。"""

    content: str = ""                          # 模型输出的文本（可能为空，比如纯工具调用时）
    tool_calls: tuple[ToolCall, ...] = ()      # 模型请求的工具调用列表
    finish_reason: str = "stop"                # "stop"=最终回答 / "tool_calls"=要调工具
    model: str = "mock"                        # 产生该响应的模型名（仅用于展示）

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass(frozen=True)
class StreamChunk:
    """流式输出的一个增量片段。只有最后一个 chunk 的 finish_reason 非 None。"""

    content: str                              # 本次新增的文本增量
    finish_reason: Optional[str] = None       # 最后一块为 "stop"；中间块为 None


class MockLLMError(RuntimeError):
    """MockLLM 自身的配置错误（例如脚本耗尽）。"""


def _coalesce(spec) -> dict:
    """把响应规格统一成 dict：字符串 -> {"content": ...}。"""
    if isinstance(spec, str):
        return {"content": spec}
    if not isinstance(spec, dict):
        raise MockLLMError(f"脚本响应必须是 str 或 dict，收到: {type(spec).__name__}")
    return spec


class MockLLM:
    """离线假 LLM。按脚本回放响应，或调用 handler 动态生成响应。"""

    def __init__(
        self,
        scripted: Optional[Sequence] = None,
        handler: Optional[Handler] = None,
        model: str = "mock",
    ):
        if scripted is None and handler is None:
            raise MockLLMError("MockLLM 至少需要 scripted 脚本或 handler 函数之一")
        if scripted is not None and handler is not None:
            raise MockLLMError("scripted 与 handler 只能二选一，避免行为不确定")
        self.scripted: list[dict] = [_coalesce(s) for s in (scripted or [])]
        self.handler = handler
        self.model = model
        self.calls: list[dict] = []  # 每次调用快照：{"messages", "tools", "params"}
        self._tc_counter = 0         # 工具调用 ID 计数器（确定性生成 call_1, call_2, ...）

    # ---- 内部：记录调用快照 ----
    def _record(self, messages: list[dict], tools: Optional[list[dict]], kwargs: dict) -> None:
        # 拷贝 messages，避免后续原地修改污染历史记录。
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "tools": tools,
            "params": dict(kwargs),   # temperature / max_tokens 等透传参数也被记录，可断言
        })

    # ---- 内部：按 handler 或脚本生成响应规格 ----
    def _respond(self, messages: list[dict], tools: Optional[list[dict]]) -> dict:
        if self.handler is not None:
            return _coalesce(self.handler(messages, tools))
        if not self.scripted:
            raise MockLLMError(
                f"MockLLM 脚本耗尽：第 {len(self.calls)} 次调用没有预设响应。"
                f"最后一条输入: {messages[-1] if messages else '(空)'}"
            )
        return self.scripted.pop(0)  # 按顺序消费脚本

    def _respond_with_ids(self, messages: list[dict], tools: Optional[list[dict]]) -> ChatResponse:
        """生成响应并为缺少 id 的工具调用补上确定性 ID（call_1, call_2, ...）。"""
        spec = self._respond(messages, tools)
        if not spec.get("tool_calls"):
            return ChatResponse(
                content=spec.get("content", ""),
                tool_calls=(),
                finish_reason="stop",
                model=self.model,
            )
        built_calls = []
        for tc in spec["tool_calls"]:
            self._tc_counter += 1
            built_calls.append(ToolCall(
                name=tc["name"],
                arguments=dict(tc.get("arguments", {})),
                id=tc.get("id") or f"call_{self._tc_counter}",
            ))
        return ChatResponse(
            content=spec.get("content", ""),
            tool_calls=tuple(built_calls),
            finish_reason="tool_calls",
            model=self.model,
        )

    # ---- 主接口 1：一次性返回（与真实客户端的 chat() 签名一致） ----
    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> ChatResponse:
        self._record(messages, tools, kwargs)
        return self._respond_with_ids(messages, tools)

    # ---- 主接口 2：流式返回（逐块 yield；文本拼接结果 == chat().content） ----
    def stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        chunk_size: int = 4,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """模拟流式输出：把本次响应文本按 chunk_size 切片逐块吐出。

        说明：真实流式协议中工具调用的增量片段较复杂，Mock 只流式文本部分；
        若响应是工具调用，则只产出一块 finish_reason="tool_calls" 的空内容。
        """
        self._record(messages, tools, kwargs)
        response = self._respond_with_ids(messages, tools)
        if response.has_tool_calls or not response.content:
            yield StreamChunk(content="", finish_reason=response.finish_reason)
            return
        text = response.content
        for start in range(0, len(text), chunk_size):
            piece = text[start:start + chunk_size]
            is_last = start + chunk_size >= len(text)
            yield StreamChunk(content=piece, finish_reason="stop" if is_last else None)

    # ---- 供测试断言用的便捷方法 ----
    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_user_text(self) -> str:
        """取最近一次调用中最后一条 user 消息的文本（常用于写 handler 的路由逻辑）。"""
        return messages_text(self.calls[-1]["messages"])

    def reset_script(self, scripted: Sequence) -> None:
        """重新填充脚本（测试复用同一个 MockLLM 时使用）。"""
        self.scripted = [_coalesce(s) for s in scripted]


def messages_text(messages: list[dict]) -> str:
    """把整段对话渲染成一段文本（含工具结果），便于测试断言与 handler 路由。"""
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):  # 兼容多段 content（暂未用到，防御性处理）
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        role = m.get("role", "?")
        if m.get("tool_calls"):
            calls = ", ".join(str(tc) for tc in m["tool_calls"])
            parts.append(f"[{role}] 调用工具: {calls}")
        elif m.get("name") is not None or m.get("tool_call_id") is not None:
            parts.append(f"[tool:{m.get('name', m.get('tool_call_id'))}] {content}")
        else:
            parts.append(f"[{role}] {content}")
    return "\n".join(parts)
