"""统一 LLM 客户端：默认离线 Mock，USE_REAL_LLM=1 时切换到真实 OpenAI 兼容接口。

设计原则：
1. 无论 Mock 还是真实客户端，chat()/stream() 的签名与产出完全一致，
   上层业务代码对两者无感知 —— 这正是"面向接口编程"的示范。
2. 真实调用用标准库 urllib 直连 /chat/completions（OpenAI 兼容格式），
   不引入任何三方 SDK，保持依赖最小化。

环境变量约定：
- USE_REAL_LLM     "1" 时启用真实调用（默认 "0"，离线 Mock）。
- OPENAI_API_KEY   真实调用的密钥（OpenAI 或任意兼容网关的 Key）。
- OPENAI_BASE_URL  兼容接口地址，默认 https://api.openai.com/v1
                   （可指向本地 ollama：http://127.0.0.1:11434/v1）。
- LEARNING_MODEL   模型名，默认 gpt-4o-mini。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterator, Optional

from common.mock_llm import ChatResponse, MockLLM, ToolCall


class LLMError(RuntimeError):
    """真实 LLM 调用失败（网络/鉴权/协议解析），或配置不完整。

    status_code: HTTP 状态码（网络级失败为 None）。重试策略据此分类：
    429/5xx 可重试，401/400 等不可重试。
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def use_real_llm() -> bool:
    """是否启用真实 LLM 调用。默认关闭，保证零依赖离线可跑。"""
    return os.environ.get("USE_REAL_LLM", "0") == "1"


def parse_sse_lines(text: str) -> list[dict]:
    """把 OpenAI SSE 文本流解析成 dict 列表（纯函数，离线可测）。

    规则：
    - 空行与 ":" 开头的注释行跳过；
    - "data: " 前缀剥掉；"data: [DONE]" 表示流结束；
    - 其余 data 行按 JSON 解析。
    """
    events: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            events.append(json.loads(data))
    return events


class OpenAICompatClient:
    """直连任意 OpenAI 兼容 /chat/completions 接口的最小客户端（纯标准库）。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("LEARNING_MODEL", "gpt-4o-mini")
        self.timeout_seconds = timeout_seconds

    # ---- 内部：构造请求 ----
    def _request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    # ---- 主接口 1：一次性返回 ----
    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> ChatResponse:
        payload: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        try:
            with urllib.request.urlopen(self._request(payload), timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # 读取失败也要保留状态码信息
                body = ""
            raise LLMError(f"LLM HTTP {exc.code}: {body or exc.reason}", status_code=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"LLM 连接失败（{self.base_url}）: {exc}") from exc

        return self._parse(data)

    # ---- 主接口 2：流式返回（逐块 yield 文本增量） ----
    def stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[str]:
        """基于真实 SSE 协议的流式输出，逐块 yield 文本增量。

        说明：真实流式协议里工具调用会以 delta 片段分散到达，解析复杂且
        各网关实现不一；本教程的 stream 只处理文本增量（教学主线），
        工具调用请用 chat()。
        """
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        try:
            with urllib.request.urlopen(self._request(payload), timeout=self.timeout_seconds) as resp:
                for raw in resp:  # 逐行读取 chunked 响应
                    line = raw.decode("utf-8").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue  # 跳过个别坏行，不让整条流崩掉
                        choices = chunk.get("choices") or []
                        if choices:
                            piece = (choices[0].get("delta") or {}).get("content")
                            if piece:
                                yield piece
        except urllib.error.HTTPError as exc:
            raise LLMError(f"LLM 流式 HTTP {exc.code}", status_code=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"LLM 流式连接失败（{self.base_url}）: {exc}") from exc

    def _parse(self, data: dict) -> ChatResponse:
        """从 OpenAI 兼容响应中提取统一 ChatResponse。"""
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"响应缺少 choices: {str(data)[:300]}")
        message = choices[0].get("message", {})
        raw_calls = message.get("tool_calls") or []
        parsed_calls = []
        for call in raw_calls:
            function = call.get("function", {})
            raw_args = function.get("arguments", "{}")
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                # 模型偶尔输出非法 JSON：宁可显式失败，也不静默吞掉。
                raise LLMError(f"工具参数不是合法 JSON: {raw_args!r}") from None
            parsed_calls.append(ToolCall(name=function.get("name", ""), arguments=arguments))
        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=tuple(parsed_calls),
            finish_reason=choices[0].get("finish_reason", "stop"),
            model=data.get("model", self.model),
        )


def make_client(mock_client: Optional[MockLLM] = None):
    """客户端工厂：本教程所有示例都通过它获取 LLM。

    - USE_REAL_LLM=1 且设置了 OPENAI_API_KEY -> OpenAICompatClient（真实调用）
    - 否则 -> 传入的 mock_client（不传则新建一个空的 MockLLM）
    """
    if use_real_llm():
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError(
                "USE_REAL_LLM=1 但未设置 OPENAI_API_KEY。"
                "请导出密钥，或取消 USE_REAL_LLM 以使用离线 Mock 模式。"
            )
        return OpenAICompatClient()
    return mock_client if mock_client is not None else MockLLM(scripted=["（未配置脚本的默认 Mock 响应）"])
