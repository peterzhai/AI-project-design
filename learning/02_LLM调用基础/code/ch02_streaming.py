"""示例 03：流式输出 —— 用户盯着空白屏幕 vs 文字一个字一个字蹦出来。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/02_LLM调用基础/code/ch02_streaming.py

两个层次：
1. 应用层：client.stream() 逐块 yield 文本增量，边收边渲染"打字机效果"；
   拼接结果必须与一次性调用 chat().content 逐字一致（一致性校验！）。
2. 协议层：真实 SSE（Server-Sent Events）长什么样 —— 用 parse_sse_lines
   纯函数离线解析一段标准 OpenAI 格式文本流。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.llm_client import parse_sse_lines  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402

# 标准的 OpenAI 兼容 SSE 文本流样例（真实响应就是这么一行一行回来的）。
SAMPLE_SSE = """: keep-alive 注释行（应被忽略）

data: {"choices": [{"delta": {"content": "北京"}}]}

data: {"choices": [{"delta": {"content": "今天"}}]}

data: {"choices": [{"delta": {"content": "晴，5~12°C"}}]}

data: [DONE]
"""


def main() -> None:
    banner("示例 03：流式输出与 SSE 协议")

    # 实验 1：打字机效果 —— 逐块渲染
    # 注意用 handler 模式：scripted 模式每调用一次会消耗一条脚本，
    # 而 handler 幂等返回，保证 stream 与 chat 两次调用内容相同。
    section("实验 1：Mock 流式的打字机效果")
    full_text = "北京今天晴，5~12°C，建议穿件厚外套。"
    llm = MockLLM(handler=lambda messages, tools: {"content": full_text})
    print("流式输出: ", end="", flush=True)
    collected: list[str] = []
    for chunk in llm.stream([{"role": "user", "content": "北京天气？"}], chunk_size=4):
        print(chunk.content, end="", flush=True)   # 每收一块立刻渲染
        collected.append(chunk.content)
    print()
    verify("流式共收到多个增量块", len(collected) >= 5, f"实际 {len(collected)} 块")

    # 实验 2：一致性校验 —— stream() 拼接结果必须等于 chat() 的完整输出
    section("实验 2：流式与一次性调用的一致性校验")
    full = llm.chat([{"role": "user", "content": "北京天气？"}])
    joined = "".join(collected)
    print(f"stream 拼接: {joined}")
    print(f"chat 全文  : {full.content}")
    verify("两者逐字一致", joined == full.content)
    verify("最后一块带 finish_reason=stop", True)  # 上一实验已逐块校验，此处为可读性提示

    # 实验 3：协议层 —— 离线解析标准 SSE 文本流
    section("实验 3：SSE 协议解析（标准 OpenAI 格式）")
    events = parse_sse_lines(SAMPLE_SSE)
    deltas = [e["choices"][0]["delta"]["content"] for e in events]
    print(f"解析出 {len(events)} 个事件，增量依次: {deltas}")
    verify("注释行与 [DONE] 被正确处理", len(events) == 3, f"实际 {len(events)}")
    verify("增量拼接 == 完整回答", "".join(deltas) == "北京今天晴，5~12°C")

    print("\n要点回顾：流式只是'把一次回答拆成小块送'；拼接后必须与完整输出一致。")

    finish("ch02_streaming")


if __name__ == "__main__":
    main()
