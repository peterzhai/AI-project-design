"""示例 01：轨迹追踪 —— 给每次 LLM 调用装上"行车记录仪"。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/10_评估、追踪与可观测性/code/ch10_run_records.py

Agent 出错时最大的痛点是"黑盒"：不知道模型当时看到什么、回答什么、花了多少。
TracedLLM 包装任意 LLM 客户端，记录每次调用的输入输出、耗时、伪 token ——
接口不变（还是 chat），上层代码零改动，这就是第 2 章"统一接口"设计的回报。
记录可导出 JSONL（每行一条 JSON），供离线分析与回放。
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402


@dataclass
class CallRecord:
    """一次 LLM 调用的完整记录。"""

    seq: int                # 本次运行内的调用序号
    prompt_chars: int       # 输入字符数（伪 token 口径：1 字符 ≈ 1 伪 token）
    output_chars: int       # 输出字符数
    duration_ms: float      # 耗时（毫秒）
    prompt_preview: str     # 输入预览（最后一条消息，截断）
    output_preview: str     # 输出预览

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "prompt_chars": self.prompt_chars,
            "output_chars": self.output_chars,
            "duration_ms": round(self.duration_ms, 2),
            "prompt_preview": self.prompt_preview,
            "output_preview": self.output_preview,
        }


class TracedLLM:
    """透明代理：包装任意有 chat() 的客户端，逐调用落记录。"""

    def __init__(self, inner, agent_name: str = "agent"):
        self.inner = inner
        self.agent_name = agent_name
        self.records: list[CallRecord] = []
        self._clock = time.perf_counter  # 可注入时钟（测试零等待）

    def chat(self, messages: list[dict], tools=None, **kwargs):
        start = self._clock()
        response = self.inner.chat(messages, tools=tools, **kwargs)
        duration_ms = (self._clock() - start) * 1000
        last_message = str(messages[-1].get("content", "")) if messages else ""
        self.records.append(CallRecord(
            seq=len(self.records) + 1,
            prompt_chars=sum(len(str(m.get("content", ""))) for m in messages),
            output_chars=len(response.content),
            duration_ms=duration_ms,
            prompt_preview=last_message[:50],
            output_preview=response.content[:50],
        ))
        return response

    @property
    def total_pseudo_tokens(self) -> int:
        """本次运行的伪 token 总消耗（输入+输出）。"""
        return sum(r.prompt_chars + r.output_chars for r in self.records)

    def to_jsonl(self) -> str:
        """导出 JSONL：每行一条调用记录。"""
        return "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in self.records)


@dataclass
class RunRecord:
    """一次完整运行的记录：任务 + 各 Agent 的调用记录。"""

    run_id: str
    task: str
    traced_agents: list[TracedLLM] = field(default_factory=list)

    @property
    def total_pseudo_tokens(self) -> int:
        return sum(a.total_pseudo_tokens for a in self.traced_agents)

    def to_jsonl(self) -> str:
        """导出运行级 JSONL：每行 = 一条带 run_id/agent 的调用记录。"""
        lines = []
        for agent in self.traced_agents:
            for r in agent.records:
                line = {"run_id": self.run_id, "task": self.task,
                        "agent": agent.agent_name, **r.to_dict()}
                lines.append(json.dumps(line, ensure_ascii=False))
        return "\n".join(lines)


def main() -> None:
    banner("示例 01：轨迹追踪（TracedLLM）")

    # 模拟一个两步 Agent：查天气 -> 总结
    inner_llm = MockLLM(scripted=[
        {"content": "查询动作：get_weather(北京)"},
        {"content": "北京晴，5~12°C，适合出行。"},
    ])
    traced = TracedLLM(inner_llm, agent_name="天气Agent")

    section("运行一个两步 Agent")
    traced.chat([{"role": "user", "content": "查北京天气"}])
    traced.chat([
        {"role": "user", "content": "查北京天气"},
        {"role": "assistant", "content": "查询动作：get_weather(北京)"},
        {"role": "tool", "name": "get_weather", "content": "晴，5~12°C"},
    ])
    print(f"调用记录 {len(traced.records)} 条:")
    for r in traced.records:
        print(f"  #{r.seq}: 输入 {r.prompt_chars} 字符 / 输出 {r.output_chars} 字符 / {r.duration_ms:.2f}ms")
        print(f"      输入末条: {r.prompt_preview!r}")
        print(f"      输出预览: {r.output_preview!r}")

    verify("每次调用都有一条记录", len(traced.records) == 2)
    verify("第 2 次调用的输入包含工具观察（可回看）",
           "晴" in traced.records[1].prompt_preview or "晴" in json.dumps(traced.records[1].to_dict(), ensure_ascii=False))
    verify("伪 token 口径可累计", traced.total_pseudo_tokens > 0)
    verify("耗时记录为非负数", all(r.duration_ms >= 0 for r in traced.records))

    section("导出 JSONL 并回放校验")
    run = RunRecord(run_id="run_001", task="查北京天气", traced_agents=[traced])
    jsonl = run.to_jsonl()
    print(jsonl)
    lines = [json.loads(line) for line in jsonl.splitlines()]
    verify("每行都是合法 JSON 且带 run_id/agent", all("run_id" in l and "agent" in l for l in lines))
    verify("JSONL 行数 == 调用记录数", len(lines) == len(traced.records))
    verify("运行级总消耗 == 各调用之和",
           sum(l["prompt_chars"] + l["output_chars"] for l in lines) == run.total_pseudo_tokens)

    finish("ch10_run_records")


if __name__ == "__main__":
    main()
