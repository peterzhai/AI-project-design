"""示例 02：输出护栏 —— 模型的输出在交还用户前，再做一遍安检。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/11_安全、护栏与成本控制/code/ch11_output_guardrails.py

实现位于 common/guards.py；本示例演示三类输出风险的处置：
1. 密钥/手机号泄漏 -> 自动打码（留痕 redactions）；
2. 幻觉引用（不存在的来源 id）-> 喂回模型修正（重试一次）；
3. 越权承诺（"保证全额退款"）-> 整体拦截，换安全话术。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.guards import OutputGuardrail  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402

# 合法的知识块 id 集合（第 7 章 RAG 检索结果；输出引用必须出自这里）
VALID_CHUNK_IDS = {"policy#0", "policy#1", "policy#2"}


def guarded_llm_output(guard: OutputGuardrail, llm, question: str) -> str:
    """带输出护栏的调用：模型输出先过安检，有幻觉引用时重试一次。"""
    response = llm.chat([{"role": "user", "content": question}])
    result = guard.check(response.content)
    if result.violations and any("幻觉引用" in v for v in result.violations):
        # 幻觉 -> 把违规信息喂回去让模型修正（第 3 章修复闭环的输出侧版本）
        retry = llm.chat([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": f"你的回答引用了不存在的来源 {result.violations}，"
                                        "请只依据真实来源重新回答。"},
        ])
        result = guard.check(retry.content)
        return result.text
    return result.text


def main() -> None:
    banner("示例 02：输出护栏")

    guard = OutputGuardrail(valid_chunk_ids=VALID_CHUNK_IDS)

    # 实验 1：密钥泄漏被自动打码
    section("实验 1：敏感凭据打码")
    llm = MockLLM(scripted=["已为您查询，调试密钥是 sk-abcdefgh12345678，请勿外传。"])
    result = guarded_llm_output(guard, llm, "查一下配置")
    print(f"处理后输出: {result}")
    verify("密钥被替换为占位符", "sk-abcdefgh12345678" not in result and "[已脱敏]" in result)

    # 实验 2：幻觉引用 -> 喂回修正
    section("实验 2：幻觉引用触发修正闭环")
    llm2 = MockLLM(scripted=[
        {"content": "根据公司政策，年假五天 [policy#99]。"},               # 引用了不存在的块
        {"content": "根据公司政策，年假五天 [policy#0]。"},               # 修正后引用真实块
    ])
    result2 = guarded_llm_output(guard, llm2, "年假几天？")
    print(f"处理后输出: {result2}")
    verify("第一次的幻觉引用触发重试", llm2.call_count == 2)
    verify("修正后的引用是合法块", "[policy#0]" in result2 and "policy#99" not in result2)

    # 实验 3：越权承诺 -> 整体拦截
    section("实验 3：越权承诺拦截")
    llm3 = MockLLM(scripted=["放心，我们保证全额退款，绝对没问题！"])
    result3 = guarded_llm_output(guard, llm3, "能退款吗")
    print(f"处理后输出: {result3}")
    verify("越权承诺被整体拦截（换成安全话术）", "人工客服" in result3)
    verify("原始承诺文本不再出现", "保证全额退款" not in result3)

    # 实验 4：正常输出零损伤
    section("实验 4：正常输出原样通过")
    llm4 = MockLLM(scripted=["根据公司政策，入职满一年享有五天带薪年假 [policy#0]。"])
    result4 = guarded_llm_output(guard, llm4, "年假政策是什么？")
    verify("正常输出未被改动", "五天带薪年假" in result4 and "policy#0" in result4)

    # 实验 5：脱敏与拦截可叠加（密钥打码留痕，承诺拦截换话术）
    section("实验 5：多项护栏叠加")
    llm5 = MockLLM(scripted=["密钥 sk-zzzz99998888 已记录，并且我们保证到账！"])
    llm5_out = guard.check(llm5.chat([]).content)
    verify("密钥不出现在最终输出（拦截后用户看不到原文）", "sk-zzzz99998888" not in llm5_out.text)
    verify("打码动作留痕在 redactions（可审计）", llm5_out.redactions == ["敏感凭据打码"])
    verify("承诺随后被拦截", llm5_out.blocked and any("未授权承诺" in v for v in llm5_out.violations))
    verify("违规清单完整（可审计）", len(llm5_out.violations) == 1)

    finish("ch11_output_guardrails")


if __name__ == "__main__":
    main()
