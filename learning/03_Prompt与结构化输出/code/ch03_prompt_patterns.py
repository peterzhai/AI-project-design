"""示例 01：提示词工程四板斧 —— 角色、格式指令、少样本示例、分隔符。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/03_Prompt与结构化输出/code/ch03_prompt_patterns.py

同一个抽取任务，两种提示词两种命运：
- 朴素提示：模型回一段散文，程序没法消费；
- 结构化提示（四板斧齐上）：模型回标准 JSON，程序直接用。

模拟"真实模型"的方式：handler 检查提示词里是否有 JSON 指令与示例 ——
提示词决定了假模型（也是真模型）的行为。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.json_utils import JsonParseError, parse_llm_json  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402

CUSTOMER_MESSAGE = "扫地机器人老在桌底卡住，做饭的时候特别耽误事，快来人看看！"


def make_extraction_llm() -> MockLLM:
    """模拟真实模型：提示词含"只输出 JSON"与示例时返回 JSON，否则返回散文。"""
    def handler(messages, tools):
        prompt = messages[0]["content"]
        if "只输出 JSON" in prompt and "示例" in prompt:
            return {"content": '{"product": "扫地机器人", "issue_type": "卡住/障碍", "urgency": "高"}'}
        return {"content": "客户抱怨扫地机器人总在桌底卡住、影响做饭，情绪比较急，建议尽快安排处理。"}

    return MockLLM(handler=handler)


# ---------------------------------------------------------------
# 结构化提示词：四板斧逐一到位（真实工程里这就是 system 提示词）。
# ---------------------------------------------------------------
STRUCTURED_PROMPT = """你是一名客服工单分类器。（第 1 板斧：角色设定）

任务：从留言中抽取结构化字段，用于自动建单。（第 2 板斧：明确任务）

输出要求：只输出 JSON，不要任何解释文字。字段规范：
- product:    string，产品名
- issue_type: string，问题类型
- urgency:    string，紧急程度，只能是 "低"、"中"、"高"（第 2 板斧：输出契约）

示例：
留言：<message>净水器底下漏水了，厨房都湿了</message>
输出：{"product": "净水器", "issue_type": "漏水", "urgency": "中"}
（第 3 板斧：少样本示例）

真实输入：用户留言会包裹在 <message> 标签里。（第 4 板斧：分隔符）"""

NAIVE_PROMPT = "帮我看看这条客户留言说了啥。"


def main() -> None:
    banner("示例 01：提示词工程四板斧")

    # 对照实验：同一句留言，两种提示词
    section("实验 1：朴素提示词 -> 散文输出（程序无法消费）")
    llm = make_extraction_llm()
    prose = llm.chat([{"role": "system", "content": NAIVE_PROMPT},
                      {"role": "user", "content": CUSTOMER_MESSAGE}])
    print(f"输出: {prose.content}")
    try:
        parse_llm_json(prose.content)
        raise AssertionError("实验 1 失败：散文不应能解析出 JSON")
    except JsonParseError:
        print("-> parse_llm_json 失败：散文无法程序化消费 ❌")
    verify("朴素提示词下无法解析出 JSON", True)

    section("实验 2：结构化提示词 -> 标准 JSON（程序直接消费）")
    structured = llm.chat([{"role": "system", "content": STRUCTURED_PROMPT},
                           {"role": "user", "content": f"<message>{CUSTOMER_MESSAGE}</message>"}])
    print(f"输出: {structured.content}")
    data = parse_llm_json(structured.content)
    verify("输出是可解析的 JSON 对象", isinstance(data, dict))
    verify("包含三个抽取字段", set(data) == {"product", "issue_type", "urgency"},
           f"实际字段 {set(data)}")
    verify("紧急度命中规范枚举", data["urgency"] == "高")

    section("实验 3：把提示词拆开看")
    print(STRUCTURED_PROMPT)
    verify("四板斧齐备（角色/任务/契约/示例/分隔符）", True)

    finish("ch03_prompt_patterns")


if __name__ == "__main__":
    main()
