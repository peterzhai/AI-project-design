"""示例 02：JSON 输出的解析与"回炉修复"循环。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/03_Prompt与结构化输出/code/ch03_json_extraction.py

现实：即使提示词说"只输出 JSON"，模型也常夹带解释文字、Markdown 围栏，
甚至输出尾逗号等非法 JSON。工程对策两步走：
1. 宽容解析：剥围栏 -> 定位花括号 -> json.loads（common/json_utils.py）；
2. 修复循环：解析失败就把「原输出 + 错误信息」喂回模型，让它自己修。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.json_utils import JsonParseError, parse_llm_json  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402

JSON_TASK_PROMPT = "你是数据抽取器。只输出 JSON，不要解释。"


def ask_for_json(llm, task_prompt: str, user_input: str, max_attempts: int = 3) -> object:
    """要求 LLM 输出 JSON；解析失败把错误信息喂回去让模型自己修。

    这是生产系统处理"结构化输出"的最小可靠闭环。
    """
    messages = [
        {"role": "system", "content": task_prompt},
        {"role": "user", "content": user_input},
    ]
    last_error: Exception
    for attempt in range(1, max_attempts + 1):
        response = llm.chat(messages)
        try:
            data = parse_llm_json(response.content)
            print(f"第 {attempt} 次尝试: 解析成功 -> {data}")
            return data
        except JsonParseError as exc:
            last_error = exc
            print(f"第 {attempt} 次尝试: 解析失败（{exc}）")
            # 关键动作：把坏输出与错误原因都放进上下文，让模型"看到自己错在哪"
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": f"你的输出无法解析：{exc}\n请重新输出，只输出合法 JSON，不要解释。",
            })
    raise JsonParseError(f"重试 {max_attempts} 次后仍无法解析；最后错误: {last_error}", raw_text="")


def main() -> None:
    banner("示例 02：JSON 解析与回炉修复循环")

    # 实验 1：宽容解析 —— 三种常见的"不干净"输出
    section("实验 1：宽容解析（剥围栏 / 定位花括号）")
    dirty_outputs = [
        '```json\n{"name": "张三", "age": 30}\n```',       # Markdown 围栏
        '好的，结果如下：{"name": "张三", "age": 30} 请查收。',  # 夹带解释文字
        '{"name": "张三", "age": 30}',                     # 干净输出
    ]
    parsed = [parse_llm_json(text) for text in dirty_outputs]
    verify("三种脏输出都解析出同一 dict", parsed == [{"name": "张三", "age": 30}] * 3)

    section("实验 2：非法 JSON 被明确拒绝（携带原文）")
    try:
        parse_llm_json('{"name": "张三",}')  # 尾逗号：LLM 最常见的 JSON 错误
        raise AssertionError("实验 2 失败：尾逗号应当解析失败")
    except JsonParseError as exc:
        # 注意：except 块结束后 Python 会删除 exc 变量，断言必须写在块内。
        print(f"按预期失败: {exc}")
        verify("JsonParseError 携带原始文本供回炉", exc.raw_text == '{"name": "张三",}')

    # 实验 3：修复闭环 —— 第一轮输出坏 JSON，喂回错误后第二轮输出好 JSON
    section("实验 3：解析失败 -> 喂回错误 -> 模型自修")
    llm = MockLLM(scripted=[
        '{"order_id": "A1001",}',              # 第 1 轮：尾逗号坏 JSON
        '{"order_id": "A1001", "amount": 199}',  # 第 2 轮：修复成功
    ])
    result = ask_for_json(llm, JSON_TASK_PROMPT, "抽取订单信息：订单号A1001，金额199")
    verify("闭环 2 次调用后拿到合法数据", llm.call_count == 2, f"实际 {llm.call_count}")
    verify("最终结果是合法 dict", result == {"order_id": "A1001", "amount": 199})
    verify("第 2 次调用上下文包含错误反馈", "无法解析" in llm.calls[1]["messages"][-1]["content"])

    # 实验 4：彻底失败 —— 模型怎么修都不行，明确报错而不是静默吞掉
    section("实验 4：重试预算耗尽 -> 显式失败")
    hopeless = MockLLM(handler=lambda m, t: {"content": "我实在不会输出JSON，抱歉。"})
    try:
        ask_for_json(hopeless, JSON_TASK_PROMPT, "再试一次", max_attempts=2)
        raise AssertionError("实验 4 失败：应当抛 JsonParseError")
    except JsonParseError as exc:
        print(f"按预期失败: {exc}")
    verify("预算耗尽前共尝试 2 次", hopeless.call_count == 2, f"实际 {hopeless.call_count}")

    finish("ch03_json_extraction")


if __name__ == "__main__":
    main()
