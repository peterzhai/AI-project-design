"""示例 03：schema 校验闭环 —— 解析成功≠数据正确，还要"验字段"。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/03_Prompt与结构化输出/code/ch03_schema_validation.py

两级防线：
1. 语法级：能 json.loads（示例 02 已覆盖）；
2. 语义级：字段齐全、类型正确、枚举合法（common/validators.py）。
语义级失败的修复方式与语法级相同：把「哪条不合规」喂回模型让它改。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.json_utils import JsonParseError, parse_llm_json  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402
from common.validators import ValidationError, validate  # noqa: E402

# 工单抽取 schema：与示例 01 提示词里的"输出契约"一一对应。
TICKET_SCHEMA = {
    "type": "object",
    "required": ["product", "issue_type", "urgency"],
    "additionalProperties": False,
    "properties": {
        "product": {"type": "string", "minLength": 2},
        "issue_type": {"type": "string", "minLength": 2},
        "urgency": {"type": "string", "enum": ["低", "中", "高"]},
    },
}

JSON_TASK_PROMPT = "你是客服工单抽取器。只输出 JSON，字段: product/issue_type/urgency。"


def ask_for_valid_json(llm, task_prompt: str, user_input: str, schema: dict,
                       max_attempts: int = 3) -> dict:
    """JSON 解析 + schema 校验的双重修复闭环。

    无论哪一级失败（语法/语义），都把具体错误列表喂回模型。
    """
    messages = [
        {"role": "system", "content": task_prompt},
        {"role": "user", "content": user_input},
    ]
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        response = llm.chat(messages)
        try:
            data = parse_llm_json(response.content)
        except JsonParseError as exc:
            errors = [f"语法级: {exc}"]
        else:
            if not isinstance(data, dict):
                errors = [f"语义级: 顶层必须是 JSON 对象，实际为 {type(data).__name__}"]
            else:
                errors = [f"语义级: {e}" for e in validate(data, schema)]
                if not errors:
                    print(f"第 {attempt} 次尝试: 语法与语义校验全部通过 -> {data}")
                    return data
        print(f"第 {attempt} 次尝试: 未通过 -> {'; '.join(errors)}")
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": "输出未通过校验，逐条修复：\n- " + "\n- ".join(errors)
                       + "\n请重新只输出合法且合规的 JSON。",
        })
    raise ValidationError(errors)


def main() -> None:
    banner("示例 03：schema 校验闭环")

    # 实验 1：校验器单独工作 —— 语法对但语义错
    section("实验 1：语法合法、语义违规的输出")
    bad = {"product": "扫地机器人", "issue_type": "卡住", "urgency": "超级紧急"}
    errors = validate(bad, TICKET_SCHEMA)
    print(f"违规输出: {bad}")
    for e in errors:
        print(f"  - {e}")
    verify("枚举外的紧急度被抓出", any("urgency" in e and "枚举" in e for e in errors))
    verify("错误路径带 $ 前缀（可读可定位）", any(e.startswith("$.urgency") for e in errors))

    section("实验 2：缺字段 / 多字段 / 类型错")
    errors2 = validate({"product": "机", "urgency": "高", "note": 1}, TICKET_SCHEMA)
    print(f"违规输出2: 共 {len(errors2)} 条违规")
    for e in errors2:
        print(f"  - {e}")
    verify("缺字段被点名", any("缺少必填字段「issue_type」" in e for e in errors2))
    verify("额外字段被点名", any("未定义的字段「note」" in e for e in errors2))
    verify("长度不达标被点名", any("$.product: 长度少于 2" in e for e in errors2))

    # 实验 3：完整闭环 —— 枚举违规 -> 喂回 -> 模型改对
    section("实验 3：修复闭环（语义违规喂回模型）")
    llm = MockLLM(scripted=[
        '{"product": "扫地机器人", "issue_type": "卡住", "urgency": "超级紧急"}',  # 枚举违规
        '{"product": "扫地机器人", "issue_type": "卡住", "urgency": "高"}',        # 修正
    ])
    ticket = ask_for_valid_json(llm, JSON_TASK_PROMPT, "<message>桌底老卡住</message>", TICKET_SCHEMA)
    verify("闭环 2 次调用后拿到合规工单", llm.call_count == 2, f"实际 {llm.call_count}")
    verify("urgency 修正为合法枚举", ticket["urgency"] == "高")
    verify("修复提示中包含具体违规描述", "枚举" in llm.calls[1]["messages"][-1]["content"])

    # 实验 4：语法与语义双重防线都触发
    section("实验 4：先语法错、再语义错、最后成功")
    llm2 = MockLLM(scripted=[
        '{"product": "净水器", "urgency": "中",}',                               # 语法错（尾逗号）
        '{"product": "净水器", "issue_type": "漏水", "urgency": "爆表", "x": 1}',  # 语义错（枚举+多字段）
        '{"product": "净水器", "issue_type": "漏水", "urgency": "中"}',           # 通过
    ])
    ticket2 = ask_for_valid_json(llm2, JSON_TASK_PROMPT, "<message>漏水了</message>", TICKET_SCHEMA)
    verify("三级递进闭环 3 次调用", llm2.call_count == 3, f"实际 {llm2.call_count}")
    verify("最终工单完全合规", ticket2 == {"product": "净水器", "issue_type": "漏水", "urgency": "中"})

    finish("ch03_schema_validation")


if __name__ == "__main__":
    main()
