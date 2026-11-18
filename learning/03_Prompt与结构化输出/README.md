# 第 3 章 · Prompt 与结构化输出

> 本章目标：让模型的输出**能被程序直接消费** —— 这是从"演示"走向"产品"的分水岭。
> 三个关键词：提示词四板斧、宽容解析、校验修复闭环。
>
> 学习时间：约 2 小时。前置要求：完成第 2 章。

---

## 1. 为什么"会聊天"不够，还要"会出 JSON"

第 1 章的 Agent 依赖模型做出正确的工具选择，第 2 章的调用只是拿到文本。
但真正接入业务时你会发现：**散文是无法编程的**。

```text
     散文输出                    结构化输出
  "客户抱怨扫地机器人…         {"product": "扫地机器人",
   建议尽快处理"                 "issue_type": "卡住",
      │                          "urgency": "高"}
      ▼                              │
  只能人工阅读                   ┌─────┴──────┐
  （演示可以，产品不行）          ▼            ▼
                            自动建单       路由到不同队列
```

本章全部示例围绕同一个任务：**从客户留言抽取工单字段**（product / issue_type / urgency）。

## 2. 代码走读

### 2.1 示例 01 —— 提示词四板斧

文件：[code/ch03_prompt_patterns.py](code/ch03_prompt_patterns.py)

朴素提示（"帮我看看这条留言说了啥"）只能换回散文。结构化提示把四样东西写进 system（摘自源码，逐字一致）：

```python
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
```

四板斧的作用：

| 板斧 | 解决什么问题 |
| --- | --- |
| 角色设定 | 收窄行为空间，减少跑题 |
| 输出契约 | 告诉模型"什么算合格"，枚举值在这里定死 |
| 少样本示例 | 比一千字描述更能教会模型格式（1~3 个即可） |
| 分隔符 | 把"要处理的文本"与"指令"隔开，防注入、防混淆 |

示例用 handler 假模型做了对照实验：含 JSON 指令与示例的提示词 → 标准 JSON；朴素提示 → 散文。

### 2.2 示例 02 —— 宽容解析与"回炉修复"循环

文件：[code/ch03_json_extraction.py](code/ch03_json_extraction.py)

即使有输出契约，模型仍会犯三类错：夹解释文字、包 Markdown 围栏、输出非法 JSON（尾逗号）。
对策一：**宽容解析**（`common/json_utils.py`）—— 剥围栏 `strip_code_fence` →
定位花括号 `extract_json_object` → `parse_llm_json`，失败时抛 `JsonParseError` 并**携带原文**。

对策二：**修复循环**（摘自源码 `ask_for_json`，逐字一致）：

```python
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
```

灵魂在于注释那行：**把坏输出 + 错误原因一起喂回**。模型看到"我上一轮输出了什么、错在哪"，
修复成功率远高于干巴巴地重复指令。示例实验 3 实证：尾逗号 → 喂回 → 2 次调用内修复。

### 2.3 示例 03 —— schema 校验闭环：解析成功 ≠ 数据正确

文件：[code/ch03_schema_validation.py](code/ch03_schema_validation.py)

语法合法的 JSON 可能语义全错：缺字段、枚举外取值（`urgency: "超级紧急"`）、多余字段。
`common/validators.py` 是一个纯标准库迷你校验器（覆盖 type/required/properties/enum/
items/min/max/additionalProperties），错误消息带 `$.字段` 路径：

```text
$.urgency: 值 '超级紧急' 不在允许的枚举 ['低', '中', '高'] 中
$: 缺少必填字段「issue_type」
$: 出现 schema 未定义的字段「note」
```

`ask_for_valid_json` 把两级防线合成一个闭环：**语法错或语义错，都把具体错误列表喂回模型**。
示例实验 4 演示了三级递进：语法错 → 语义错 → 完全合规，恰好 3 次调用。

> 为什么不用 pydantic/jsonschema？—— 教程依赖最小化红线。迷你校验器约 100 行、零依赖、
> 行为可离线测试；工程中你可按需替换为 pydantic，接口思路完全一致。

## 3. 常见坑

1. **只要求"输出 JSON"却不在提示词里给字段规范与示例** —— 格式靠猜，十次九歪。
2. **解析失败直接重试而不带错误信息** —— 模型不知道错哪，同样的错再来一遍。
3. **把修复循环的 `max_attempts` 设得过大** —— 每次失败都是一次真实调用费用；2~3 次是经验值。
4. **校验只查"有没有"，不查"值域"** —— `urgency: "超级紧急"` 这种枚举外值会污染下游数据。
5. **提示词与代码中的 schema 各写一份** —— 两处漂移后永远修不好。本教程的 schema 是常量，
   提示词与校验器引用同一份（示例 03 的 `TICKET_SCHEMA` 与提示词描述一一对应）。
6. **Python 细节**：`except ... as exc` 块结束后 `exc` 会被删除，相关断言要写在块内
   （示例 02 开发时就踩了这个坑）。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch03_prompt_patterns.py](code/ch03_prompt_patterns.py) | 示例 01：四板斧对照实验 |
| [code/ch03_json_extraction.py](code/ch03_json_extraction.py) | 示例 02：宽容解析 + 修复循环 |
| [code/ch03_schema_validation.py](code/ch03_schema_validation.py) | 示例 03：schema 校验闭环 |
| [tests/test_ch03_examples.py](tests/test_ch03_examples.py) | 10 个测试 |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

公共层新组件：`common/json_utils.py`（宽容解析）、`common/validators.py`（迷你 schema 校验）。

---

> 下一篇：[第 4 章 · Tool Calling 与函数调用](../04_Tool_Calling与函数调用/README.md) ——
> 把第 1 章里的"名字 -> 函数"字典，升级为带 JSON Schema 描述、参数校验与错误分类的正规军。
