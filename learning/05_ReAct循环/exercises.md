# 第 5 章 · 练习

## 练习 1（热身）：经典文本协议 ReAct

在 Tool Calling 普及之前，ReAct 靠解析模型输出的**文本协议**：

    Thought: 我需要先查天气
    Action: get_weather[{"city": "北京"}]

请实现 `parse_react_text(text) -> (thought, tool_name, arguments_dict)`，
用正则提取三个字段，并构造 handler MockLLM（返回这种文本格式）驱动一个最小循环。

**验收标准**：
- 能解析 Thought/Action 两行，arguments 是 JSON 字符串转 dict；
- 解析失败（缺 Action）返回 `(thought, None, None)`；
- pytest 覆盖成功与失败两种解析。

**参考答案要点**：
- 正则如 `r"Thought:\s*(.+)"` 与 `r"Action:\s*(\w+)\[(.*)\]"`；`json.loads` 解析参数。
- 感受一下：文本协议需要容错解析（模型少个换行、多个空格都会崩），而 Tool-Calling 是
  结构化协议、无需解析 —— 这正是业界从文本 ReAct 迁移到原生 Tool Calling 的原因。
- 但**思想完全一致**：思考显式化、行动结构化、观察回填。

## 练习 2（进阶）：轨迹回放断言（回放测试）

写 `assert_trace_invariants(trace) -> list[str]`：对轨迹做完整性检查，返回违规列表：
- 行动步必须有 observation，observation 不能为空字符串；
- final_answer 只能出现在最后一步；
- 每步 turn 严格递增且从 1 开始。

**验收标准**：
- 对示例 01 的正常轨迹返回空列表；
- 手工构造一条"中间步骤就给了 final_answer"的坏轨迹，能被抓出；
- pytest 覆盖三条规则（各构造一条坏轨迹）。

**参考答案要点**：
- 这是"轨迹不变量"思想：比逐个断言更系统，第 10 章评估时会作为标准检查项。
- final_answer 位置规则提示了循环的实现约定：终止步不再产生新行动。

## 练习 3（挑战）：预算终止条件

写 `token_budget_detector(max_chars: int) -> StopCondition`：累计所有 observation 的
字符数，超过预算即终止，原因里注明"已用 X / 预算 Y"。

**验收标准**：
- 工具返回很长文本时，Agent 在预算处止损，`termination` 包含已用/预算数字；
- 未超预算时正常完成；
- 与 `repeated_call_detector` 同时挂载时，任一命中即停（测试两种命中顺序）。

**参考答案要点**：
- 每步检查"累计 observation"而非"单步"，因为多次小观察也会累积超支。
- 这就是成本护栏的雏形（第 11 章正式展开）：**预算检查要在"下一次行动"之前**，
  而不是事后；终止条件的插拔签名让它成为两行代码的事 —— 好的抽象让安全功能"顺路"实现。
