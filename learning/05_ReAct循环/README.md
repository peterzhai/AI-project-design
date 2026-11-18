# 第 5 章 · ReAct 循环

> 本章目标：把 Agent 循环升级为显式的 **Thought → Action → Observation** 三段式 ——
> 模型每一步"想什么、做什么、看到什么"全部落盘，黑盒变白盒。
> 同时解决两个工程问题：轨迹持久化（断点续跑）与终止条件（怎么"知道该停"）。
>
> 学习时间：约 2~3 小时。前置要求：第 1、4 章。

---

## 1. ReAct 是什么：让"推理"成为上下文的一部分

ReAct（Reasoning + Acting，Yao et al. 2022）的核心洞察：

```text
   普通循环（第 1 章）              ReAct 循环（本章）
   ┌────────────┐                 ┌────────────┐
   │ 行动: 调工具 │                 │ 思考: 我要先查天气， │
   │ 观察: 结果   │                 │       因为用户要去玩 │
   │ ...        │                 │ 行动: get_weather   │
   └────────────┘                 │ 观察: 晴，5~12°C    │
                                  │ 思考: 天气不错，接下来…│
                                  └────────────┘
```

思考被写进上下文后有三重收益：

1. **模型更准**：显式推理让下一步决策建立在"说清楚的理由"上（思维链效应）；
2. **轨迹可读**：出问题时逐帧回看"它当时在想什么"，黑盒变白盒；
3. **天然可审计**：每一步都是结构化数据（本章的 `Step`/`Trace`），第 10 章可观测性直接复用。

现代实现中，"思考"就是 assistant 消息的 `content` 字段，"行动"是 `tool_calls` 字段 ——
不需要解析 "Thought:/Action:" 文本协议（那是 Tool Calling 普及前的经典做法，练习题里会回望）。

## 2. 代码走读

### 2.1 示例 01 —— 三段式循环与轨迹记录

文件：[code/ch05_react_loop.py](code/ch05_react_loop.py)

数据结构：`Step`（turn / thought / action / observation / final_answer）与 `Trace`（task / steps / termination）。
循环主体（摘自 `ReActAgent.run`，逐字一致）：

```python
        for turn in range(1, self.max_turns + 1):
            response = self.llm.chat(messages, tools=self.toolbox.schemas())
            if not response.tool_calls:
                # 终止：模型给出最终回答
                trace.steps.append(Step(turn=turn, thought=response.content,
                                        final_answer=response.content))
                trace.termination = f"完成：模型在第 {turn} 步给出最终回答"
                messages.append({"role": "assistant", "content": response.content})
                return trace
            # 行动：执行响应中的全部工具调用（观察合并为一段文本）
            messages.append({
                "role": "assistant",
                "content": response.content,   # ReAct 灵魂：思考显式进入上下文
                "tool_calls": response.tool_calls,
            })
```

与第 1 章循环的差异只有一处实质不同：`messages.append` 时带上了 `content`（思考）——
但这正是全部魔法所在。`Trace.render()` 输出人可读的三段式轨迹（示例输出见运行结果）。

### 2.2 示例 02 —— 轨迹持久化与断点续跑

文件：[code/ch05_trace_replay.py](code/ch05_trace_replay.py)

`trace_to_json` / `trace_from_json` 双向转换（ToolCall 序列化为 `{name, arguments, id}`）；
`messages_from_trace` 把轨迹还原成 LLM 消息序列 —— **续跑时模型看到的上下文与中断前完全一致**。
三个实验：序列化（落盘 JSON）、往返等价性（`restored.render() == trace.render()`）、
断点续跑（环境结果注入，模型接上历史继续推理）。

为什么要续跑而不是重跑？—— 每一步都可能花真金白银的 token；生产系统中进程崩溃、
超时、限流后，从轨迹断点恢复是标准操作（第 10 章追踪、第 12 章部署都会用到）。

### 2.3 示例 03 —— 终止条件：怎么"知道该停"

文件：[code/ch05_termination.py](code/ch05_termination.py)

`GuardedAgent` 支持可插拔终止条件（签名：`轨迹 -> 终止原因或 None`），开闭原则 ——
新条件随时插入，循环主体零改动。本章实现三种：

1. **正常完成**：模型不再请求工具；
2. **重复调用检测**（摘自源码，逐字一致）：

```python
    def check(trace: Trace) -> Optional[str]:
        counts: Counter = Counter()
        for step in trace.steps:
            if step.action is None:
                continue
            key = (step.action.name,
                   json.dumps(step.action.arguments, sort_keys=True, ensure_ascii=False))
            counts[key] += 1
            if counts[key] >= times:
                return f"重复调用检测：{key[0]}({key[1]}) 已累计 {counts[key]} 次，疑似死循环"
        return None
```

   注意 `sort_keys=True` 的必要性：`{"a":1,"b":2}` 与 `{"b":2,"a":1}` 是同一个调用，
   键序不同不该误杀（测试 `test_repeated_call_detector_canonicalizes_args` 钉死了这一点）。
3. **保险丝** `max_turns`：一切条件都不命中时的最后防线。

四个场景实验：正常完成 / 第 3 次重复止损 / 参数不同不误杀 / 永不满足时保险丝兜底。

## 3. 常见坑

1. **思考不进上下文** —— 模型"失忆"于自己刚才的推理，决策质量骤降。`content` 必须随消息append。
2. **重复检测漏了参数 canonical 化** —— 字典键序不同被当成新调用，检测形同虚设。
3. **轨迹不落盘** —— 出问题时无据可查，只能重跑烧钱。
4. **续跑时上下文重建不完整** —— 少了一条历史消息，模型行为漂移。用 `render()` 等价性测试兜底。
5. **终止条件写死在循环里** —— 加新条件要改核心代码，容易引入回归。保持"函数(轨迹)->原因"的插拔签名。
6. **只止损不告知** —— 被终止时 `termination` 原因要回传给上层与用户，否则"为什么停了"永远是悬案。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch05_react_loop.py](code/ch05_react_loop.py) | 示例 01：Step/Trace/ReActAgent 三段式循环 |
| [code/ch05_trace_replay.py](code/ch05_trace_replay.py) | 示例 02：JSON 序列化与断点续跑 |
| [code/ch05_termination.py](code/ch05_termination.py) | 示例 03：可插拔终止条件（重复检测/保险丝） |
| [tests/test_ch05_examples.py](tests/test_ch05_examples.py) | 11 个测试 |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

> `Step`/`Trace` 定义在示例 01 中，示例 02/03 跨文件导入（同章内复用），
> 第 10 章将把它们升级为统一的追踪记录结构。

---

> 下一篇：[第 6 章 · 记忆系统](../06_记忆系统（短期_长期_摘要）/README.md) ——
> 对话越长上下文越贵：滑动窗口、长期记忆、摘要压缩三板斧。
