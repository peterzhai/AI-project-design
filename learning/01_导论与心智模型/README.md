# 第 1 章 · 导论与心智模型

> 本章目标：不写任何"魔法"，用 3 个可离线运行的示例，建立贯穿全教程的心智模型：
> **Agent = LLM（大脑）+ Tools（手脚）+ Loop（心跳）**。
>
> 学习时间：约 1~2 小时。前置要求：会基础 Python（函数、类、字典）。

---

## 1. 什么是 Agent：从"会聊天"到"会做事"

LLM 本身只会一件事：**给定一段文本，预测下一段文本**。它没有手、没有眼睛、没有记忆。
所谓 AI Agent，就是给这个"大脑"配上三样东西，让它能在真实世界里完成任务：

| 配件 | 角色 | 类比 |
| --- | --- | --- |
| Tools（工具） | 让模型能"出手"：查数据、算数字、发请求 | 手脚 |
| Memory（记忆） | 让模型记得"刚才发生了什么" | 工作台 |
| Loop（循环） | 让模型"做完一步看结果，再决定下一步" | 心跳 |

**一句话判断标准**：如果你的系统只是"输入 -> 一次 LLM 调用 -> 输出"，那它是应用，不是 Agent；
如果 LLM 在运行时**自己决定**要不要调工具、调哪个、调几次，那它就是 Agent。

## 2. 心智模型：感知 - 决策 - 行动 - 观察循环

所有 Agent 框架（LangChain、AutoGen、OpenAI Assistants…）剥掉外衣后都是同一个循环：

```text
            ┌─────────────────────────────────────────┐
            │                                         │
            ▼                                         │
   ┌──────────────┐    决策: 调哪个工具？   ┌────────────┐
   │  上下文组装   │ ────────────────────▶ │   LLM 大脑  │
   │ (Memory)     │                       └─────┬──────┘
   └──────▲───────┘                             │
          │                        ┌────────────┴────────────┐
          │                        ▼                         ▼
          │                 要用工具(行动)              不要工具(完成)
          │                        │                         │
          │                        ▼                         ▼
          │                 ┌────────────┐            最终答案返回
          │                 │ ToolBox 执行 │
          └──观察结果写回────└────────────┘
```

这个循环在本教程第 1 章只有十几行，之后每一章都是给它换更强的零件：

| 章节 | 换掉的零件 |
| --- | --- |
| 02 章 | 大脑升级：LLM 调用细节（参数、错误、流式） |
| 03 章 | 大脑升级：让模型输出严格 JSON（结构化输出） |
| 04 章 | 手脚升级：真正的 Tool Calling 协议 |
| 05 章 | 心跳升级：ReAct 显式"思考-行动-观察" |
| 06 章 | 记忆升级：滑动窗口 / 长期记忆 / 摘要压缩 |
| 07 章 | 感知升级：RAG 外部知识检索 |
| 08 章 | 决策升级：任务分解与规划 |
| 09 章 | 单体升级：多 Agent 协作 |
| 10-11 章 | 加上仪表盘（可观测）、安全带（护栏）、计价器（成本） |
| 12 章 + capstone | 整车出厂：工程化部署 + 综合项目 |

## 3. 代码走读：三个递进示例

### 3.1 示例 01 —— 十几行的最小 Agent

文件：[code/ch01_hello_agent.py](code/ch01_hello_agent.py)
运行：`python learning/01_导论与心智模型/code/ch01_hello_agent.py`（离线 Mock，无需 Key）

核心就是下面这个循环（摘自文件中的 `run_minimal_agent`，与源码逐字一致）：

```python
def run_minimal_agent(llm, tools: dict, task: str, max_turns: int = 5) -> str:
    """最小 Agent 循环。

    参数:
        llm:      任何提供 chat(messages, tools) -> ChatResponse 的客户端
        tools:    {工具名: 函数} 字典
        task:     用户的自然语言任务
        max_turns: 最大思考-行动轮数（防止无限循环的保险丝）
    """
    messages = [{"role": "user", "content": task}]
    for _ in range(max_turns):
        response = llm.chat(messages, tools=list(tools))
        if not response.tool_calls:               # 模型不再请求工具 => 已给出最终答案
            return response.content
        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": response.tool_calls,    # 记录模型的"行动决定"
        })
        for call in response.tool_calls:
            result = tools[call.name](**call.arguments)   # 真正执行工具
            messages.append({
                "role": "tool",
                "name": call.name,
                "content": str(result),           # 工具结果作为"观察"回传给模型
            })
    raise RuntimeError(f"超过最大轮数 {max_turns}：模型可能陷入了工具调用循环")
```

逐行理解：

1. `messages = [{"role": "user", ...}]` —— 任务进入记忆，这是模型的全部初始信息。
2. `llm.chat(...)` —— 问大脑："拿着这些上下文和工具清单，你决定怎么办？"
3. `if not response.tool_calls` —— 大脑不再要工具，说明它认为任务完成，循环出口。
4. `result = tools[call.name](**call.arguments)` —— **Agent 和聊天机器人的分水岭**：代码真的去执行了工具。
5. `messages.append({"role": "tool", ...})` —— 观察结果写回上下文，下一轮大脑就能"看到"。
6. `raise RuntimeError(...)` —— 保险丝。真实模型可能陷入"调工具-看结果-再调工具"的死循环，必须设上限。

示例内部跑三组对照实验并自检：带工具任务（2 次 LLM 调用）、无工具任务（1 次调用直接回答）、死循环兜底（`max_turns` 截断）。运行后你会看到全部 ✅。

### 3.2 示例 02 —— 四大组件的工程化拆分

文件：[code/ch01_agent_anatomy.py](code/ch01_agent_anatomy.py)

示例 01 把所有东西写在一个函数里便于理解，但工程上要把零件拆开。本示例定义了三个类：

- `Memory`：按顺序保存消息，`render()` 输出完整上下文。**存的是副本**，外部篡改不影响历史（第 6 章会大幅升级它）。
- `ToolBox`：工具注册表，统一执行入口。设计上有两个关键决策（摘自文件，与源码逐字一致）：

```python
    def execute(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return f"错误：未注册的工具「{name}」，可用工具: {self.names()}"
        try:
            return str(self._tools[name](**arguments))
        except Exception as exc:  # 工具报错 -> 作为观察回传，而非让整个 Agent 崩溃
            return f"工具「{name}」执行失败: {exc}"
```

  > 注意注释里的原则：**工具失败不抛异常崩溃，而是把错误文本作为"观察"回传给模型**，让模型自己决定重试、换路还是向用户道歉。示例里演示了模型点名不存在的工具 `send_email`、以及用 `__import__('os')` 攻击计算工具，两者都被优雅拦截。

- `AgentSkeleton`：用前两者组装出的 Agent 骨架（循环逻辑与示例 01 相同，只是每条消息都写入 `Memory`）。

工具 `calculate` 用 **ast 白名单**解析算术表达式而不是 `eval`（防止提示注入执行任意代码），这个安全习惯在第 11 章会展开。

### 3.3 示例 03 —— 流水线 vs Agent：谁在决定下一步？

文件：[code/ch01_agent_vs_pipeline.py](code/ch01_agent_vs_pipeline.py)

同一个任务"写一首关于秋天的诗"：

- `WeatherPipeline`（步骤写死）：第一步永远查天气，第二步把天气塞进提示词。于是**写诗也查了天气**（示例用 `weather_calls` 计数器实证了这一点）。
- Agent（LLM 决定）：假 LLM 的 handler 模拟了真实模型的判断力 —— 诗题直接创作、0 次工具调用；天气题先查再答。

结论：**流水线便宜、可控、笨；Agent 灵活但需要护栏**。生产系统常见混合形态：Agent 做路由与规划，流水线做固定工序。

## 4. 常见坑

1. **没有 `max_turns` 保险丝** —— 模型反复调同一个工具，账单和延迟都会爆炸。任何循环必须有出口。
2. **工具报错直接抛异常** —— 一次网络抖动就让整个 Agent 崩溃。正确做法是让错误成为"观察"。
3. **工具结果用 `eval` 处理** —— 用户一句"帮我执行 `__import__('os')`"就是远程代码执行。用白名单解析。
4. **把"能用"当"正确"** —— 示例 03 证明：流水线能跑通不代表它跑得合理。评估是第 10 章的主题。
5. **混淆 Mock 与真实** —— 本教程所有示例默认离线 Mock。想连真实模型，设置 `USE_REAL_LLM=1` 和 `OPENAI_API_KEY`（详见根目录 [README](../README.md)）。

## 5. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch01_hello_agent.py](code/ch01_hello_agent.py) | 示例 01：最小 Agent 循环 |
| [code/ch01_agent_anatomy.py](code/ch01_agent_anatomy.py) | 示例 02：Memory / ToolBox / AgentSkeleton 组件化 |
| [code/ch01_agent_vs_pipeline.py](code/ch01_agent_vs_pipeline.py) | 示例 03：流水线 vs Agent 对照实验 |
| [tests/test_ch01_examples.py](tests/test_ch01_examples.py) | 13 个测试：端到端 + 精确行为断言 |
| [exercises.md](exercises.md) | 3 道练习题 + 参考答案要点 |

---

> 下一篇：[第 2 章 · LLM 调用基础](../02_LLM调用基础/README.md) —— 把"大脑"拆开：消息角色、参数、错误重试与流式输出。
