# 第 9 章 · 多 Agent 协作

> 本章目标：一个 Agent 管不过来时怎么办 —— 路由器（转给专职的人）、
> 编排者-执行者（拆给多人再汇总）、消息传递与黑板（协作显式化）。
>
> 学习时间：约 2~3 小时。前置要求：第 4、5、8 章。

---

## 1. 什么时候需要多 Agent：分工的收益与代价

单 Agent 痛点：工具一多（20+），描述挤爆上下文、模型选择准确率下降、一个提示词背负所有职责。
多 Agent = 把职责拆给专职的"人"：

```text
   路由器模式                编排者-执行者模式            消息传递/黑板
   用户 ──▶ Router ──▶ 专员A     Orchestrator ─┬─ Worker1    AgentA ──消息──▶ AgentB
                      ──▶ 专员B                    ├─ Worker2       │
                      ──▶ 专员C                    └─ Worker3   Blackboard（共享板）
   （转给一个人）             （拆给多人，汇总）           （协作内容显式落盘）
```

代价也要清醒：每次协作都是真实 LLM 调用（钱与延迟）、错误会在 Agent 间传播、
调试难度上升 —— 所以第 10 章的可观测性对多 Agent 系统是刚需。

## 2. 代码走读

### 2.1 示例 01 —— 路由器：找对人，再办事

文件：[code/ch09_router.py](code/ch09_router.py)

`SpecialistAgent` = 名字 + 描述（给 Router 看）+ 自己的 LLM。
`Router` 提供两种路由（摘自 `route_by_llm`，逐字一致）：

```python
    def route_by_llm(self, task: str) -> SpecialistAgent:
        """LLM 路由：把任务与各 Agent 描述一起给模型，让它选人。"""
        roster = "\n".join(f"- {a.name}: {a.description}" for a in self.agents.values())
        response = self.router_llm.chat([
            {"role": "system", "content": f"你是任务路由器。根据用户请求选择一个最合适的处理人，只输出其名字。\n候选人：\n{roster}"},
            {"role": "user", "content": task},
        ])
        name = response.content.strip()
        if name not in self.agents:
            raise KeyError(f"Router 返回了未知处理人: {name!r}")
        return self.agents[name]
```

关键细节：Router 返回未知处理人时**显式报错**而不是兜底乱派 —— 路由层的错误要大声失败。
`route_by_rules` 用关键词零成本兜底（生产配方：规则先试、LLM 收尾）。
实验用 handled 审计列表证明：三类请求各自找到了对的专员，互不串台。

### 2.2 示例 02 —— 编排者-执行者：拆、派、收、汇

文件：[code/ch09_orchestrator.py](code/ch09_orchestrator.py)

`Orchestrator` 的完整编排（摘自 `run`，逐字一致）：

```python
    def run(self, task: str, workers: dict[str, Worker]) -> tuple[str, list[SubTask]]:
        """完整编排：拆 -> 派 -> 收 -> 汇。"""
        subtasks = self.plan(task)
        for sub in subtasks:
            if sub.worker_name not in workers:
                sub.output = f"（未找到执行者 {sub.worker_name}，该子任务被跳过）"
                continue
            sub.output = workers[sub.worker_name].work(sub.instruction)
        return self.synthesize(task, subtasks), subtasks
```

设计要点：
- **Planner 与 Synthesizer 是两个独立 LLM** —— 职责分离，各自可测可换；
- 子任务"指令自包含"（不依赖其他子任务产出）→ 三个 Worker 天然可并行；
- 未知 Worker 的子任务**安全跳过**（写明原因到 output），编排不崩溃；
- 汇总输入是结构化的汇报文本（`【执行者】产出`），模型汇总时对得上号。

### 2.3 示例 03 —— 消息传递与黑板：协作要显式落盘

文件：[code/ch09_message_passing.py](code/ch09_message_passing.py)

两个协作原语：
- `MessageBus`：点对点投递（收件人队列 + 全量审计日志）；
- `Blackboard`：key -> (写入者, 内容)，后写覆盖。

协作流水线：数据分析员把结论写到黑板、给撰写员发通知；撰写员收到通知后**从黑板读取**结论写报告。
核心原则（示例标题里那句话）：

> **协作内容必须显式落盘成数据，而不是藏在各自上下文里靠猜。**

隐式协作（"我以为你知道"）是多 Agent 系统最难查的 bug —— 显式协议让每一步可回放、可审计。

## 3. 常见坑

1. **一上来就多 Agent** —— 单 Agent 能解决就别拆；拆分的收益必须大于调用成本与调试成本。
2. **Router 输出不校验** —— 模型输出了不存在的名字就 KeyError 崩溃是好事（显式失败），
   静默兜底到随机 Agent 才是灾难。
3. **子任务互相依赖却并行派发** —— "指令自包含"是并行的前提；有依赖的用第 8 章计划图。
4. **协作靠隐式约定** —— 不落盘的协作不可审计、不可回放、不可测试。
5. **汇总阶段丢来源** —— 汇总提示词里保留 `【执行者】产出` 的结构，答案才能追溯到人。
6. **忘了每次协作都是钱** —— 三层编排 = 拆 1 次 + 执行 N 次 + 汇 1 次，预算要提前算（第 11 章）。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch09_router.py](code/ch09_router.py) | 示例 01：Router（LLM 路由 + 规则兜底）+ 三个专职 Agent |
| [code/ch09_orchestrator.py](code/ch09_orchestrator.py) | 示例 02：Orchestrator（拆/派/收/汇）+ 三个调研 Worker |
| [code/ch09_message_passing.py](code/ch09_message_passing.py) | 示例 03：MessageBus + Blackboard 协作流水线 |
| [tests/test_ch09_examples.py](tests/test_ch09_examples.py) | 11 个测试 |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

---

> 下一篇：[第 10 章 · 评估、追踪与可观测性](../10_评估、追踪与可观测性/README.md) ——
> Agent 产出的质量如何量化？每次运行的轨迹如何回看？成本如何归因到每一步？
