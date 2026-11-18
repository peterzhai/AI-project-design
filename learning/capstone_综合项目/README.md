# capstone · 综合项目：客服 Agent 整车装配

> 把第 1~12 章的零件装成一台整车：**电商客服 Agent** —— 政策问答（RAG 带引用）、
> 订单/积分查询（工具循环）、多轮记忆、注入拦截、预算熔断、轨迹追踪。
> 全部离线可跑：`python learning/capstone_综合项目/code/capstone_agent.py`
>
> 前置要求：第 1~12 章全部完成。

---

## 1. 整车架构图

```text
 用户输入
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ 预算熔断（每次 LLM 调用前检查）── common/guards.BudgetGuard   │
│ 输入护栏（注入拦截）──────────── common/guards.InputGuardrail │
│ RAG 检索（低于阈值不注入）────── ch07 切块 + 向量检索           │
│ 上下文组装（system+知识+窗口）── ch06 ContextWindow            │
│ ReAct 循环（思考/行动/观察）──── ch05 循环 + ch04 ToolBox      │
│ 输出护栏（引用回溯/打码/拦截）── common/guards.OutputGuardrail │
│ 记忆沉淀 + 成本逐次入账 ──────── ch06 + ch10 口径              │
│ 轨迹追踪（JSONL 导出）────────── ch10 TracedLLM/RunRecord      │
└────────────────────────────────────────────────────────────┘
    │
    ▼  {"answer", "citations", "blocked", "turns"}
 用户 / HTTP 服务（ch12 的 /chat 即可直接接本类）
```

## 2. 零件来源映射表（讲义 ↔ 代码 ↔ 章节）

| 零件 | 实现位置 | 来源章节 |
| --- | --- | --- |
| `ContextWindow` | [../06_记忆系统（短期_长期_摘要）/code/ch06_sliding_window.py](../06_记忆系统（短期_长期_摘要）/code/ch06_sliding_window.py) | 第 6 章 |
| `chunk_by_paragraph` / `VectorIndex` | [../07_RAG与知识检索/code/](../07_RAG与知识检索/code/ch07_chunking.py) | 第 7 章 |
| `ToolBox`（schema 自动生成 + 三层防线） | [../common/toolbox.py](../common/toolbox.py) | 第 4 章 |
| ReAct 循环（融合版） | [code/capstone_agent.py](code/capstone_agent.py) `CustomerServiceAgent.answer` | 第 5 章 |
| `TracedLLM` / `RunRecord` | [../10_评估、追踪与可观测性/code/ch10_run_records.py](../10_评估、追踪与可观测性/code/ch10_run_records.py) | 第 10 章 |
| `BudgetGuard` / `InputGuardrail` / `OutputGuardrail` | [../common/guards.py](../common/guards.py) | 第 11 章 |
| 预算/追踪统一口径 | `tokens_of`（1 字符 ≈ 1 伪 token） | 第 6、10 章 |

> 唯一没有搬进整车的大脑是真实 LLM —— `make_llm()` 返回 handler Mock。
> 换真实模型只需 `TracedLLM(llm_client.make_client(), ...)`（第 2 章工厂），
> 其余零件一行不动 —— 这就是全教程"面向接口编程"的验收。

## 3. 代码走读：装配的关键接缝

### 3.1 整车主流程（摘自 `CustomerServiceAgent.answer`，逐字一致）

```python
        # ReAct 循环（第 5 章）；预算熔断在每次 LLM 调用前，消耗逐次入账，
        # 与 TracedLLM 的轨迹口径完全一致（第 10 章的成本数字 = 第 11 章的账本）
        output_guard = OutputGuardrail(valid_chunk_ids=self.valid_chunk_ids)
        final_answer = ""
        turns = 0
        for turn in range(1, self.max_turns + 1):
            self.budget.check()                      # 熔断发生在每一次调用前
            response = self.traced.chat(messages, tools=self.toolbox.names)
            self.budget.charge_and_check(tokens_of(messages, response.content))
            turns = turn
```

两个装配要点：

1. **口径统一**：预算逐次入账（而不是 answer 末尾一次入账），保证
   `TracedLLM 累计 == BudgetGuard.spent`（测试 `test_trace_and_budget_ledger_are_consistent`
   钉死这个不变量 —— 第 10 章的报表和第 11 章的账本从此是同一本账）；
2. **熔断语义**：check 在每次 LLM 调用前，绝不中途掐断一次回答。

### 3.2 知识库装配的一个真实坑

多文档入库时 `chunk_by_paragraph` 的块序号每次从 0 重启，三个文档会全部叫
`policy#0`（互相冲突）。capstone 用 `dataclasses.replace(chunk, chunk_id=...)`
全局重编号 —— 知识块 id 的唯一性是引用回溯的前提（第 7 章教训的延伸）。

### 3.3 七个端到端场景

| 场景 | 验证什么 |
| --- | --- |
| 1 政策咨询 | RAG 命中 → 回答带合法引用 `[policy#0]` |
| 2 订单查询 | ReAct 工具循环（turns≥2）→ 回答基于观察 |
| 3 积分查询 + 多轮记忆 | 窗口满载、早轮被挤出可追溯 |
| 4 注入拦截 | 模型零调用、预算零消耗 |
| 5 超纲问题 | 检索低分不注入 → 诚实告知"无法回答" |
| 6 预算熔断 | 第二轮 LLM 调用前止损 |
| 7 轨迹与成本 | JSONL 可解析、账本与轨迹一致 |

## 4. 扩展方向（学以致用）

1. **接真实模型**：`make_llm()` 换成 `llm_client.make_client()`（第 2 章），配上
   `AgentConfig(llm_mode="real")`（第 12 章）—— 提示注入、幻觉、超纲问题的真实行为
   会立刻不同，正好用第 10 章评估集量化。
2. **上 HTTP 服务**：第 12 章的 `ChatHandler` 把 `server.service` 换成
   `CustomerServiceAgent` 的薄包装即可（/chat 返回 `answer` + `remaining_budget`）。
3. **扩展工具**：给 ToolBox 注册 `create_refund`（有副作用的动作）—— 想想第 11 章
   说的"幂等性"问题：重复调用会发生什么？
4. **评估集**：把本 README 的 7 个场景固化为第 10 章的 EvalSuite 案例，
   每次改提示词/换模型后重跑 —— 这就是"评估驱动开发"的完整闭环。

## 5. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/capstone_agent.py](code/capstone_agent.py) | 整车装配 + 7 场景端到端演示（全部 verify 自检） |
| [tests/test_capstone.py](tests/test_capstone.py) | 11 个端到端测试 |
