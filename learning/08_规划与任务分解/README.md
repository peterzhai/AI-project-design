# 第 8 章 · 规划与任务分解

> 本章目标：让 Agent 学会"先画路线图再出发" —— LLM 把目标拆成带依赖的步骤，
> 校验过的计划交给执行器（重试/跳过），执行失败时把观察喂回 Planner 动态重规划。
>
> 学习时间：约 2~3 小时。前置要求：第 3 章（结构化输出）、第 4 章（ToolBox）。

---

## 1. ReAct 与 Planning：走一步看一步 vs 先画路线图

第 5 章的 ReAct 是"走一步看一步"，适合探索型任务；但当任务有明确的结构
（先订场地才能发通知），规划更稳：

```text
   ReAct 模式                       Planning 模式
   思考→行动→观察→思考→…            目标 → [计划: 步骤1←步骤2←步骤3] → 执行
   （每步都要 LLM 决策）             （LLM 决策一次，执行器机械执行）
   优点：灵活                       优点：便宜（省 LLM 调用）、可控、可并行
   缺点：调用多、易跑偏             缺点：计划错了就一路错下去
                                    → 所以需要动态重规划
```

成熟系统的配方：**Planner 出计划 + Executor 机械执行 + 失败时 Re-plan**，
而 ReAct 用于计划之外的不确定环节。

## 2. 代码走读

### 2.1 示例 01 —— 任务分解：计划是结构化输出，必须校验

文件：[code/ch08_task_decomposition.py](code/ch08_task_decomposition.py)

计划 JSON 与第 3 章的 schema 校验同一套路：`PLAN_SCHEMA` 约定
`{goal, steps: [{id, description, depends_on?, tool?, arguments?}]}`，
步骤可携带工具调用（执行器直接执行，见示例 02）。
schema 之外还有**三条结构合法性检查**（摘自 `make_plan`，逐字一致）：

```python
    # 检查 1：id 必须唯一（重复 id 会让依赖指向产生歧义）
    ids = [s.id for s in steps]
    if len(ids) != len(set(ids)):
        raise ValidationError([f"步骤 id 重复: {sorted(ids)}"])
    # 检查 2：依赖必须指向已存在的步骤
    missing = [dep for s in steps for dep in s.depends_on if dep not in set(ids)]
    if missing:
        raise ValidationError([f"依赖引用了不存在的步骤 id: {sorted(set(missing))}"])
    # 检查 3：依赖图必须无环（有环 = 死锁）
    cycle = _find_cycle(steps)
    if cycle:
        raise ValidationError([f"依赖存在环，无法执行: {' -> '.join(map(str, cycle))}"])
```

为什么环检测独立于 schema？—— schema 只能查"每个字段长什么样"，
查不出"字段之间的关系"。`depends_on: [1]` 挂在步骤 1 自己头上（自指环），
每条数据都合法，整体却永远执行不了。示例演示了自环、双步环、重复 id、悬空依赖四种非法计划。

### 2.2 示例 02 —— 计划执行器：重试与失败传播

文件：[code/ch08_plan_executor.py](code/ch08_plan_executor.py)

执行器（摘自 `_run_step`，逐字一致）：

```python
    def _run_step(self, step: PlanStep) -> StepResult:
        """执行单步：调用工具，失败重试至 max_attempts 次。"""
        observation = ""
        for attempt in range(1, self.max_attempts + 1):
            ok, observation = self.toolbox.try_execute(step.tool, step.arguments)
            if ok:
                return StepResult(step_id=step.id, status="done",
                                  output=observation, attempts=attempt)
        return StepResult(step_id=step.id, status="failed",
                          output=observation, attempts=self.max_attempts)
```

三个设计点：
1. `toolbox.try_execute`（本章新增到 common）返回 `(是否成功, 观察文本)` ——
   程序化判定需要布尔标志，不能靠字符串前缀猜；
2. **重试的是"同一参数的同一调用"**，适合瞬时故障（预订系统抖动），不适合业务性失败；
3. **失败传播**：前置 failed/skipped 的步骤自动 skipped —— 不浪费调用、不产生半成品。
   四个场景实验：全成功 / 抖动重试（attempts=2）/ 彻底失败传播 / 多依赖部分失败即跳过。

### 2.3 示例 03 —— 动态重规划：失败不是终点，是新的输入

文件：[code/ch08_replanning.py](code/ch08_replanning.py)

对照实验：计划 A 假设"有直飞航班"（工具直接抛 `ValueError("无直飞航班…")`）。
静态执行 → 步骤 1 failed → 步骤 2 skipped，任务失败；
`run_with_replanning` 则把失败观察喂回 Planner（摘自源码，逐字一致）：

```python
    current = plan
    replans = 0
    while True:
        results = PlanExecutor(toolbox).execute(current)
        if all(r.status == "done" for r in results):
            return current, results, replans
        if replans >= max_replans:
            return current, results, replans
        # 失败观察喂回 Planner，生成新计划
        prompt = replan_context(current, results)
        current = make_plan_with_llm(planner_llm, prompt)
        replans += 1
```

Planner 看到"无直飞航班"后改道高铁（计划 B），一次重规划后全部 done。
对照 3 证明了**预算纪律**：Planner 固执地每次返回同一个坏计划，重规划 2 次后停手、
诚实报告未完成 —— 没有预算的重规划就是死循环（呼应第 5 章终止条件）。

> 工具约定提醒：业务失败（无直飞）要**抛异常**而不是返回"错误：…"文本 ——
> 只有异常才会被 ToolBox 防线 3 捕获并让步骤判定为 failed（本章开发时踩过这个坑）。

## 3. 常见坑

1. **计划不校验就执行** —— 环、悬空依赖会让执行器死循环或 KeyError。
2. **重试无差别套用** —— 瞬时故障该重试；业务失败（无直飞）重试一万次也没用，应触发重规划。
3. **重规划没有预算** —— Planner 固执时就是死循环；必须设 max_replans。
4. **失败后静默继续** —— skipped 是有意的，但最终报告必须如实呈现（示例的执行报告）。
5. **计划步骤没有工具映射** —— description 写得再好，执行器也不知道调哪个函数；
   计划 JSON 里的 tool/arguments 是"计划可执行"的关键。
6. **又见 except 作用域** —— `except ValidationError as exc` 块外不能再用 exc
   （本章示例 1 开发时踩了本教程第二次，特此再记录）。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch08_task_decomposition.py](code/ch08_task_decomposition.py) | 示例 01：Plan/PlanStep + schema + 三条结构检查 + 环检测 |
| [code/ch08_plan_executor.py](code/ch08_plan_executor.py) | 示例 02：PlanExecutor（重试/跳过/执行报告） |
| [code/ch08_replanning.py](code/ch08_replanning.py) | 示例 03：静态 vs 动态重规划对照 |
| [tests/test_ch08_examples.py](tests/test_ch08_examples.py) | 13 个测试 |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

公共层变更：`common/toolbox.py` 新增 `try_execute`（返回成功标志）；`execute` 行为不变。

---

> 下一篇：[第 9 章 · 多 Agent 协作](../09_多Agent协作/README.md) ——
> 一个 Agent 管不过来时：路由器、编排者-执行者、Agent 之间的消息传递。
