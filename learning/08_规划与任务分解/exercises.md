# 第 8 章 · 练习

## 练习 1（热身）：并行执行无依赖步骤

`PlanExecutor` 目前严格按列表顺序执行。请实现 `ParallelPlanExecutor`：
把"当前所有依赖已满足"的步骤并发执行（教程用 `concurrent.futures.ThreadPoolExecutor`
即可，注意工具要线程安全或无共享状态）。

**验收标准**：
- 三个无依赖步骤的完成顺序可能交错，但 StepResult 列表仍按 step_id 排序返回；
- 有依赖的步骤等待全部前置 done；
- pytest 用计数器断言并行度 > 1（两个步骤的执行区间有重叠）。

**参考答案要点**：
- 拓扑分层：同一层并发，层间串行（这是最简单的正确并行）。
- 线程安全：`audit` 列表 append 是原子的，但真实工具未必 —— 第 12 章部署会展开。

## 练习 2（进阶）：带预估成本的计划

给 PlanStep 增加 `estimated_cost: int`（伪 token 预估）。实现 `check_budget(plan, max_cost)`：
- 总预估超预算时抛 ValidationError，错误信息列出各步骤成本与总额；
- 执行器执行时累计"实际成本"（每次工具调用 +100 伪 token），超预算时后续步骤 skipped，
  状态为 `skipped`，output 注明"预算耗尽"。

**验收标准**：
- 计划阶段拦截超预算计划；
- 执行阶段预算耗尽时后续 skipped，且报告诚实标注；
- pytest 覆盖两级拦截。

**参考答案要点**：
- 计划期拦截（便宜）优先于执行期拦截（贵）—— 与第 11 章成本护栏的"事前预算"一致。
- estimated_cost 也是 Planner 产出的一部分，schema 加字段即可（复用第 3 章套路）。

## 练习 3（挑战）：计划与 ReAct 的混合执行

实现 `HybridAgent`：确定性环节走计划（示例 02 的执行器），计划里标记
`"tool": null` 的步骤表示"交给 ReAct 子 Agent 处理"。子 Agent 用第 5 章的
ReActAgent 完成该步骤（自由探索），产出结果填回 StepResult。

**验收标准**：
- 混合计划执行：纯工具步骤 done、ReAct 步骤也 done（用 Mock 验证子循环发生了）；
- ReAct 子步骤失败按同一套 failed/skipped 传播；
- pytest 断言子 Agent 的 LLM 调用次数（证明确实走了子循环）。

**参考答案要点**：
- 这是"规划器 + 执行器 + 子 Agent"的层次化架构（hierarchical agent）原型。
- 关键接口：子 Agent 的产出要收敛成字符串 output，才能填进统一的 StepResult ——
  统一数据结构是混合系统可维护的前提。
- 权衡讨论：什么时候值得动用子 Agent？（不确定性高、需要多轮工具选择的环节）
