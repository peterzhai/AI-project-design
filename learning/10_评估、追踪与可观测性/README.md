# 第 10 章 · 评估、追踪与可观测性

> 本章目标：Agent 不再是"黑盒玄学" —— 每次调用有记录（追踪）、每次改版有跑分（评估）、
> 每一分消耗有归属（成本归因）。这也是调试多 Agent 系统（第 9 章）的唯一有效手段。
>
> 学习时间：约 2~3 小时。前置要求：第 5 章（Trace）、第 6 章（伪 token 口径）。

---

## 1. 可观测性三支柱

```text
   ┌─ 追踪（Logging/Tracing）─  发生了什么？每次调用的输入/输出/耗时/token
   ├─ 评估（Evaluation）─      做得好不好？数据集跑分、失败逐条归因
   └─ 成本（Cost Attribution）─ 花了多少？按 Agent/步骤分摊账单
        三者共用同一份底料：调用记录（CallRecord）
```

本章全部基于一个统一的伪 token 口径（1 字符 ≈ 1 伪 token，第 6 章约定），
离线确定性；真实系统换算成价格即可，结构完全相同。

## 2. 代码走读

### 2.1 示例 01 —— TracedLLM：行车记录仪

文件：[code/ch10_run_records.py](code/ch10_run_records.py)

`TracedLLM` 包装任意 LLM 客户端 —— **接口不变，上层零改动**（第 2 章统一接口的回报）。
记录逻辑（摘自 `chat`，逐字一致）：

```python
    def chat(self, messages: list[dict], tools=None, **kwargs):
        start = self._clock()
        response = self.inner.chat(messages, tools=tools, **kwargs)
        duration_ms = (self._clock() - start) * 1000
        last_message = str(messages[-1].get("content", "")) if messages else ""
        self.records.append(CallRecord(
            seq=len(self.records) + 1,
            prompt_chars=sum(len(str(m.get("content", ""))) for m in messages),
            output_chars=len(response.content),
            duration_ms=duration_ms,
            prompt_preview=last_message[:50],
            output_preview=response.content[:50],
        ))
        return response
```

三个细节值得注意：
- `_clock` 可注入 —— 测试里用迭代器模拟"耗时 1 秒"，断言毫秒精度而**零真实等待**；
- 记录的是**字符数**与**预览**（截断 50），落盘体积可控；
- `RunRecord.to_jsonl()` 导出每行一条 JSON（带 run_id/agent），离线可分析、可回放。

### 2.2 示例 02 —— 评估器：跑分与失败归因

文件：[code/ch10_evaluators.py](code/ch10_evaluators.py)

评估三板斧（摘自 `evaluate_one`，逐字一致）：

```python
    def evaluate_one(self, case: EvalCase) -> EvalResult:
        answer = self.agent(case.question)
        failures: list[str] = []
        for keyword in case.must_contain:
            if keyword not in answer:
                failures.append(f"缺少关键信息「{keyword}」")
        for forbidden in case.must_not_contain:
            if forbidden in answer:
                failures.append(f"出现禁用内容「{forbidden}」（疑似编造）")
        if case.pattern and not re.search(case.pattern, answer):
            failures.append(f"未匹配格式要求: {case.pattern}")
        return EvalResult(case_name=case.name, passed=not failures,
                          answer=answer, failures=failures)
```

- **must_contain** 防漏（关键信息必须给）、**must_not_contain** 防编造、
  **pattern** 防格式漂移；
- 失败必须**逐条给出原因**（不是一句"不通过"）—— 修提示词才知道往哪修；
- 示例演示了完整的"评估驱动开发"闭环：v1 通过率 2/3（兜底话术不含"不确定"被抓）→
  修复 → v2 3/3。**评估集就是 Agent 的回归测试**。

> 断言式评估器是起点；模糊任务的评分（如"回答是否礼貌"）才需要 LLM 裁判 ——
> 那是生产话题，本教程聚焦可确定性验证的部分。

### 2.3 示例 03 —— 成本归因：账单摊到人头

文件：[code/ch10_cost_report.py](code/ch10_cost_report.py)

`cost_report` 把多个 RunRecord 的消耗按 Agent 聚合、降序排列、画出占比条。
两个实证：

1. 一次三 Agent 协作中，**持长上下文的 Planner 是最贵环节**；
2. 上下文瘦身（长背景 → 短背景）后总消耗下降超过 50% —— 优化从"感觉"变成"数字"。

最后一环预告第 11 章：`run_after.total_pseudo_tokens <= budget` ——
成本报表的数字正是护栏的输入。

## 3. 常见坑

1. **只记输出不记输入** —— 模型答错往往因为输入就错了（上下文组装 bug），输入必须落盘。
2. **评估集太小或太偏** —— 5 个案例的"通过率 100%"说明不了问题；案例要覆盖正常/边界/刁钻。
3. **失败不归因** —— 只报通过率不报原因，等于没评；逐条失败原因才可行动。
4. **成本不看明细只看总数** —— 总数不涨不代表健康，可能是最贵的环节在悄悄翻倍。
5. **追踪影响主流程** —— TracedLLM 必须透明（异常原样上抛、返回原样透传）；记录器坏了不该弄垮业务。
6. **伪 token 与真实 token 混用** —— 教程统一 1 字符 ≈ 1 伪 token；真实计费另算，别混在一张报表里。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch10_run_records.py](code/ch10_run_records.py) | 示例 01：CallRecord/TracedLLM/RunRecord + JSONL 导出 |
| [code/ch10_evaluators.py](code/ch10_evaluators.py) | 示例 02：EvalCase/EvalSuite + 评估驱动开发闭环 |
| [code/ch10_cost_report.py](code/ch10_cost_report.py) | 示例 03：成本报表 + 优化前后对比 |
| [tests/test_ch10_examples.py](tests/test_ch10_examples.py) | 14 个测试 |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

---

> 下一篇：[第 11 章 · 安全、护栏与成本控制](../11_安全、护栏与成本控制/README.md) ——
> 给 Agent 系上安全带：输入护栏、输出护栏、循环检测、预算硬熔断。
