# 第 2 章 · 练习

## 练习 1（热身）：给假 LLM 加"温度人格"

改造 `make_translator_llm`：handler 读取 `llm.calls[-1]["params"]` 里无法直接拿到
temperature（Mock 不模拟采样），改为**用参数路由行为**的变体 —— 写一个
`make_echo_llm()`，当调用参数里带 `style="formal"` 时返回 `[正式] …`，
带 `style="casual"` 时返回 `[随意] …`，无参数返回 `[默认] …`。

**验收标准**：
- 三种参数组合各有对应输出前缀；
- `pytest` 断言三种情况（提示：参数会记录在 `llm.calls[n]["params"]`，但 handler
  拿不到 kwargs —— 所以要用 `chat(..., style=...)` 时在 handler 外断言，或给
  handler 传闭包变量）。

**参考答案要点**：
- 最简做法：不用 handler 读参数，而是在**调用侧**把 style 拼进 system 提示词 ——
  这正是真实工程的做法：参数层面只有 temperature 等，"风格"永远走提示词。
- 陷阱预警：如果尝试让 handler 依赖 kwargs，会发现 handler 签名只有
  `(messages, tools)` —— 这是刻意设计：**模型在生成时感知不到采样参数**。

## 练习 2（进阶）：抖动退避（Jitter）

真实系统在指数退避上加随机抖动（jitter），避免成千上万个客户端同步重试造成
"重试风暴"。请给 `with_retries` 写一个包装版 `with_jitter_retries`：
`delay = base_delay * 2**attempt`，再乘以 `random.uniform(0.5, 1.5)`。

**验收标准**：
- 注入固定种子的 `random.Random(42).uniform` 作为随机源，断言退避序列可复现；
- 多个"客户端"（不同种子）的退避序列互不相同。

**参考答案要点**：
- 依赖注入是关键：`rng: random.Random` 作为参数传入，测试给固定种子。
- 这同时是第 10 章"可复现性"的前置练习：**随机性必须可注入、可种子化**。

## 练习 3（挑战）：带"费用感知"的调用包装器

写 `BudgetedLLM` 包装类：包住任意客户端，构造时给定 `max_chars`（本教程用字符数
近似 token 数，不做真实计费）；每次调用后累计本次输出字符数；超预算时抛
`BudgetExceededError`（自定义异常），并让该错误**不可重试**。

**验收标准**：
- 正常调用透传结果，累计值正确；
- 超预算时立刻抛错，不再发起真实调用；
- 与 `with_retries` 组合时，`BudgetExceededError` 不会被重试
  （提示：用 `retry_on=lambda e: not isinstance(e, BudgetExceededError)`）。

**参考答案要点**：
- 包装器模式（decorator pattern）：实现与被包装者相同的 `chat()` 签名即可无缝替换
  —— 这正是 `make_client` 工厂 + 统一接口设计的直接收益。
- 累计状态放实例属性；"先检查后调用"还是"调用后检查"取决于语义：预算用于
  **防止下一次**调用，所以应在发起调用前检查累计值。
- 这个练习是第 11 章"成本护栏"的雏形。
