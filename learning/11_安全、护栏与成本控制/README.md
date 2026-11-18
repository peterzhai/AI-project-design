# 第 11 章 · 安全、护栏与成本控制

> 本章目标：给 Agent 系上安全带 —— 恶意输入进不来（输入护栏）、危险输出出不去
> （输出护栏）、账单失控刹得住（预算熔断）。三者合成一条 GuardedPipeline。
>
> 学习时间：约 2~3 小时。前置要求：第 7 章（引用回溯）、第 10 章（伪 token）。

---

## 1. 护栏的站位：在错误发生之前与之后

```text
   用户输入 ──▶ [输入护栏] ──▶ LLM ──▶ [输出护栏] ──▶ 用户
                │ 拒绝/脱敏           │ 打码/拦截/重试
                │                    │
   预算检查 ────┘（调用前熔断）        └── 入账（消耗累计）
```

三条铁律：

1. **护栏便宜且必跑**：正则级别的护栏每次微秒级，不构成成本；LLM 自审反而费钱（可作为第二层）。
2. **拦截要留痕**：拒绝什么、为什么、命中哪条规则，全部进审计清单 —— 没有审计的护栏是黑箱。
3. **失败要诚实**：拦截后明确告知用户"被拦截"，而不是静默吞掉；预算耗尽明确说"为什么不能继续"。

## 2. 代码走读

### 2.1 示例 01 —— 输入护栏：注入拦截 + 敏感信息脱敏

文件：[code/ch11_input_guardrails.py](code/ch11_input_guardrails.py)

`InputGuardrail` 两种注入模式（实现位于 [common/guards.py](../common/guards.py)，摘自 `check` 核心，逐字一致）：

```python
        injection_hits = [p for p in INJECTION_PATTERNS if re.search(p, text)]
        for pattern in injection_hits:
            findings.append(f"疑似提示注入（命中模式: {pattern}）")
        if injection_hits and self.injection_mode == "reject":
            return GuardResult(allowed=False, reason="检测到提示注入，已拒绝本次请求",
                               findings=findings)
        sanitized = text
        if injection_hits:
            # strip 模式：按句子切分，剥掉命中注入模式的句子，保留正常请求
            sentences = re.split(r"(?<=[。？！])", sanitized)
            kept = [s for s in sentences if not any(re.search(p, s) for p in INJECTION_PATTERNS)]
            sanitized = "".join(kept).strip()
```

- **reject 模式**：整条拒绝，模型零调用机会；
- **strip 模式**：按句剥离注入句，保留正常请求 —— 用户一句话夹带私货时不误杀全句；
- **脱敏顺序有讲究**：先身份证（18 位、更特异）后手机号（11 位）——
  反过来手机号正则会把身份证内嵌的 11 位子串误伤掉（本章开发时真实踩坑，讲义如实记录）。

### 2.2 示例 02 —— 输出护栏：打码、幻觉引用、越权承诺

文件：[code/ch11_output_guardrails.py](code/ch11_output_guardrails.py)（实现位于 [common/guards.py](../common/guards.py)）

`OutputGuardrail` 三道检查叠加：

| 检查 | 动作 | 理由 |
| --- | --- | --- |
| 密钥/手机号出现 | 自动打码（留痕 redactions） | 泄漏不可逆，能打则打 |
| 引用了不存在的块 id | 喂回模型修正（重试 1 次） | 复用第 7 章引用回溯，输出侧闭环 |
| "保证全额退款"类承诺 | **整体拦截**，换安全话术 | 宁可不回话，不许乱承诺 |

拦截与脱敏可以叠加：同一条输出先打码、再因承诺被拦截 —— 审计清单里两项都有记录。

### 2.3 示例 03 —— 预算硬熔断 + 综合流水线

文件：[code/ch11_budget_guard.py](code/ch11_budget_guard.py)

`BudgetGuard` 的纪律（实现位于 [common/guards.py](../common/guards.py)，摘自 `check`，逐字一致）：

```python
    def check(self) -> None:
        remaining = self.max_tokens - self.spent
        if remaining <= 0:
            self.audit.append(f"熔断：预算耗尽（{self.spent}/{self.max_tokens}）")
            raise BudgetExceededError(
                f"预算已耗尽：已用 {self.spent}/{self.max_tokens} 伪 token，"
                "请提高预算或缩短上下文后重试。")
        self.audit.append(f"放行：剩余 {remaining} 伪 token")
```

关键约定：**熔断发生在下一次调用之前**（先 check 再调用、调用完 charge_and_check），
绝不在对话中途掐断 —— 半截对话比超支更难收拾。

`GuardedPipeline` 串起全链路：预算检查 → 输入护栏 → LLM → 入账。
实验证明一个漂亮的组合性质：**被注入拦截的请求零消耗**（模型没被调用、账本不动）——
安全和省钱在这里是同一件事。

## 3. 常见坑

1. **护栏只拦输入不管输出** —— 模型把内部日志/密钥复述出来照样泄漏，输出侧必须设卡。
2. **脱敏正则顺序错误** —— 短模式（手机号）先跑会误伤长模式（身份证）内嵌子串；先长后短。
3. **拦截不留痕** —— 拒绝了多少、为什么拒绝，没有审计就是黑箱，无法改进规则。
4. **熔断发生在对话中途** —— 半截对话状态不一致；熔断永远在"下一次"调用前。
5. **预算只有总额没有明细** —— 超支后不知道钱烧在哪个环节（第 10 章成本归因补上这一环）。
6. **把护栏当摆设** —— 护栏代码存在但主流程没接（"我写了但没调用"）；GuardedPipeline
   把护栏焊进调用路径，避免靠自觉。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch11_input_guardrails.py](code/ch11_input_guardrails.py) | 示例 01：InputGuardrail（reject/strip + 脱敏） |
| [code/ch11_output_guardrails.py](code/ch11_output_guardrails.py) | 示例 02：OutputGuardrail（打码/幻觉引用/承诺拦截） |
| [code/ch11_budget_guard.py](code/ch11_budget_guard.py) | 示例 03：BudgetGuard 熔断 + GuardedPipeline 综合流水线 |
| [tests/test_ch11_examples.py](tests/test_ch11_examples.py) | 19 个测试 |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

> 架构说明：护栏实现统一收进公共层 [common/guards.py](../common/guards.py)
> （第 12 章的 HTTP 服务与 capstone 都直接复用），本章示例文件只保留演示逻辑。

---

> 下一篇：[第 12 章 · 工程化部署](../12_工程化部署/README.md) ——
> 从脚本到服务：配置管理、并发、缓存、健康检查，把 Agent 变成可上线的 API。
