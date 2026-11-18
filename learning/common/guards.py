"""护栏与预算基础设施（第 11 章的公共层提升版，capstone 直接复用）。

三类组件：
- InputGuardrail:  输入护栏（提示注入拦截 + 敏感信息脱敏）
- OutputGuardrail: 输出护栏（凭据打码 + 幻觉引用校验 + 越权承诺拦截）
- BudgetGuard / GuardedPipeline: 预算硬熔断与综合护栏流水线

铁律（第 11 章 README）：护栏便宜且必跑；拦截要留痕；失败要诚实。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 提示注入的特征模式（教程示例集，生产要持续扩充并配合分类模型）
INJECTION_PATTERNS = [
    r"忽略(之前|以上|前面)(的)?(所有|全部)?指令",
    r"无视(之前|以上)的?(所有|全部)?规则",
    r"你现在是一个(没有|不受)(任何)?限制的",
    r"请输出你的?(系统提示词|system prompt)",
]

# 未授权承诺的模式（示例集）
COMMITMENT_PATTERNS = [
    r"保证(退款|到账|赔偿|赔付)",
    r"(我们|公司)(承诺|保证).{0,6}(全额|双倍)(退款|赔偿)",
]

CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_#]+)\]")


# =====================================================================
# 输入护栏
# =====================================================================
@dataclass
class GuardResult:
    """输入护栏判定结果。"""

    allowed: bool
    reason: str = ""
    sanitized_text: str = ""   # 脱敏后的安全文本
    findings: list[str] = field(default_factory=list)  # 命中的威胁清单


class InputGuardrail:
    """输入护栏：注入检测 + 敏感信息脱敏。"""

    def __init__(self, injection_mode: str = "reject"):
        """injection_mode: reject=整条拒绝 / strip=剥离注入句后放行。"""
        if injection_mode not in ("reject", "strip"):
            raise ValueError("injection_mode 只能是 reject 或 strip")
        self.injection_mode = injection_mode
        self._phone_pattern = re.compile(r"1[3-9]\d{9}")
        self._id_pattern = re.compile(r"\d{17}[\dXx]")

    def check(self, text: str) -> GuardResult:
        findings: list[str] = []
        # 检查 1：提示注入
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
        # 检查 2：敏感信息脱敏（先长模式后短模式：身份证先于手机号，
        # 否则手机号正则会误伤身份证号内嵌的 11 位子串）
        masked, id_masked = self._id_pattern.subn("110***********0000", sanitized)
        if id_masked:
            findings.append("已脱敏：身份证号")
        sanitized = masked
        masked = self._phone_pattern.sub("138****0000", sanitized)
        if masked != sanitized:
            findings.append("已脱敏：手机号")
        sanitized = masked
        return GuardResult(allowed=True, sanitized_text=sanitized, findings=findings)


# =====================================================================
# 输出护栏
# =====================================================================
@dataclass
class OutputGuardResult:
    """输出护栏判定结果。"""

    text: str                       # 处理后的最终文本
    blocked: bool = False           # 是否整体拦截
    redactions: list[str] = field(default_factory=list)   # 脱敏动作
    violations: list[str] = field(default_factory=list)   # 违规清单


class OutputGuardrail:
    """输出护栏：敏感信息打码 + 幻觉引用校验 + 越权承诺拦截。"""

    def __init__(self, valid_chunk_ids: set[str]):
        self.valid_chunk_ids = valid_chunk_ids
        self._secret_pattern = re.compile(r"(sk-[A-Za-z0-9]{8,}|1[3-9]\d{9})")  # API密钥/手机号

    def check(self, text: str) -> OutputGuardResult:
        violations: list[str] = []
        redactions: list[str] = []
        result_text = text

        # 检查 1：密钥/手机号泄漏 -> 打码
        def mask(match: re.Match) -> str:
            redactions.append("敏感凭据打码")
            return "[已脱敏]"

        result_text = self._secret_pattern.sub(mask, result_text)

        # 检查 2：幻觉引用（引用了不存在的块 id）
        cited = CITATION_PATTERN.findall(result_text)
        fake = [c for c in cited if c not in self.valid_chunk_ids]
        if fake:
            violations.append(f"幻觉引用（不存在的来源）: {fake}")

        # 检查 3：越权承诺 -> 整体拦截（宁可不回话，不许乱承诺）
        for pattern in COMMITMENT_PATTERNS:
            if re.search(pattern, result_text):
                violations.append(f"未授权承诺（命中: {pattern}）")
        if any("未授权承诺" in v for v in violations):
            return OutputGuardResult(
                text="抱歉，我无法就此作出承诺，请您联系人工客服确认。",
                blocked=True, violations=violations, redactions=redactions)

        return OutputGuardResult(text=result_text, redactions=redactions, violations=violations)


# =====================================================================
# 预算熔断与综合流水线
# =====================================================================
class BudgetExceededError(RuntimeError):
    """预算耗尽，调用被熔断。"""


@dataclass
class BudgetGuard:
    """预算护栏：按伪 token 硬熔断。"""

    max_tokens: int
    spent: int = 0
    audit: list[str] = field(default_factory=list)

    def charge_and_check(self, tokens: int) -> None:
        """把上一次调用的消耗入账，并判定下一次是否放行。

        约定：调用前先 check()，调用后 charge_and_check() —— 熔断永远
        发生在"下一次"调用前，绝不在中途掐断（避免半截对话）。
        """
        self.spent += tokens
        self.audit.append(f"入账 {tokens}，累计 {self.spent}/{self.max_tokens}")

    def check(self) -> None:
        remaining = self.max_tokens - self.spent
        if remaining <= 0:
            self.audit.append(f"熔断：预算耗尽（{self.spent}/{self.max_tokens}）")
            raise BudgetExceededError(
                f"预算已耗尽：已用 {self.spent}/{self.max_tokens} 伪 token，"
                "请提高预算或缩短上下文后重试。")
        self.audit.append(f"放行：剩余 {remaining} 伪 token")

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.spent)


def tokens_of(messages: list[dict], output: str) -> int:
    """一次调用的伪 token 消耗（第 10 章口径：1 字符 ≈ 1 伪 token）。"""
    return sum(len(str(m.get("content", ""))) for m in messages) + len(output)


class GuardedPipeline:
    """完整的带护栏调用流水线：输入护栏 -> 预算检查 -> LLM -> 入账。"""

    def __init__(self, llm, guard: InputGuardrail, budget: BudgetGuard):
        self.llm = llm
        self.guard = guard
        self.budget = budget

    def run(self, user_text: str) -> str:
        # 第 1 道闸：预算检查（熔断发生在调用前）
        self.budget.check()
        # 第 2 道闸：输入护栏
        guard_result: GuardResult = self.guard.check(user_text)
        if not guard_result.allowed:
            return "您的请求包含不被允许的内容，已被安全护栏拦截。"
        # 调用 LLM（用脱敏后的文本）
        messages = [{"role": "user", "content": guard_result.sanitized_text}]
        response = self.llm.chat(messages)
        # 入账（输入+输出）
        self.budget.charge_and_check(tokens_of(messages, response.content))
        return response.content
