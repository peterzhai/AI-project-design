"""示例 02：生产级必备 —— 错误分类、超时与指数退避重试。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/02_LLM调用基础/code/ch02_errors_retry.py

核心思想：
- LLM 服务会限流（429）、网关抖动（5xx）、瞬时断网 —— 这些"过一会儿就好"的错误要重试；
- 401/400 这类"重试也不会好"的错误要立刻上抛；
- 退避节奏用指数增长（0.5s -> 1s -> 2s...），避免在服务最痛的时候继续踩。
重试器支持注入 sleep，因此本示例零真实等待、秒级跑完。
"""

import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.llm_client import LLMError  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402
from common.retry import RetryExhaustedError, with_retries  # noqa: E402


# ---------------------------------------------------------------
# 教学用"抽风客户端"：前 fail_times 次抛指定状态码错误，之后转发后备 LLM。
# 模拟真实世界：限流几次后恢复，或密钥配置错误（一直 401）。
# ---------------------------------------------------------------
class FlakyClient:
    def __init__(self, fail_times: int, status_code: int, fallback: MockLLM):
        self.fail_times = fail_times
        self.status_code = status_code
        self.fallback = fallback
        self.calls = 0

    def chat(self, messages: list[dict], tools=None, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise LLMError(f"LLM HTTP {self.status_code}（模拟）", status_code=self.status_code)
        return self.fallback.chat(messages, tools, **kwargs)


def main() -> None:
    banner("示例 02：错误分类与指数退避重试")

    # 实验 1：限流 429 两次 -> 指数退避后成功
    section("实验 1：429 限流两次，重试后成功（退避序列 0.5s -> 1s）")
    flaky = FlakyClient(fail_times=2, status_code=429, fallback=MockLLM(scripted=["重试成功了！"]))
    sleeps: list[float] = []  # 注入的"时钟"：只记录不真睡，示例秒级跑完
    answer = with_retries(
        lambda: flaky.chat([{"role": "user", "content": "你好"}]),
        max_retries=3,
        base_delay=0.5,
        sleep_fn=sleeps.append,
    )
    print(f"结果: {answer.content}；尝试 {flaky.calls} 次，退避节奏 {sleeps}")
    verify("共尝试 3 次（2 失败 + 1 成功）", flaky.calls == 3, f"实际 {flaky.calls}")
    verify("指数退避序列正确", sleeps == [0.5, 1.0], f"实际 {sleeps}")

    # 实验 2：401 密钥错误 -> 不重试，立刻上抛
    section("实验 2：401 鉴权失败，重试无意义，立刻失败")
    auth_fail = FlakyClient(fail_times=99, status_code=401, fallback=MockLLM(scripted=["不会到这"]))
    sleeps2: list[float] = []
    try:
        with_retries(
            lambda: auth_fail.chat([{"role": "user", "content": "你好"}]),
            max_retries=3,
            base_delay=0.5,
            sleep_fn=sleeps2.append,
        )
        raise AssertionError("实验 2 失败：401 应当立刻上抛")
    except LLMError as exc:
        print(f"按预期立刻失败: {exc}")
    verify("401 只尝试了 1 次（不浪费重试预算）", auth_fail.calls == 1, f"实际 {auth_fail.calls}")
    verify("零等待（没有进入退避）", sleeps2 == [])

    # 实验 3：一直 429 -> 重试预算耗尽，抛 RetryExhaustedError
    section("实验 3：重试预算耗尽（3 次重试全失败）")
    always = FlakyClient(fail_times=99, status_code=429, fallback=MockLLM(scripted=["不会到这"]))
    sleeps3: list[float] = []
    try:
        with_retries(
            lambda: always.chat([{"role": "user", "content": "你好"}]),
            max_retries=3,
            base_delay=0.5,
            sleep_fn=sleeps3.append,
        )
        raise AssertionError("实验 3 失败：应当抛 RetryExhaustedError")
    except RetryExhaustedError as exc:
        print(f"按预期预算耗尽: {exc}")
    verify("总尝试 1+3=4 次", always.calls == 4, f"实际 {always.calls}")
    verify("退避节奏 0.5 -> 1 -> 2（指数增长）", sleeps3 == [0.5, 1.0, 2.0], f"实际 {sleeps3}")

    print("\n要点回顾：可重试（429/5xx/网络）才重试；退避指数增长；预算耗尽要显式失败。")

    finish("ch02_errors_retry")


if __name__ == "__main__":
    main()
