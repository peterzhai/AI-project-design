"""通用重试器：指数退避 + 可注入时钟，离线可测。

LLM 服务常见失败：限流（429）、网关抖动（502/503/504）、瞬时网络错误。
这类"过一会儿就好"的错误值得重试；而 401（密钥错）、400（参数错）重试多少次
都不会好 —— 所以重试前必须先做错误分类。
"""

from __future__ import annotations

import time
from typing import Callable, Sequence


class RetryExhaustedError(RuntimeError):
    """重试预算耗尽。保留最后一次原始异常便于排查。"""


def is_retryable(error: Exception) -> bool:
    """错误分类：网络级失败（无状态码）与 429/5xx 可重试，其余不重试。"""
    status = getattr(error, "status_code", None)
    if status is None:
        return True  # 连接失败/超时等网络级错误
    return status == 429 or status >= 500


def with_retries(
    fn: Callable,
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    sleep_fn: Callable[[float], None] = time.sleep,
    retry_on: Callable[[Exception], bool] = is_retryable,
) -> object:
    """执行 fn()，可重试失败按指数退避重试，非可重试异常立即抛出。

    参数:
        fn:          无参调用函数（需要参数时用 lambda/偏函数包一层）
        max_retries: 首次调用之外的最大重试次数（总尝试 = 1 + max_retries）
        base_delay:  首次重试等待秒数；第 n 次重试等待 base_delay * 2**(n-1)
        sleep_fn:    注入的等待函数；测试传记录器，避免真实 sleep
        retry_on:    判定异常是否可重试的谓词（默认 is_retryable）
    返回:
        fn() 的返回值
    抛出:
        非可重试异常直接上抛；预算耗尽抛 RetryExhaustedError（__cause__ 为原异常）
    """
    last_error: Exception
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not retry_on(exc):
                raise  # 例如鉴权失败：重试没有意义，让调用方立刻知道
            last_error = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** attempt)
            sleep_fn(delay)
    raise RetryExhaustedError(
        f"在 {max_retries + 1} 次尝试后仍失败，最后错误: {last_error!r}"
    ) from last_error
