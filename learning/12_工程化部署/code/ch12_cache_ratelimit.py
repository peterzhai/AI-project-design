"""示例 02：缓存与限速 —— 同样的提问不重复花钱，突发流量不压垮服务。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/12_工程化部署/code/ch12_cache_ratelimit.py

两个部署期优化：
1. LRU 响应缓存：完全相同的消息列表直接复用上次回答 —— 重复问题零 LLM 调用；
   （注意：temperature>0 的生成任务慎用缓存，教程示例是确定性场景）
2. 限速器：滑动窗口限流，超频的请求被拒绝并提示等待时间 —— 保护下游 LLM 服务。
时间源可注入，测试零真实等待。
"""

import json
import sys
import time
from collections import deque
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402


def json_dumps(obj) -> str:
    """稳定序列化（键排序），用作缓存键的一部分。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


class CachedLLM:
    """带 LRU 响应缓存的 LLM 代理：接口不变，命中缓存零调用。"""

    def __init__(self, inner, capacity: int = 128, stats: dict | None = None):
        self.inner = inner
        self.capacity = capacity
        self._cache: dict[tuple, object] = {}   # key -> 完整响应对象（frozen，可安全共享）
        self.stats = stats if stats is not None else {"hits": 0, "misses": 0}

    @staticmethod
    def _key(messages: list[dict], tools) -> tuple:
        """缓存键：消息内容 + 工具清单的稳定序列化。"""
        return (json_dumps(messages), json_dumps(tools or []))

    def chat(self, messages: list[dict], tools=None, **kwargs):
        key = self._key(messages, tools)
        if key in self._cache:                   # 命中：移到"最新"位置并直接返回
            response = self._cache.pop(key)
            self._cache[key] = response
            self.stats["hits"] += 1
            return response
        # 未命中：调用真实 LLM，完整响应入缓存；超容量时淘汰最旧
        response = self.inner.chat(messages, tools=tools, **kwargs)
        self._cache[key] = response
        if len(self._cache) > self.capacity:
            self._cache.pop(next(iter(self._cache)))
        self.stats["misses"] += 1
        return response


class RateLimitExceededError(RuntimeError):
    """触发限速。message 中带建议等待秒数。"""


class RateLimiter:
    """滑动窗口限速器：窗口内最多 max_calls 次。时间源可注入。"""

    def __init__(self, max_calls: int, per_seconds: float, clock=None):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._clock = clock or time.monotonic
        self._timestamps: deque = deque()

    def check(self) -> None:
        now = self._clock()
        # 弹出窗口外的时间戳
        while self._timestamps and now - self._timestamps[0] >= self.per_seconds:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_calls:
            wait = self.per_seconds - (now - self._timestamps[0])
            raise RateLimitExceededError(
                f"请求过于频繁：窗口内已有 {len(self._timestamps)} 次，"
                f"请 {wait:.2f} 秒后重试。")
        self._timestamps.append(now)


def main() -> None:
    banner("示例 02：缓存与限速")

    # 实验 1：缓存命中 —— 相同问题只调一次 LLM
    section("实验 1：LRU 响应缓存")
    inner = MockLLM(handler=lambda m, t: {"content": f"针对「{m[-1]['content']}」的回答"})
    stats: dict = {"hits": 0, "misses": 0}
    cached = CachedLLM(inner, capacity=4, stats=stats)
    question = [{"role": "user", "content": "北京天气怎么样"}]
    answer1 = cached.chat(question).content
    answer2 = cached.chat([{"role": "user", "content": "北京天气怎么样"}]).content
    print(f"第一次: {answer1}")
    print(f"第二次: {answer2}")
    verify("第二次命中缓存（答案一致且模型只被调用一次）",
           answer1 == answer2 and inner.call_count == 1)
    verify("命中/未命中统计正确", stats == {"hits": 1, "misses": 1})

    # 实验 2：不同问题不命中；容量淘汰最旧
    section("实验 2：容量淘汰（LRU）")
    for q in ["上海天气", "广州天气", "深圳天气", "杭州天气"]:
        cached.chat([{"role": "user", "content": q}])
    verify("5 个不同问题全部未命中（inner 被调 5 次）", inner.call_count == 5)
    verify("容量=4，最旧的北京天气已被淘汰",
           "北京天气怎么样" not in json_dumps(list(cached._cache.keys())))
    # 再问北京 -> 未命中（重新调用）
    cached.chat([{"role": "user", "content": "北京天气怎么样"}])
    verify("被淘汰的键再问会重新调用", inner.call_count == 6 and stats["hits"] == 1)

    # 实验 3：限速器 —— 注入时钟，零真实等待
    section("实验 3：滑动窗口限速（可注入时钟）")
    fake_time = {"now": 100.0}
    limiter = RateLimiter(max_calls=3, per_seconds=10.0, clock=lambda: fake_time["now"])
    for i in range(3):
        limiter.check()                       # 前 3 次放行
    # 注意：except 变量只在块内有效，断言写在块内（本教程第三次遇到该坑）
    try:
        limiter.check()
        raise AssertionError("实验 3 失败：第 4 次应当被限速")
    except RateLimitExceededError as exc:
        print(f"第 4 次被限速: {exc}")
        verify("窗口内第 4 次被拒（提示等待时长）", "秒后重试" in str(exc))
    # 时间推进 11 秒 -> 窗口清空 -> 恢复放行
    fake_time["now"] += 11.0
    limiter.check()
    verify("窗口滑过后恢复放行", True)

    # 实验 4：缓存与限速协同 —— 缓存命中不占限速额度
    section("实验 4：组合策略（先查缓存，再过限速）")
    calls_allowed = {"n": 0}
    fake_time2 = {"now": 200.0}
    limiter2 = RateLimiter(max_calls=2, per_seconds=10.0, clock=lambda: fake_time2["now"])
    cached2 = CachedLLM(MockLLM(handler=lambda m, t: {"content": "确定答案"}), capacity=8)
    same_question = [{"role": "user", "content": "固定问题"}]

    def serve(question):
        before_hits = cached2.stats["hits"]
        cached2.chat(question)                # 缓存层
        if cached2.stats["hits"] == before_hits:
            # 未命中 -> 需要真实调用 -> 才过限速闸
            limiter2.check()
            calls_allowed["n"] += 1
        # 命中 -> 零真实调用，不占限速额度

    serve(same_question)                      # miss -> 占 1 个额度
    serve(same_question)                      # hit -> 不占额度
    serve(same_question)                      # hit -> 不占额度
    verify("两次缓存命中未占限速额度（白盒：限速队列只有 1 个时间戳）",
           calls_allowed["n"] == 1 and len(limiter2._timestamps) == 1,
           f"真实调用 {calls_allowed['n']} 次, 时间戳数 {len(limiter2._timestamps)}")

    finish("ch12_cache_ratelimit")


if __name__ == "__main__":
    main()
