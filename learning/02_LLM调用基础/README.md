# 第 2 章 · LLM 调用基础

> 本章目标：把"调一次大模型"这件看似简单的事做对 —— 消息结构、参数旋钮、
> 错误分类、指数退避重试、流式输出。这是所有后续章节的"大脑接口"。
>
> 学习时间：约 2 小时。前置要求：完成第 1 章。

---

## 1. 调用 LLM 的本质：一个请求，两种形态

无论哪家服务商，接口都收敛成同一种形状：**发一段消息列表，收一段回复**。

```text
┌─────────────────────── 请求 ────────────────────────┐
│ messages: [                                          │
│   {"role": "system",    "content": "你是法语翻译"},   │ ← 定身份（可省略）
│   {"role": "user",      "content": "你好"},          │ ← 用户输入
│   {"role": "assistant", "content": "Bonjour!"},      │ ← 模型历史回答
│   {"role": "user",      "content": "谢谢"}           │ ← 新输入
│ ]                                                    │
│ + 参数: temperature / max_tokens / tools ...          │
└──────────────────────────────────────────────────────┘
                     │                 ▲
                     ▼                 │ 一次性: ChatResponse（全文）
              ┌────────────┐           │ 流式:   StreamChunk × N（增量）
              │  LLM 服务  │ ──────────┘
              └────────────┘
```

三条关键认知：

1. **模型没有记忆**。所谓"多轮对话"，只是每次把历史消息全部重新发一遍。
2. **system 是导演**。同一句 user 输入，换 system 提示词，输出天差地别。
3. **流式只是拆分**。`stream()` 的增量块拼接后必须等于 `chat()` 的全文 —— 这是契约，可测试。

## 2. 代码走读

### 2.1 示例 01 —— 消息角色与参数

文件：[code/ch02_messages_and_params.py](code/ch02_messages_and_params.py)

示例用一个 handler 假 LLM 模拟"有身份感"的模型：读 system 提示词决定行为。

- **实验 1**：带 `你是法语翻译` 的 system → 输出 Bonjour；去掉 system → 原样回应。证明行为由 system 塑造。
- **实验 2**：`temperature` / `max_tokens` 随调用透传，MockLLM 把它们记录进 `calls[n]["params"]`，可审计可断言：

```python
    llm2.chat(
        [{"role": "user", "content": "讲个笑话"}],
        temperature=0.2,   # 低温度：更确定，适合抽取类任务
        max_tokens=100,    # 上限：控制成本
    )
    recorded = llm2.calls[0]["params"]
```

  参数速查：`temperature`（随机性，抽取/工具调用建议 0~0.3，创意写作 0.7+）、
  `max_tokens`（输出上限，直接决定成本）、`stop`（提前截断序列）。
- **实验 3**：多轮对话 = 不断增长的消息列表。第 2 次调用时上下文是
  `[system, user, assistant, user]` 四条 —— 历史回答必须带上，否则模型"失忆"。

### 2.2 示例 02 —— 错误分类与指数退避重试

文件：[code/ch02_errors_retry.py](code/ch02_errors_retry.py)

真实服务一定会抽风。第一步是**给错误分类**（`common/retry.py` 的 `is_retryable`）：

| 错误 | 状态码 | 重试？ | 理由 |
| --- | --- | --- | --- |
| 限流 | 429 | ✅ | 过一会儿配额恢复 |
| 网关抖动 | 500/502/503/504 | ✅ | 瞬时故障 |
| 网络失败/超时 | 无状态码 | ✅ | 可能没发出去 |
| 鉴权失败 | 401/403 | ❌ | 重试一万次也不会好 |
| 参数错误 | 400 | ❌ | 请求本身有问题 |

第二步是**指数退避**（`common/retry.py` 的 `with_retries`，核心循环与源码逐字一致）：

```python
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
```

为什么 `sleep_fn` 可注入？—— 测试与示例传一个"记录器"进去，只记录不真睡，
于是 0.5s → 1s → 2s 的退避序列可以**在 0.04 秒内被断言**，不用真等 3.5 秒。
这是"依赖注入让时间可测试"的经典手法，第 12 章部署时还会用到。

示例的三个实验：429 两次后成功（退避 `[0.5, 1.0]`）；401 立刻上抛（零重试零等待）；
预算耗尽抛 `RetryExhaustedError`（尝试 1+3=4 次）。

### 2.3 示例 03 —— 流式输出与 SSE 协议

文件：[code/ch02_streaming.py](code/ch02_streaming.py)

应用层（摘自示例，与源码逐字一致）：

```python
    collected: list[str] = []
    for chunk in llm.stream([{"role": "user", "content": "北京天气？"}], chunk_size=4):
        print(chunk.content, end="", flush=True)   # 每收一块立刻渲染
        collected.append(chunk.content)
```

协议层：真实响应是 SSE 文本流，形如 `data: {"choices":[{"delta":{...}}]}` 一行行到达，
以 `data: [DONE]` 结束。`common/llm_client.parse_sse_lines` 是处理它的**纯函数**，
不联网也能单测（注释行跳过、`[DONE]` 截断、增量拼接）。

**一致性铁律**：`"".join(stream 增量) == chat().content`。示例 2 的实验 2 就是这条契约的实测。
> 教学诚实声明：`OpenAICompatClient.stream()` 走真实网络，本教程环境无法实测；
> 已离线验证的是 `parse_sse_lines`（协议解析）与 Mock 的 stream 行为。

## 3. 常见坑

1. **不区分错误类型就盲目重试** —— 401 重试 5 次只是把故障时间拉长 5 倍。
2. **重试没有退避** —— 被限流后立刻重试只会被限得更死。
3. **多轮对话丢了 assistant 历史** —— 模型每轮都"失忆"，答非所问。
4. **流式拼接不做一致性校验** —— 中途丢块，前端展示缺字，还以为模型抽风。
5. **忘记 max_tokens** —— 一次失控的生成就烧掉一天预算（第 11 章细讲成本护栏）。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch02_messages_and_params.py](code/ch02_messages_and_params.py) | 示例 01：角色、参数、多轮上下文 |
| [code/ch02_errors_retry.py](code/ch02_errors_retry.py) | 示例 02：错误分类 + 指数退避（FlakyClient 模拟） |
| [code/ch02_streaming.py](code/ch02_streaming.py) | 示例 03：流式输出 + SSE 解析 |
| [tests/test_ch02_examples.py](tests/test_ch02_examples.py) | 10 个测试 |
| [exercises.md](exercises.md) | 3 道练习题 + 参考答案要点 |

公共层新组件：`common/mock_llm.StreamChunk`（流式增量）、`common/llm_client.parse_sse_lines`（SSE 纯函数）、
`common/retry.py`（重试器与错误分类）。

---

> 下一篇：[第 3 章 · Prompt 与结构化输出](../03_Prompt与结构化输出/README.md) ——
> 让模型不只会"说话"，还会输出能被程序直接消费的 JSON。
