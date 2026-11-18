# 第 12 章 · 工程化部署

> 本章目标：把 Agent 从"脚本"变成"服务" —— 配置收敛与 fail-fast、缓存与限速、
> 纯标准库 HTTP 服务（离线可启动、可探活、可对话）。
>
> 学习时间：约 2~3 小时。前置要求：第 11 章（GuardedPipeline）。

---

## 1. 从脚本到服务：多出来的四件事

```text
   脚本形态                     服务形态
   python agent.py 跑完即止      常驻进程，7x24 接客
   配置写死在代码里              配置来自环境变量，启动即校验
   单次调用                      并发请求 + 限速 + 缓存
   print 调试                    /health 探活 + 结构化错误 + 预算熔断
```

教程实现全部用标准库（`http.server`/`dataclasses`/`collections`），零新增依赖；
每层的"生产级替代品"在文中标注（配置中心/redis 缓存/网关限流/fastapi），思路一致。

## 2. 代码走读

### 2.1 示例 01 —— 配置与环境管理

文件：[code/ch12_config.py](code/ch12_config.py)

`AgentConfig` 三条纪律（摘自 `__post_init__` 校验核心，逐字一致）：

```python
        # 校验 3：真实模式必须有密钥；mock 模式不需要
        if self.llm_mode == "real" and not self.api_key:
            raise ConfigError("llm_mode=real 但缺少 api_key（禁止把密钥写进代码，请用环境变量）")
        # 校验 4：base_url 必须是 http(s)
        if not self.base_url.startswith(("http://", "https://")):
            raise ConfigError(f"base_url 非法: {self.base_url!r}")
```

- **来源分层**：环境变量 > 显式传入 > 安全默认值；`from_env(env=...)` 可注入环境字典，
  测试零污染（不会误读你机器上的真实环境变量）；
- **fail-fast**：非法配置（枚举错、范围越界、real 模式缺密钥）在启动时炸掉，
  而不是运行到第 38 个请求才出诡异问题；
- **密钥打码**：`__repr__` 把 api_key 显示为 `***(26 chars)` —— `print(config)` 永不泄漏。
  示例 6 组非法配置逐一演示"启动即炸"。

### 2.2 示例 02 —— 缓存与限速：省钱与保命

文件：[code/ch12_cache_ratelimit.py](code/ch12_cache_ratelimit.py)

`CachedLLM` 是 LRU 代理（摘自 `chat`，逐字一致）：

```python
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
```

`RateLimiter` 滑动窗口限速，时间源可注入 —— 测试用假时钟推 11 秒验证"窗口滑过恢复放行"，
零真实等待。实验 4 是本章的点睛组合：**缓存命中不占限速额度** ——
缓存层挡住的请求不需要真实调用，限速只保护真正要花钱的部分。

### 2.3 示例 03 —— HTTP 服务：薄协议层 + 厚业务层

文件：[code/ch12_http_server.py](code/ch12_http_server.py)

端点与错误语义：

| 端点 | 语义 |
| --- | --- |
| `GET /health` | 探活，部署平台靠它判断进程存活 |
| `POST /chat` | 对话入口；200 正常 / 400 格式错 / 429 预算熔断 / 500 兜底 |

架构关键（摘自 `ChatHandler.do_POST` 业务分派，逐字一致）：

```python
        # 业务层（护栏 + 预算 + LLM）
        try:
            answer = self.server.service.run(message)
            self._respond(200, {"answer": answer,
                                "remaining_budget": self.server.service.budget.remaining})
        except BudgetExceededError as exc:
            self._respond(429, {"error": str(exc)})   # 预算熔断 = 限流语义
        except Exception as exc:                       # 兜底：任何异常不裸奔给客户端
            self._respond(500, {"error": f"内部错误: {exc}"})
```

- **Handler 无状态**：业务对象 `GuardedPipeline` 由 `build_server` 注入 ——
  测试与部署共用同一条构建路径（`test_chat_endpoint_roundtrip` 就是进程内起服务的集成测试）；
- **熔断语义与 HTTP 对齐**：预算耗尽返回 429（限流语义），且发生在"下一次"调用前 ——
  示例演示了大请求正常返回、下一次 429 的完整过程；
- **端口 0**：随机空闲端口，教学脚本无冲突；示例跑完自动 shutdown。

> 生产边界：http.server 单机、无 TLS、并发能力有限。生产环境换 fastapi/uvicorn
> + gunicorn，但"协议层薄、业务层厚、错误语义明确"这三条原样适用。

## 3. 常见坑

1. **密钥进代码或进日志** —— AgentConfig 强制环境变量 + repr 打码；`.gitignore` 排除 `.env`。
2. **配置错误运行时才炸** —— 半夜三点的诡异故障十有八九是配置；启动时校验一切。
3. **缓存不加键空间隔离** —— messages 里混入时间戳/随机数会让命中率归零；键只含语义内容。
4. **限速器用真实 sleep 测试** —— 时间必须可注入（第 2 章就埋下的习惯）。
5. **Handler 里写业务逻辑** —— 协议层厚了就无法单测业务；Handler 只做协议转换。
6. **熔断掐断进行中的对话** —— 半截对话状态不一致；熔断永远在"下一次"调用前。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch12_config.py](code/ch12_config.py) | 示例 01：AgentConfig（来源分层/fail-fast/密钥打码） |
| [code/ch12_cache_ratelimit.py](code/ch12_cache_ratelimit.py) | 示例 02：CachedLLM（LRU）+ RateLimiter（滑动窗口） |
| [code/ch12_http_server.py](code/ch12_http_server.py) | 示例 03：/health + /chat 服务（离线自测后自动关闭） |
| [tests/test_ch12_examples.py](tests/test_ch12_examples.py) | 23 个测试（含 HTTP 服务进程内集成测试） |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

公共层新组件：`common/guards.py`（第 11 章护栏的提升版，本章直接复用）。

---

> 下一篇：[capstone · 综合项目](../capstone_综合项目/README.md) ——
> 把 1~12 章的零件装成一台整车：客服 Agent（RAG + 工具 + 记忆 + 护栏 + 追踪 + 预算），
> 端到端离线可跑、测试全绿。
