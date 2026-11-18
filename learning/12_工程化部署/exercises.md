# 第 12 章 · 练习

## 练习 1（热身）：优雅关闭与超时保护

`ThreadingHTTPServer.shutdown()` 会等当前请求处理完。请给 `build_server` 增加
`shutdown_with_timeout(server, timeout=5)`：超时仍未关闭时强制 `server_close()`
并发出警告日志。用线程 + `join(timeout)` 实现。

**验收标准**：
- 正常情况下干净关闭（无警告）；
- 模拟卡住的请求（Handler sleep 超过 timeout）触发强制关闭与警告；
- pytest 用注入的短 timeout 覆盖两条路径。

**参考答案要点**：
- 这是部署的最后一公里：进程要"走得体面"。K8s 的 terminationGracePeriodSeconds
  干的就是这件事。
- 强杀可能丢请求 —— 生产上通常配合 /healthz 摘流量后再关（先摘流量，再排水）。

## 练习 2（进阶）：请求日志中间件

给 ChatHandler 加结构化访问日志：每个请求记录 `{"ts", "method", "path", "status", "duration_ms"}`
到 JSONL（复用第 10 章 TracedLLM 的 JSONL 手法），日志器可注入（测试里指向内存列表）。

**验收标准**：
- /health 与 /chat 的请求都留痕，字段完整；
- 注入自定义日志器后不写文件（零副作用测试）；
- pytest 断言日志条数与关键字段。

**参考答案要点**：
- 日志是服务的另一条"可观测性"生命线（第 10 章三支柱在服务形态的落地）。
- 注意别把请求体原样落日志 —— 里面有用户隐私（第 11 章脱敏思想同样适用于日志）。

## 练习 3（挑战）：多 Worker 并发压测（离线模拟）

实现 `load_test(service, n_requests=50, concurrency=5)`：线程池并发调用 service.run，
统计成功/熔断/拦截计数与 P50/P95 延迟（伪 token 口径计时可用注入时钟模拟）。

**验收标准**：
- 50 个请求全部得到响应（无异常泄漏）；预算熔断的请求数与 BudgetGuard.audit 一致；
- 输出分位数报表；
- pytest 断言三类计数的守恒关系：成功 + 熔断 + 拦截 == 总请求数。

**参考答案要点**：
- 压测前先想清楚"成功"的定义：护栏拦截也算被服务正确处理（200）——
  分桶统计比单一成功率诚实得多（第 10 章评估思想）。
- 分位数排序后取第 50/95 百分位即可，不必引入统计库。
