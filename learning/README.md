# AI Agent 系统学习教程（learning/）

> 一套**可运行、可复现、循序渐进**的 AI Agent 工程教程。
> 所有示例默认**离线 Mock 模式**：无网络、无 API Key 也能跑通；一行环境变量切换到真实模型。

## 你将学到什么

按"换零件"的路径逐步搭出一台完整的 Agent：

| 章节 | 主题 | 你会实现的核心物 |
| --- | --- | --- |
| [01](01_导论与心智模型/README.md) | 导论与心智模型 | 十几行的最小 Agent 循环、四大组件、流水线 vs Agent |
| [02](02_LLM调用基础/README.md) | LLM 调用基础 | 统一客户端、参数、错误重试、流式输出 |
| [03](03_Prompt与结构化输出/README.md) | Prompt 与结构化输出 | 系统提示词工程、严格 JSON 输出与校验 |
| [04](04_Tool_Calling与函数调用/README.md) | Tool Calling 与函数调用 | JSON Schema 工具注册、真实 Tool-Calling 协议 |
| [05](05_ReAct循环/README.md) | ReAct 循环 | 显式"思考-行动-观察"循环 |
| [06](06_记忆系统（短期_长期_摘要）/README.md) | 记忆系统 | 滑动窗口、长期存储、摘要压缩 |
| [07](07_RAG与知识检索/README.md) | RAG 与知识检索 | 切块、检索、引用回溯（离线向量 Mock） |
| [08](08_规划与任务分解/README.md) | 规划与任务分解 | 任务分解、计划执行、动态重规划 |
| [09](09_多Agent协作/README.md) | 多 Agent 协作 | 路由器、编排者-执行者、消息传递 |
| [10](10_评估、追踪与可观测性/README.md) | 评估与可观测性 | 数据集评估、轨迹追踪、成本统计 |
| [11](11_安全、护栏与成本控制/README.md) | 安全、护栏与成本控制 | 输入/输出护栏、循环检测、预算控制 |
| [12](12_工程化部署/README.md) | 工程化部署 | 配置、并发、缓存、生产 API 服务 |
| [capstone](capstone_综合项目/README.md) | 综合项目 | 端到端可运行的综合 Agent（可带工具、记忆、护栏） |

## 环境搭建

```bash
# 1. Python 3.10+（教程在 3.14.0 上验证）
python3 --version

# 2. 安装依赖（只有 pytest，示例本身零三方依赖）
pip install -r learning/requirements.txt
```

## 如何运行

```bash
# 运行任意示例（全部默认离线 Mock，无需 Key）
python learning/01_导论与心智模型/code/ch01_hello_agent.py

# 打开 DEBUG 日志观察每一步（部分示例支持）
LEARNING_VERBOSE=1 python learning/01_导论与心智模型/code/ch01_agent_anatomy.py

# 跑全部测试
pytest learning/ -q

# 切换真实 LLM（OpenAI 兼容接口；默认仍是 Mock）
export USE_REAL_LLM=1
export OPENAI_API_KEY=sk-xxx
# 可选：export OPENAI_BASE_URL=http://127.0.0.1:11434/v1   （指向本地 ollama）
# 可选：export LEARNING_MODEL=gpt-4o-mini
```

## 教程约定

1. **零占位**：所有代码都是完整实现，没有 TODO/伪代码。
2. **自检内置**：每个示例结尾有 `verify()` 自检区，全绿才算跑通。
3. **Mock 优先**：默认离线可跑；真实调用是可选增强，接口完全一致（`common/llm_client.make_client`）。
4. **讲义与代码同源**：README 里引用的片段与 `code/` 中文件逐字一致，测试持续验证。
5. **中文讲解、英文命名**、依赖最小化（见 `requirements.txt`）。

## 学习建议

- 每章先读 README 原理，再**亲手跑一遍**示例，最后做 [exercises.md](01_导论与心智模型/exercises.md) 练习。
- 想看"真模型 vs 假模型"行为差异时再打开 `USE_REAL_LLM=1`，注意成本。
- 教程的进度与交接文档见 [PROGRESS.md](PROGRESS.md)。
