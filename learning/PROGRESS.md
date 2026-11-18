# PROGRESS · 进度看板与交接文档

> 状态机文档：任何一轮工作开始前先读本文件确定"下一个待办项"。
> 结束前必须更新本文件（进度、已知问题、环境假设、STATUS）。

STATUS: DONE

## 1. 总进度表

| 章节 | 状态 | 说明 |
| --- | --- | --- |
| 公共层 common/ | ✅ 已完成 | MockLLM(+stream/params/tool_call.id)、SSE 解析、retry、json_utils、validators、toolbox、guards、日志、check；55 测试全绿 |
| 01_导论与心智模型 | ✅ 已完成 | 3 示例 + 13 测试全绿 + README 讲义 + exercises |
| 02_LLM调用基础 | ✅ 已完成 | 3 示例（消息参数/错误重试/流式SSE）+ 10 测试全绿 + 讲义 + exercises |
| 03_Prompt与结构化输出 | ✅ 已完成 | 3 示例（四板斧/JSON修复闭环/schema校验闭环）+ 10 测试全绿 + 讲义 + exercises |
| 04_Tool_Calling与函数调用 | ✅ 已完成 | 3 示例（schema自动生成/三层防线/协议还原+并行调用）+ 9 测试全绿 + 讲义 + exercises |
| 05_ReAct循环 | ✅ 已完成 | 3 示例（三段式/轨迹持久化续跑/可插拔终止条件）+ 11 测试全绿 + 讲义 + exercises；Step/Trace 定义于 ch05_react_loop.py 供后续章复用 |
| 06_记忆系统（短期/长期/摘要） | ✅ 已完成 | 3 示例（滑动窗口/长期记忆检索回注/摘要压缩同口径对比）+ 13 测试全绿 + 讲义 + exercises |
| 07_RAG与知识检索 | ✅ 已完成 | 3 示例（切块/伪嵌入向量检索/全链路+幻觉拦截+诚实拒答）+ 14 测试全绿 + 讲义 + exercises |
| 08_规划与任务分解 | ✅ 已完成 | 3 示例（分解+环检测/执行器重试跳过/动态重规划对照）+ 13 测试全绿 + 讲义 + exercises；toolbox 新增 try_execute |
| 09_多Agent协作 | ✅ 已完成 | 3 示例（路由器/编排者执行者/消息传递+黑板）+ 11 测试全绿 + 讲义 + exercises |
| 10_评估、追踪与可观测性 | ✅ 已完成 | 3 示例（TracedLLM+JSONL/评估器三断言/成本归因报表）+ 14 测试全绿 + 讲义 + exercises |
| 11_安全、护栏与成本控制 | ✅ 已完成 | 3 示例（输入护栏reject+strip/输出护栏三道检查/预算熔断+GuardedPipeline）+ 19 测试全绿 + 讲义 + exercises；护栏实现提升至 common/guards.py |
| 12_工程化部署 | ✅ 已完成 | 3 示例（AgentConfig fail-fast/缓存+限速/纯标准库HTTP服务）+ 23 测试全绿（含服务集成测试）+ 讲义 + exercises |
| capstone_综合项目 | ✅ 已完成 | 客服 Agent 整车装配：RAG+工具+记忆+护栏+追踪+预算，7 场景端到端 + 11 测试全绿 |
| learning/README.md 总览 | ✅ 已完成 | 章节表含全部链接；新章完成后同步校对链接 |
| learning/README.md 总览 | ✅ 已完成 | 章节表含全部链接；新章完成后同步校对链接 |

## 2. 下一个待办项

（无 —— 大纲全部章节完成，STATUS: DONE。后续可扩展方向见 capstone README 第 4 节：接真实模型、上 HTTP 服务、扩展工具、固化评估集。）

## 3. 已确认可用的运行命令

```bash
# 从仓库根目录 /Users/zhaizy/workspace/AI-project-design 执行：
python3 -m pytest learning/ -q                 # 全量测试（当前 219 passed，教程完成态）
python3 learning/capstone_综合项目/code/capstone_agent.py       # capstone 整车演示（7 场景）
# 各章示例均为 learning/NN_*/code/chNN_*.py（每章 3 个，共 37 个脚本，全部可独立运行）：
python3 learning/01_导论与心智模型/code/ch01_hello_agent.py       # 示例1
python3 learning/01_导论与心智模型/code/ch01_agent_anatomy.py     # 示例2
python3 learning/01_导论与心智模型/code/ch01_agent_vs_pipeline.py # 示例3
python3 learning/02_LLM调用基础/code/ch02_messages_and_params.py  # 示例4
python3 learning/02_LLM调用基础/code/ch02_errors_retry.py         # 示例5
python3 learning/02_LLM调用基础/code/ch02_streaming.py            # 示例6
python3 learning/03_Prompt与结构化输出/code/ch03_prompt_patterns.py     # 示例7
python3 learning/03_Prompt与结构化输出/code/ch03_json_extraction.py     # 示例8
python3 learning/03_Prompt与结构化输出/code/ch03_schema_validation.py   # 示例9
python3 learning/04_Tool_Calling与函数调用/code/ch04_schema_generation.py  # 示例10
python3 learning/04_Tool_Calling与函数调用/code/ch04_param_validation.py   # 示例11
python3 learning/04_Tool_Calling与函数调用/code/ch04_tool_protocol.py      # 示例12
python3 learning/tools/check_readme_consistency.py             # 讲义↔代码一致性校验（32 块全匹配）
python3 -m py_compile <文件>                   # 单文件语法检查
grep -rn "TODO\|FIXME\|待实现\|\bpass$" learning/ --include="*.py"  # 占位符自查（当前 0 残留）
```

- `learning/conftest.py` 已把各章 `code/` 加入 sys.path，测试可直接 `import ch01_hello_agent`。
- 示例脚本头部有 3 行路径引导（`parents[2]`），保证任意工作目录下直接 `python <脚本>` 可跑。

## 4. 已知问题与决策记录

1. **依赖刻意最小化**：仅 pytest。真实 LLM 调用用标准库 `urllib` 直连 OpenAI 兼容接口
   （`common/llm_client.OpenAICompatClient`），不引入 openai/httpx SDK —— 保证零网络零 Key 可复现。
2. **模块命名前缀 `chNN_`**：跨章导入不冲突（conftest 把所有章 code/ 都加进 sys.path）。
3. **MockLLM 双模式**：scripted（按序回放）与 handler（函数路由）二选一，构造时强制互斥，
   避免行为不确定；脚本耗尽抛 `MockLLMError` 带清晰信息，测试可断言。
4. **工具失败不抛异常**：ToolBox 把错误文本作为"观察"回传（第 1 章 README 有原则说明），
   这是贯穿教程的一致设计。
5. **目录名中文含括号**：第 6 章目录名 `06_记忆系统（短期_长期_摘要）` 用下划线替代 `/`
   （文件系统不允许 `/`）。第 10 章同理：`10_评估、追踪与可观测性`。
6. **第 1 章断言修正记录**：首版 `ch01_agent_anatomy.py` 把工具观察错位到 `render()[-3]`，
   实际在 `[-2]`（[-1] 是最终 assistant 消息）——运行时发现并已修复，测试固定该行为。
7. **scripted 模式消耗脚本**：MockLLM 脚本模式下每次 chat/stream 消费一条脚本。
   需要"同一内容多次调用"（如流式 vs 一次性一致性校验）必须用 handler 模式 ——
   第 2 章示例 3 首版踩坑（stream 消费后 chat 脚本耗尽），已改为 handler 并在讲义注明。
8. **新增 learning/tools/**：`check_readme_consistency.py` 可复用工具，每轮自检必跑；
   校验范围是 learning/ 下全部 .py 文件，后续章节无需修改即可复用。
9. **诚实声明**：`OpenAICompatClient.stream()`/`chat()` 走真实网络路径，本环境无法实测；
   离线验证覆盖协议解析纯函数（parse_sse_lines）、Mock 行为与重试逻辑。
10. **except 变量作用域**：Python 的 `except ... as exc` 在块结束时会删除 exc ——
    示例 8（ch03_json_extraction）首版把断言写在 except 外导致 UnboundLocalError，
    已改为块内断言并在讲义"常见坑"记录。
11. **schema 单一事实来源**：第 3 章确立"TICKET_SCHEMA 与提示词描述一一对应"的约定，
    第 4 章工具 schema 将直接从函数签名+docstring 生成，继续沿用该思想。
12. **一致性工具两次立功**：第 5 章讲义"magic"混排（人工发现）、第 6 章讲义三方法签名
    拼接块与源码不连续（工具抓出）—— 结论：README 只引用**源文件中连续的完整片段**。
13. **第 6 章两处实现教训**：① remember 用不同 key 演示"覆盖"导致覆盖未触发（改为同 key）；
    ② 摘要压缩对比必须同口径（两侧都含 system）且对话足够长，否则"更省"不成立；
    ③ _consolidate_oldest 批次条件 `>=1`（超窗即压 2 条），首版 `>=2` 导致只压 1 条。
14. **第 8 章三条教训**：① except 作用域坑第二次踩（断言必须写在 except 块内）；
    ② 工具约定强化——业务失败必须抛异常（返回"错误：…"文本会被判为成功）；
    ③ PlanStep 增加可选 tool/arguments 字段，计划才"可执行"（description 无法映射工具）。
15. **共享代码演进必须同步讲义**：第 8 章重构 ToolBox（execute -> try_execute）后，
    第 4 章 README 的引用块脱节，一致性工具抓出后已更新为当前实现。
    教训：凡改动 common/，必须跑 check_readme_consistency.py 并回改相关章节讲义。
16. **第 11 章两条安全教训**：① strip 模式"声明了剥离却没实现"——运行时才暴露，
    教训：护栏行为必须与文档一致且有测试钉住；② 脱敏正则顺序——短模式（手机号）
    先跑会误伤长模式（身份证）内嵌子串，必须先长后短；③ 被拦截后返回安全话术，
    打码痕迹只存在于 redactions 审计清单而非最终文本（断言曾写错，已修正）。
17. **第 12 章三条教训**：① 跨章 import 在"直接运行脚本"下不可用（conftest 只覆盖
    pytest 场景）——护栏类提升到 common/guards.py 一劳永逸，capstone 同样受益；
    ② 缓存命中路径不能引用已被重构删除的内部方法（用完整响应对象做缓存值更稳）；
    ③ 熔断语义：发生在下一次调用前，HTTP 层演示需要发出"下一次"请求才见 429；
    ④ except 作用域坑第三次出现（ch12 限速示例），已形成肌肉记忆：断言必写块内；
    ⑤ bytes 字面量不能含中文字面字符，须 .encode("utf-8")。

## 5. 环境信息

- macOS (darwin 25.3.0, arm64) / ServBay Python 3.14.0
- pytest 8.4.2（`pip install "pytest>=8.4,<9"` 安装成功）
- 网络与 Key 均未配置；所有验证都在纯离线 Mock 模式下完成
- 仓库根目录另有公众号文章等无关内容，教程全部位于 `learning/` 内，互不干扰

## 6. 每章交付物清单（保持一致的结构）

- `README.md`：原理 + ASCII 图解 + 分步讲解（引用与源码逐字一致的片段）+ 常见坑
- `code/chNN_*.py`：递进式示例，结尾 verify() 自检区，零异常退出
- `tests/test_chNN_*.py`：端到端跑 main() + 核心函数精确断言
- `exercises.md`：3 道练习（热身/进阶/挑战）+ 参考答案要点

## 7. 最终验收记录（DONE 时刻）

- 37 个示例脚本清缓存后逐个运行，全部零异常退出（退出码 0）
- 全量 pytest：219 passed / 0 failed（公共层 48 + 各章 160 + capstone 11）
- 讲义一致性：32 个 python 代码块与源文件逐字匹配（tools/check_readme_consistency.py）
- grep `TODO|FIXME|待实现|pass$|NotImplementedError`：零残留
- 依赖：仅 pytest>=8.4（requirements.txt），全部示例离线 Mock 可跑，无网络无 Key
- 停机条件判定：满足"大纲全部章节完成，且最终全量 pytest 通过"（停机条件 1）
