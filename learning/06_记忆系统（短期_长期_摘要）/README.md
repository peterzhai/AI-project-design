# 第 6 章 · 记忆系统（短期 / 长期 / 摘要）

> 本章目标：解决 Agent 的"健忘与失忆"——上下文无限增长会烧钱、撑爆窗口；
> 窗口一截断又丢关键信息。三板斧：滑动窗口保近期、长期记忆存事实、摘要压缩换空间。
>
> 学习时间：约 2~3 小时。前置要求：第 1 章（Memory 雏形）。

---

## 1. 为什么需要记忆架构：一个实验看懂

对话每多一轮，上下文就长一分 —— **每次调用都带着全部历史**（第 2 章讲过：模型没有记忆）。
长对话的成本与延迟随之线性上涨，最终撞上模型窗口上限。

```text
  无记忆架构的上下文增长：          三层记忆架构：

  轮1:  [sys][u][a]                每轮都只发送:
  轮5:  [sys][u][a]×10             [sys 身份]
  轮20: [sys][u][a]×40  ← 爆炸     [+ 历史摘要（压缩记忆）]
  轮50:    ✗ 窗口溢出              [+ 最近 N 条（滑动窗口）]
                                   [+ 长期记忆条目（按需检索回注）]
```

三个组件各管一段，职责不重叠：

| 组件 | 管什么 | 类比 |
| --- | --- | --- |
| 滑动窗口（短期） | 最近几轮的逐字原文 | 工作台上的便签 |
| 摘要压缩 | 窗口外旧对话的要点 | 会议纪要 |
| 长期记忆 | 跨会话的关键事实条目 | 档案柜 |

> 教程约定：所有"token 数"用 **1 字符 ≈ 1 伪 token** 近似（`approx_tokens`），
> 离线确定、便于断言；真实成本核算在第 11 章。

## 2. 代码走读

### 2.1 示例 01 —— 滑动窗口：system 永驻，最近原文

文件：[code/ch06_sliding_window.py](code/ch06_sliding_window.py)

`ContextWindow` 的关键设计是 **system 与滑动区分离**（摘自 `render()`，逐字一致）：

```python
    def render(self) -> list[dict]:
        """输出受窗口约束的上下文：system + 最近 max_messages 条。"""
        windowed = self._history[-self.max_messages:]
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(windowed)
        return messages
```

朴素做法（直接切最近 N 条）会丢掉 system —— 模型瞬间"忘记自己是音乐推荐助手"
（示例里有反面对照实验）。被挤出的消息进入 `evicted()` 可追溯，为示例 02 的长期记忆做准备。

### 2.2 示例 02 —— 长期记忆：存事实、按相关性取、回注上下文

文件：[code/ch06_long_term_memory.py](code/ch06_long_term_memory.py)

三个动作的入口 `remember`（写入）、`search`（检索）、`recall_for`（回注）。
回注方法的完整实现（摘自源码，逐字一致）：

```python
    def recall_for(self, query: str, top_k: int = 3) -> str:
        """把检索结果渲染成可注入 system 的附加段。"""
        hits = self.search(query, top_k=top_k)
        if not hits:
            return ""
        lines = [f"- [{item.key}] {item.content}" for item in hits]
        return "以下是关于该用户的已知记忆，回答时必须遵守：\n" + "\n".join(lines)
```

设计要点：
- **同 key 覆盖**：用户搬家了，"地址"应更新而不是新增两条矛盾记忆；
- **检索打分**：教程版用字符重合度（零依赖、确定性）；生产版换成向量检索 —— 第 7 章 RAG 讲原理；
- **回注格式**：`"以下是关于该用户的已知记忆，回答时必须遵守：…"` 作为 system 附加段 ——
  "记住"不等于"会用"，必须显式注入并要求遵守。

### 2.3 示例 03 —— 摘要压缩：拿空间换信息

文件：[code/ch06_summarizing_memory.py](code/ch06_summarizing_memory.py)

`SummarizingMemory` 的压缩内核（摘自 `_consolidate_oldest`，逐字一致）：

```python
    def _consolidate_oldest(self) -> None:
        """把最老的 2 条（仅剩 1 条超窗时压 1 条）交给 LLM 压缩，摘要累积。"""
        batch = self._buffer[:2] if len(self._buffer) - self.max_recent >= 1 else self._buffer[:1]
        self._buffer = self._buffer[len(batch):]
        prompt = (
            "把以下对话压缩成一段简洁摘要，保留关键事实（作品名/决定/约束），不要客套话：\n"
            + "\n".join(f"[{m['role']}] {m['content']}" for m in batch)
        )
        response = self.llm.chat([{"role": "user", "content": prompt}])
        self.compress_count += 1
        # 摘要累积：旧摘要在前，新摘要在后（时间顺序）
        self.summary = (self.summary + " " + response.content).strip() if self.summary else response.content
```

示例的对比实验**同口径**（两侧都含 system）：10 轮长对话，完整方案 ≈ 562 伪 token，
压缩方案（身份 + 累积摘要 + 最近 4 条）显著更省且保留早期作品名 —— 
摘要提示词里那句 **"保留关键事实"** 是压缩质量的关键。

## 3. 常见坑

1. **朴素窗口丢了 system** —— 模型失忆身份，风格与约束全崩。system 永驻头部。
2. **摘要丢事实** —— 摘要提示词不强调"保留人名/数字/决定"，压缩完模型就胡编。
3. **记忆只存不取** —— 长期记忆写了从不回注，等于没记。检索 + 注入 + "必须遵守"三件套。
4. **矛盾记忆并存** —— 新旧地址都在档案柜里，模型随机挑一个。同 key 覆盖 + 过时标记。
5. **摘要调用不加预算** —— 每超窗都触发摘要调用，费用悄悄上涨；生产中要批量、降频、用便宜模型。
6. **检索分数并列时的次序依赖** —— 教程版按插入顺序稳定排序，生产向量检索用相似度阈值，都要想到"并列"问题。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch06_sliding_window.py](code/ch06_sliding_window.py) | 示例 01：ContextWindow（system 永驻 + 滑动窗口 + 伪 token 统计） |
| [code/ch06_long_term_memory.py](code/ch06_long_term_memory.py) | 示例 02：LongTermMemory（存取/检索/回注） |
| [code/ch06_summarizing_memory.py](code/ch06_summarizing_memory.py) | 示例 03：SummarizingMemory（累积摘要 + 同口径成本对比） |
| [tests/test_ch06_examples.py](tests/test_ch06_examples.py) | 13 个测试 |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

---

> 下一篇：[第 7 章 · RAG 与知识检索](../07_RAG与知识检索/README.md) ——
> 把"检索相关内容注入上下文"做成正式架构：切块、向量检索、引用回溯。
