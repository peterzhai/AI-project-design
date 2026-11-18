# 第 7 章 · RAG 与知识检索

> 本章目标：实现完整的 RAG（Retrieval-Augmented Generation）链路 ——
> 切块、向量检索、注入生成、引用回溯、诚实拒答。让 Agent 能回答"训练数据里没有"的问题。
>
> 学习时间：约 2~3 小时。前置要求：第 3 章（结构化提示词）。

---

## 1. RAG 解决什么问题：让模型"开卷考试"

闭卷的 LLM 只知道训练数据里的事：公司年假政策、昨天上传的文档、内部系统参数 —— 它一概不知，
而且**不知道自己不知道**（会一本正经地编造）。RAG 就是开卷考试：

```text
   用户问题                      知识库（文档）
      │                              │
      │ ①问题向量化                   │ ①切块 → 每块向量化
      ▼                              ▼
   ┌────────────── ③相似度检索 top-k ──────────────┐
   │                      │                        │
   │                      ▼                        │
   │            ┌──────────────────┐               │
   │            │ 提示词 = 指令      │               │
   │            │  + [块1] 带来源id  │  ← ②注入      │
   │            │  + [块2] ...      │               │
   │            └────────┬─────────┘               │
   └─────────────────────┤                         │
                         ▼                         │
                  ④生成（必须标引用）                │
                         │                         │
                         ▼                         │
              ⑤引用回溯校验 / 低分拒答              │
```

第 6 章的长期记忆检索是它的"迷你版"（找的是用户事实）；本章升级为正式架构（找的是知识文档）。

## 2. 代码走读

### 2.1 示例 01 —— 切块：RAG 的第一道工序

文件：[code/ch07_chunking.py](code/ch07_chunking.py)

块太大 → 一块里塞了三个主题，检索到它就引入噪音；块太小 → 上下文破碎，模型读不懂。
两种基础策略，`Chunk` 始终记录来源（`doc_id` / `index` / `start`）—— 这是引用回溯的原料。
固定长度 + 重叠的核心循环（摘自 `chunk_fixed`，逐字一致）：

```python
    if overlap >= size:
        raise ValueError(f"overlap({overlap}) 必须小于 size({size})，否则切块无法前进")
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(chunk_id=f"{doc_id}#{index}", doc_id=doc_id, index=index,
                            text=text[start:end], start=start))
        if end == len(text):
            break
        start += size - overlap
        index += 1
    return chunks
```

重叠（overlap）的作用：句子被拦腰切断时，两个相邻块都保有它的部分上下文，检索任一侧都找得到。
段落感知切块（`chunk_by_paragraph`）则优先按空行切，超长段落再降级固定长度 —— 尊重语义边界。

### 2.2 示例 02 —— 离线向量检索：伪嵌入 + 余弦相似度

文件：[code/ch07_vector_search.py](code/ch07_vector_search.py)

真实系统用 embedding 模型把文本变成语义向量；本教程用**哈希伪嵌入**（摘自 `embed`，逐字一致）：

```python
    vector = [0.0] * dim
    grams = [text[i:i + 2] for i in range(len(text) - 1)] or [text]
    for gram in grams:
        digest = hashlib.md5(f"{dim}:{gram}".encode("utf-8")).hexdigest()
        vector[int(digest, 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]
```

它是"字符 bigram 投影到 64 维 + L2 归一化"：完全确定性（同一文本永远同一向量）、
零依赖、能反映词汇重合度。**检索代码与真实嵌入完全一致** —— 换 embedding 模型时只需替换 `embed`。
`VectorIndex.search` 按余弦相似度排序取 top-k。实测校准数据：
相关问题得分 0.37~0.60，无关问题 0.22~0.31 —— 这组数字直接决定了下一示例的拒答阈值。

> 为什么不用 numpy/向量库？—— 依赖最小化红线。纯 Python 的 64 维余弦在教程规模下毫秒级完成。

### 2.3 示例 03 —— RAG 全链路：生成、引用回溯、诚实拒答

文件：[code/ch07_rag_pipeline.py](code/ch07_rag_pipeline.py)

`rag_answer` 四步（摘自源码，逐字一致）：

```python
    hits = index.search(question, top_k=top_k)
    if not hits or hits[0].score < threshold:
        return RAGResult(answer="知识库中未找到相关信息，无法回答该问题。",
                         refused=True)

    response = llm.chat([
        {"role": "system", "content": build_rag_prompt(question, hits)},
        {"role": "user", "content": question},
    ])
    cited_ids = CITATION_PATTERN.findall(response.content)
    valid_ids = {hit.chunk.chunk_id for hit in hits}
    return RAGResult(
        answer=response.content,
        citations=[c for c in cited_ids if c in valid_ids],
        invalid_citations=[c for c in cited_ids if c not in valid_ids],
    )
```

两道护栏：

1. **引用回溯**：答案里的 `[chunk_id]` 必须真的在本次检索结果中 —— 引用了 `[policy#99]`
   说明模型在编造（示例场景 2 现场抓了一个）；
2. **阈值拒答**：最高得分低于阈值就不调用模型、直接说"不知道" —— 
   **拒答时模型根本不会被调用**（零成本、零幻觉机会），示例场景 3 用 `llm.call_count == 0` 证明了这一点。

> 阈值校准是硬要求：必须拿"相关问题"和"无关问题"各一批实测，取分界线。拍脑袋的阈值
> 要么拦不住幻觉（太低），要么大量误杀正常提问（太高）。

## 3. 常见坑

1. **overlap >= size** —— 切块步长为负，死循环。示例里做了显式校验。
2. **块太大或太小** —— 经验起点 200~500 字符（中文），必须结合自己的文档实测调整。
3. **阈值拍脑袋** —— 见上文校准数据；阈值是超参数，换知识库/换嵌入模型后要重新校准。
4. **不校验引用** —— 模型引用了不存在的块 id 也照单全收，幻觉披着"有据可查"的外衣。
5. **检索命中就一定回答** —— 低分命中也应该拒答；"命中"和"相关"是两回事。
6. **切块丢了来源信息** —— 没有 `doc_id/start`，答案无从追溯，用户不敢信。

## 4. 本目录文件清单

| 文件 | 说明 |
| --- | --- |
| [code/ch07_chunking.py](code/ch07_chunking.py) | 示例 01：固定长度重叠切块 + 段落感知切块 |
| [code/ch07_vector_search.py](code/ch07_vector_search.py) | 示例 02：哈希伪嵌入 + 余弦相似度 + 内存索引 |
| [code/ch07_rag_pipeline.py](code/ch07_rag_pipeline.py) | 示例 03：全链路（含幻觉拦截与诚实拒答） |
| [tests/test_ch07_examples.py](tests/test_ch07_examples.py) | 14 个测试 |
| [exercises.md](exercises.md) | 3 道练习 + 参考答案要点 |

---

> 下一篇：[第 8 章 · 规划与任务分解](../08_规划与任务分解/README.md) ——
> 复杂任务一口气做不完：让 Agent 先列计划，再逐项执行，必要时重规划。
