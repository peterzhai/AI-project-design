"""示例 02：离线向量检索 —— 伪嵌入 + 余弦相似度，无网络也能"语义"匹配。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/07_RAG与知识检索/code/ch07_vector_search.py

向量检索的本质：把文本变成向量，问题向量与知识块向量越"接近"越相关。
本教程用**哈希伪嵌入**（字符 n-gram 投影到固定维度）：
- 完全离线、确定性（同样的文本永远得到同样的向量，可断言可复现）；
- 能反映词汇重合度 —— 与第 6 章的关键词检索同源，但支持部分匹配与打分排序。
真实系统换成 embedding 模型（如 text-embedding-3-small），检索代码一行不用改。
"""

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from ch07_chunking import Chunk  # noqa: E402
from common.check import banner, finish, section, verify  # noqa: E402


def embed(text: str, dim: int = 64) -> list[float]:
    """确定性伪嵌入：字符 bigram 哈希到 dim 维词袋，再 L2 归一化。

    同一文本永远得到同一向量；词汇重合度越高，向量夹角越小。
    """
    vector = [0.0] * dim
    grams = [text[i:i + 2] for i in range(len(text) - 1)] or [text]
    for gram in grams:
        digest = hashlib.md5(f"{dim}:{gram}".encode("utf-8")).hexdigest()
        vector[int(digest, 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：1=方向完全一致，0=正交（无关）。"""
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class Hit:
    """一次检索命中：知识块 + 相似度得分。"""

    chunk: Chunk
    score: float


class VectorIndex:
    """内存向量索引：add 写入，search 按 cosine 排序取 top-k。"""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._vectors: list[tuple[Chunk, list[float]]] = []

    def add(self, chunk: Chunk) -> None:
        self._vectors.append((chunk, embed(chunk.text, self.dim)))

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        query_vec = embed(query, self.dim)
        scored = [(chunk, cosine(query_vec, vec)) for chunk, vec in self._vectors]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [Hit(chunk=chunk, score=score) for chunk, score in scored[:top_k]]


# 三段政策知识（复用第 1 个示例的段落切块）
DOCS = {
    "年假": "公司年假政策：入职满一年的员工每年享有五天带薪年假。年假需提前三个工作日在系统申请。",
    "病假": "病假规定：病假需提供医院证明，连续病假超过三天需人事部复核。病假期间工资按基本工资的百分之八十发放。",
    "报销": "报销流程：出差报销需在返回后十五天内提交发票，超过期限需部门总监特批。发票必须为增值税发票。",
}


def build_index() -> VectorIndex:
    index = VectorIndex()
    for i, (topic, text) in enumerate(DOCS.items()):
        index.add(Chunk(chunk_id=f"policy#{i}", doc_id="policy", index=i, text=text, start=i))
    return index


def main() -> None:
    banner("示例 02：离线向量检索")

    index = build_index()

    # 实验 1：基本检索 —— 问题命中最相关块
    section("实验 1：检索「怎么申请年假？」")
    hits = index.search("怎么申请年假？需要提前多久？", top_k=3)
    for hit in hits:
        print(f"  score={hit.score:.3f} [{hit.chunk.chunk_id}] {hit.chunk.text[:24]}…")
    verify("年假块排第一", hits[0].chunk.text.startswith("公司年假政策"),
           f"实际第一块: {hits[0].chunk.chunk_id}")
    verify("得分排序单调递减", hits[0].score >= hits[1].score >= hits[2].score)

    # 实验 2：不同问题命中不同块（打分能区分主题）
    section("实验 2：检索「报销发票期限」")
    hits2 = index.search("出差报销的发票有什么要求？", top_k=3)
    verify("报销块排第一", hits2[0].chunk.text.startswith("报销流程"),
           f"实际第一块: {hits2[0].chunk.chunk_id}")

    # 实验 3：确定性 —— 同一查询两次结果完全一致
    section("实验 3：伪嵌入的确定性（可复现）")
    again = index.search("怎么申请年假？需要提前多久？", top_k=3)
    verify("两次检索结果与得分完全一致",
           [(h.chunk.chunk_id, round(h.score, 9)) for h in hits] ==
           [(h.chunk.chunk_id, round(h.score, 9)) for h in again])

    # 实验 4：无关问题 —— 得分整体偏低（为拒答阈值做铺垫）
    section("实验 4：完全无关的问题「今天天气如何？」")
    hits3 = index.search("今天天气如何", top_k=1)
    print(f"最高得分: {hits3[0].score:.3f}")
    verify("无关问题得分明显偏低", hits3[0].score < 0.35, f"实际 {hits3[0].score:.3f}")
    print("-> 这就是下一示例'拒答阈值'的物理依据。")

    finish("ch07_vector_search")


if __name__ == "__main__":
    main()
