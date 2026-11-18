"""示例 01：文档切块（Chunking）—— RAG 的第一道工序。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/07_RAG与知识检索/code/ch07_chunking.py

为什么切块：模型窗口装不下整个知识库，检索也只能"按块"召回 ——
块太大则噪音多、块太小则上下文破碎。两种基础策略：
1. 固定长度 + 重叠（overlap）：滑动窗口切块，重叠防止句子被拦腰切断后两边都检不到；
2. 段落感知：优先按空行段落切，超长段落再降级到固定长度 —— 尊重语义边界。
"""

import sys
from dataclasses import dataclass
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from common.check import banner, finish, section, verify  # noqa: E402


@dataclass(frozen=True)
class Chunk:
    """一个知识块：内容 + 来源定位（引用回溯的基础）。"""

    chunk_id: str      # 如 doc1#3
    doc_id: str        # 来源文档
    index: int         # 在文档内的序号
    text: str          # 块内容
    start: int         # 在原文中的起始偏移（引用回溯到原文位置）


def chunk_fixed(text: str, doc_id: str, size: int = 40, overlap: int = 10) -> list[Chunk]:
    """固定长度 + 重叠滑动切块。

    要求 overlap < size，否则死循环；步长 = size - overlap。
    """
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


def chunk_by_paragraph(text: str, doc_id: str, max_size: int = 60) -> list[Chunk]:
    """段落感知切块：按空行切段落，超长段落再按固定长度细分。

    每个段落起点作为新的偏移基准，start 记录全局偏移。
    """
    chunks: list[Chunk] = []
    index = 0
    offset = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            offset += 2  # 跳过空行本身
            continue
        if len(para) <= max_size:
            chunks.append(Chunk(chunk_id=f"{doc_id}#{index}", doc_id=doc_id, index=index,
                                text=para, start=offset))
            index += 1
        else:
            # 超长段落降级为固定长度切块（无重叠，保持语义段独立）
            for piece_start in range(0, len(para), max_size):
                piece = para[piece_start:piece_start + max_size]
                chunks.append(Chunk(chunk_id=f"{doc_id}#{index}", doc_id=doc_id, index=index,
                                    text=piece, start=offset + piece_start))
                index += 1
        offset += len(para) + 2  # 段落 + 分隔空行
    return chunks


SAMPLE_DOC = (
    "公司年假政策：入职满一年的员工每年享有五天带薪年假。年假需提前三个工作日在系统申请。\n"
    "\n"
    "病假规定：病假需提供医院证明，连续病假超过三天需人事部复核。病假期间工资按基本工资的百分之八十发放。\n"
    "\n"
    "报销流程：出差报销需在返回后十五天内提交发票，超过期限需部门总监特批。发票必须为增值税发票。"
)


def main() -> None:
    banner("示例 01：文档切块")

    # 实验 1：固定长度 + 重叠
    section("实验 1：固定长度切块（size=40, overlap=10）")
    chunks = chunk_fixed(SAMPLE_DOC.replace("\n\n", "\n"), doc_id="policy", size=40, overlap=10)
    for c in chunks[:3]:
        print(f"  [{c.chunk_id}] start={c.start}: {c.text[:30]}…")
    verify("块数量符合步长预期", len(chunks) >= 4, f"实际 {len(chunks)}")
    verify("相邻块有重叠（衔接处内容连续）",
           chunks[0].text[-10:] == chunks[1].text[:10],
           f"块0尾部={chunks[0].text[-10:]!r} 块1头部={chunks[1].text[:10]!r}")
    verify("全部原文被覆盖（首尾齐全）",
           chunks[0].start == 0 and chunks[-1].start + len(chunks[-1].text) >= len(SAMPLE_DOC.replace("\n\n", "\n")) - 1)

    section("实验 2：overlap >= size 时显式报错（防死循环）")
    try:
        chunk_fixed("短文本", "d1", size=10, overlap=10)
        raise AssertionError("实验 2 失败：应当抛 ValueError")
    except ValueError as exc:
        print(f"按预期报错: {exc}")
    verify("非法重叠被拒绝", True)

    # 实验 3：段落感知切块
    section("实验 3：段落感知切块（尊重语义边界）")
    pchunks = chunk_by_paragraph(SAMPLE_DOC, doc_id="policy", max_size=60)
    print(f"共 {len(pchunks)} 块：")
    for c in pchunks:
        print(f"  [{c.chunk_id}] start={c.start}: {c.text[:28]}…")
    verify("每个块都是完整段落（不跨语义边界）",
           all("\n" not in c.text for c in pchunks))
    verify("三个政策主题各自独立成块",
           any("年假" in c.text for c in pchunks)
           and any("病假" in c.text for c in pchunks)
           and any("报销" in c.text for c in pchunks))
    verify("块内包含可回溯的原文偏移", all(c.start >= 0 for c in pchunks))

    finish("ch07_chunking")


if __name__ == "__main__":
    main()
