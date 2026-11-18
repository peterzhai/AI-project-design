"""示例 03：RAG 全链路 —— 检索、注入、生成、引用回溯与诚实拒答。

运行（离线 Mock，无需任何 API Key 与网络）：
    python learning/07_RAG与知识检索/code/ch07_rag_pipeline.py

RAG = Retrieval-Augmented Generation。完整四步：
  1. 检索：向量索引 top-k 命中知识块；
  2. 注入：块内容带 [chunk_id] 标注拼进提示词；
  3. 生成：要求模型回答必须标注引用；
  4. 校验：答案里的引用 id 必须真的在检索结果里（引用回溯）；
     检索得分过低时直接拒答（诚实地说"不知道"，而不是编造）。
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEARNING_DIR))

from ch07_vector_search import Hit, VectorIndex, build_index  # noqa: E402
from common.check import banner, finish, section, verify  # noqa: E402
from common.mock_llm import MockLLM  # noqa: E402

# 引用格式约定：[chunk_id]，如 [policy#0]
CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_#]+)\]")


@dataclass
class RAGResult:
    """一次 RAG 问答的完整产物（答案 + 引用 + 状态）。"""

    answer: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)  # 引用了不存在的块（幻觉信号）
    refused: bool = False                                        # 是否触发拒答


def build_rag_prompt(question: str, hits: list[Hit]) -> str:
    """把检索块拼进 system 提示词，块带 [chunk_id] 标注。"""
    context_lines = [f"[{hit.chunk.chunk_id}] {hit.chunk.text}" for hit in hits]
    return (
        "你是企业政策问答助手。只依据下面提供的知识块回答，"
        "并在每个事实后标注来源块 id，格式如 [policy#0]。知识块里没有的信息不要编造。\n\n"
        "知识块：\n" + "\n".join(context_lines)
    )


def rag_answer(llm, index: VectorIndex, question: str,
               top_k: int = 3, threshold: float = 0.35) -> RAGResult:
    """RAG 主流程：检索 -> 阈值拒答 -> 生成 -> 引用校验。

    阈值是超参数：必须用真实数据实测校准（本教程知识库的相关问题得分
    0.37~0.60，无关问题 0.22~0.31，取 0.35 为分界线）。"""
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


def main() -> None:
    banner("示例 03：RAG 全链路（检索/注入/生成/引用回溯）")
    index = build_index()

    # 场景 1：正常问答 —— 答案带合法引用
    section("场景 1：正常问答「怎么申请年假？」")
    llm = MockLLM(scripted=[
        {"content": "入职满一年的员工每年享有五天带薪年假，需提前三个工作日在系统申请 [policy#0]。"}
    ])
    result = rag_answer(llm, index, "怎么申请年假？需要提前多久？")
    print(f"答案: {result.answer}")
    verify("答案包含合法引用", result.citations == ["policy#0"], f"实际 {result.citations}")
    verify("无非法引用（无幻觉信号）", result.invalid_citations == [])
    verify("上下文里注入了知识块", "[policy#0]" in llm.calls[0]["messages"][0]["content"])

    # 场景 2：模型幻觉 —— 引用了不存在的块 id，校验器当场抓住
    section("场景 2：幻觉拦截「模型引用了不存在的 policy#99」")
    llm2 = MockLLM(scripted=[
        {"content": "年假可以无限累积 [policy#99]。"}  # 编造的知识 + 编造的引用
    ])
    result2 = rag_answer(llm2, index, "年假可以累积吗？")
    print(f"答案: {result2.answer}")
    print(f"非法引用: {result2.invalid_citations}")
    verify("幻觉引用被抓出", result2.invalid_citations == ["policy#99"])
    verify("合法引用列表为空", result2.citations == [])

    # 场景 3：知识库没有的问题 —— 低分拒答，诚实说不知道
    section("场景 3：诚实拒答「公司年会几点开始？」（知识库无此信息）")
    llm3 = MockLLM(scripted=[{"content": "（模型本想编造，但根本没有被调用）"}])
    result3 = rag_answer(llm3, index, "公司年会几点开始？")
    print(f"答案: {result3.answer}")
    verify("低分触发拒答", result3.refused is True)
    verify("拒答时不调用模型（零成本、零幻觉）", llm3.call_count == 0)
    verify("拒答没有引用", result3.citations == [])

    # 场景 4：多块命中 —— 答案可引用多个来源
    section("场景 4：跨主题问题引用多个块")
    llm4 = MockLLM(scripted=[
        {"content": "病假需医院证明，超过三天人事部复核 [policy#1]；"
                    "报销必须用增值税发票，十五天内提交 [policy#2]。"}
    ])
    result4 = rag_answer(llm4, index, "病假和报销分别要注意什么？")
    print(f"答案: {result4.answer}")
    verify("两个合法引用都被识别", result4.citations == ["policy#1", "policy#2"],
           f"实际 {result4.citations}")

    print("\n要点回顾：检索给'依据'，注入给'约束'，引用给'可追溯'，阈值给'诚实'。")

    finish("ch07_rag_pipeline")


if __name__ == "__main__":
    main()
