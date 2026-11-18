"""第 7 章测试：切块、伪嵌入向量检索、RAG 全链路。"""

import pytest

import ch07_chunking as chunking_demo
import ch07_rag_pipeline as rag_demo
import ch07_vector_search as vector_demo
from ch07_chunking import Chunk, chunk_by_paragraph, chunk_fixed
from ch07_vector_search import VectorIndex, build_index, cosine, embed
from common.mock_llm import MockLLM


# ---------- 示例 01：切块 ----------
def test_chunking_main_end_to_end():
    chunking_demo.main()


def test_fixed_chunks_have_overlap_and_coverage():
    text = "A" * 100 + "B" * 30
    chunks = chunk_fixed(text, doc_id="d", size=40, overlap=10)
    assert chunks[0].text[-10:] == chunks[1].text[:10]      # 重叠区逐字一致
    assert chunks[0].chunk_id == "d#0"
    assert chunks[-1].start + len(chunks[-1].text) >= len(text)  # 尾部覆盖


def test_fixed_chunk_overlap_validation():
    with pytest.raises(ValueError, match="必须小于"):
        chunk_fixed("任意文本足够长", "d", size=10, overlap=10)


def test_paragraph_chunking_respects_boundaries():
    doc = "段落一内容较短。\n\n段落二也比较短。\n\n" + "超长段落" * 30
    chunks = chunk_by_paragraph(doc, doc_id="d", max_size=50)
    short_chunks = [c for c in chunks if len(c.text) < 50]
    assert all("\n" not in c.text for c in short_chunks)    # 短段不跨边界
    assert any("段落一" in c.text and "段落二" not in c.text for c in chunks)


# ---------- 示例 02：伪嵌入向量检索 ----------
def test_vector_search_main_end_to_end():
    vector_demo.main()


def test_embed_is_deterministic_and_normalized():
    v1 = embed("年假申请流程", dim=64)
    v2 = embed("年假申请流程", dim=64)
    assert v1 == v2                                       # 确定性
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9       # L2 归一化


def test_cosine_higher_for_similar_texts():
    a = cosine(embed("年假需要提前三个工作日申请"), embed("年假提前三个工作日在系统申请"))
    b = cosine(embed("年假需要提前三个工作日申请"), embed("出差报销发票期限"))
    assert a > b > 0


def test_index_search_orders_by_relevance():
    index = build_index()
    hits = index.search("怎么申请年假？", top_k=3)
    assert hits[0].chunk.text.startswith("公司年假政策")
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert len(hits) == 3


# ---------- 示例 03：RAG 全链路 ----------
def test_rag_pipeline_main_end_to_end():
    rag_demo.main()


def test_rag_answer_with_valid_citation():
    llm = MockLLM(scripted=[{"content": "五天带薪年假 [policy#0]。"}])
    result = rag_demo.rag_answer(llm, build_index(), "年假有几天？")
    assert result.citations == ["policy#0"]
    assert result.invalid_citations == []
    assert result.refused is False


def test_rag_detects_hallucinated_citation():
    llm = MockLLM(scripted=[{"content": "编造内容 [policy#42]。"}])
    result = rag_demo.rag_answer(llm, build_index(), "年假有几天？")
    assert result.invalid_citations == ["policy#42"]
    assert result.citations == []


def test_rag_refuses_when_below_threshold():
    llm = MockLLM(scripted=[{"content": "不该被调用"}])
    result = rag_demo.rag_answer(llm, build_index(), "公司年会几点开始？")  # 实测得分 0.312 < 0.35
    assert result.refused is True
    assert llm.call_count == 0
    assert result.citations == []


def test_rag_refuses_on_empty_index():
    llm = MockLLM(scripted=[{"content": "不该被调用"}])
    result = rag_demo.rag_answer(llm, VectorIndex(), "任何问题")
    assert result.refused is True


def test_citation_pattern_extraction():
    assert rag_demo.CITATION_PATTERN.findall("依据 [policy#0] 和 [policy#2]") == ["policy#0", "policy#2"]
    assert rag_demo.CITATION_PATTERN.findall("没有引用") == []
