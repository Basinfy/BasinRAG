import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from basinrag.indexer.summarizer import BasinSummarizer
from basinrag.core.topology import TopologicalBasin


def test_extract_score():
    summarizer = BasinSummarizer(engine=MagicMock(), provider="mock", model_name="mock")
    assert summarizer._extract_score("Excelente. SCORE: 9") == 9
    assert summarizer._extract_score("Score: 10/10") == 10
    assert summarizer._extract_score("O score atribuído é 7") == 7
    assert summarizer._extract_score("Sem número") == 5


@pytest.mark.asyncio
@patch("basinrag.indexer.summarizer.UniversalLLM")
async def test_summarizer_draft_critique_refine(mock_llm_cls):
    mock_llm_instance = AsyncMock()
    # 1. Draft
    # 2. Critique (< 8 triggers refine)
    # 3. Refine
    # 4. Critique 2 (>= 8 finishes)
    mock_llm_instance.generate.side_effect = [
        "TÍTULO: Draft Inicial\nTEMAS: a\nENTIDADES: b\nRESUMO: c",
        "Pode melhorar detalhes. SCORE: 6",
        "TÍTULO: Refinado\nTEMAS: a, b\nENTIDADES: x\nRESUMO: y",
        "Agora está excelente! SCORE: 9",
    ]
    mock_llm_cls.return_value = mock_llm_instance
    
    mock_engine = MagicMock()
    basin = TopologicalBasin("b1")
    basin.add_node("n1", hops=0, data={"text": "Texto do chunk 1", "source": "doc1"})
    basin.add_node("n2", hops=1, data={"text": "Texto do chunk 2", "source": "doc1"})
    mock_engine.basins = {"b1": basin}

    summarizer = BasinSummarizer(engine=mock_engine, provider="mock", model_name="mock")
    summary = await summarizer.summarize_basin("b1", verbose=False)

    assert "TÍTULO: Refinado" in summary
    assert mock_llm_instance.generate.call_count == 4


@pytest.mark.asyncio
async def test_summarizer_empty_basin():
    mock_engine = MagicMock()
    mock_engine.basins = {}
    summarizer = BasinSummarizer(engine=mock_engine, provider="mock", model_name="mock")
    res = await summarizer.summarize_basin("inexistente")
    assert res == ""
