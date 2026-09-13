from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import basinrag.cli as cli
from basinrag.core import llm as llm_module
from basinrag.core.llm import UniversalLLM
from basinrag.retriever.fusion import (
    apply_hop_prior,
    multi_signal_drf,
    ranked_ids,
    tanh_soft_brake,
    weighted_rrf,
)
from basinrag.retriever.reranker import CrossEncoderReranker


def _set_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["basinrag", *args])


def test_cli_help_and_version_exit_before_creating_rag(monkeypatch, capsys):
    monkeypatch.setattr(cli.BasinRAG, "create", Mock(side_effect=AssertionError("must stay lazy")))
    _set_argv(monkeypatch, "--help")
    with pytest.raises(SystemExit) as help_exit:
        cli.main()
    assert help_exit.value.code == 0
    assert "reindex" in capsys.readouterr().out

    _set_argv(monkeypatch, "--version")
    with pytest.raises(SystemExit) as version_exit:
        cli.main()
    assert version_exit.value.code == 0
    assert "1.1.0" in capsys.readouterr().out


def test_cli_without_command_shows_help_and_exits(monkeypatch, capsys):
    _set_argv(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "usage: basinrag" in capsys.readouterr().out


def test_cli_ingest_reindex_and_sync_forward_options(monkeypatch, capsys):
    rag = SimpleNamespace(
        ingest=Mock(return_value=3),
        reindex=Mock(return_value=4),
        sync=Mock(return_value=0),
    )
    create = Mock(return_value=rag)
    monkeypatch.setattr(cli.BasinRAG, "create", create)

    _set_argv(monkeypatch, "ingest", "docs", "--storage-dir", "store-a")
    cli.main()
    assert create.call_args.kwargs == {"load_existing_index": True, "storage_dir": "store-a"}
    rag.ingest.assert_called_once_with("docs")
    assert "3 nós criados" in capsys.readouterr().out

    create.reset_mock()
    _set_argv(monkeypatch, "reindex", "docs", "--storage-dir", "store-b")
    cli.main()
    assert create.call_args.kwargs == {"load_existing_index": False, "storage_dir": "store-b"}
    rag.reindex.assert_called_once_with("docs")
    assert "Reindexação completa" in capsys.readouterr().out

    _set_argv(monkeypatch, "sync", "docs")
    cli.main()
    rag.sync.assert_called_once_with("docs")
    assert "0 nós ingeridos" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["ingest", "reindex"])
def test_cli_empty_ingest_or_reindex_exits_with_failure(monkeypatch, command, capsys):
    rag = SimpleNamespace(ingest=Mock(return_value=0), reindex=Mock(return_value=0))
    monkeypatch.setattr(cli.BasinRAG, "create", Mock(return_value=rag))
    _set_argv(monkeypatch, command, "empty-source")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "Nenhum nó gerado" in capsys.readouterr().out


def test_cli_query_requires_loaded_index_and_prints_results(monkeypatch, capsys):
    rag = SimpleNamespace(_loaded=False)
    monkeypatch.setattr(cli.BasinRAG, "create", Mock(return_value=rag))
    _set_argv(monkeypatch, "query", "question")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1

    doc = SimpleNamespace(page_content="retrieved passage")
    rag._loaded = True
    rag.query = Mock(return_value=[doc])
    _set_argv(monkeypatch, "query", "question", "--type", "global", "--top-k", "2")
    cli.main()
    rag.query.assert_called_once_with("question", search_type="global", top_k=2)
    assert "retrieved passage" in capsys.readouterr().out

    rag.query.return_value = []
    _set_argv(monkeypatch, "query", "question")
    cli.main()
    assert "Nenhum trecho recuperado" in capsys.readouterr().out


def test_cli_chat_saves_on_exit_and_streams_tokens(monkeypatch, capsys):
    async def chat(_question):
        yield "Olá"
        yield " mundo"

    summarizer = AsyncMock()
    rag = SimpleNamespace(
        _loaded=True,
        config=SimpleNamespace(
            enable_background_l3=True, provider="openai", allow_remote_l3_egress=True
        ),
        chat=chat,
        start_background_summarizer=summarizer,
        persistence=SimpleNamespace(save_topology=Mock()),
        engine=object(),
    )
    monkeypatch.setattr(cli.BasinRAG, "create", Mock(return_value=rag))
    answers = iter(["pergunta", "sair"])
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))
    _set_argv(monkeypatch, "chat", "--enable-background-l3", "--allow-remote-l3-egress")
    cli.main()
    out = capsys.readouterr().out
    assert "Olá mundo" in out
    assert summarizer.await_count == 1
    rag.persistence.save_topology.assert_called_once_with(rag.engine)


def test_cli_chat_unloaded_index_exits(monkeypatch, capsys):
    monkeypatch.setattr(cli.BasinRAG, "create", Mock(return_value=SimpleNamespace(_loaded=False)))
    _set_argv(monkeypatch, "chat")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "Nenhum indice carregado" in capsys.readouterr().out


def test_cli_serve_forwards_security_and_runtime_options(monkeypatch):
    api_package = ModuleType("basinrag.api")
    api_package.__path__ = []
    api_server = ModuleType("basinrag.api.server")
    key_check = Mock()
    api_server.require_api_key_for_public_bind = key_check
    uvicorn = SimpleNamespace(run=Mock())
    monkeypatch.setitem(sys.modules, "basinrag.api", api_package)
    monkeypatch.setitem(sys.modules, "basinrag.api.server", api_server)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    _set_argv(
        monkeypatch, "serve", "--host", "0.0.0.0", "--port", "8123",
        "--storage-dir", "release-index", "--enable-background-l3", "--allow-remote-l3-egress",
    )
    cli.main()
    key_check.assert_called_once_with("0.0.0.0")
    uvicorn.run.assert_called_once_with(
        "basinrag.api.server:app", host="0.0.0.0", port=8123, reload=False, ws_max_size=8192
    )
    assert cli.os.environ["BASINRAG_STORAGE_DIR"] == "release-index"
    assert cli.os.environ["BASINRAG_ENABLE_BACKGROUND_L3"] == "true"
    assert cli.os.environ["BASINRAG_ALLOW_REMOTE_L3_EGRESS"] == "true"


def test_cli_serve_auth_failure_exits_before_uvicorn(monkeypatch, capsys):
    api_package = ModuleType("basinrag.api")
    api_package.__path__ = []
    api_server = ModuleType("basinrag.api.server")

    def deny(_host):
        raise RuntimeError("key required")

    api_server.require_api_key_for_public_bind = deny
    uvicorn = SimpleNamespace(run=Mock())
    monkeypatch.setitem(sys.modules, "basinrag.api", api_package)
    monkeypatch.setitem(sys.modules, "basinrag.api.server", api_server)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    _set_argv(monkeypatch, "serve")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "key required" in capsys.readouterr().out
    uvicorn.run.assert_not_called()


def test_local_tokenizer_is_cached_and_uses_offline_safe_options(monkeypatch):
    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            if add_special_tokens is False:
                raise TypeError("old tokenizer signature")
            return list(text.split())

    class AutoTokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            cls.calls.append((model, kwargs))
            return Tokenizer()

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    llm = UniversalLLM("ollama", "local-model")
    assert llm.count_tokens("two tokens") == 2
    assert llm.count_tokens("three local tokens") == 3
    assert len(AutoTokenizer.calls) == 1
    assert AutoTokenizer.calls[0] == (
        "local-model", {"local_files_only": True, "trust_remote_code": False}
    )


def test_token_counting_uses_tiktoken_and_unknown_model_fallback(monkeypatch):
    class Encoding:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [text, "token"]

    class Tiktoken:
        @staticmethod
        def encoding_for_model(_model):
            raise KeyError("unknown model")

        @staticmethod
        def get_encoding(name):
            assert name == "cl100k_base"
            return Encoding()

    monkeypatch.setitem(sys.modules, "tiktoken", Tiktoken)
    llm = UniversalLLM("openai", "made-up-model")
    assert llm.count_tokens("example") == 2


def test_token_count_falls_back_conservatively_by_script_and_handles_empty(monkeypatch):
    class BrokenTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise OSError("offline cache miss")

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = BrokenTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    llm = UniversalLLM("ollama", "uncached")
    assert llm.count_tokens("") == 0
    assert llm.count_tokens("abcd 你好!!") == 5
    assert llm.count_tokens("abcde") == 2


def test_openai_and_ollama_streamers_yield_provider_content(monkeypatch):
    messages_module = ModuleType("langchain_core.messages")

    class Message:
        def __init__(self, content):
            self.content = content

    messages_module.SystemMessage = Message
    messages_module.HumanMessage = Message

    class StreamingModel:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        async def astream(self, messages):
            assert [message.content for message in messages] == ["system", "question"]
            yield SimpleNamespace(content="one")
            yield SimpleNamespace(content="two")

    openai_module = ModuleType("langchain_openai")
    openai_module.ChatOpenAI = StreamingModel
    ollama_module = ModuleType("langchain_ollama")
    ollama_module.ChatOllama = StreamingModel
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_module)
    monkeypatch.setitem(sys.modules, "langchain_openai", openai_module)
    monkeypatch.setitem(sys.modules, "langchain_ollama", ollama_module)

    async def collect(llm):
        return [token async for token in llm.chat_stream("system", "question")]

    openai = UniversalLLM("openai", "gpt-test")
    assert asyncio.run(collect(openai)) == ["one", "two"]
    assert asyncio.run(collect(openai)) == ["one", "two"]
    assert len(StreamingModel.instances) == 1
    assert StreamingModel.instances[0].kwargs == {"model": "gpt-test", "temperature": 0.3}

    ollama = UniversalLLM("ollama", "llama-test")
    assert asyncio.run(collect(ollama)) == ["one", "two"]
    assert StreamingModel.instances[-1].kwargs == {"model": "llama-test", "temperature": 0.3}


@pytest.mark.parametrize("provider,module_name,error_text", [
    ("openai", "langchain_openai", "langchain-openai não instalado"),
    ("ollama", "langchain_ollama", "langchain-ollama não instalado"),
])
def test_missing_provider_extra_has_actionable_error(monkeypatch, provider, module_name, error_text):
    monkeypatch.setitem(sys.modules, module_name, None)
    llm = UniversalLLM(provider)

    async def collect():
        return [token async for token in llm.chat_stream("system", "question")]

    if provider == "ollama":
        with pytest.raises(ImportError, match=error_text):
            llm._ensure_ollama()
    else:
        with pytest.raises(ImportError, match=error_text):
            asyncio.run(collect())


def test_unsupported_provider_and_uninitialized_ollama_are_clear_errors(monkeypatch):
    unsupported = UniversalLLM("other")

    async def collect(llm):
        return [token async for token in llm.chat_stream("system", "question")]

    with pytest.raises(ValueError, match="não suportado"):
        asyncio.run(collect(unsupported))

    ollama = UniversalLLM("ollama")
    monkeypatch.setattr(ollama, "_ensure_ollama", lambda: None)
    with pytest.raises(RuntimeError, match="Motor Ollama não inicializado"):
        asyncio.run(collect(ollama))


@pytest.mark.asyncio
async def test_generate_retries_timeouts_with_exponential_backoff(monkeypatch):
    delays = AsyncMock()
    monkeypatch.setattr(llm_module.asyncio, "sleep", delays)
    monkeypatch.setattr(llm_module, "RETRY_BASE_DELAY", 0.25)
    llm = UniversalLLM("openai")
    attempts = 0

    def provider_stream(_system, _user):
        async def stream():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                if False:
                    yield "unreachable"
                raise TimeoutError("provider timed out")
            yield "final"

        return stream()

    monkeypatch.setattr(llm, "chat_stream", provider_stream)
    assert await llm.generate("system", "question") == "final"
    assert attempts == 3
    assert [call.args[0] for call in delays.await_args_list] == [0.25, 0.5]


@pytest.mark.asyncio
async def test_generate_raises_provider_error_after_retry_budget(monkeypatch, caplog):
    delays = AsyncMock()
    monkeypatch.setattr(llm_module.asyncio, "sleep", delays)
    llm = UniversalLLM("openai")
    attempts = 0

    def broken_stream(_system, _user):
        async def stream():
            nonlocal attempts
            attempts += 1
            if False:
                yield "unreachable"
            raise RuntimeError("provider unavailable")

        return stream()

    monkeypatch.setattr(llm, "chat_stream", broken_stream)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await llm.generate("system", "question")
    assert attempts == llm_module.MAX_RETRIES
    assert delays.await_count == llm_module.MAX_RETRIES - 1
    assert "LLM falhou após" in caplog.text


def test_reranker_skips_empty_input_without_loading_model(monkeypatch):
    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_load_model", Mock(side_effect=AssertionError("empty input must skip")))
    assert reranker.predict_scores("q", []) == []
    assert reranker.rerank("q", []) == []
    assert reranker.rerank_items("q", []) == []
    assert reranker.rerank_with_scores("q", []) == []


def test_reranker_load_is_revision_pinned_and_negative_tail_stays_ordered(monkeypatch):
    calls = []

    class Model:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))

        def predict(self, pairs, **kwargs):
            assert pairs == [["q", "tail"], ["q", "head"], ["q", "tie"]]
            assert kwargs == {"batch_size": 32, "show_progress_bar": False}
            return [-9.0, -1.0, -1.0]

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.CrossEncoder = Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    reranker = CrossEncoderReranker("cross-encoder", max_length=128, revision="rev-123")
    items = [("tail-id", "tail"), ("head-id", "head"), ("tie-id", "tie")]
    assert reranker.rerank_items("q", items, top_k=3) == [items[1], items[2], items[0]]
    assert reranker.rerank_with_scores("q", ["tail", "head", "tie"], top_k=2) == [
        ("head", -1.0), ("tie", -1.0)
    ]
    assert calls == [("cross-encoder", {
        "max_length": 128, "trust_remote_code": False, "revision": "rev-123"
    })]


def test_reranker_load_failure_is_explicit_and_chained(monkeypatch):
    class BrokenCrossEncoder:
        def __init__(self, *_args, **_kwargs):
            raise OSError("model missing from local cache")

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.CrossEncoder = BrokenCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    reranker = CrossEncoderReranker("missing-model", revision="fixed-rev")
    with pytest.raises(RuntimeError, match="fixed-rev") as exc:
        reranker.predict_scores("q", ["doc"])
    assert isinstance(exc.value.__cause__, OSError)


def test_weighted_rrf_respects_rankings_and_depth():
    scores = weighted_rrf(
        ["both", "lexical", "deep"],
        ["both", "semantic"],
        alpha=0.6,
        k=10,
        max_depth=2,
    )
    assert set(scores) == {"both", "lexical", "semantic"}
    assert scores["both"] == pytest.approx(0.6 / 11 + 0.4 / 11)
    assert scores["lexical"] > scores["semantic"]
    assert ranked_ids(scores, 2) == ["both", "lexical"]
    assert ranked_ids(scores, 0) == []


def test_hop_prior_enabled_disabled_penalty_and_empty_inputs():
    scores = {"near": 1.0, "far": 1.0, "unseen": 1.0}
    assert apply_hop_prior(scores, {}, enabled=False) == scores
    neutral = apply_hop_prior(scores, {"near": 0, "far": 5})
    assert neutral["unseen"] == neutral["near"]
    penalized = apply_hop_prior(scores, {"near": 0, "far": 5}, missing="penalty")
    assert penalized["unseen"] < penalized["far"] < penalized["near"]
    assert apply_hop_prior({}, {}) == {}


def test_multi_signal_drf_normalizes_available_signals_and_is_opt_in():
    raw = {"a": 0.5, "b": 0.0}
    assert multi_signal_drf(raw, {"a": 1}, enabled=False) == raw
    enabled = multi_signal_drf(
        raw,
        {"a": 1},
        cohesion={"a": 0.8},
        centrality={"b": 0.4},
        memory={"a": 0.6, "b": 0.2},
        enabled=True,
    )
    assert set(enabled) == {"a", "b"}
    assert 0.5 < enabled["a"] < 1.0
    assert enabled["a"] > enabled["b"]
    assert multi_signal_drf({}, {}, enabled=True) == {}
    assert 0.0 < tanh_soft_brake(0.5) < 1.0
