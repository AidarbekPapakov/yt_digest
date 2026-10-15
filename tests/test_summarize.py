import pytest
import requests

from src.cs2_digest.summarize import (
    ChunkSummary,
    SummaryParseError,
    VideoSummary,
    _chat_structured,
    _strip_think,
    chunk_turns,
    summarize_turns,
)


def _turn(speaker, text):
    return {"speaker": speaker, "text": text}


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _mock_post(monkeypatch, contents):
    """Makes requests.post return _FakeResponse(contents[i]) on the i-th call,
    and returns the list of prompts sent so tests can inspect them."""
    prompts = []

    def fake_post(url, headers=None, json=None, timeout=None):
        prompts.append(json["messages"][0]["content"])
        return _FakeResponse(contents[len(prompts) - 1])

    monkeypatch.setattr(requests, "post", fake_post)
    return prompts


def test_chunk_turns_single_chunk_when_small():
    turns = [_turn("speaker_0", "short line")] * 5
    chunks = chunk_turns(turns, token_budget=1000)
    assert len(chunks) == 1
    assert sum(len(c) for c in chunks) == 5


def test_chunk_turns_splits_on_budget():
    long_text = "word " * 50  # ~250 chars
    turns = [_turn("speaker_0", long_text) for _ in range(10)]
    # budget tiny enough that each turn alone nearly fills a chunk
    chunks = chunk_turns(turns, token_budget=100)  # 400 char budget
    assert len(chunks) > 1
    # no turn lost, order preserved
    flat = [t for c in chunks for t in c]
    assert flat == turns


def test_chunk_turns_never_returns_empty_list():
    assert chunk_turns([]) == [[]]


def test_strip_think_removes_thinking_block():
    raw = '<think>let me reason about this</think>{"tldr": "ok"}'
    assert _strip_think(raw) == '{"tldr": "ok"}'


def test_strip_think_removes_code_fence():
    raw = '```json\n{"tldr": "ok"}\n```'
    assert _strip_think(raw) == '{"tldr": "ok"}'


def test_strip_think_passthrough_when_clean():
    raw = '{"tldr": "ok"}'
    assert _strip_think(raw) == '{"tldr": "ok"}'


def test_chat_structured_valid_json(monkeypatch):
    content = (
        '{"summary": "ok", "positions": {}, "entities": {"teams": [], "players": [], "events": []}}'
    )
    _mock_post(monkeypatch, [content])

    result = _chat_structured("prompt", ChunkSummary, "http://x", "m", "key")
    assert isinstance(result, ChunkSummary)
    assert result.summary == "ok"


def test_chat_structured_strips_think_and_code_fence(monkeypatch):
    content = (
        "<think>reasoning about the transcript</think>\n"
        '```json\n{"summary": "ok", "positions": {}, '
        '"entities": {"teams": [], "players": [], "events": []}}\n```'
    )
    _mock_post(monkeypatch, [content])

    result = _chat_structured("prompt", ChunkSummary, "http://x", "m", "key")
    assert result.summary == "ok"


def test_chat_structured_retries_after_invalid_json_then_succeeds(monkeypatch):
    prompts = _mock_post(
        monkeypatch,
        [
            "not json at all",
            '{"summary": "fixed", "positions": {}, '
            '"entities": {"teams": [], "players": [], "events": []}}',
        ],
    )

    result = _chat_structured("prompt", ChunkSummary, "http://x", "m", "key")
    assert result.summary == "fixed"
    assert len(prompts) == 2
    assert "invalid" in prompts[1].lower()


def test_chat_structured_retries_after_wrong_shape_then_succeeds(monkeypatch):
    prompts = _mock_post(
        monkeypatch,
        [
            '{"summary": "ok"}',  # missing positions/entities
            '{"summary": "ok", "positions": {}, '
            '"entities": {"teams": [], "players": [], "events": []}}',
        ],
    )

    result = _chat_structured("prompt", ChunkSummary, "http://x", "m", "key")
    assert result.summary == "ok"
    assert len(prompts) == 2


def test_chat_structured_raises_summary_parse_error_after_exhausting_retries(monkeypatch):
    prompts = _mock_post(monkeypatch, ["still not json"] * 5)

    with pytest.raises(SummaryParseError):
        _chat_structured("prompt", ChunkSummary, "http://x", "m", "key")
    assert len(prompts) == 3  # 1 initial attempt + 2 retries


def test_summarize_turns_single_chunk_validates_llm_json(monkeypatch):
    content = (
        '{"summary": "a test summary", "positions": {"speaker_0": "argued X"}, '
        '"entities": {"teams": ["Vitality"], "players": [], "events": []}}'
    )
    _mock_post(monkeypatch, [content])

    turns = [_turn("speaker_0", "some commentary")]
    result = summarize_turns(turns)
    assert isinstance(result, VideoSummary)
    assert result.tldr == "a test summary"
    assert result.positions == {"speaker_0": "argued X"}
    assert result.entities.teams == ["Vitality"]


def test_summarize_turns_reduces_multiple_chunks(monkeypatch):
    partial = (
        '{"summary": "partial", "positions": {}, '
        '"entities": {"teams": [], "players": [], "events": []}}'
    )
    final = (
        '{"tldr": "final merged summary", "positions": {}, '
        '"entities": {"teams": [], "players": [], "events": []}}'
    )
    prompts = _mock_post(monkeypatch, [partial, partial, final])

    import src.cs2_digest.summarize as summarize_mod

    monkeypatch.setattr(summarize_mod, "CHUNK_TOKEN_BUDGET", 10)  # force a split with short turns

    turns = [_turn("speaker_0", "word " * 50), _turn("speaker_1", "word " * 50)]
    result = summarize_turns(turns)
    assert result.tldr == "final merged summary"
    assert len(prompts) == 3  # 2 map + 1 reduce
