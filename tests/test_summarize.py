from src.cs2_digest.summarize import _strip_think, chunk_turns, summarize_turns


def _turn(speaker, text):
    return {"speaker": speaker, "text": text}


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


def test_summarize_turns_single_chunk_validates_llm_json(monkeypatch):
    import src.cs2_digest.summarize as summarize_mod

    def fake_chat_json(prompt, base_url, model, api_key, timeout_s=180):
        return {
            "summary": "a test summary",
            "positions": {"speaker_0": "argued X"},
            "entities": {"teams": ["Vitality"], "players": [], "events": []},
        }

    monkeypatch.setattr(summarize_mod, "_chat_json", fake_chat_json)

    turns = [_turn("speaker_0", "some commentary")]
    result = summarize_turns(turns)
    assert result["tldr"] == "a test summary"
    assert result["positions"] == {"speaker_0": "argued X"}
    assert "teams" in result["entities"]


def test_summarize_turns_reduces_multiple_chunks(monkeypatch):
    import src.cs2_digest.summarize as summarize_mod

    calls = []

    def fake_chat_json(prompt, base_url, model, api_key, timeout_s=180):
        calls.append(prompt)
        if len(calls) <= 2:  # the two map calls
            return {"summary": "partial", "positions": {}, "entities": {"teams": [], "players": [], "events": []}}
        return {"tldr": "final merged summary", "positions": {}, "entities": {"teams": [], "players": [], "events": []}}

    monkeypatch.setattr(summarize_mod, "_chat_json", fake_chat_json)
    monkeypatch.setattr(summarize_mod, "CHUNK_TOKEN_BUDGET", 10)  # force a split with short turns

    turns = [_turn("speaker_0", "word " * 50), _turn("speaker_1", "word " * 50)]
    result = summarize_turns(turns)
    assert result["tldr"] == "final merged summary"
    assert len(calls) == 3  # 2 map + 1 reduce
