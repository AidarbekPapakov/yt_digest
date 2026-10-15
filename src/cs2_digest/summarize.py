"""Map-reduce summarization over ASR turns via an OpenAI-compatible endpoint.

Uses `requests` directly against the chat/completions endpoint rather than
the `openai` SDK: LiteLLM's endpoint is a plain HTTP API, and pulling in the
SDK for one call type isn't worth the extra dependency (Simplicity First).
"""

import json
import logging
import re

import requests
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class Entities(BaseModel):
    teams: list[str]
    players: list[str]
    events: list[str]


class ChunkSummary(BaseModel):
    summary: str
    positions: dict[str, str]
    entities: Entities


class VideoSummary(BaseModel):
    tldr: str
    positions: dict[str, str]
    entities: Entities


class SummaryParseError(Exception):
    """Raised when the LLM's JSON output still fails to parse or validate after retries."""


DEFAULT_BASE_URL = "http://litellm:4000"
DEFAULT_MODEL = "local-llm"
# Qwen3's context is 16k; ~10k tokens of transcript leaves room for the
# prompt and completion.
CHUNK_TOKEN_BUDGET = 10_000
# Rough chars-per-token for English transcript text; good enough for chunking,
# not for billing.
CHARS_PER_TOKEN = 4

MAP_PROMPT = """You are summarising part of a CS2 (Counter-Strike 2) esports commentary video.
The transcript below is machine-transcribed and may contain typos. Speakers
are labelled speaker_0, speaker_1, etc.

Transcript chunk {chunk_num}/{total_chunks}:
{transcript}

Return ONLY valid JSON with this shape:
{{
  "summary": "2-4 sentence summary of this chunk",
  "positions": {{"speaker_0": "what this speaker argued/said in this chunk", ...}},
  "entities": {{"teams": [...], "players": [...], "events": [...]}}
}}"""

REDUCE_PROMPT = """You are merging partial summaries of a CS2 esports commentary video
into one final digest entry. Here are the partial summaries, in order:

{partials}

Return ONLY valid JSON with this shape:
{{
  "tldr": "2-3 sentence overall summary",
  "positions": {{"speaker_0": "this speaker's overall position/argument across the video", ...}},
  "entities": {{"teams": [...], "players": [...], "events": [...]}}
}}
Merge duplicate entities and per-speaker positions rather than concatenating them."""


def _strip_think(content: str) -> str:
    """Strip Qwen3 thinking-mode <think> blocks, as asr/src/normalize._extract_json does."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    return content


def _chat_structured(
    prompt: str,
    schema: type[BaseModel],
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: int = 180,
    max_retries: int = 2,
) -> BaseModel:
    """Calls the chat endpoint and validates the reply against `schema`.

    On a `json.JSONDecodeError` or `ValidationError`, retries up to
    `max_retries` times, appending the error message to the original prompt
    each time so the model can fix it. Raises SummaryParseError once retries
    are exhausted.
    """
    original_prompt = prompt
    current_prompt = prompt
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": current_prompt}],
                "temperature": 0.2,
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            data = json.loads(_strip_think(content))
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            current_prompt = (
                f"{original_prompt}\n\nYour previous response was invalid: {e}\n"
                "Return ONLY valid JSON matching the required shape."
            )
    raise SummaryParseError(
        f"LLM output failed validation after {max_retries + 1} attempts: {last_error}"
    )


def chunk_turns(turns: list[dict], token_budget: int = CHUNK_TOKEN_BUDGET) -> list[list[dict]]:
    """Split turns into chunks of about `token_budget` tokens each, never
    splitting a single turn across chunks."""
    char_budget = token_budget * CHARS_PER_TOKEN
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for turn in turns:
        turn_chars = len(turn["text"]) + len(turn["speaker"]) + 4
        if current and current_chars + turn_chars > char_budget:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(turn)
        current_chars += turn_chars
    if current:
        chunks.append(current)
    return chunks or [[]]


def summarize_turns(
    turns: list[dict],
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
) -> VideoSummary:
    """Returns a validated VideoSummary."""
    chunks = chunk_turns(turns, token_budget=CHUNK_TOKEN_BUDGET)

    if len(chunks) == 1:
        transcript = "\n".join(f"[{t['speaker']}] {t['text']}" for t in chunks[0])
        prompt = MAP_PROMPT.format(chunk_num=1, total_chunks=1, transcript=transcript)
        partial = _chat_structured(prompt, ChunkSummary, base_url, model, api_key)
        return VideoSummary(
            tldr=partial.summary, positions=partial.positions, entities=partial.entities
        )

    partials = []
    for i, chunk in enumerate(chunks, start=1):
        transcript = "\n".join(f"[{t['speaker']}] {t['text']}" for t in chunk)
        prompt = MAP_PROMPT.format(chunk_num=i, total_chunks=len(chunks), transcript=transcript)
        partials.append(_chat_structured(prompt, ChunkSummary, base_url, model, api_key))

    reduce_prompt = REDUCE_PROMPT.format(
        partials=json.dumps([p.model_dump() for p in partials], indent=2)
    )
    return _chat_structured(reduce_prompt, VideoSummary, base_url, model, api_key)
