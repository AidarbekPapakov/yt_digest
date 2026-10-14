# cs2_digest

A daily Airflow job that reads me the CS2 scene so I don't have to watch
YouTube all day. Every morning it checks a list of CS2 personalities for new
videos, transcribes the audio, asks a local LLM "who said what," and emails
me a short digest instead of a two-hour VOD.

I built this to solve a real annoyance: following Counter-Strike commentary
means a dozen channels each posting 10-40 minute videos, most of which is one
or two interesting takes buried in filler. This pulls out the takes.

## What it actually does

```
RSS feed → yt-dlp → ASR + diarization → LLM summary → email
```

1. **Discover** — polls each channel's YouTube RSS feed (no API key needed),
   filters out Shorts and multi-hour VODs, and skips anything it's already
   processed.
2. **Fetch** — downloads just the audio with `yt-dlp` and downmixes it to
   mono 16kHz with `ffmpeg`.
3. **Transcribe** — sends the audio to a GigaAM-based ASR service with
   speaker diarization, boosted with a list of ~95 CS2 player/team/event
   names so names like "ZywOo" or "m0NESY" don't come out mangled.
4. **Summarize** — a local LLM (any OpenAI-compatible endpoint works) reads
   the transcript and writes a TL;DR, pulls out what each speaker argued,
   and lists the entities mentioned.
5. **Email** — if there's anything new, one HTML digest goes out covering
   every video since the last run. If there's nothing new, it stays quiet.

Every video's progress is tracked in ClickHouse, so a crashed run or a bad
network connection doesn't mean losing work or sending duplicate emails —
the next run just picks up from wherever that video got stuck.

## Why this stack

- **GigaAM over Whisper/Parakeet** — benchmarked all three on real CS2 clips
  (see `notebooks/`). GigaAM lands close on accuracy but uses meaningfully
  less VRAM and runs faster, which matters when it's sharing an 8GB GPU with
  a locally-hosted LLM.
- **ClickHouse as the job ledger** — this isn't "big data," it's used here
  because `ReplacingMergeTree` gives cheap idempotent upserts, which is
  exactly the shape of "mark this video's pipeline stage, retry safely if it
  fails."
- **Airflow TaskFlow API with dynamic task mapping** — each video fans out
  into its own task instance per stage, so one bad video (a dead link, a
  weird audio format) fails on its own and doesn't take the rest of the
  batch down with it.
- **Any OpenAI-compatible LLM endpoint** — defaults to a local Qwen3 model
  through LiteLLM, but swapping in a frontier model for comparison is just a
  config change, not a code change.

## Project layout

```
src/cs2_digest/
  discover.py      RSS polling + "have I seen this video" filtering
  fetch.py         download + downmix audio
  transcribe.py    talks to the ASR service, handles hotword boosting
  summarize.py     map-reduce summarization + JSON parsing of LLM output
  store.py         ClickHouse schema and all reads/writes
  email.py         renders the digest HTML (pure function, easy to test)
  config/
    channels.yaml  which YouTube channels to watch
    hotwords.yaml  CS2 names to boost during transcription
dags/
  cs2_digest_dag.py  the Airflow DAG that wires all of the above together
tests/               unit tests + a ClickHouse integration test
```

## Running it

```bash
docker compose up -d          # starts this project's own ClickHouse
pip install -r requirements.txt -r requirements-dev.txt
pytest                         # unit tests only
pytest -m integration          # also run the ClickHouse integration test
```

The integration test is marked `integration` and skipped by default (it talks
to a real ClickHouse and skips itself cleanly if one isn't reachable — no
mocking the database just to make CI green).

The DAG itself is deployed into a separate, shared Airflow instance; see
`dags/cs2_digest_dag.py`'s module docstring for how the two copies stay in
sync.

## Where this is headed

Right now the digest says "speaker_1 argued X" because speakers are
anonymous. The next phase adds voice-based speaker identification (pyannote
embeddings matched against enrolled voiceprints of each personality), so the
digest can instead say "Thorin argued X." The full phased plan — including
the ASR benchmark writeup — is in `PLAN.md`.

## A note on how this was built

I built this with Claude Code doing most of the typing, under close
supervision — I made the architecture calls, caught the design bugs, and
decided what was worth fixing versus what could wait. I'm including that
plainly rather than hiding it: using AI well is itself a skill, and I'd
rather show the real process than pretend otherwise.
