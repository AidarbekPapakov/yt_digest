# cs2_digest: plan

Two linked portfolio projects, planned on 2026-09-29. Nothing is built yet.

1. **cs2_digest** (this repo): a daily Airflow batch job. It finds new YouTube
   videos from CS2 personalities, transcribes them with
   `audio_analytics`'s ASR service, summarises them with an LLM, and emails a
   digest.
2. **Speaker identification**: a new stage inside the `audio_analytics` repo's
   `asr` package. Nemotron 3 Diarization finds who spoke
   when, pyannote embeddings match each diarized speaker against enrolled
   voiceprints, and the result fills `speaker_map` automatically. The digest
   can then say "Thorin argues X" instead of "speaker_1 argues X".

Build order: Stage 1 → 2 → 3 (project 1 works end to end with anonymous
speakers) → 4 (project 2) → 5 (the digest uses names).

## Decisions (made with the user, don't relitigate)

| Topic | Decision |
|---|---|
| Repo layout | New repo `cs2_digest` (DAG, discovery, summariser, ClickHouse). Speaker ID goes **into** `audio_analytics/asr` as a pipeline stage. |
| ASR choice | Benchmark in a Jupyter notebook: GigaAM vs Whisper vs Parakeet. The write-up must **highlight GigaAM's much lower VRAM and higher speed**, and that its decoding can be improved further with beam search and hotwords (the service already has `DECODER=beam` and hotword-boosted CTC prefix beam search). |
| Summary LLM | Any **OpenAI-compatible endpoint**, configured through env/Variables, so frontier models can be swapped in for testing. The default is `local-llm` (Qwen3-8B) through LiteLLM. |
| Cadence | Daily. One email covers all videos since the last run, and the email is skipped when there is nothing new. |
| Sources | YouTube only (RSS discovery + `yt-dlp`). The list will grow, so keep it in a config file. |
| Digest content, per video | TL;DR · who argued what (per-speaker positions and disagreements) · entities mentioned (teams, players, events, patches). No timestamp links. |
| Storage | A **new ClickHouse inside cs2_digest** (its own compose and network). |
| Email | Airflow's existing Gmail SMTP (`airflow.utils.email.send_email`, as the retrain and price DAGs use it). HTML body. |
| Diarizer | **Nemotron is the default, pyannote community-1 the fallback** (`DIARIZATION_BACKEND=nemotron|pyannote`). The user's private benchmarks show Nemotron far ahead. The CS2-podcast DER comparison is only a formality for the public repo. |

## Channels (v1)

To go into `config/channels.yaml`. For each channel, resolve the `channel_id`
(`UC…`) for the RSS URL, and have the user confirm it.

- Thorin
- launders
- Mauisnake
- Pimp
- HLTV Confirmed
- kassad
- Last Free Nation CS. The user named these shows alongside it: "Hot Take",
  "Point Made", "Counter Points", "Snake & Banter". **Check whether these are
  shows on the Last Free Nation channel or separate channels**, and ask the
  user if it's unclear.

Each entry: `name`, `channel_id`, optional `known_speakers` (used in phase 5),
optional `min_duration_s` / `max_duration_s` (to skip Shorts and 6-hour VODs).

---

## Stage 1: ASR (+ diarization) benchmark notebook

`cs2_digest/notebooks/01_asr_benchmark.ipynb`

1. **Pick about 10 clips** of 2–3 minutes each from different channels. Include
   segments with many player and team names, crosstalk, and 3 or more speakers.
   Store them under `data/bench/`, which is gitignored (it holds audio of real
   people).
2. **Write reference transcripts** by hand: run Whisper, then correct the output
   by hand. For the diarization comparison, also add speaker labels with rough
   turn boundaries (RTTM).
3. **Candidates:**
   - GigaAM (the engine currently in asr_service) in three modes: greedy · beam
     · beam + CS2 hotwords
   - Whisper large-v3-turbo
   - Parakeet (NeMo; pick the current English/multilingual TDT checkpoint)
4. **Metrics:** WER, **entity recall** (the share of player/team names spelled
   correctly), RTF, peak VRAM, and CPU vs GPU where it matters. The
   GPU is an 8 GB RTX 3070 that llama.cpp mostly fills. Record what fits
   alongside it.
5. **Diarization, for completeness:** DER of Nemotron vs pyannote community-1 on
   the same clips.
6. **Output:** a results table for the README. Frame it as a trade-off. GigaAM
   is cheaper and faster, and beam search plus hotwords may close the accuracy gap.

Verify: the notebook runs top to bottom, and the table has every model × every metric.

Open check: GigaAM's current checkpoint is `multilingual_large_ctc`. Confirm
how well it handles English before assuming anything.

## Stage 2: prepare asr_service for English YouTube audio

Where: the `audio_analytics` repo's `asr` package. Respect its existing
conventions: the service is domain-neutral, and `HOTWORDS = {}` stays empty.

1. **Engine:** keep GigaAM, or add a switchable engine if Stage 1 says so. Add
   no CS2-specific code to the service.
2. **Hotwords:** cs2_digest sends the CS2 roster **per request**
   (`hotwords` form field). The roster lives in `cs2_digest/config/hotwords.yaml`.
3. **Stereo:** YouTube stereo is one mix, not one speaker per channel. cs2_digest
   downmixes before uploading (`ffmpeg -ac 1 -ar 16000`), so the service diarizes
   the audio and needs no change for this.
4. **Long audio:** everything so far was tested on 23–28 s clips. Test a
   60–90 minute file. Watch memory, the HTTP timeout, and LitServe request
   limits. CPU diarization at the measured speed (~0.5× real time) means about 40–60 min per
   90 min of audio. Nemotron on GPU should cut this, but it competes with
   llama.cpp for the 8 GB. Decide here whether to chunk on the client side.

Verify: `/analyze` on a full-length downmixed episode returns sensible
turns, and the results are recorded in audio_analytics' own notes.

## Stage 3: cs2_digest pipeline (anonymous speakers)

Planned layout:

```
cs2_digest/
  compose.yml            ClickHouse (own network: cs2_digest_default)
  config/channels.yaml   channels (above)
  config/hotwords.yaml   CS2 player/team roster, ~100 names
  src/cs2_digest/
    discover.py          RSS → new video ids (not in ledger)
    fetch.py             yt-dlp audio-only + ffmpeg mono 16 kHz
    transcribe.py        POST asr_service /analyze (hotwords, run_llm=0)
    summarize.py         OpenAI-compatible client, map-reduce
    store.py             ClickHouse reads/writes
    email.py             HTML digest rendering
  dags/cs2_digest_dag.py deployed copy lives in the shared Airflow instance
                         (`airflowbl4` repo)
  notebooks/
  tests/
```

Follow the shared Airflow instance's conventions. The origin repo keeps the
DAG, and a hand-synced copy is deployed in the `airflowbl4` repo's `dags/`.
Shared code is copied into its `src/`. Use the TaskFlow API (`airflow.sdk`),
polars, and `logging`. Config goes in Airflow Variables through
`provision_variables.sh`.

Steps:

1. **ClickHouse** (`compose.yml`). Pick a host port that's free: 8123, 8124,
   8023 and 9000 are taken. Tables:
   - `videos`: `video_id`, `channel`, `title`, `published_at`, `duration_s`,
     `status` (discovered/transcribed/summarised/emailed/failed), `error`,
     `updated_at`. This is the idempotency ledger (ReplacingMergeTree on
     `updated_at`).
   - `transcripts`: `video_id`, turns (speaker, start, end, text), and
     `asr_meta` JSON.
   - `summaries`: `video_id`, `model`, `tldr`, `positions` (JSON per speaker),
     `entities` (arrays per type), `created_at`.
2. **Networks:** airflow's `docker-compose.yaml` joins `cs2_digest_default` and
   `audio_analytics_default`, which lets it reach `asr_service:8400` and
   `litellm:4000` by name.
3. **discover:** fetch `https://www.youtube.com/feeds/videos.xml?channel_id=<id>`,
   parse it, drop ids that are already in `videos`, and filter by duration once it's known.
4. **fetch:** `yt-dlp -x` into a temp dir, then downmix to mono 16 kHz. Delete the
   audio after transcription. The downloaded audio is never stored long term.
5. **transcribe:** call `/analyze` with the hotwords and a generous timeout.
   Store the turns.
6. **summarize:** use the `openai` client with `LLM_BASE_URL` / `LLM_MODEL` /
   `LLM_API_KEY`, defaulting to `http://litellm:4000` / `local-llm` /
   `LITELLM_MASTER_KEY`.
   - **Map:** split the turns into chunks of about 10k tokens (Qwen3's context is 16k),
     then extract a partial summary, per-speaker positions and entities as JSON.
   - **Reduce:** merge the partials into the final `{tldr, positions, entities}`.
   - Strip `<think>` (Qwen3 thinking mode) the way
     `audio_analytics/asr/src/normalize._extract_json` does, or turn it off.
   - Calls to local-llm are traced in Langfuse automatically through LiteLLM.
7. **email:** a `@task.short_circuit` that skips when nothing new was summarised,
   then `send_email` with HTML grouped by channel. Mark the videos `emailed`.
8. **DAG** `cs2_digest`, `@daily`,
   `discover → fetch → transcribe → summarize → email`. Map over videos with
   dynamic task mapping, and cap the concurrency of the ASR stage at 1. A single failed
   video is marked `failed` and doesn't block the others.

Verify: unit tests for the RSS parsing, chunking and reduce-prompt JSON
validation. Then a manual DAG run on one real short video, which should
produce a correct row in each table and one email.

## Stage 4: speaker identification (in the audio_analytics repo's asr package)

1. **Nemotron backend.** Add `DIARIZATION_BACKEND=nemotron|pyannote` (default
   `nemotron`) in `config.py`/`diarization.py`. It must produce the same output
   shape (one zero-filled track per speaker, ordered by first onset), so the
   rest of the pipeline stays unchanged. If Nemotron fails, fall back to
   pyannote, with the reason in `meta.diarization`. Model:
   `nvidia/Nemotron-3-Diarization` (100M params, up to 8 speakers, offline
   and streaming). Expect NeMo to add image size. Decide on GPU or CPU from
   Stage 2's timing.
2. **Embeddings.** Use the pyannote embedding model already in the community-1
   snapshot (`model_weights/.../community-1/embedding`). For each diarized
   speaker, embed only non-overlapping turns of 2 s or more, then average and
   L2-normalise the embeddings. Enrolment and queries **must use the same
   embedding model**.
3. **Gallery (enrolment).** Store 3–5 clean clips of about 30 s per person,
   taken from solo videos. An enrolment script (in cs2_digest) writes
   `voiceprints/<name>.npy`. The service reads the gallery from a mounted
   `VOICEPRINTS_DIR`, so it stays domain-neutral: the voiceprints are data,
   not code.
4. **Matching.** This is open-set identification. For each speaker, take the
   gallery entry with the highest cosine similarity. Accept it above a
   threshold, otherwise label the speaker `unknown_<k>`. No two speakers in one
   file get the same name (use Hungarian assignment or greedy matching by score).
   Turn it on per request with `identify=1`. The matches fill `speaker_map`, and
   an explicit `speaker_map` in the request still wins. Report the scores in
   `meta.identification`.
5. **Evaluation notebook** (cs2_digest `notebooks/02_speaker_id.ipynb`):
   hand-label about 20 segments and report identification accuracy, EER (which
   sets the threshold), and the confusion between similar voices. Also report
   the Nemotron vs community-1 DER from Stage 1. This table is what the
   portfolio shows.

Verify: unit tests for the matching (threshold, no duplicates, the unknown
path) on synthetic embeddings, inside the image
(see audio_analytics' own docs, "Running and checking"). Then `/verify` on a
real clip with enrolled speakers.

## Stage 5: names in the digest

cs2_digest sends `identify=1`. The `known_speakers` in `channels.yaml` can
narrow the gallery for each channel. The summariser prompt then uses real names,
and `unknown_<k>` speakers are called "a guest".

## Privacy and licensing (applies to both)

- Voiceprints are biometric data about real people. Keep enrolment audio,
  `voiceprints/`, benchmark clips, downloaded audio and full transcripts
  **out of git**, and don't publish them. Publish the code and the
  aggregate metrics only.
- The digest email goes only to the user.

## Open items

- Resolve the channel ids, and settle the Last Free Nation show names (above).
- The ASR engine outcome from Stage 1.
- Whether long-audio handling needs client-side chunking (Stage 2).
- The ClickHouse host port for cs2_digest.
