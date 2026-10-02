"""Integration test against a real ClickHouse (not mocked): requires
`docker compose up -d` in this repo. Skipped automatically if unreachable."""

import os
from datetime import datetime, timezone

import pytest

from src.cs2_digest import store

CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD")


def _get_client():
    return store.get_client(
        host="localhost", port=8125, username="cs2_digest",
        password=CLICKHOUSE_PASSWORD, database="cs2_digest",
    )


@pytest.fixture
def client():
    if not CLICKHOUSE_PASSWORD:
        pytest.skip("CLICKHOUSE_PASSWORD not set; export it or source .env")
    try:
        c = _get_client()
    except Exception as e:
        pytest.skip(f"ClickHouse unreachable: {e}")
    store.ensure_tables(c)
    return c


def test_full_ledger_lifecycle(client):
    video_id = f"test_{datetime.now().timestamp()}"
    channel = "test_channel"

    store.upsert_video(
        client, video_id, channel, "Test Video",
        datetime.now(timezone.utc), 180, "discovered",
    )
    # Not yet terminal: still eligible for (re)discovery, e.g. if the run that
    # would have advanced it crashed before reaching a terminal status.
    assert video_id not in store.known_video_ids(client, channel)

    store.mark_status(client, video_id, channel, "fetched")
    store.mark_status(client, video_id, channel, "transcribed")

    turns = [{"speaker": "speaker_0", "text": "hello world", "start": 0.0, "end": 1.0}]
    store.insert_transcript(client, video_id, turns, {"source": "test"})

    store.insert_summary(
        client, video_id, "local-llm", "a test tldr",
        {"speaker_0": "said hello"}, {"teams": [], "players": [], "events": []},
    )
    store.mark_status(client, video_id, channel, "summarised")

    pending = store.videos_pending_email(client)
    pending_ids = {v["video_id"] for v in pending}
    assert video_id in pending_ids

    row = next(v for v in pending if v["video_id"] == video_id)
    assert row["tldr"] == "a test tldr"
    assert row["positions"] == {"speaker_0": "said hello"}

    store.mark_status(client, video_id, channel, "emailed")
    assert video_id in store.known_video_ids(client, channel)


def test_known_video_ids_retries_stuck_but_not_failed(client):
    channel = "test_channel_retry"
    stuck_id = f"test_stuck_{datetime.now().timestamp()}"
    failed_id = f"test_failed_{datetime.now().timestamp()}"
    done_id = f"test_done_{datetime.now().timestamp()}"

    for vid, status in [(stuck_id, "transcribed"), (failed_id, "failed"), (done_id, "emailed")]:
        store.upsert_video(
            client, vid, channel, "Test Video",
            datetime.now(timezone.utc), 180, status,
        )

    known = store.known_video_ids(client, channel)
    assert stuck_id not in known  # retry it: never reached a terminal status
    assert failed_id in known  # permanent failure, don't auto-retry
    assert done_id in known  # already emailed, don't reprocess
