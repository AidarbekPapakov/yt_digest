from src.cs2_digest.email import render_digest_html


def test_render_digest_html_groups_by_channel_and_escapes():
    videos = [
        {
            "video_id": "v1",
            "channel": "Thorin",
            "title": "<script>alert(1)</script>",
            "tldr": "A summary",
            "positions": {"speaker_0": "argued X"},
            "entities": {"teams": ["Vitality"], "players": [], "events": []},
        },
    ]
    html = render_digest_html(videos)
    assert "Thorin" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "Vitality" in html


def test_render_digest_html_empty_list():
    assert render_digest_html([]) == "<h1>cs2_digest</h1>"
