"""Windows-client upload guards: error rendering, pre-upload conversion, waiting.

Three real failures from the field drove this module, and each one is a case
that used to produce an *unactionable* message:

* ``HTTP 413: <html>...413 Request Entity Too Large...nginx/1.24.0 (Ubuntu)`` --
  nginx refuses the body **before** the app sees it, so the app's own clean JSON
  413 can never fire through the proxy. The client dumped the markup.
* ``HTTP 409: A transcription is already in progress`` -- correct, but it named
  neither the file that owns the server nor whether it can be cancelled.
* A 300 MB MP4 was uploaded unchanged, so it could only ever fail at the proxy.

The client now strips the video track and transcodes to 16 kHz mono FLAC before
uploading, describes proxy vs server errors differently, and waits out a 409.
"""

import json
from pathlib import Path
import time

import pytest

from conftest import transcribe_missing as transcribe_missing  # noqa: F401

pytestmark = transcribe_missing

NGINX_413 = (
    "<html>\r\n<head><title>413 Request Entity Too Large</title></head>\r\n"
    "<body>\r\n<center><h1>413 Request Entity Too Large</h1></center>\r\n"
    "<hr><center>nginx/1.24.0 (Ubuntu)</center>\r\n</body>\r\n</html>\r\n"
)


# --------------------------------------------------------------------------
# _describe_http_error
# --------------------------------------------------------------------------

def test_nginx_413_html_is_named_not_echoed(client):
    out = client._describe_http_error(413, NGINX_413, {"Content-Type": "text/html"})
    assert "nginx" in out or "reverse proxy" in out
    assert "200 MB" in out
    # The actionable half: what to do about it.
    assert "ffmpeg" in out and "--no-convert" in out
    # Never dump the markup.
    assert "<html>" not in out and "nginx/1.24.0" not in out


def test_server_json_413_is_reported_as_the_servers_own_limit(client):
    """If the app answered, the proxy is not what refused it."""
    body = json.dumps({"detail": "File too large (max 200MB)"})
    out = client._describe_http_error(413, body, {"Content-Type": "application/json"})
    assert "File too large (max 200MB)" in out
    assert "reverse proxy" not in out


def test_409_names_the_running_file_and_its_age(client):
    started = time.time() - 7 * 60
    body = json.dumps({"detail": "A transcription is already in progress",
                       "job": {"filename": "meeting.mp4", "started_at": started,
                               "can_cancel": True}})
    out = client._describe_http_error(409, body, {"Content-Type": "application/json"})
    assert "meeting.mp4" in out
    assert "7 min ago" in out
    assert "cancelled" in out


def test_409_for_another_users_job_does_not_offer_a_cancel(client):
    body = json.dumps({"detail": "A transcription is already in progress",
                       "job": {"filename": "other.mp4", "can_cancel": False}})
    out = client._describe_http_error(409, body, {"Content-Type": "application/json"})
    assert "other.mp4" in out
    assert "cancelled" not in out


def test_409_without_a_job_falls_back_to_the_raw_detail(client):
    body = json.dumps({"detail": "A transcription is already in progress"})
    out = client._describe_http_error(409, body, {"Content-Type": "application/json"})
    assert "already in progress" in out
    assert "the server is already transcribing" not in out


def test_401_reports_the_token_as_the_problem(client):
    body = json.dumps({"detail": "Invalid API key"})
    out = client._describe_http_error(401, body, {"Content-Type": "application/json"})
    assert "Invalid API key" in out


def test_html_500_is_reported_as_an_html_page_not_dumped(client):
    out = client._describe_http_error(500, "<html><body><h1>500</h1></body></html>",
                                      {"Content-Type": "text/html"})
    assert "HTML error page" in out
    assert "<h1>" not in out


def test_empty_body_still_says_something_useful(client):
    out = client._describe_http_error(502, "", {})
    assert "502" in out and "no error detail" in out


def test_fastapi_validation_list_is_flattened(client):
    body = json.dumps({"detail": [{"msg": "field required", "loc": ["file"]},
                                  {"msg": "invalid", "loc": ["language"]}]})
    out = client._describe_http_error(422, body, {"Content-Type": "application/json"})
    assert "field required" in out and "invalid" in out


def test_json_content_type_is_not_required_for_detection(client):
    """A proxy that drops the header must not change the message."""
    body = json.dumps({"detail": "Invalid API key"})
    assert "Invalid API key" in client._describe_http_error(401, body, {})


def test_malformed_json_is_not_fatal(client):
    out = client._describe_http_error(500, "{not json", {"Content-Type": "application/json"})
    assert "500" in out


def test_long_bodies_are_truncated(client):
    out = client._describe_http_error(500, "x" * 5000, {})
    assert len(out) < 400


# --------------------------------------------------------------------------
# prepare_upload -- what actually gets uploaded
# --------------------------------------------------------------------------

def _sparse(path: Path, size_mb: float) -> Path:
    """A file of `size_mb` MB that occupies no blocks.

    The limits under test are 200-300 MB; writing that for real filled the
    disk. ftruncate leaves a hole, and st_size is what prepare_upload reads.
    """
    import os
    with open(path, "wb") as fh:
        fh.truncate(int(size_mb * 1024 * 1024))
    assert path.stat().st_size == int(size_mb * 1024 * 1024)
    return path


def _fake(client, tmp_path, name, size_mb, *, has_video=True, converted_mb=None):
    """A Path of `size_mb` MB plus stubs for the probe/converter."""
    src = _sparse(tmp_path / name, size_mb)

    calls = {"probe": 0, "convert": 0}

    def probe(_path):
        calls["probe"] += 1
        return has_video

    def convert(path, dest=None):
        calls["convert"] += 1
        mb = converted_mb if converted_mb is not None else size_mb / 10
        return _sparse(path.with_name(f"{path.stem}.16k.flac"), mb)

    client.probe_has_video = probe
    client.convert_to_audio = convert
    return src, calls


def test_small_audio_is_uploaded_untouched(client, tmp_path):
    src, calls = _fake(client, tmp_path, "note.mp3", 1.0)
    assert client.prepare_upload(src) == src
    assert calls["convert"] == 0


def test_video_container_is_always_converted(client, tmp_path):
    src, calls = _fake(client, tmp_path, "clip.mp4", 1.0, converted_mb=0.2)
    out = client.prepare_upload(src)
    assert out != src and out.suffix == ".flac"
    assert calls["convert"] == 1


def test_large_audio_over_the_proxy_limit_is_converted(client, tmp_path, monkeypatch):
    """No video track, but the file would not survive the proxy."""
    monkeypatch.setattr(client, "PROXY_MAX_UPLOAD_BYTES", 8 * 1024 * 1024)
    src, calls = _fake(client, tmp_path, "long.m4a", 30.0, converted_mb=3.0)
    out = client.prepare_upload(src)
    assert calls["convert"] == 1
    assert out.stat().st_size < src.stat().st_size
    assert out.stat().st_size <= 8 * 1024 * 1024


def test_large_audio_that_already_fits_is_left_alone(client, tmp_path):
    """30 MB of FLAC fits the proxy; converting it would just cost the user a wait.

    This is deliberate and is why TRANSCODE_ABOVE_BYTES alone is not the rule.
    """
    src, calls = _fake(client, tmp_path, "already.flac", 30.0)
    assert client.prepare_upload(src) == src
    assert calls["convert"] == 0


def test_mislabelled_video_container_that_fits_is_uploaded_as_audio(client, tmp_path):
    src, calls = _fake(client, tmp_path, "notreallyvideo.mp4", 5.0, has_video=False)
    assert client.prepare_upload(src) == src
    assert calls["convert"] == 0


def test_no_convert_leaves_a_large_file_alone_when_it_fits(client, tmp_path):
    src, calls = _fake(client, tmp_path, "big.mp4", 50.0)
    assert client.prepare_upload(src, convert=False) == src
    assert calls["convert"] == 0


def test_no_convert_over_the_proxy_limit_fails_with_the_actual_size(client, tmp_path):
    """The real 413 the user hit: 250 MB with --no-convert."""
    src, _ = _fake(client, tmp_path, "huge.mp4", 250.0)
    with pytest.raises(client.ClientError) as exc:
        client.prepare_upload(src, convert=False)
    msg = str(exc.value)
    assert "250 MB" in msg and "200 MB" in msg and "--no-convert" in msg


def test_a_conversion_that_does_not_get_under_the_limit_is_reported(client, tmp_path):
    """Stripping the video is not a guarantee; do not claim it succeeded."""
    src, _ = _fake(client, tmp_path, "stillhuge.mp4", 300.0, converted_mb=250.0)
    with pytest.raises(client.ClientError) as exc:
        client.prepare_upload(src)
    assert "still" in str(exc.value) and "250 MB" in str(exc.value)


# --------------------------------------------------------------------------
# wait_for_idle -- the 409 retry path
# --------------------------------------------------------------------------

class _Stream:
    """What open_request returns: an iterable of byte lines, and closeable.

    Both wait_for_idle and transcribe_file call resp.close() in a finally, so a
    plain list iterator is not a faithful stand-in.
    """

    def __init__(self, lines, headers=None):
        self._lines = list(lines)
        self.headers = headers or {"Content-Type": "text/event-stream"}
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def read(self, n=-1):
        return b""

    def close(self):
        self.closed = True


def _sse(*events):
    return [f"data: {json.dumps(e)}\n".encode() for e in events]


def test_wait_returns_immediately_when_the_server_is_idle(client, monkeypatch):
    calls = []

    def fake_open(method, url, **kw):
        calls.append(url)
        return _Stream(_sse({"active": False, "job": None}))

    monkeypatch.setattr(client, "open_request", fake_open)
    client.wait_for_idle("http://x", "tok")
    assert calls == ["http://x/api/asr/activity/stream"]


def test_wait_blocks_until_the_running_job_finishes(client, monkeypatch):
    monkeypatch.setattr(client.time, "monotonic", lambda: 0.0)
    seen = []

    def fake_open(method, url, **kw):
        seen.append("opened")
        # snapshot: busy; keep-alive; then the finish transition
        return _Stream([
            b": ping\n",
            *_sse({"active": True, "job": {"filename": "meeting.mp4"}}),
            b": ping\n",
            *_sse({"active": False, "job": None}),
        ])

    monkeypatch.setattr(client, "open_request", fake_open)
    client.wait_for_idle("http://x", "tok", timeout=60)
    assert seen == ["opened"]


def test_wait_reports_what_it_is_waiting_for(client, monkeypatch, capsys):
    monkeypatch.setattr(client.time, "monotonic", lambda: 0.0)

    def fake_open(method, url, **kw):
        return _Stream(_sse({"active": True, "job": {"filename": "meeting.mp4"}},
                             {"active": False}))

    monkeypatch.setattr(client, "open_request", fake_open)
    client.wait_for_idle("http://x", "tok", timeout=60)
    err = capsys.readouterr().err
    assert "meeting.mp4" in err and "waiting" in err


def test_wait_times_out_instead_of_blocking_forever(client, monkeypatch):
    # Every event re-checks the deadline; step the clock past it each time.
    clock = {"t": 0.0}
    monkeypatch.setattr(client.time, "monotonic", lambda: clock["t"])
    clock_bump = {"n": 0}

    def fake_open(method, url, **kw):
        def gen():
            for _ in range(50):
                clock["t"] += 30.0
                yield _sse({"active": True, "job": {"filename": "x.mp4"}})[0]
        return _Stream(list(gen()))

    monkeypatch.setattr(client, "open_request", fake_open)
    with pytest.raises(client.ClientError) as exc:
        client.wait_for_idle("http://x", "tok", timeout=60)
    assert "Timed out" in str(exc.value)


def test_wait_gives_up_immediately_when_disabled(client, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("should not open a stream")

    monkeypatch.setattr(client, "open_request", boom)
    client.wait_for_idle("http://x", "tok", timeout=0)


def test_a_broken_activity_stream_does_not_block_the_upload(client, monkeypatch):
    """If we cannot tell whether the server is busy, let the upload try."""
    def fake_open(method, url, **kw):
        raise OSError("connection reset")

    monkeypatch.setattr(client, "open_request", fake_open)
    client.wait_for_idle("http://x", "tok", timeout=60)   # must not raise


def test_transcribe_retries_once_after_a_409(client, tmp_path, monkeypatch):
    src, _ = _fake(client, tmp_path, "a.wav", 1.0)
    posts = []

    def fake_open(method, url, **kw):
        posts.append(url)
        if len(posts) == 1:
            raise client.ClientError("HTTP 409: the server is already transcribing 'b.mp4'.")
        return _Stream(_sse({"stage": "done", "result": {
            "results": [{"start": 0.0, "end": 1.0, "speaker": "Gergely Papp",
                         "text": "hello"}],
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "Gergely Papp"}],
            "speakers": ["Gergely Papp"],
        }}))

    waited = []
    monkeypatch.setattr(client, "open_request", fake_open)
    monkeypatch.setattr(client, "wait_for_idle",
                        lambda *a, **kw: waited.append(kw))
    monkeypatch.setattr(client, "encode_multipart", lambda *a, **kw: (b"", "text/plain"))

    paths, status = client.transcribe_file("http://x", "tok", src, "en")
    assert len(posts) == 2, "the upload was not retried after the 409"
    assert waited, "wait_for_idle was not called"
    assert paths and paths[0].name == "a.txt"
    assert status == "ok"


def test_transcribe_gives_up_after_one_retry(client, tmp_path, monkeypatch):
    src, _ = _fake(client, tmp_path, "a.wav", 1.0)
    posts = []

    def fake_open(method, url, **kw):
        posts.append(url)
        raise client.ClientError("HTTP 409: the server is already transcribing 'b.mp4'.")

    monkeypatch.setattr(client, "open_request", fake_open)
    monkeypatch.setattr(client, "wait_for_idle", lambda *a, **kw: None)
    monkeypatch.setattr(client, "encode_multipart", lambda *a, **kw: (b"", "text/plain"))

    with pytest.raises(client.ClientError) as exc:
        client.transcribe_file("http://x", "tok", src, "en")
    assert "409" in str(exc.value)
    assert len(posts) == 2, "the retry must not loop"


def test_a_413_is_not_retried(client, tmp_path, monkeypatch):
    """Waiting cannot help with a size problem; retrying would hang."""
    src, _ = _fake(client, tmp_path, "a.wav", 1.0)
    posts = []

    def fake_open(method, url, **kw):
        posts.append(url)
        raise client.ClientError("HTTP 413: the reverse proxy rejected the upload")

    monkeypatch.setattr(client, "open_request", fake_open)
    monkeypatch.setattr(client, "wait_for_idle", lambda *a, **kw: pytest.fail("waited"))
    monkeypatch.setattr(client, "encode_multipart", lambda *a, **kw: (b"", "text/plain"))

    with pytest.raises(client.ClientError):
        client.transcribe_file("http://x", "tok", src, "en")
    assert len(posts) == 1


def test_no_wait_disables_the_retry(client, tmp_path, monkeypatch):
    src, _ = _fake(client, tmp_path, "a.wav", 1.0)
    posts = []

    def fake_open(method, url, **kw):
        posts.append(url)
        raise client.ClientError("HTTP 409: the server is already transcribing 'b.mp4'.")

    monkeypatch.setattr(client, "open_request", fake_open)
    monkeypatch.setattr(client, "wait_for_idle", lambda *a, **kw: pytest.fail("waited"))
    monkeypatch.setattr(client, "encode_multipart", lambda *a, **kw: (b"", "text/plain"))

    with pytest.raises(client.ClientError):
        client.transcribe_file("http://x", "tok", src, "en", wait=False)
    assert len(posts) == 1