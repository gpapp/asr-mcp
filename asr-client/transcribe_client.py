import http.client
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus", ".mkv", ".mp4"}

#: All inbound files are converted to 16 kHz mono FLAC before upload so the
#: server receives a compact, processable format regardless of the original
#: size or container. The converted file is temporary and is never left behind.
CONVERT_SAMPLE_RATE = "16000"
CONVERT_CHANNELS = "1"
CONVERT_CODEC = "flac"

#: How long to wait for a running job to finish before giving up, and how often
#: to re-check. The server runs one transcription at a time.
JOB_WAIT_TIMEOUT_SEC = 60 * 60
JOB_POLL_INTERVAL_SEC = 5.0

#: Shown instead of a speaker name when the server could not attribute the
#: utterance to a confident speaker (see the server's uncertainty policy).
UNKNOWN_SPEAKER_LABEL = "UNKNOWN"

USAGE = """Usage:
  transcribe <audio file> [more files...]   Transcribe files (writes <name>.txt next to each)
                                            Optional: --language <code|auto> (overrides .env LANGUAGE)
                                            Optional: --json (also write <name>.json with raw segments)
                                            Optional: --no-wait (fail instead of waiting for a busy server)
  status                                    Check server/token connectivity
  voiceprints                               List speakers and voiceprints
  voiceprint-add <name> <file> [start] [end]   Create/refine a voiceprint from audio
                                             (replaces the speaker's earlier
                                              snippets from this same file)
  voiceprint-refine <name>                  Rebuild a voiceprint from its snippets

All inbound files are converted locally to 16 kHz mono FLAC (processable audio)
before upload and streamed to the server; the temporary converted file is removed
afterwards and never left behind.

If the server is already transcribing something, this waits for it to finish
(override with --no-wait). The server accepts one transcription at a time.

Segments the server could not attribute to a confident speaker are written as
[UNKNOWN (<reason>)] with their text kept — the .txt is never silently clean.

Configuration: edit .env next to this script (SERVER_URL + TOKEN, optional LANGUAGE=auto,
optional WRITE_JSON=1)."""


class ClientError(Exception):
    pass


def load_env():
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def get_language():
    """Forced transcription language: LANGUAGE (or TRANSCRIBE_LANGUAGE) from .env, default auto."""
    env = load_env()
    lang = (env.get("LANGUAGE") or env.get("TRANSCRIBE_LANGUAGE") or "auto").strip().lower()
    return lang or "auto"


def get_json_sidecar() -> bool:
    """WRITE_JSON=1 in .env (or --json) also writes <name>.json with raw segments."""
    return _JSON_SIDECAR


_JSON_SIDECAR = (load_env().get("WRITE_JSON") or "").strip().lower() in ("1", "true", "yes", "on")


def get_config():
    env = load_env()
    base = (env.get("SERVER_URL") or "").strip().rstrip("/")
    token = (env.get("TOKEN") or "").strip()
    if not base:
        raise ClientError(f"SERVER_URL is missing — copy .env.example to {ENV_PATH.name} and set it "
                          "(e.g. http://127.0.0.1:8087/asr-mcp)")
    if not token:
        raise ClientError("TOKEN is missing — open the web UI, use 'Generate token' in the "
                          "Windows Client Token card, and paste it into .env")
    return base, token


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _describe_http_error(code: int, body: str, headers=None) -> str:
    """Turn an error body into a sentence a human can act on.

    Three shapes arrive here:
      * the server's JSON  {"detail": "..."} / {"error": "..."};
      * nginx's HTML 413 page, which the client cannot parse and which used to
        be dumped verbatim into the console;
      * anything else, truncated.
    """
    ctype = (headers or {}).get("Content-Type", "") if headers else ""

    detail = None
    job = None
    looks_json = "json" in ctype.lower() or body.lstrip().startswith(("{", "["))
    if looks_json:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                job = parsed.get("job")
                for key in ("detail", "error", "message"):
                    if parsed.get(key):
                        detail = parsed[key]
                        break
                if detail is None and isinstance(parsed.get("detail"), list):
                    detail = "; ".join(str(d) for d in parsed["detail"])
        except ValueError:
            pass

    # --- nginx (and other proxies) reject oversized bodies before the app ---
    # Only when the body is NOT the server's own JSON: if the server answered,
    # the proxy is not what refused it.
    if detail is None and (code in (413, 414) or "Request Entity Too Large" in body):
        return (
            f"HTTP {code}: the reverse proxy rejected the upload before it reached "
            "the server. The client converts all inbound files to 16 kHz mono FLAC "
            "before upload to keep them small; if this still occurs, the file is too "
            "long for the server/proxy limit."
        )

    # --- a 409 carries the running job, which is the actionable part --------
    if code == 409 and isinstance(job, dict):
        who = job.get("filename") or "another file"
        age = ""
        started = job.get("started_at")
        if isinstance(started, (int, float)):
            mins = int(max(time.time() - started, 0) // 60)
            age = f" (started {mins} min ago)" if mins else ""
        lines = [f"HTTP 409: the server is already transcribing '{who}'{age}."]
        if job.get("can_cancel"):
            lines.append(
                "That job is yours and can be cancelled from the web UI "
                "(Transcription tab) if it is stuck."
            )
        return " ".join(lines)

    if detail is None:
        stripped = body.strip()
        if "<" in stripped[:200] and ">" in stripped[:200]:
            # An HTML page from a proxy: name the status, never echo the markup.
            return f"HTTP {code}: the server or proxy returned an HTML error page"
        detail = stripped[:300] or f"no error detail (HTTP {code})"
    if not isinstance(detail, str):
        detail = json.dumps(detail)
    return f"HTTP {code}: {detail}"


def _http_error_to_client_error(e: urllib.error.HTTPError) -> ClientError:
    if e.code in (301, 302, 303, 307, 308):
        loc = e.headers.get("Location", "")
        return ClientError(
            f"Server redirected to {loc} — check SERVER_URL (include any /asr-mcp prefix) "
            "and that TOKEN is valid"
        )
    body = e.read().decode("utf-8", "replace")
    return ClientError(_describe_http_error(e.code, body, e.headers))


def open_request(method, url, *, data=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=data, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        return _OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise _http_error_to_client_error(e) from None
    except urllib.error.URLError as e:
        raise ClientError(f"Cannot reach server at {url}: {e.reason}") from None


def request_json(method, url, *, data=None, headers=None, timeout=60):
    resp = open_request(method, url, data=data, headers=headers, timeout=timeout)
    raw = resp.read().decode("utf-8", "replace")
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = {"raw": raw[:500]}
    return resp.status, parsed


class _MultipartStream:
    """Readable multipart/form-data body that streams the file content.

    Implements __len__ so urllib can set Content-Length, and read() so the
    file payload is read in chunks instead of buffering the whole upload.
    """

    def __init__(self, file_field, file_path: Path, fields=None, upload_name=None):
        self._path = Path(file_path)
        self._boundary = "asrclient7f3a9c2e" + format(
            abs(hash((str(self._path), self._path.stat().st_size))), "x")
        name = upload_name or self._path.name
        head = bytearray()
        for fname, value in (fields or {}).items():
            head += (f"--{self._boundary}\r\n"
                     f"Content-Disposition: form-data; name=\"{fname}\"\r\n\r\n"
                     f"{value}\r\n").encode("utf-8")
        head += (f"--{self._boundary}\r\n"
                 f"Content-Disposition: form-data; name=\"{file_field}\"; "
                 f"filename=\"{name}\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n").encode("utf-8")
        self._head = bytes(head)
        self._tail = f"\r\n--{self._boundary}--\r\n".encode("utf-8")
        self._file_size = self._path.stat().st_size
        self._length = len(self._head) + self._file_size + len(self._tail)
        self._pos = 0
        self._fh = None

    @property
    def content_type(self):
        return f"multipart/form-data; boundary={self._boundary}"

    def __len__(self):
        return self._length

    def _ensure_open(self):
        if self._fh is None:
            self._fh = open(self._path, "rb")

    def read(self, size=-1):
        if self._pos >= self._length:
            return b""
        if size is None or size < 0:
            size = self._length - self._pos
        out = bytearray()
        while len(out) < size and self._pos < self._length:
            want = size - len(out)
            if self._pos < len(self._head):
                take = self._head[self._pos:self._pos + want]
                out += take
                self._pos += len(take)
                continue
            file_end = len(self._head) + self._file_size
            if self._pos < file_end:
                self._ensure_open()
                chunk = self._fh.read(min(want, file_end - self._pos))
                if not chunk:
                    # Should not happen if size/pos are consistent; advance to tail.
                    self._pos = file_end
                    continue
                out += chunk
                self._pos += len(chunk)
                continue
            tail_off = self._pos - file_end
            take = self._tail[tail_off:tail_off + want]
            out += take
            self._pos += len(take)
        if self._pos >= self._length and self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        return bytes(out)

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


def encode_multipart(file_field, file_path: Path, fields=None, upload_name=None):
    """Return a streaming multipart body and its Content-Type.

    The body is streamed from disk in chunks (see _MultipartStream) so large
    uploads do not need to be buffered in memory.
    """
    stream = _MultipartStream(file_field, file_path, fields=fields, upload_name=upload_name)
    return stream, stream.content_type


def _fmt_secs(sec):
    sec = max(0, int(round(float(sec))))
    minutes, seconds = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def render_progress(label, evt):
    parts = [label]
    stage = evt.get("stage")
    if stage:
        parts.append(str(stage))
    progress = evt.get("progress")
    if isinstance(progress, (int, float)):
        parts.append(f"{int(progress * 100)}%")
    if evt.get("elapsed_sec") is not None:
        parts.append(f"{_fmt_secs(evt['elapsed_sec'])} elapsed")
    if evt.get("predicted_remaining_sec") is not None:
        parts.append(f"~{_fmt_secs(evt['predicted_remaining_sec'])} left")
    line = " | ".join(parts)
    width = shutil.get_terminal_size((120, 24)).columns
    sys.stdout.write("\r" + line[: width - 1].ljust(max(width - 1, 1)))
    sys.stdout.flush()


def finish_progress_line():
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fmt_hms(sec):
    sec = max(0, int(round(float(sec))))
    hours, rem = divmod(sec, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _profiles_banner(result):
    """SPEAKER VOICE PROFILES banner lines (empty list when no profiles)."""
    profiles = result.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        return []
    rows = []
    for name, p in profiles.items():
        if not isinstance(p, dict):
            continue
        parts = []
        pitch = p.get("pitch_hz")
        if isinstance(pitch, (int, float)) and pitch > 0:
            pstd = p.get("pitch_std")
            if isinstance(pstd, (int, float)) and pstd > 0:
                parts.append(f"pitch={pitch:.0f}Hz (±{pstd:.0f}Hz)")
            else:
                parts.append(f"pitch={pitch:.0f}Hz")
        energy = p.get("energy_rms")
        if isinstance(energy, (int, float)):
            parts.append(f"energy={energy:.4f}")
        speech = p.get("total_speech_sec")
        if isinstance(speech, (int, float)):
            parts.append(f"speech={speech:.0f}s")
        if parts:
            rows.append((float(speech) if isinstance(speech, (int, float)) else 0.0,
                         f"  {name}: " + "  ".join(parts)))
    if not rows:
        return []
    rows.sort(key=lambda r: -r[0])
    rule = "=" * 60
    return [rule, "SPEAKER VOICE PROFILES", rule] + [r[1] for r in rows] + [rule]


def _paragraphs_from_segments(segments):
    """Group a run's segments into paragraphs: break on pause >=1.5s or long text."""
    paragraphs = []
    current = []
    current_len = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if current:
            prev_end = float(current[-1].get("end") or current[-1].get("start") or 0.0)
            gap = start - prev_end
            last_text = (current[-1].get("text") or "").strip()
            sentence_end = last_text.endswith((".","!","?","\u2026"))
            if gap >= 1.5 or current_len >= 800 and sentence_end or current_len >= 1600:
                paragraphs.append(current)
                current = []
                current_len = 0
        current.append(seg)
        current_len += len(text) + 1
    if current:
        paragraphs.append(current)
    return paragraphs


def _paragraph_confidence(segs):
    """Duration-weighted mean confidence as 0..100 int, or None if unavailable."""
    num = den = 0.0
    for s in segs:
        c = s.get("confidence")
        if c is None:
            continue
        try:
            c = float(c)
            w = float(s.get("end") or 0.0) - float(s.get("start") or 0.0)
        except (TypeError, ValueError):
            continue
        if w <= 0:
            w = 1.0
        num += c * w
        den += w
    if den <= 0:
        return None
    return max(0, min(100, int(round(100.0 * num / den))))


def build_transcript(filename, result, date_str):
    lines = []
    lines.append(f"Audio: {filename}")
    lines.append(f"Date: {date_str}")
    lines.append(f"Speakers: {result.get('total_speakers', 0)}")
    lines.append(f"Duration: {result.get('audio_duration_sec', 0):.1f}s")
    lines.append("")

    banner = _profiles_banner(result)
    if banner:
        lines.extend(banner)
        lines.append("")

    results = result.get("results") or []
    for r in results:
        speaker = r.get("speaker") or UNKNOWN_SPEAKER_LABEL
        if r.get("uncertain"):
            reason = r.get("attribution_reason") or "unspecified"
            speaker = f"{UNKNOWN_SPEAKER_LABEL} ({reason})"
        segs = [s for s in (r.get("segments") or [])
                if isinstance(s, dict) and (s.get("text") or "").strip()]
        if not segs:
            text = r.get("text", "")
            start = r.get("start")
            end = r.get("end")
            if start is not None and end is not None:
                body = text.replace("\n", "\n    ")
                lines.append(f"[{speaker}] {start:.1f}s - {end:.1f}s: {body}")
            elif text:
                lines.append(text)
            continue
        for para in _paragraphs_from_segments(segs):
            p_start = float(para[0].get("start") or 0.0)
            text = " ".join((s.get("text") or "").strip() for s in para).strip()
            if not text:
                continue
            conf = _paragraph_confidence(para)
            conf_s = f" ({conf}%)" if conf is not None else ""
            lines.append(f"[{_fmt_hms(p_start)}] {speaker}{conf_s}: {text}")

    errors = [r.get("error") for r in results if r.get("error")]
    if errors:
        lines.append("")
        lines.append("Errors: " + "; ".join(str(e) for e in errors))

    uncertain = [r for r in results if r.get("uncertain")]
    if uncertain:
        lines.append("")
        lines.append(
            f"WARNING: {len(uncertain)} of {len(results)} segment(s) have an UNKNOWN speaker "
            "(identity could not be established; the text was kept). Re-run if you need clean labels."
        )

    return "\n".join(lines)


def _has_text(result):
    return any((r.get("text") or "").strip() for r in (result.get("results") or []))


def _write_sidecar(out_path: Path, result, done_evt, status, date_str,
                  _pending_profiles=None):
    """Optional machine-readable sidecar next to the .txt transcript."""
    payload = {
        "generated_at": date_str,
        "status": status,
        "processing_time_sec": (done_evt or {}).get("processing_time_sec"),
        "speedup": (done_evt or {}).get("speedup"),
        "total_speakers": result.get("total_speakers", 0),
        "uncertain_segments": result.get("uncertain_segments", 0),
        "audio_duration_sec": result.get("audio_duration_sec", 0),
        "pending_profiles": _pending_profiles,
        "results": [
            {
                "start": r.get("start"),
                "end": r.get("end"),
                "speaker": r.get("speaker"),
                "uncertain": bool(r.get("uncertain")),
                "speaker_source": r.get("speaker_source"),
                "speaker_confidence": r.get("speaker_confidence"),
                "attribution_reason": r.get("attribution_reason"),
                "text": r.get("text", ""),
                "segments": r.get("segments") or [],
                "error": r.get("error"),
            }
            for r in (result.get("results") or [])
        ],
        "errors": [r.get("error") for r in (result.get("results") or []) if r.get("error")],
    }
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar


def find_ffmpeg() -> str | None:
    """Locate ffmpeg: PATH first, then the usual Windows install locations."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for env in ("FFMPEG", "FFMPEG_PATH"):
        cand = os.environ.get(env)
        if cand and Path(cand).is_file():
            return cand
    local = SCRIPT_DIR / "ffmpeg.exe"
    if local.is_file():
        return str(local)
    for base in (r"C:\Program Files\ffmpeg\bin", r"C:\ffmpeg\bin", r"C:\ProgramData\chocolatey\bin"):
        cand = Path(base) / "ffmpeg.exe"
        if cand.is_file():
            return str(cand)
    return None


def probe_has_video(path: Path) -> bool:
    """True when the container holds a video stream (best effort)."""
    exe = find_ffmpeg()
    if not exe:
        return False
    try:
        out = subprocess.run(
            [exe, "-v", "quiet", "-print_format", "json", "-show_format", "-i", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode not in (0, 1) or not out.stdout:
            return False
        for stream in json.loads(out.stdout).get("streams", []) or []:
            if stream.get("codec_type") == "video":
                return True
    except Exception:
        return False
    return False


def convert_to_audio(path: Path) -> Path:
    """Transcode an inbound file to a temporary 16 kHz mono FLAC.

    All inbound files are converted so the upload is compact and in a format
    the server can process. The returned path is in a temporary location and
    MUST be removed by the caller when no longer needed.
    """
    exe = find_ffmpeg()
    if not exe:
        raise ClientError(
            f"'{path.name}' needs converting but ffmpeg was not found. Install it "
            "(https://www.gyan.dev/ffmpeg/builds/, then put ffmpeg.exe beside this "
            "script or on PATH)."
        )
    fd, tmp_name = tempfile.mkstemp(prefix="asr-upload-", suffix=".flac")
    os.close(fd)
    dest = Path(tmp_name)
    cmd = [exe, "-v", "error", "-y", "-i", str(path), "-vn",
           "-ac", CONVERT_CHANNELS, "-ar", CONVERT_SAMPLE_RATE, "-c:a", CONVERT_CODEC, str(dest)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise ClientError(f"Converting '{path.name}' timed out") from None
    if proc.returncode != 0 or not dest.is_file():
        tail = (proc.stderr or "").strip().splitlines()
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise ClientError(
            f"Could not convert '{path.name}': {tail[-1] if tail else 'ffmpeg failed'}"
        )
    return dest


def prepare_upload(path: Path) -> tuple[Path, Path | None]:
    """Convert an inbound file to a temporary processable upload.

    Returns (upload_path, cleanup_path). cleanup_path is the temporary
    converted file that must be deleted after use, or None if no temporary
    file was created.
    """
    before = path.stat().st_size
    dest = convert_to_audio(path)
    after = dest.stat().st_size
    print(f"  converted for upload: {before / (1024 * 1024):.1f} MB -> "
          f"{after / (1024 * 1024):.1f} MB (16 kHz mono FLAC)")
    return dest, dest


def wait_for_idle(base, token, *, timeout=JOB_WAIT_TIMEOUT_SEC):
    """Block until the server has no transcription running.

    The server runs one job at a time (one GPU, one decoder), so a second
    upload is refused with 409 while the first is in flight. Waiting here is
    better than failing: the files were queued for a reason.

    Uses GET /api/asr/activity/stream, which emits an immediate snapshot
    ({"active": ..., "job": ...}) and then only start/finish transitions, with
    a comment keep-alive every 20s.
    """
    if timeout <= 0:
        return
    url = f"{base}/api/asr/activity/stream"
    deadline = time.monotonic() + timeout
    announced = False
    resp = None
    try:
        resp = open_request("GET", url, headers={"X-API-Key": token}, timeout=None)
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if line.startswith(":"):
                continue  # keep-alive
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                evt = json.loads(payload)
            except ValueError:
                continue
            if not evt.get("active"):
                return
            if not announced:
                job = evt.get("job") or {}
                who = job.get("filename")
                started = job.get("started_at")
                age = ""
                if isinstance(started, (int, float)):
                    mins = int(max(time.time() - started, 0) // 60)
                    age = f", started {mins} min ago" if mins else ""
                print(f"  server is busy transcribing '{who or 'a file'}'{age} — waiting "
                      f"(up to {int(timeout // 60)} min)",
                      file=sys.stderr)
                announced = True
            if time.monotonic() > deadline:
                raise ClientError(
                    f"Timed out after {int(timeout // 60)} min waiting for the server to "
                    "finish the transcription it is already running"
                )
    except (ClientError, OSError, http.client.HTTPException) as e:
        if isinstance(e, ClientError) and "Timed out" in str(e):
            raise
        # Cannot tell whether the server is busy; let the upload try and
        # surface its own error rather than blocking on a broken stream.
        return
    finally:
        if resp is not None:
            resp.close()


def _post_upload(base, token, url, upload_path: Path, upload_name: str):
    body, ctype = encode_multipart("file", upload_path, upload_name=upload_name)
    try:
        return open_request("POST", url, data=body,
                            headers={"X-API-Key": token, "Content-Type": ctype},
                            timeout=None)
    finally:
        try:
            body.close()
        except Exception:
            pass


def transcribe_file(base, token, path: Path, language: str = "auto", *,
                    wait: bool = True):
    label = path.name
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        raise ClientError(f"Unsupported file type '{path.suffix}' ({label})")
    if not path.is_file():
        raise ClientError(f"File not found: {path}")

    upload_path, cleanup_path = prepare_upload(path)

    url = (f"{base}/api/asr/transcribe/upload?save=false"
           f"&language={urllib.parse.quote(language or 'auto')}")
    upload_name = path.stem + ".flac"
    try:
        try:
            resp = _post_upload(base, token, url, upload_path, upload_name)
        except ClientError as e:
            # A 409 means another transcription owns the server; wait for it and
            # retry once rather than losing the file.
            if wait and "HTTP 409" in str(e):
                print(f"  server is busy; {e}", file=sys.stderr)
                wait_for_idle(base, token)
                resp = _post_upload(base, token, url, upload_path, upload_name)
            else:
                raise

        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            raw = resp.read(2048).decode("utf-8", "replace")
            raise ClientError(f"Unexpected server response: {raw[:300]}")

        result = None
        error = None
        done_evt = None
        pending_profiles = []
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    evt = json.loads(payload)
                except ValueError:
                    continue
                stage = evt.get("stage")
                # Learned speakers arrive on diarization_complete, BEFORE done --
                # that is the only stream that carries them, so remember them.
                if evt.get("pending_profiles"):
                    pending_profiles.extend(evt["pending_profiles"])
                if stage == "error":
                    error = evt.get("error") or "unknown server error"
                    break
                if stage == "cancelled":
                    error = evt.get("message") or "Cancelled from the web UI"
                    break
                if stage == "done":
                    result = evt.get("result") or {}
                    done_evt = evt
                    render_progress(label, {"stage": "done", "progress": 1.0})
                    break
                render_progress(label, evt)
        except (OSError, http.client.HTTPException) as e:
            raise ClientError(
                "Connection lost while waiting for the result — the server keeps "
                "processing; the transcript will be saved and can be downloaded "
                "from the web UI (Transcription history)"
            ) from e
        finally:
            finish_progress_line()
            resp.close()
    finally:
        if cleanup_path is not None:
            try:
                cleanup_path.unlink(missing_ok=True)
            except Exception:
                pass

    if error:
        raise ClientError(str(error))
    if result is None:
        raise ClientError("Server stream ended without a result")

    date_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = (done_evt or {}).get("status") or ("ok" if _has_text(result) else "error")

    out_path = path.with_suffix(".txt")
    out_path.write_text(build_transcript(path.name, result, date_str), encoding="utf-8")

    written = [out_path]
    if get_json_sidecar():
        written.append(_write_sidecar(out_path, result, done_evt, status,
                                      date_str, pending_profiles))

    processing = (done_evt or {}).get("processing_time_sec")
    if processing:
        audio_sec = result.get("audio_duration_sec") or 0
        speedup = (done_evt or {}).get("speedup")
        summary = f"processed {_fmt_secs(audio_sec)} of audio in {_fmt_secs(processing)}"
        if speedup:
            summary += f" ({speedup:g}x realtime)"
        print(f"  {summary}")

    if pending_profiles:
        # Learned speakers are excluded from voiceprint matching until named,
        # so this run (and every run until the user acts) reports them UNKNOWN.
        # Saying so here is the only place the user is told.
        print(f"  Learned {len(pending_profiles)} new speaker(s), not yet named: "
              + ", ".join(p.get("name", "?") for p in pending_profiles))
        print("  They are EXCLUDED from matching until named. Name them in the "
              "web UI: Voiceprints tab -> Unnamed speakers.")

    results = result.get("results") or []
    n_uncertain = sum(1 for r in results if r.get("uncertain"))
    if n_uncertain:
        print(
            f"  WARNING: {n_uncertain}/{len(results)} segment(s) have an "
            f"{UNKNOWN_SPEAKER_LABEL} speaker — text was kept, but the "
            "identity could not be established."
        )
    if not _has_text(result):
        # Never present a metadata-only file as a clean success.
        print("  WARNING: the server returned no text for this file", file=sys.stderr)

    if status == "error" and _has_text(result):
        print("  WARNING: server reported a transcription error (partial text kept)",
              file=sys.stderr)

    return written, status


def cmd_status():
    base, token = get_config()
    status, data = request_json("GET", f"{base}/api/token", headers={"X-API-Key": token})
    if status != 200:
        raise ClientError(f"HTTP {status}: {data}")
    if data.get("has_token"):
        print(f"OK — server reachable, token valid for user '{data.get('user_id', '?')}'")
    else:
        print("OK — server reachable, but no token on file for this user")


def cmd_voiceprints():
    base, token = get_config()
    status, data = request_json("GET", f"{base}/api/voiceprint/speakers",
                                headers={"X-API-Key": token})
    if status != 200:
        raise ClientError(f"HTTP {status}: {data}")
    speakers = data.get("speakers") or []
    if not speakers:
        print("No speakers or voiceprints yet.")
        return
    print(f"{'Name':<30} {'Snippets':>8} {'Duration':>10} {'Voiceprint':>10}")
    for s in speakers:
        has_vp = "yes" if s.get("has_voiceprint") else "no"
        print(f"{s.get('name', '?'):<30} {s.get('snippet_count', 0):>8} "
              f"{s.get('total_duration_sec', 0):>9.1f}s {has_vp:>10}")


def cmd_voiceprint_add(name, file_path: Path, start, end):
    base, token = get_config()
    if not file_path.is_file():
        raise ClientError(f"File not found: {file_path}")
    query = ""
    if start is not None:
        query += f"?start_sec={start}"
        if end is not None:
            query += f"&end_sec={end}"
    quoted = urllib.parse.quote(name, safe="")
    upload_path, cleanup_path = prepare_upload(file_path)
    body, ctype = encode_multipart("file", upload_path, upload_name=file_path.stem + ".flac")
    try:
        status, data = request_json(
            "POST", f"{base}/api/voiceprint/speakers/{quoted}/upload{query}",
            data=body, headers={"X-API-Key": token, "Content-Type": ctype},
        )
    finally:
        try:
            body.close()
        except Exception:
            pass
        if cleanup_path is not None:
            try:
                cleanup_path.unlink(missing_ok=True)
            except Exception:
                pass
    if status != 200 or data.get("error"):
        raise ClientError(f"Snippet upload failed: {data.get('error') or data}")
    print(f"Added snippet: {data.get('duration_sec', '?')}s ({data.get('file_path', '')})")
    # The server replaces this speaker's earlier snippets from the SAME file
    # (one file, one contribution) rather than stacking a second copy of one
    # recording onto the profile. Report it -- the removed audio is gone.
    if data.get("replaced_snippets"):
        print(f"Replaced {data['replaced_snippets']} earlier snippet(s) "
              f"from {file_path.name}")

    status, data = request_json(
        "POST", f"{base}/api/voiceprint/speakers/{quoted}/refine",
        headers={"X-API-Key": token}, timeout=300,
    )
    if status != 200 or data.get("error"):
        raise ClientError(f"Refine failed: {data.get('error') or data}")
    print(f"Voiceprint '{name}' refined: {data.get('snippet_count', '?')} snippets, "
          f"{data.get('total_duration_sec', '?')}s total")


def cmd_voiceprint_refine(name):
    base, token = get_config()
    quoted = urllib.parse.quote(name, safe="")
    status, data = request_json(
        "POST", f"{base}/api/voiceprint/speakers/{quoted}/refine",
        headers={"X-API-Key": token}, timeout=300,
    )
    if status != 200 or data.get("error"):
        raise ClientError(f"Refine failed: {data.get('error') or data}")
    print(f"Voiceprint '{name}' refined: {data.get('snippet_count', '?')} snippets, "
          f"{data.get('total_duration_sec', '?')}s total, "
          f"pitch {data.get('pitch_hz', '?')} Hz")


def cmd_transcribe(paths):
    global _JSON_SIDECAR
    base, token = get_config()
    args = list(paths)
    language = get_language()
    if "--language" in args:
        i = args.index("--language")
        if i + 1 >= len(args):
            print("--language requires a code (e.g. hu) or 'auto'", file=sys.stderr)
            return 2
        language = args[i + 1].strip().lower() or "auto"
        args = args[:i] + args[i + 2:]
    if "--json" in args:
        _JSON_SIDECAR = True
        args = [a for a in args if a != "--json"]
    if "--no-convert" in args:
        print("  note: --no-convert ignored; all inbound files are converted to 16 kHz mono FLAC before upload",
              file=sys.stderr)
        args = [a for a in args if a != "--no-convert"]
    wait = "--no-wait" not in args
    args = [a for a in args if a != "--no-wait"]
    if not args:
        print("No files given.")
        print(USAGE)
        return 2
    failures = 0
    partial = 0
    for raw in args:
        path = Path(raw).expanduser()
        print(f"Transcribing {path.name} (language={language}) ...")
        try:
            written, status = transcribe_file(base, token, path, language=language,
                                              wait=wait)
            for p in written:
                print(f"  saved {p}")
            if status != "ok":
                partial += 1
        except ClientError as e:
            failures += 1
            print(f"  FAILED: {e}", file=sys.stderr)
    if failures:
        print(f"{failures} of {len(args)} file(s) failed", file=sys.stderr)
        return 1
    if partial:
        # Text was recovered but at least one segment is uncertain or errored:
        # report it, but do not fail the run.
        print(
            f"{partial} of {len(args)} file(s) finished with uncertain/partial speaker data",
            file=sys.stderr,
        )
    return 0


def main(argv):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0 if argv else 2

    cmd = argv[0]
    try:
        if cmd in ("transcribe", "--transcribe"):
            return cmd_transcribe(argv[1:])
        if cmd == "status":
            cmd_status()
            return 0
        if cmd == "voiceprints":
            cmd_voiceprints()
            return 0
        if cmd == "voiceprint-add":
            if len(argv) < 3:
                print("usage: voiceprint-add <name> <file> [start_sec] [end_sec]", file=sys.stderr)
                return 2
            start = float(argv[3]) if len(argv) > 3 else None
            end = float(argv[4]) if len(argv) > 4 else None
            cmd_voiceprint_add(argv[1], Path(argv[2]).expanduser(), start, end)
            return 0
        if cmd == "voiceprint-refine":
            if len(argv) < 2:
                print("usage: voiceprint-refine <name>", file=sys.stderr)
                return 2
            cmd_voiceprint_refine(argv[1])
            return 0
        return cmd_transcribe(argv)
    except ClientError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
