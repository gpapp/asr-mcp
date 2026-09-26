import http.client
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus", ".mkv", ".mp4"}

USAGE = """Usage:
  transcribe <audio file> [more files...]   Transcribe files (writes <name>.txt next to each)
  status                                    Check server/token connectivity
  voiceprints                               List speakers and voiceprints
  voiceprint-add <name> <file> [start] [end]   Create/refine a voiceprint from audio
  voiceprint-refine <name>                  Rebuild a voiceprint from its snippets

Configuration: edit .env next to this script (SERVER_URL + TOKEN)."""


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


def _http_error_to_client_error(e: urllib.error.HTTPError) -> ClientError:
    if e.code in (301, 302, 303, 307, 308):
        loc = e.headers.get("Location", "")
        return ClientError(
            f"Server redirected to {loc} — check SERVER_URL (include any /asr-mcp prefix) "
            "and that TOKEN is valid"
        )
    body = e.read().decode("utf-8", "replace")
    detail = body
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and parsed.get("detail"):
            detail = parsed["detail"]
            if not isinstance(detail, str):
                detail = json.dumps(detail)
    except ValueError:
        pass
    return ClientError(f"HTTP {e.code}: {detail}")


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


def encode_multipart(file_field, file_path: Path, fields=None):
    boundary = "asrclient7f3a9c2e" + format(abs(hash((str(file_path), file_path.stat().st_size))), "x")
    body = bytearray()
    for name, value in (fields or {}).items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode("utf-8")
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
             f"filename=\"{file_path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
             ).encode("utf-8")
    body += file_path.read_bytes()
    body += b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


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


def build_transcript(filename, result, date_str):
    lines = []
    lines.append(f"Audio: {filename}")
    lines.append(f"Date: {date_str}")
    lines.append(f"Speakers: {result.get('total_speakers', 0)}")
    lines.append(f"Duration: {result.get('audio_duration_sec', 0):.1f}s")
    lines.append("")

    results = result.get("results") or []
    for r in results:
        text = r.get("text", "")
        speaker = r.get("speaker", "")
        start = r.get("start")
        end = r.get("end")
        if speaker and start is not None and end is not None:
            body = text.replace("\n", "\n    ")
            lines.append(f"[{speaker}] {start:.1f}s - {end:.1f}s: {body}")
        else:
            lines.append(text)

    errors = [r.get("error") for r in results if r.get("error")]
    if errors:
        lines.append("")
        lines.append("Errors: " + "; ".join(str(e) for e in errors))

    return "\n".join(lines)


def transcribe_file(base, token, path: Path):
    label = path.name
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        raise ClientError(f"Unsupported file type '{path.suffix}' ({label})")
    if not path.is_file():
        raise ClientError(f"File not found: {path}")

    url = f"{base}/api/asr/transcribe/upload?save=false"
    body, ctype = encode_multipart("file", path)
    resp = open_request("POST", url, data=body,
                        headers={"X-API-Key": token, "Content-Type": ctype},
                        timeout=None)

    content_type = resp.headers.get("Content-Type", "")
    if "text/event-stream" not in content_type:
        raw = resp.read(2048).decode("utf-8", "replace")
        raise ClientError(f"Unexpected server response: {raw[:300]}")

    result = None
    error = None
    done_evt = None
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

    if error:
        raise ClientError(str(error))
    if result is None:
        raise ClientError("Server stream ended without a result")

    date_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_path = path.with_suffix(".txt")
    out_path.write_text(build_transcript(path.name, result, date_str), encoding="utf-8")

    processing = (done_evt or {}).get("processing_time_sec")
    if processing:
        audio_sec = result.get("audio_duration_sec") or 0
        speedup = (done_evt or {}).get("speedup")
        summary = f"processed {_fmt_secs(audio_sec)} of audio in {_fmt_secs(processing)}"
        if speedup:
            summary += f" ({speedup:g}x realtime)"
        print(f"  {summary}")
    return out_path


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
    body, ctype = encode_multipart("file", file_path)
    status, data = request_json(
        "POST", f"{base}/api/voiceprint/speakers/{quoted}/upload{query}",
        data=body, headers={"X-API-Key": token, "Content-Type": ctype},
    )
    if status != 200 or data.get("error"):
        raise ClientError(f"Snippet upload failed: {data.get('error') or data}")
    print(f"Added snippet: {data.get('duration_sec', '?')}s ({data.get('file_path', '')})")

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
    base, token = get_config()
    if not paths:
        print("No files given.")
        print(USAGE)
        return 2
    failures = 0
    for raw in paths:
        path = Path(raw).expanduser()
        print(f"Transcribing {path.name} ...")
        try:
            out_path = transcribe_file(base, token, path)
            print(f"  saved {out_path}")
        except ClientError as e:
            failures += 1
            print(f"  FAILED: {e}", file=sys.stderr)
    if failures:
        print(f"{failures} of {len(paths)} file(s) failed", file=sys.stderr)
        return 1
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
