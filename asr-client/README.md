# asr-client — standalone transcription client

Drop-file Windows client for the ASR MCP server. Pure Python standard
library — nothing to install beyond Python 3.8+. The package is
self-contained: it carries its own `.env` config, `requirements.txt`,
and (optionally) a private `.venv`. No server code or server
dependencies are required.

## Setup

1. Unzip anywhere.
2. Run `transcribe.bat` once — it creates `.env` from `.env.example`
   (and a private `.venv` on the next run).
3. Edit `.env`:
   - `SERVER_URL` — full base URL, include the `/asr-mcp` prefix when
     behind nginx (e.g. `http://host:8087/asr-mcp`)
   - `TOKEN` — generate in the web UI ("Windows Client Token" card)
     or `POST /api/token` while logged in; one token per user
   - `LANGUAGE` — ISO 639-1 code (e.g. `hu`) or `auto` (default)
4. Drag audio files onto `transcribe.bat`, or use the CLI below.

## Commands

| Command | Description |
|---------|-------------|
| Drag files onto `transcribe.bat` | Transcribe; writes `<name>.txt` next to each file |
| `transcribe.bat status` | Check server/token connectivity |
| `transcribe.bat voiceprints` | List speakers and voiceprints |
| `transcribe.bat voiceprint-add <name> <file> [start] [end]` | Create/refine a voiceprint from a clip |
| `transcribe.bat voiceprint-refine <name>` | Rebuild a voiceprint from its snippets |
| `transcribe.bat <file...> [--language <code>]` | Transcribe (language overrides `.env`) |

Supported inputs: wav, mp3, flac, ogg, m4a, webm, opus, mkv, mp4.

## Notes

- Client transcriptions use `?save=false` — nothing is stored in
  server transcript history. If the connection drops mid-job, the
  server finishes the run and saves the result to transcript history
  (download from the web UI). Voiceprint snippets uploaded by the
  client ARE kept (they feed refinement).
- `transcribe.bat` creates `.venv` next to the script and runs the
  client with it; `requirements.txt` is installed there when it
  declares dependencies. Without a working venv it falls back to
  `py`/`python` on PATH (the client is stdlib-only, so this always
  works).
- `.env`, `.venv/`, and `__pycache__/` are ignored by git and are
  never included in the distributed zip.
