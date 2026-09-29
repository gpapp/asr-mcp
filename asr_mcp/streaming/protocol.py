"""Wire protocol for the live (WebSocket) streaming path.

Two frame shapes are accepted on ``/api/asr/ws/stream``:

**Legacy raw PCM** (unchanged, still fully supported)::

    struct.pack("<II", channel, sequence) + int16 LE PCM (16 kHz, mono)

The server runs its own :class:`~asr_mcp.streaming.turn_detector.TurnDetector`
on the mic channel and cuts turns itself.

**Turn frames** (preferred; the client owns endpointing)::

    magic "LVT1" | version | msg_type | channel | start_sample
                    | sample_count | sequence | int16 LE PCM

A turn frame carries one already-detected utterance plus its exact position
on the file timeline, so the server skips turn detection entirely and the live
boundaries and the offline diarization boundaries are the same ones.

A turn frame is recognised by its 4-byte magic, so a client may send either
shape on the same connection and detection needs no configuration flag.

The client (``asr-client/live_client.py``) implements this independently —
it is shipped as a standalone zip and must not import server code.
``tests/test_streaming_protocol.py`` asserts the two stay in agreement by
reading the client's constants, so a change here cannot silently desync them.
"""

from __future__ import annotations

import struct

#: 4-byte magic that distinguishes a turn frame from legacy raw PCM.
TURN_MAGIC = b"LVT1"

#: Protocol version. Bumped only for breaking layout changes; the server
#: rejects a version it does not know instead of misparsing the frame.
PROTOCOL_VERSION = 1

#: ``msg_type`` values.
MSG_TURN = 1
MSG_FLUSH = 2  # end-of-stream marker; the server flushes and drains

#: Fixed part of a turn frame header, immediately before the PCM payload.
TURN_HEADER = struct.Struct("<4sBBHQI I")
TURN_HEADER_SIZE = TURN_HEADER.size  # 24

#: Legacy raw-PCM header (channel, sequence) — kept for existing clients.
RAW_HEADER = struct.Struct("<II")
RAW_HEADER_SIZE = RAW_HEADER.size  # 8

#: Canonical wire sample rate.  Anything else must be resampled by the client.
SAMPLE_RATE = 16000

assert TURN_HEADER_SIZE == 24, "turn header layout changed - bump PROTOCOL_VERSION"


def is_turn_frame(data: bytes) -> bool:
    """True when ``data`` starts with the turn-frame magic."""
    return len(data) >= len(TURN_MAGIC) and bytes(data[: len(TURN_MAGIC)]) == TURN_MAGIC


def pack_turn(channel: int, start_sample: int, pcm: bytes, sequence: int = 0,
              version: int = PROTOCOL_VERSION) -> bytes:
    """Build one turn frame. ``pcm`` must be int16 LE, mono, 16 kHz."""
    if len(pcm) % 2:
        raise ValueError("PCM payload must be a whole number of int16 samples")
    n_samples = len(pcm) // 2
    return TURN_HEADER.pack(
        TURN_MAGIC, version, MSG_TURN, int(channel), int(start_sample),
        n_samples, int(sequence),
    ) + bytes(pcm)


def pack_control(msg_type: int, version: int = PROTOCOL_VERSION) -> bytes:
    """Build a control frame (e.g. :data:`MSG_FLUSH`) with no audio payload."""
    return TURN_HEADER.pack(
        TURN_MAGIC, version, int(msg_type), 0, 0, 0, 0,
    )


def unpack_turn(data: bytes) -> tuple[dict, bytes]:
    """Parse a turn frame.

    Returns ``(header_dict, pcm)``.  Raises :class:`ValueError` when the frame
    is too short, carries an unknown version, or declares more samples than the
    payload actually holds.  The server counts these as malformed and keeps the
    connection open — a single bad frame must never drop a live session.
    """
    if len(data) < TURN_HEADER_SIZE:
        raise ValueError(
            f"truncated turn frame: {len(data)} bytes (need >= {TURN_HEADER_SIZE})"
        )
    magic, version, msg_type, channel, start_sample, n_samples, sequence = \
        TURN_HEADER.unpack_from(data, 0)
    if magic != TURN_MAGIC:
        raise ValueError("not a turn frame")
    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported protocol version {version} (server speaks {PROTOCOL_VERSION})"
        )
    pcm = bytes(data[TURN_HEADER_SIZE:])
    if n_samples * 2 != len(pcm):
        raise ValueError(
            f"declared {n_samples} samples but payload holds {len(pcm) // 2}"
        )
    return (
        {
            "version": version,
            "msg_type": msg_type,
            "channel": int(channel),
            "start_sample": int(start_sample),
            "n_samples": int(n_samples),
            "sequence": int(sequence),
        },
        pcm,
    )
