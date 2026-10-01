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

The client owns the timeline in a turn frame, so its ``start_sample`` is a
*claim*, not a measurement.  :class:`TimelineGuard` checks the claim against
what the server has already seen and reports a channel whose declared clock
cannot be true; :func:`unpack_turn` only covers what a single frame can prove
on its own (shape, magic, version, and a turn that declares no audio at all).

The client (``asr-client/live_client.py``) implements this independently —
it is shipped as a standalone zip and must not import server code.
``tests/test_streaming_protocol.py`` asserts the two stay in agreement by
reading the client's constants, so a change here cannot silently desync them.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

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

#: Fault codes reported for a client-declared timeline that cannot be true.
#: They are surfaced as ``dropped`` messages and in the shutdown ``stats``
#: frame, never as a session failure.
FAULT_REGRESSION = "start_sample_regression"
FAULT_AHEAD = "start_sample_ahead_of_stream"

#: Longest ``[reason, start]`` record set per fault reason.
_FAULT_DETAIL_LIMIT = 10

_TIMELINE_DEFAULTS = {
    # Turns cut one after another on the SAME channel never overlap (the
    # detector is serial and pre-roll starts after the previous turn closed),
    # so anything beyond this tolerance is a client-side timeline fault.
    "timeline_overlap_tolerance_sec": 0.05,
    # How far past everything the server has actually seen a declared start may
    # sit before it is called impossible. The gap between the audio a client has
    # recorded and the turn frames it has sent is bounded by the coalescer's
    # merge window (<= streaming.merge_gap_sec) plus a slow socket, so a
    # generous slack keeps this from firing on a merely lagging client.
    "timeline_max_lead_sec": 10.0,
    # ...and the same slack again, as a fraction of how long the connection has
    # been up. A resampler whose rate is off by a fraction of a percent drifts
    # by seconds over an hour, and a client's declared clock must not be
    # declared impossible for that.
    "timeline_max_lead_ratio": 0.05,
}

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
    is too short, carries an unknown version, declares more samples than the
    payload actually holds, or is a turn frame with no audio at all.  The server
    counts these as malformed and keeps the connection open — a single bad frame
    must never drop a live session.  What a *single* frame cannot show — whether
    its ``start_sample`` is consistent with the frames before it — is
    :class:`TimelineGuard`'s job.
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
    if msg_type == MSG_TURN and n_samples == 0:
        # A turn with no audio decodes to nothing and declares a zero-length
        # span; it is a client bug, not a turn. MSG_FLUSH legitimately carries
        # no payload, so this is checked per msg_type.
        raise ValueError("turn frame declares 0 samples")
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


class TimelineGuard:
    """Check that a client-declared turn timeline is internally consistent.

    A turn frame carries the client's own ``start_sample``, and the server used
    to copy it through verbatim.  Nothing else on the wire pins it down: one
    dropped capture block, a ``getDisplayMedia`` stream that starts late, a
    counter reset on reconnect -- each silently shifts every later boundary on
    that channel, and because the *recording* is what the shutdown
    re-attribution runs against, the text then lands on the wrong speaker with
    nothing in the transcript to say so.

    Two things are checkable without trusting the client:

    * **monotonicity** -- turns cut one after another on one channel must not
      overlap or move backwards.  A regression is clamped to the previous
      turn's end so the emitted boundary never goes backwards, and flagged;
    * **plausibility** -- a client streams audio as it is captured, so a start
      sample far beyond every sample the server has seen on either channel (and
      beyond the connection's own elapsed time, when the caller supplies it)
      means the declared clock is not the recording's clock.

    Both are *flagged*, never fatal: one bad frame must not end a session, and
    dropping the audio would lose the words it carries.
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = dict(_TIMELINE_DEFAULTS)
        self.cfg.update(cfg or {})
        self._last_end: Dict[int, int] = {}
        self._high_water = 0
        self._counts: Dict[str, int] = {}
        self._drifted: set = set()
        self.detail: List[dict] = []

    # -- state ------------------------------------------------------------
    @property
    def faults(self) -> int:
        return sum(self._counts.values())

    def drifted_channels(self) -> List[int]:
        """Channels whose declared timeline has been contradicted at least once."""
        return sorted(self._drifted)

    def stats(self) -> dict:
        return {
            "faults": self.faults,
            "by_reason": dict(self._counts),
            "channels": self.drifted_channels(),
            "detail": list(self.detail),
        }

    # -- validation -------------------------------------------------------
    def note_end(self, end_sample: int) -> None:
        """Record audio the server saw on its own path (legacy raw PCM).

        A connection that mixes both frame shapes must not have its turn-frame
        timeline judged against a high-water mark that ignores the raw stream.
        """
        self._high_water = max(self._high_water, int(end_sample))

    def observe(self, channel: int, start_sample: int, n_samples: int,
                elapsed_sec: Optional[float] = None
                ) -> Tuple[int, Optional[dict]]:
        """Validate one declared span.

        Returns ``(start_sample_to_use, fault_or_None)``.  The returned start
        is the declared one unless it regressed, in which case it is clamped to
        where the previous turn on that channel ended.
        """
        channel = int(channel)
        start = int(start_sample)
        end = start + int(n_samples)
        previous = self._last_end.get(channel)
        fault = None

        if previous is not None and start < previous - int(
                float(self.cfg["timeline_overlap_tolerance_sec"]) * SAMPLE_RATE):
            fault = self._record(
                FAULT_REGRESSION, channel, start, previous,
                "turn starts before the previous turn on this channel ended",
            )
            start = previous

        plausible = self._high_water
        lead = float(self.cfg["timeline_max_lead_sec"]) * SAMPLE_RATE
        if elapsed_sec is not None:
            plausible = max(plausible, int(float(elapsed_sec) * SAMPLE_RATE))
            lead = max(lead, float(elapsed_sec)
                       * float(self.cfg["timeline_max_lead_ratio"]) * SAMPLE_RATE)
        if start > plausible + lead:
            ahead = self._record(
                FAULT_AHEAD, channel, start, plausible,
                "declared start is beyond every sample seen so far",
            )
            if fault is None:
                # A frame can trip both checks. Both are counted; the regression
                # is the one reported to the client, because it is the one that
                # was acted on (the span was clamped).
                fault = ahead

        self._last_end[channel] = max(end, start)
        self._high_water = max(self._high_water, end)
        return start, fault

    def _record(self, reason: str, channel: int, declared: int, expected: int,
                note: str) -> dict:
        self._counts[reason] = self._counts.get(reason, 0) + 1
        fault = {
            "reason": reason,
            "channel": channel,
            "declared_start_sec": round(declared / SAMPLE_RATE, 3),
            "expected_start_sec": round(expected / SAMPLE_RATE, 3),
            "delta_sec": round((declared - expected) / SAMPLE_RATE, 3),
            "note": note,
        }
        # Bounded: a client stuck in a loop must not be able to grow the stats
        # frame without limit. The counts above are exact; the detail is a
        # sample.
        if len(self.detail) < _FAULT_DETAIL_LIMIT:
            self.detail.append(fault)
        self._drifted.add(channel)
        return fault
