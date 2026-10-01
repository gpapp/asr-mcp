"""Adaptive endpointing for the live streaming path.

The previous implementation (inline in ``streaming/handler.py``) used a fixed
RMS gate of 0.02 with no hysteresis and hard ``>0.5s`` / ``<0.3s`` duration
filters.  That suppressed short responses ("Yes", "No", "Hello") and split
single utterances on every low-energy consonant gap.

This detector replaces it with:

* an **adaptive noise floor** (per-frame RMS relative to a slowly tracked
  minimum) instead of a fixed threshold;
* **separate start/end thresholds** (hysteresis), so a frame above the start
  threshold does not have to stay above it to keep the turn open;
* **start confirmation** over N frames, so a single loud transient (door slam,
  keyboard click) does not open a turn;
* **silence hangover** before the turn is actually closed, so a 200-300ms
  intra-utterance dip does not split it;
* **pre-roll / post-roll** padding, so word onsets/offsets are not clipped;
* a **minimum voiced duration** (short) and a **maximum turn length** (force
  split on pathological input).

Pure stdlib + numpy so it can be unit tested without any model loaded.
All timings come from the config (``thresholds.json`` -> ``streaming``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger("asr_mcp.streaming.turn_detector")

SAMPLE_RATE = 16000

_DEFAULTS = {
    "frame_ms": 32.0,
    "noise_floor_min": 0.0005,
    "noise_floor_max": 0.05,
    "start_threshold_ratio": 2.5,
    "end_threshold_ratio": 1.6,
    "start_threshold_min": 0.0012,
    "end_threshold_min": 0.0006,
    "start_confirm_frames": 2,
    "hangover_ms": 320,
    "pre_roll_ms": 160,
    "post_roll_ms": 200,
    "min_voiced_ms": 120,
    "max_turn_sec": 30.0,
    # Turn coalescing (see TurnCoalescer): a closed turn is held briefly and
    # merged with the next one when the pause between them is short.
    "merge_gap_sec": 1.0,
    "max_merge_sec": 15.0,
}


def _cfg() -> dict:
    merged = dict(_DEFAULTS)
    try:
        from asr_mcp.config import get_config

        raw = get_config() or {}
        section = raw.get("streaming") or {}
        if isinstance(section, dict):
            merged.update(section)
    except Exception:
        pass
    return merged


def config() -> dict:
    """Streaming section of thresholds.json, merged over the defaults."""
    return _cfg()


def frame_size(cfg: dict | None = None) -> int:
    cfg = cfg or _cfg()
    return max(64, int(SAMPLE_RATE * float(cfg["frame_ms"]) / 1000.0))


def rms(chunk: bytes) -> float:
    """Root-mean-square of int16 PCM bytes (0..1 range)."""
    if not chunk:
        return 0.0
    usable = len(chunk) - (len(chunk) % 2)
    if usable <= 0:
        return 0.0
    samples = np.frombuffer(chunk[:usable], dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2)))


@dataclass
class Turn:
    """One detected speech turn, on the sample/sequence timeline."""

    start_sample: int
    end_sample: int
    audio: np.ndarray
    reason: str = "hangover"
    peak_rms: float = 0.0
    mean_rms: float = 0.0
    #: Where the audio really ends on the channel timeline, when that is not
    #: ``end_sample``.  A coalesced turn declares the span it *carries*
    #: (:class:`TurnCoalescer`), which is shorter than the audio it was cut
    #: from because the inter-turn pause is not concatenated in; this keeps the
    #: true position so ``covered_sec`` still reports how much of the recording
    #: the server actually saw.
    audio_end_sample: Optional[int] = None

    @property
    def duration_sec(self) -> float:
        return max(0, self.end_sample - self.start_sample) / SAMPLE_RATE

    @property
    def start_sec(self) -> float:
        return self.start_sample / SAMPLE_RATE

    @property
    def end_sec(self) -> float:
        return self.end_sample / SAMPLE_RATE


def samples_as_float32(audio) -> np.ndarray:
    """Normalise any accepted turn payload to float32 in ``[-1.0, 1.0]``.

    ``Turn.audio`` has two legitimate representations depending on where the
    turn was produced:

    * the **server** (``TurnDetector._close_turn`` and
      ``handler._turn_from_frame``) already stores a float32 numpy array;
    * the **client** (``live_client.py``) keeps raw int16 ``pcm`` bytes,
      because that is what it packs into the wire frame.

    Both must be readable, and :func:`asr_mcp.streaming.handler` needs the
    float form for ``extract_embedding`` / ``compute_pitch``.  Anything that
    blindly calls ``np.frombuffer(x, dtype=np.int16)`` on a float32 array
    silently **doubles the length** and yields the IEEE-754 *bit patterns* as
    int16 -- i.e. noise.  That produced random voiceprint distances
    (0.78-0.87, confidence 0.00) for every live turn, which looked like
    "randomly assigns speakers".  Always go through this helper.
    """
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(audio, np.ndarray):
        if np.issubdtype(audio.dtype, np.floating):
            return audio.astype(np.float32, copy=False)
        return audio.astype(np.float32) / 32768.0
    if isinstance(audio, (bytes, bytearray, memoryview)):
        raw = bytes(audio)
        usable = len(raw) - (len(raw) % 2)
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(raw[:usable], dtype=np.int16).astype(np.float32) / 32768.0
    arr = np.asarray(audio)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    return arr.astype(np.float32) / 32768.0


class TurnDetector:
    """Frame-based endpointing with an adaptive noise floor.

    Feed it raw int16 PCM bytes for the **mic channel only** (the speaker
    channel must never influence this state machine).  Completed turns are
    returned from :meth:`feed`; an in-progress turn is available via
    :meth:`flush` on disconnect.
    """

    def __init__(self, cfg: dict | None = None, sample_rate: int = SAMPLE_RATE):
        self.cfg = {**_DEFAULTS, **(cfg or {})}
        if cfg is None:
            self.cfg = _cfg()
        self.sample_rate = sample_rate
        self.frame_samples = frame_size(self.cfg)
        self.frame_ms = self.frame_samples / sample_rate * 1000.0

        self._pending = bytearray()      # incomplete frame bytes
        self._frame_idx = 0              # frames accepted so far
        self._noise_floor = float(self.cfg["noise_floor_min"])
        self._in_speech = False
        self._start_run = 0              # consecutive above-start frames
        self._below_run = 0              # consecutive below-end frames
        self._turn_start_frame = 0
        self._turn_peak = 0.0
        self._turn_rms_sum = 0.0
        self._turn_rms_n = 0
        self._turn_voiced_frames = 0     # frames above the end threshold
        # Pre-roll ring buffer of (frame_index, frame) and the open turn frames.
        self._pre_roll_frames: List[tuple] = []
        self._turn_frames: List[bytes] = []
        self._turns_emitted = 0
        self._dropped_turns = 0

    # -- introspection ---------------------------------------------------
    @property
    def turns_emitted(self) -> int:
        return self._turns_emitted

    @property
    def dropped_turns(self) -> int:
        return self._dropped_turns

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    # -- thresholds ------------------------------------------------------
    def _start_threshold(self) -> float:
        # `noise_floor_min` only keeps the tracked floor from collapsing to
        # zero; it must NOT also act as the absolute detection gate. It used
        # to (`max(floor * ratio, noise_floor_min)`), which put a hard
        # 0.004 * 2.5 = 0.01 RMS (-40 dBFS) floor on every start decision: a
        # quiet headset mic speaking normally never crossed it, so 59s of
        # speech produced zero turns. The absolute gate is
        # `start_threshold_min`, and it is low enough to admit a real mic
        # while still rejecting digital near-silence.
        return max(
            self._noise_floor * float(self.cfg["start_threshold_ratio"]),
            float(self.cfg["start_threshold_min"]),
        )

    def _end_threshold(self) -> float:
        # Strictly below the start threshold -> hysteresis.
        return max(
            self._noise_floor * float(self.cfg["end_threshold_ratio"]),
            float(self.cfg["end_threshold_min"]),
            self._start_threshold() * 0.6,
        )

    # -- frame handling --------------------------------------------------
    def feed(self, data: bytes) -> List[Turn]:
        """Consume PCM bytes; return any turns completed by this call."""
        if not data:
            return []
        self._pending.extend(data)
        need = self.frame_samples * 2
        turns: List[Turn] = []
        while len(self._pending) >= need:
            frame = bytes(self._pending[:need])
            del self._pending[:need]
            t = self._process_frame(frame)
            if t is not None:
                turns.append(t)
        return turns

    def _process_frame(self, frame: bytes) -> Optional[Turn]:
        c = self.cfg
        e = rms(frame)

        # Adaptive noise floor: only track DOWN quickly (a genuine pause) and
        # UP slowly, so a long utterance cannot drag the floor with it.
        if e < self._noise_floor:
            self._noise_floor = 0.7 * self._noise_floor + 0.3 * e
        else:
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * e
        self._noise_floor = min(
            max(self._noise_floor, float(c["noise_floor_min"])),
            float(c["noise_floor_max"]),
        )

        start_th = self._start_threshold()
        end_th = self._end_threshold()

        if not self._in_speech:
            # Keep the ring buffer of recent silence for pre-roll.
            self._pre_roll_frames.append((self._frame_idx, frame))
            max_pre = max(1, int(float(c["pre_roll_ms"]) / self.frame_ms))
            if len(self._pre_roll_frames) > max_pre:
                del self._pre_roll_frames[0 : len(self._pre_roll_frames) - max_pre]

            if e >= start_th:
                self._start_run += 1
            else:
                self._start_run = 0

            need_frames = max(1, int(c["start_confirm_frames"]))
            if self._start_run >= need_frames:
                confirm_idx = self._frame_idx - need_frames + 1
                older = [(i, f) for i, f in self._pre_roll_frames if i < confirm_idx]
                self._in_speech = True
                self._turn_peak = 0.0
                self._turn_rms_sum = 0.0
                self._turn_rms_n = 0
                self._turn_voiced_frames = 0
                if older:
                    # Pre-roll so the word onset is not clipped.
                    self._turn_start_frame = older[0][0]
                    self._turn_frames = [f for _, f in older]
                else:
                    self._turn_start_frame = confirm_idx
                    self._turn_frames = []
                self._turn_frames.extend(
                    f for i, f in self._pre_roll_frames if i >= confirm_idx
                )
                for i, f in self._pre_roll_frames:
                    if i >= confirm_idx:
                        fe = rms(f)
                        self._turn_peak = max(self._turn_peak, fe)
                        self._turn_rms_sum += fe
                        self._turn_rms_n += 1
                        if fe >= end_th:
                            self._turn_voiced_frames += 1
                self._below_run = 0
            self._frame_idx += 1
            return None

        # -- in speech --
        self._turn_frames.append(frame)
        self._turn_peak = max(self._turn_peak, e)
        self._turn_rms_sum += e
        self._turn_rms_n += 1
        if e >= end_th:
            self._turn_voiced_frames += 1
        self._frame_idx += 1

        max_frames = int(float(c["max_turn_sec"]) * 1000.0 / self.frame_ms)
        if len(self._turn_frames) >= max_frames:
            return self._close_turn("max_turn", emit=True)

        if e < end_th:
            self._below_run += 1
        else:
            self._below_run = 0

        hangover_frames = max(1, int(round(float(c["hangover_ms"]) / self.frame_ms)))
        if self._below_run >= hangover_frames:
            return self._close_turn("hangover", emit=True)
        return None

    def _close_turn(self, reason: str, emit: bool) -> Optional[Turn]:
        """Close the open turn, applying post-roll and min-duration filters."""
        c = self.cfg
        frames = self._turn_frames
        self._turn_frames = []
        self._in_speech = False
        self._start_run = 0
        self._below_run = 0
        self._pre_roll_frames = []
        voiced_frames = self._turn_voiced_frames
        self._turn_voiced_frames = 0
        if not frames:
            return None

        # Trim the trailing silence: the hangover frames plus post-roll.
        hangover_frames = max(1, int(round(float(c["hangover_ms"]) / self.frame_ms)))
        post_roll = max(0, int(round(float(c["post_roll_ms"]) / self.frame_ms)))
        keep = max(1, len(frames) - hangover_frames + post_roll)
        frames = frames[:keep]

        start_sample = self._turn_start_frame * self.frame_samples
        end_sample = start_sample + len(frames) * self.frame_samples
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0

        # The min-voiced gate looks at the VOICED frames, not the padded span:
        # pre/post-roll would otherwise make every click look long enough.
        voiced_ms = voiced_frames * self.frame_ms
        if voiced_ms < float(c["min_voiced_ms"]):
            self._dropped_turns += 1
            logger.debug("Dropped turn %.0fms voiced (< min_voiced_ms)", voiced_ms)
            return None

        turn = Turn(
            start_sample=start_sample,
            end_sample=end_sample,
            audio=audio,
            reason=reason,
            peak_rms=self._turn_peak,
            mean_rms=(self._turn_rms_sum / self._turn_rms_n) if self._turn_rms_n else 0.0,
        )
        self._turns_emitted += 1
        return turn if emit else None

    def flush(self) -> Optional[Turn]:
        """Close any open turn (disconnect). Returns it once, or None."""
        if self._in_speech:
            return self._close_turn("flush", emit=True)
        return None

    def stats(self) -> dict:
        return {
            "frames": self._frame_idx,
            "turns": self._turns_emitted,
            "dropped_turns": self._dropped_turns,
            "noise_floor": round(self._noise_floor, 5),
            "in_speech": self._in_speech,
        }


def _rebuild_turn(turn, start_sample: int, end_sample: int, audio, peak_rms: float,
                  audio_end_sample: Optional[int] = None):
    """Build a merged turn of the same class as ``turn``.

    The server's :class:`Turn` carries a float32 numpy array in ``audio`` while
    the client's mirror carries raw ``bytes`` in ``pcm``; the two constructors
    are otherwise identical, so try the server shape first and fall back.  A
    single implementation therefore serves both endpoints and the two can never
    disagree about what was merged.

    ``end_sample`` is the span the merged payload *declares* (see
    :func:`payload_samples`) while ``audio_end_sample`` is where the audio
    really sits on the channel timeline.
    """
    try:
        return type(turn)(
            start_sample=start_sample, end_sample=end_sample,
            audio=audio, reason="merged", peak_rms=peak_rms,
            audio_end_sample=audio_end_sample,
        )
    except TypeError:
        return type(turn)(
            start_sample=start_sample, end_sample=end_sample,
            pcm=audio, reason="merged", peak_rms=peak_rms,
        )


def _turn_payload(turn):
    """The turn's audio, whatever the concrete Turn class calls it.

    Never use ``a or b`` on these: a numpy array has no single truth value.
    """
    payload = getattr(turn, "audio", None)
    if payload is None:
        payload = getattr(turn, "pcm", None)
    if payload is None:
        return b""
    return payload


def _silence_like(payload, n_samples: int):
    """``n_samples`` of digital silence in the same representation as ``payload``.

    Used to keep a merged turn's audio on its true timeline: the payload is a
    slice of a turn's span, so a merge that just did ``head + tail`` would land
    the tail earlier than it occurred.
    """
    if n_samples <= 0:
        return None
    if isinstance(payload, np.ndarray):
        return np.zeros(int(n_samples), dtype=np.float32)
    return b"\x00" * (int(n_samples) * 2)


def payload_samples(payload) -> int:
    """How many samples a turn payload carries.

    The declared span of a turn must equal this: a span longer than the audio
    makes the decoder read audio that is not there and pushes every downstream
    attribution (span, ``covered_sec``, the shutdown re-attribution against the
    recording) past the end of what was actually sent.
    """
    if payload is None:
        return 0
    if isinstance(payload, np.ndarray):
        return int(payload.size)
    return int(len(payload)) // 2


# Public alias.  Edge refinement lives in ``speech_gate`` (it needs Silero, which
# this module must not import) and rebuilds a turn with a different span, so the
# two representations of "the same turn, different bounds" cannot drift apart.
rebuild_turn = _rebuild_turn
turn_payload = _turn_payload
silence_like = _silence_like


class TurnCoalescer:
    """Hold a closed turn briefly and merge it into the next one.

    ``hangover_ms`` is deliberately short (320 ms) so a live turn closes
    promptly, but natural speech contains 300-800 ms pauses in the middle of a
    sentence.  Every one of those closes a turn, and a 0.4 s fragment is
    useless twice over:

    * Whisper *hallucinates* on it -- a 0.5 s sniff reliably decodes to
      "Thank you." with ``avg_logprob`` high enough to look confident;
    * ECAPA cannot identify anyone from 0.4 s of audio, so the fragment can
      only ever come back ``UNKNOWN``.

    So a closed turn is not submitted immediately.  It is held as *pending* and
    merged with the next turn on the same channel when the pause between them
    is shorter than ``merge_gap_sec`` and the merged span stays under
    ``max_merge_sec``.  Otherwise the pending turn is released.

    The latency this adds is bounded by ``merge_gap_sec`` and only applies to
    pauses shorter than that.  A real conversational pause is longer, so it
    still submits at once and the live feel is unchanged -- and the ASR queue
    is the bottleneck anyway (~3 s per turn on this hardware), not this delay.

    The merged payload is ``head + silence + tail``: the silence puts the tail's
    audio back at the offset where it actually occurred.  The pad is measured
    from the head's own length rather than from the turn boundary, because a
    payload is only a *slice* of its turn's span (the detector trims hangover
    frames), so ``head + tail`` would place the tail up to ``merge_gap_sec``
    early.  Two spans are therefore kept distinct, and they mean different
    things:

    * ``end_sample`` -- the end of the audio actually carried, which is what the
      decoder reads.  Declaring the tail's real end while carrying only
      ``head + tail`` made the decoder read audio that is not in the payload and
      shifted every downstream consumer of the span (item timestamps,
      attribution, the shutdown re-attribution against the recording).
    * ``audio_end_sample`` -- where the audio really ended *on the recording*,
      used for the next merge decision (whose gap is a timeline measurement) and
      for the handler's ``covered_sec``.  Measuring the next gap against the
      trimmed audio end instead would inflate it by the trim and silently stop
      merges.
    """

    def __init__(self, cfg: Optional[dict] = None, sample_rate: int = SAMPLE_RATE,
                 clock=None):
        self.cfg = dict(_DEFAULTS)
        self.cfg.update(cfg or {})
        self.sample_rate = sample_rate
        self._clock = clock or time.monotonic
        self._pending = None
        # Real end of the pending turn's audio on the channel timeline; equal
        # to _pending.end_sample until the first merge shortens the span.
        self._audio_end = None
        self._due = None
        self._merged = 0
        self._released = 0

    @property
    def pending_sec(self) -> float:
        return float(getattr(self._pending, "duration_sec", 0.0) or 0.0)

    def _defer(self, now) -> None:
        self._due = (now if now is not None else self._clock()) + float(
            self.cfg["merge_gap_sec"])

    def _hold(self, turn) -> None:
        self._pending = turn
        self._audio_end = int(turn.end_sample)

    def submit(self, turn, now=None) -> List[Turn]:
        """Offer a completed turn; return the turns that are ready to send."""
        if turn is None:
            return []
        pending = self._pending
        if pending is None:
            self._hold(turn)
            self._defer(now)
            return []

        pending_end = self._audio_end
        if pending_end is None:
            pending_end = int(pending.end_sample)
        gap = int(turn.start_sample) - pending_end
        span = (int(turn.end_sample) - int(pending.start_sample)) / self.sample_rate

        if (0 <= gap <= float(self.cfg["merge_gap_sec"]) * self.sample_rate
                and span <= float(self.cfg["max_merge_sec"])):
            head = _turn_payload(pending)
            tail = _turn_payload(turn)
            start = int(pending.start_sample)
            # Digital silence between the two payloads so the tail sits at its
            # true offset.  This is not just the inter-turn gap: a payload is a
            # *slice* of its turn's span (the detector trims hangover frames), so
            # the pad is measured from the head's own length, not from
            # pending.end_sample.  With it, the merged turn's declared end equals
            # the tail's true audio end and nothing downstream is shifted.
            pad_n = int(turn.start_sample) - start - payload_samples(head)
            pad = _silence_like(head, pad_n)
            if isinstance(head, np.ndarray) or isinstance(tail, np.ndarray):
                audio = np.concatenate([
                    head if isinstance(head, np.ndarray) else np.frombuffer(
                        head, dtype=np.int16).astype(np.float32) / 32768.0,
                    pad if pad is not None else np.zeros(0, dtype=np.float32),
                    tail if isinstance(tail, np.ndarray) else np.frombuffer(
                        tail, dtype=np.int16).astype(np.float32) / 32768.0,
                ])
            else:
                audio = head + (pad or b"") + tail
            declared_end = start + payload_samples(audio)
            self._pending = _rebuild_turn(
                pending, start, declared_end, audio,
                max(float(getattr(pending, "peak_rms", 0.0) or 0.0),
                    float(getattr(turn, "peak_rms", 0.0) or 0.0)),
                audio_end_sample=int(turn.end_sample),
            )
            # The merge decision and covered_sec stay on the RECORDING's timeline
            # (turn.end_sample), not on the trimmed audio end -- measuring the
            # next gap against the trim would inflate it by the hangover and stop
            # merges that used to happen.
            self._audio_end = int(turn.end_sample)
            self._merged += 1
            self._defer(now)
            return []

        self._released += 1
        self._hold(turn)
        self._defer(now)
        return [pending]

    def poll(self, now=None) -> List[Turn]:
        """Release the held turn once the merge window has elapsed.

        Without this a turn is only ever released by the *next* turn or by
        end-of-stream, so a session that ends after one sentence would send
        nothing until shutdown. The deadline is what makes the extra latency
        bounded by ``merge_gap_sec`` rather than by the pause length.
        """
        if self._pending is None or self._due is None:
            return []
        if (now if now is not None else self._clock()) < self._due:
            return []
        pending, self._pending, self._due = self._pending, None, None
        self._audio_end = None
        self._released += 1
        return [pending]

    def flush(self) -> List[Turn]:
        """Release the held turn (end of stream / shutdown)."""
        pending, self._pending = self._pending, None
        self._audio_end = None
        self._due = None
        if pending is None:
            return []
        self._released += 1
        return [pending]

    def stats(self) -> dict:
        return {
            "merged": self._merged,
            "released": self._released,
            "pending": self.pending_sec > 0,
        }
