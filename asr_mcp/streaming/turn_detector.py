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
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger("asr_mcp.streaming.turn_detector")

SAMPLE_RATE = 16000

_DEFAULTS = {
    "frame_ms": 32.0,
    "noise_floor_ratio": 3.0,
    "noise_floor_min": 0.0005,
    "noise_floor_max": 0.05,
    "start_threshold_ratio": 2.5,
    "end_threshold_ratio": 1.6,
    "start_threshold_min": 0.0012,
    "end_threshold_min": 0.0006,
    "start_confirm_frames": 2,
    "end_confirm_frames": 3,
    "hangover_ms": 320,
    "pre_roll_ms": 160,
    "post_roll_ms": 200,
    "min_voiced_ms": 120,
    "max_turn_sec": 30.0,
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

    @property
    def duration_sec(self) -> float:
        return max(0, self.end_sample - self.start_sample) / SAMPLE_RATE

    @property
    def start_sec(self) -> float:
        return self.start_sample / SAMPLE_RATE

    @property
    def end_sec(self) -> float:
        return self.end_sample / SAMPLE_RATE


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
