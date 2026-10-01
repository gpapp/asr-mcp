"""Decide whether a live turn contains speech before spending ASR on it.

An energy-based endpointing detector cannot tell a breath, a sniff, a keyboard
click or a chair creak from a syllable: all of them are short bursts above the
noise floor, and the detector's job is only to cut *when* speech starts and
stops, not to decide *whether* it is speech at all.  Whatever survives is sent
to Whisper, and Whisper's documented failure mode on non-speech is a
**hallucination**: half a second of room noise decodes to "Thank you." with a
confident ``avg_logprob``, and the client shows ``UNKNOWN: Thank you`` as if
the speaker had said it.

This module is the *only* effective non-speech guard on the live path. Whisper's
own ``no_speech_prob`` is not: measured in-container against the real backend
(faster-whisper 1.2.1 / CTranslate2 4.8.2 / large-v3-turbo) it came back
0.0000 for every segment -- real speech, white noise and digital silence alike
-- and ``avg_logprob`` is no help either (silence decodes to "Thank you." at
-0.29, real speech at -0.30).  See the module comment in
``transcribers/whisper.py`` before reintroducing a decoder-side check.

Measured with the shipped thresholds, real Silero against real audio:

===========================  ================  ===========
input                        mean prob         verdict
===========================  ================  ===========
real speech, 1.5 s           0.46 - 0.57       pass
white noise -50 dBFS         0.23              reject
white noise -40 dBFS         0.08              reject
white noise -30 dBFS         0.09              reject
3 kHz hiss -35 dBFS          0.14              reject
sniff / click, 0.4 s         0.04              reject
digital silence              0.04              reject
===========================  ================  ===========

The gate is deliberately **fail-open**: a missing model, an exception or an
unset threshold lets the turn through. Refusing to transcribe real speech is a
worse failure than transcribing a noise fragment.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from asr_mcp.streaming.turn_detector import (
    _turn_payload,
    payload_samples,
    rebuild_turn,
)

logger = logging.getLogger("asr_mcp.streaming.speech_gate")

_DEFAULTS = {
    "speech_gate_enabled": True,
    # Mean Silero speech probability over the turn.
    "min_speech_prob": 0.30,
    # Fraction of Silero frames above 0.5 that must be speech.
    "min_speech_ratio": 0.35,
    "vad_frame_threshold": 0.5,
    # --- edge trim -------------------------------------------------------
    # Trim leading/trailing frames whose Silero probability is below
    # `edge_frame_threshold`.  The client places a boundary by an energy
    # crossing plus a fixed pre/post-roll, so it is systematically early at the
    # onset and late at the offset (up to ~0.2 s nominal).  The VAD session is
    # already resident for the gate above, so trimming costs no extra model and
    # no measurable latency.
    "edge_trim_enabled": True,
    "edge_frame_threshold": 0.35,
    # Never trim more than this much from either end, and never trim away more
    # than this fraction of the turn -- a turn that is *mostly* silence is the
    # gate's business, not the trim's, and must reach the gate intact.
    "edge_max_trim_sec": 0.40,
    "edge_max_trim_ratio": 0.30,
}


def _cfg(cfg: Optional[dict] = None) -> dict:
    merged = dict(_DEFAULTS)
    merged.update(cfg or {})
    return merged


def _as_float32(audio, frame_size: int):
    """Coerce a turn payload to a flat float32 array in [-1, 1]."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size and (samples.max() > 1.5 or samples.min() < -1.5):
        samples = samples / 32768.0  # int16 payloads arrive unscaled
    return samples


def frame_speech_probs(audio, vad_session, sample_rate: int = 16000,
                       frame_size: int = 512) -> "list[float]":
    """Per-frame Silero speech probability for one turn.

    The per-frame form is what :func:`trim_turn_edges` needs: the energy
    detector places a boundary by a threshold crossing plus a fixed pre/post
    roll, so the audio either side of the true onset/offset is padding, and
    Silero can say exactly which frames are padding.
    """
    import numpy as np

    samples = _as_float32(audio, frame_size)
    if samples.size < frame_size:
        return []

    input_names = [inp.name for inp in vad_session.get_inputs()]
    input_name = input_names[0]
    h = np.zeros((2, 1, 128), dtype=np.float32)
    c = np.zeros((2, 1, 128), dtype=np.float32)
    sr_np = np.array([sample_rate], dtype=np.int64)

    probs = []
    for i in range(0, samples.size - frame_size + 1, frame_size):
        feed = {input_name: samples[i:i + frame_size][np.newaxis]}
        if "state" in input_names:
            feed["state"] = h
        if "sr" in input_names:
            feed["sr"] = sr_np
        outputs = vad_session.run(None, feed)
        prob = outputs[0]
        probs.append(float(prob[0][0]) if np.ndim(prob) else float(prob))
        if len(outputs) >= 3:
            h, c = outputs[1], outputs[2]
    return probs


def probe_speech(audio, vad_session, sample_rate: int = 16000,
                 frame_size: int = 512, threshold: float = 0.5
                 ) -> Tuple[float, float]:
    """Run Silero over one turn and return ``(mean_prob, speech_ratio)``.

    Unlike :func:`asr_mcp.speaker.vad.run_vad_onnx` this does **no** trigger
    logic and applies **no** minimum-duration filter: a live turn is 0.3-8 s and
    the file path's 250 ms floor would discard exactly the short replies that
    matter here.  Every 512-sample frame is scored, and the two numbers are
    reported so the caller can gate on either.
    """
    import numpy as np

    samples = _as_float32(audio, frame_size)
    if samples.size < frame_size:
        return 0.0, 0.0

    probs = frame_speech_probs(samples, vad_session,
                               sample_rate=sample_rate, frame_size=frame_size)
    if not probs:
        return 0.0, 0.0
    arr = np.asarray(probs, dtype=np.float32)
    return float(arr.mean()), float((arr >= threshold).mean())


def trim_turn_edges(turn, vad_session=None, cfg: Optional[dict] = None,
                    frame_size: int = 512, sample_rate: int = 16000):
    """Move a live turn's onset/offset onto the speech, not onto the energy gate.

    Returns ``(turn, trimmed_sec)``.  ``turn`` is the *same object* when nothing
    was trimmed, so a caller can compare identity to detect a no-op.

    Why this exists: neither live client can place a boundary better than an
    energy crossing plus a fixed pre/post-roll, so the onset is up to
    ``pre_roll_ms`` (160 ms) early and the offset up to ``post_roll_ms`` (200 ms)
    late, on a 32 ms grid -- up to ~0.5 s against a true VAD onset for quiet
    speech.  Every one of those padded frames is decoded by the ASR and every
    one of them widens the item span that the shutdown re-attribution matches
    against the diarization turns, so a boundary error of that size can turn
    into an ``UNKNOWN (boundary_crossing)`` on the final transcript.

    The Silero session is already resident for :func:`turn_has_speech`, so this
    costs no extra model and no extra model load -- it is one more scoring pass
    over the same frames.

    **The timeline is preserved, never shortened.**  ``start_sample`` advances by
    exactly the number of trimmed samples and ``audio_end_sample`` is carried
    through unchanged, so the turn still occupies its true interval on the
    channel; only the audio inside it and the span it declares change.

    Fails open on every unavailable path (no probe, knob off, probe returns
    ``None``, probe raises, nothing to trim, trim would exceed the caps), because
    a worse outcome than "the boundary is 0.2 s early" is losing the turn.
    """
    c = _cfg(cfg)
    if not bool(c.get("edge_trim_enabled", True)) or vad_session is None:
        return turn, 0.0

    payload = _turn_payload(turn)
    total = payload_samples(payload)
    if total < 2 * frame_size:
        return turn, 0.0

    try:
        probs = frame_speech_probs(payload, vad_session,
                                   sample_rate=sample_rate,
                                   frame_size=frame_size)
    except Exception as e:
        logger.warning("Edge trim probe failed, keeping the boundary: %s", e)
        return turn, 0.0
    if not probs:
        return turn, 0.0

    thresh = float(c.get("edge_frame_threshold", 0.35))
    arr = np.asarray(probs, dtype=np.float32)
    voiced = np.flatnonzero(arr >= thresh)
    if voiced.size == 0:
        return turn, 0.0  # nothing but padding; the gate decides this turn

    # Only the first/last scored frame can be padding; a silence in the middle
    # belongs to the turn (the coalescer already merged across pauses).
    first = int(voiced[0])
    last = int(voiced[-1])
    max_frames = int(float(c.get("edge_max_trim_sec", 0.40))
                     * sample_rate / frame_size)
    # Never trim away more than `edge_max_trim_ratio` of the turn: a turn that is
    # mostly silence is the gate's business, and must reach it intact.
    cap = min(max_frames, int(total * float(c.get("edge_max_trim_ratio", 0.30))
                              / frame_size))
    lead = min(first, max(cap, 0))
    trail = min(len(arr) - 1 - last, max(cap - lead, 0))
    if lead <= 0 and trail <= 0:
        return turn, 0.0

    start = int(lead * frame_size)
    end = total - int(trail * frame_size)
    if end - start < frame_size:
        return turn, 0.0

    if isinstance(payload, np.ndarray):
        audio = payload[start:end]
    else:
        audio = payload[start * 2:end * 2]

    trimmed = (total - (end - start)) / float(sample_rate)
    rebuilt = rebuild_turn(
        turn,
        int(turn.start_sample) + start,
        int(turn.start_sample) + end,
        audio,
        getattr(turn, "peak_rms", 0.0),
        audio_end_sample=getattr(turn, "audio_end_sample", None),
    )
    # Keep the reason the detector gave; only the bounds moved.
    try:
        rebuilt.reason = getattr(turn, "reason", "client_turn")
    except Exception:  # pragma: no cover - a frozen dataclass
        pass
    return rebuilt, float(trimmed)


def turn_has_speech(turn, probe=None, cfg: Optional[dict] = None
                    ) -> Tuple[bool, float, str]:
    """``(transcribe?, score, reason)`` for one turn.

    ``probe`` is injectable -- ``probe(turn) -> (mean_prob, speech_ratio)`` --
    so the gating logic is unit-testable without loading a model.  ``None``
    means "no detector available" and always lets the turn through.

    The probe may ALSO return ``None`` at call time, meaning "I have no
    detector right now".  That is the late-loading case: a live session can
    connect before the VAD model is resident, and it must not be gated for its
    whole duration by a decision made once at connect.  The probe is therefore
    always installed and resolves the model per turn (lesson 26).
    """
    c = _cfg(cfg)
    if not bool(c.get("speech_gate_enabled", True)) or probe is None:
        return True, -1.0, ""

    try:
        probed = probe(turn)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Speech probe failed, transcribing anyway: %s", e)
        return True, -1.0, "speech_probe_failed"

    if probed is None:
        return True, -1.0, "speech_probe_unavailable"
    mean_prob, ratio = probed

    mean_prob = float(mean_prob or 0.0)
    ratio = float(ratio or 0.0)
    score = mean_prob

    if mean_prob < float(c["min_speech_prob"]):
        return False, score, "no_speech_low_prob"
    if ratio < float(c["min_speech_ratio"]):
        return False, score, "no_speech_low_ratio"
    return True, score, ""
