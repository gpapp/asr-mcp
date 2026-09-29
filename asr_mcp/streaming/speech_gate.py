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

logger = logging.getLogger("asr_mcp.streaming.speech_gate")

_DEFAULTS = {
    "speech_gate_enabled": True,
    # Mean Silero speech probability over the turn.
    "min_speech_prob": 0.30,
    # Fraction of Silero frames above 0.5 that must be speech.
    "min_speech_ratio": 0.35,
    "vad_frame_threshold": 0.5,
}


def _cfg(cfg: Optional[dict] = None) -> dict:
    merged = dict(_DEFAULTS)
    merged.update(cfg or {})
    return merged


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

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size < frame_size:
        return 0.0, 0.0
    if samples.max() > 1.5 or samples.min() < -1.5:
        samples = samples / 32768.0  # int16 payloads arrive unscaled

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

    if not probs:
        return 0.0, 0.0
    arr = np.asarray(probs, dtype=np.float32)
    return float(arr.mean()), float((arr >= threshold).mean())


def turn_has_speech(turn, probe=None, cfg: Optional[dict] = None
                    ) -> Tuple[bool, float, str]:
    """``(transcribe?, score, reason)`` for one turn.

    ``probe`` is injectable -- ``probe(turn) -> (mean_prob, speech_ratio)`` --
    so the gating logic is unit-testable without loading a model.  ``None``
    means "no detector available" and always lets the turn through.
    """
    c = _cfg(cfg)
    if not bool(c.get("speech_gate_enabled", True)) or probe is None:
        return True, -1.0, ""

    try:
        mean_prob, ratio = probe(turn)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Speech probe failed, transcribing anyway: %s", e)
        return True, -1.0, "speech_probe_failed"

    mean_prob = float(mean_prob or 0.0)
    ratio = float(ratio or 0.0)
    score = mean_prob

    if mean_prob < float(c["min_speech_prob"]):
        return False, score, "no_speech_low_prob"
    if ratio < float(c["min_speech_ratio"]):
        return False, score, "no_speech_low_ratio"
    return True, score, ""
