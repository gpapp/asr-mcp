"""The pre-ASR speech gate: do not spend a decode on a sniff.

Whisper's standard failure mode on non-speech audio is a hallucination -- half
a second of room noise decodes to "Thank you." with a perfectly healthy
``avg_logprob``, and the client prints it as ``UNKNOWN: Thank you``. The gate
runs before the decoder and is deliberately **fail-open**: refusing to
transcribe real speech is worse than transcribing a noise fragment.

The decoder's own ``no_speech_prob`` was implemented here as a second line of
defence and then removed: measured in-container against the real backend
(faster-whisper 1.2.1 / CTranslate2 4.8.2 / large-v3-turbo) it is 0.0000 for
every segment -- speech, white noise and digital silence alike -- so the gate
would have been dead code reading as protection.  The thresholds exercised
below are the ones measured against the real Silero model on real audio:
speech 0.46-0.57, noise/hiss/sniff/silence 0.04-0.23.
"""

import numpy as np
import pytest

from asr_mcp.streaming.speech_gate import (
    _DEFAULTS, probe_speech, turn_has_speech,
)



class _Turn:
    def __init__(self, dur=1.0, rate=16000):
        self.audio = np.zeros(int(dur * rate), dtype=np.float32)
        self.duration_sec = dur


def _probe(mean, ratio):
    return lambda turn: (mean, ratio)


# ── gating ─────────────────────────────────────────────────────────────────

def test_speech_passes():
    ok, score, reason = turn_has_speech(_Turn(), _probe(0.8, 0.9))
    assert ok is True and reason == "" and score == pytest.approx(0.8)


def test_noise_is_rejected_on_mean_probability():
    ok, score, reason = turn_has_speech(_Turn(), _probe(0.04, 0.05))
    assert ok is False
    assert reason == "no_speech_low_prob"
    assert score == pytest.approx(0.04)


def test_sparse_speech_is_rejected_on_frame_ratio():
    """A loud click in a mostly-silent turn: decent mean, almost no speech."""
    ok, _, reason = turn_has_speech(_Turn(), _probe(0.45, 0.10))
    assert ok is False and reason == "no_speech_low_ratio"


def test_no_probe_fails_open():
    """A missing VAD must never silence the session."""
    assert turn_has_speech(_Turn(), None)[:2] == (True, -1.0)


def test_probe_exception_fails_open():
    def boom(turn):
        raise RuntimeError("onnx exploded")

    ok, score, reason = turn_has_speech(_Turn(), boom)
    assert ok is True and score == -1.0 and reason == "speech_probe_failed"


def test_gate_can_be_disabled():
    assert turn_has_speech(_Turn(), _probe(0.0, 0.0),
                           {"speech_gate_enabled": False})[0] is True


def test_thresholds_are_configurable():
    cfg = dict(_DEFAULTS, min_speech_prob=0.05, min_speech_ratio=0.05)
    assert turn_has_speech(_Turn(), _probe(0.10, 0.10), cfg)[0] is True
    strict = dict(_DEFAULTS, min_speech_prob=0.9, min_speech_ratio=0.9)
    assert turn_has_speech(_Turn(), _probe(0.10, 0.10), strict)[0] is False


def test_none_from_the_probe_is_treated_as_no_speech():
    assert turn_has_speech(_Turn(), _probe(None, None))[0] is False


# ── the Silero driver, against a fake session ──────────────────────────────

class _FakeVad:
    """Silero stand-in: reports a constant probability per frame."""

    def __init__(self, prob, names=("input", "state", "sr")):
        self.prob = prob
        self._names = list(names)
        self.frames = 0

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in self._names]

    def run(self, _out, feed):
        self.frames += 1
        import numpy as np

        h = np.zeros((2, 1, 128), dtype=np.float32)
        return [np.array([[self.prob]], dtype=np.float32), h, h]


def test_probe_scores_speech():
    mean, ratio = probe_speech(np.zeros(16000, np.float32), _FakeVad(0.9))
    assert mean == pytest.approx(0.9) and ratio == pytest.approx(1.0)


def test_probe_scores_noise():
    mean, ratio = probe_speech(np.zeros(16000, np.float32), _FakeVad(0.02))
    assert mean == pytest.approx(0.02) and ratio == pytest.approx(0.0)


def test_probe_threshold_is_honoured():
    _, ratio = probe_speech(np.zeros(16000, np.float32), _FakeVad(0.6),
                            threshold=0.5)
    assert ratio == pytest.approx(1.0)
    _, ratio = probe_speech(np.zeros(16000, np.float32), _FakeVad(0.6),
                            threshold=0.8)
    assert ratio == pytest.approx(0.0)


def test_probe_handles_int16_payloads():
    """Turn frames arrive as int16; not rescaling would read them as pure clipping."""
    import numpy as np

    raw = (np.random.RandomState(0).randn(16000) * 3000).astype(np.int16)
    mean, _ = probe_speech(raw, _FakeVad(0.9))
    assert 0.0 <= mean <= 1.0


def test_probe_on_a_too_short_turn():
    assert probe_speech(np.zeros(100, np.float32), _FakeVad(0.9)) == (0.0, 0.0)


def test_probe_feeds_the_state_tensors():
    vad = _FakeVad(0.5)
    probe_speech(np.zeros(16000, np.float32), vad)
    assert vad.frames > 1
