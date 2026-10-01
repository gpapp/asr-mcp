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
    _DEFAULTS, probe_speech, trim_turn_edges, turn_has_speech,
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


# ── late-loading probe ─────────────────────────────────────────────────────
# The gate is installed ONCE at connect but the model is lazy-loaded, so a
# session that connects before the VAD is resident used to be decided
# unfiltered for its whole duration.  The probe now reports "unavailable"
# per turn instead, which must still fail OPEN.

def test_a_probe_reporting_unavailable_fails_open():
    ok, score, reason = turn_has_speech(_Turn(), lambda turn: None)
    assert ok is True and reason == "speech_probe_unavailable"
    assert score == -1.0


def test_no_probe_at_all_fails_open_without_a_reason():
    ok, score, reason = turn_has_speech(_Turn(), None)
    assert ok is True and reason == "" and score == -1.0


def test_a_raising_probe_still_fails_open():
    def boom(turn):
        raise RuntimeError("model unloaded")

    ok, _, reason = turn_has_speech(_Turn(), boom)
    assert ok is True and reason == "speech_probe_failed"


# ── edge trim ──────────────────────────────────────────────────────────────
# The live boundary is an energy crossing plus a fixed pre/post-roll, so it is
# early at the onset and late at the offset by up to ~0.2 s.  Those padded
# frames are decoded and they widen the item span that the shutdown
# re-attribution matches against the diarization turns.

class _ScriptedVad:
    """Silero stand-in whose probability is read off a fixed list.

    One entry per 512-sample frame; anything past the end repeats the last.
    """

    def __init__(self, probs, names=("input",)):
        self.probs = list(probs)
        self._names = list(names)
        self.calls = 0

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in self._names]

    def run(self, _out, feed):
        self.calls += 1
        import numpy as np

        idx = self.calls - 1
        p = self.probs[idx] if idx < len(self.probs) else self.probs[-1]
        return [np.array([[p]], dtype=np.float32)]


class _RealTurn:
    """Stand-in with the server Turn's shape (float32 `audio`).

    Keyword-accepting like the real dataclass, because ``rebuild_turn``
    reconstructs by keyword -- that is the contract the trim relies on.
    """

    def __init__(self, start_sample=0, end_sample=None, audio=None,
                 duration_sec=None, peak_rms=0.2, reason="client_turn",
                 audio_end_sample=None, rate=16000, dur=None, start=None):
        if audio is None:
            audio = np.zeros(int((dur if dur is not None else 2.0) * rate),
                             dtype=np.float32)
        self.audio = np.asarray(audio, dtype=np.float32)
        if start is not None:
            start_sample = start
        self.start_sample = int(start_sample)
        self.end_sample = int(end_sample if end_sample is not None
                              else self.start_sample + self.audio.size)
        self.audio_end_sample = (int(audio_end_sample)
                                 if audio_end_sample is not None
                                 else self.end_sample)
        self.duration_sec = (float(duration_sec) if duration_sec is not None
                             else (self.end_sample - self.start_sample) / rate)
        self.peak_rms = peak_rms
        self.reason = reason

    @property
    def start_sec(self):
        return self.start_sample / 16000.0


def test_trim_removes_the_leading_silence():
    # 4 frames of padding, then speech, then speech to the end.
    vad = _ScriptedVad([0.02, 0.02, 0.02, 0.02, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    turn = _RealTurn()
    trimmed, sec = trim_turn_edges(turn, vad_session=vad, frame_size=512)
    assert sec == pytest.approx(4 * 512 / 16000.0, abs=1e-6)
    assert trimmed.start_sample == 4 * 512
    assert trimmed.end_sample == turn.end_sample
    assert trimmed.audio.size == turn.audio.size - 4 * 512


def test_trim_removes_the_trailing_silence():
    # The script must cover every frame: 2.0 s is 62 frames, and anything past
    # the end of the list repeats the last value.
    turn = _RealTurn()
    frames = int(turn.audio.size // 512)
    vad = _ScriptedVad([0.9] * (frames - 4) + [0.02] * 4)
    trimmed, sec = trim_turn_edges(turn, vad_session=vad, frame_size=512)
    assert sec == pytest.approx(4 * 512 / 16000.0, abs=1e-6)
    assert trimmed.start_sample == turn.start_sample
    assert trimmed.end_sample == turn.end_sample - 4 * 512


def test_trim_never_shortens_the_timeline():
    """The turn still occupies its true interval; only the audio inside moves."""
    vad = _ScriptedVad([0.02, 0.02, 0.9, 0.9, 0.9, 0.02])
    turn = _RealTurn(start=16000)
    trimmed, sec = trim_turn_edges(turn, vad_session=vad, frame_size=512)
    assert sec > 0
    # start advanced by exactly the trimmed samples, so the audio still starts
    # where it did; audio_end_sample is carried through untouched.
    assert trimmed.start_sample == 16000 + 2 * 512
    assert trimmed.audio_end_sample == turn.audio_end_sample
    assert trimmed.end_sample == trimmed.start_sample + trimmed.audio.size


def test_trim_is_a_no_op_when_the_turn_is_all_speech():
    vad = _ScriptedVad([0.9] * 12)
    turn = _RealTurn()
    trimmed, sec = trim_turn_edges(turn, vad_session=vad, frame_size=512)
    assert sec == 0.0
    assert trimmed is turn


def test_an_all_silent_turn_is_left_for_the_gate_to_reject():
    vad = _ScriptedVad([0.02] * 12)
    turn = _RealTurn()
    trimmed, sec = trim_turn_edges(turn, vad_session=vad, frame_size=512)
    assert sec == 0.0 and trimmed is turn


def test_trim_is_capped_by_edge_max_trim_sec():
    # 20 frames = 0.64 s of padding, but the cap is 0.40 s.
    vad = _ScriptedVad([0.01] * 20 + [0.9] * 20)
    turn = _RealTurn(dur=2.56)
    _, sec = trim_turn_edges(turn, vad_session=vad, frame_size=512,
                             cfg={"edge_max_trim_sec": 0.40,
                                  "edge_max_trim_ratio": 1.0})
    assert sec <= 0.40 + 1e-9


def test_trim_is_capped_by_edge_max_trim_ratio():
    # 30 frames of padding out of 60 = 50%, but the cap is 30%.
    vad = _ScriptedVad([0.01] * 30 + [0.9] * 30)
    turn = _RealTurn(dur=1.92)
    _, sec = trim_turn_edges(turn, vad_session=vad, frame_size=512,
                             cfg={"edge_max_trim_sec": 10.0,
                                  "edge_max_trim_ratio": 0.30})
    assert sec == pytest.approx(1.92 * 0.30, abs=0.05)


def test_the_trim_threshold_is_configurable():
    """A HIGHER bar calls more frames silence, so it trims more."""
    probs = [0.30, 0.30, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    turn = _RealTurn()
    frames = int(turn.audio.size // 512)
    probs = probs + [0.9] * (frames - len(probs))
    lenient, sec_a = trim_turn_edges(turn, vad_session=_ScriptedVad(probs),
                                     frame_size=512,
                                     cfg={"edge_frame_threshold": 0.20})
    strict, sec_b = trim_turn_edges(turn, vad_session=_ScriptedVad(probs),
                                    frame_size=512,
                                    cfg={"edge_frame_threshold": 0.80})
    assert sec_a == 0.0          # 0.30 counts as speech, so nothing is padding
    assert sec_b == pytest.approx(2 * 512 / 16000.0, abs=1e-6)


def test_trim_can_be_switched_off():
    vad = _ScriptedVad([0.01, 0.01, 0.9, 0.9, 0.9, 0.9])
    turn = _RealTurn()
    trimmed, sec = trim_turn_edges(turn, vad_session=vad, frame_size=512,
                                   cfg={"edge_trim_enabled": False})
    assert sec == 0.0 and trimmed is turn


def test_trim_without_a_session_is_a_no_op():
    turn = _RealTurn()
    trimmed, sec = trim_turn_edges(turn, vad_session=None)
    assert sec == 0.0 and trimmed is turn


def test_a_raising_probe_leaves_the_boundary_alone():
    class _Boom:
        def get_inputs(self):
            return [type("I", (), {"name": "input"})()]

        def run(self, *_a):
            raise RuntimeError("session gone")

    turn = _RealTurn()
    trimmed, sec = trim_turn_edges(turn, vad_session=_Boom())
    assert sec == 0.0 and trimmed is turn


def test_a_turn_shorter_than_two_frames_is_untouched():
    turn = _RealTurn(dur=0.032)
    trimmed, sec = trim_turn_edges(turn, vad_session=_ScriptedVad([0.01]),
                                   frame_size=512)
    assert sec == 0.0 and trimmed is turn


def test_the_trimmed_turn_keeps_the_detectors_reason():
    vad = _ScriptedVad([0.01, 0.9, 0.9, 0.9, 0.9, 0.9])
    turn = _RealTurn(reason="client_turn")
    trimmed, sec = trim_turn_edges(turn, vad_session=vad, frame_size=512)
    assert sec > 0
    assert trimmed.reason == "client_turn"


def test_defaults_expose_the_trim_knobs():
    for key in ("edge_trim_enabled", "edge_frame_threshold",
                "edge_max_trim_sec", "edge_max_trim_ratio"):
        assert key in _DEFAULTS


def test_threshold_json_matches_the_module_defaults():
    from asr_mcp.config import get_config

    shipped = get_config().get("streaming", {})
    for key, value in _DEFAULTS.items():
        if key in shipped:
            assert shipped[key] == pytest.approx(value), key
