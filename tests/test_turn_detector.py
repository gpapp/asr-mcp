"""Unit tests for the live-streaming turn detector.

The old implementation used a fixed RMS gate (0.02) with hard >0.5s / <0.3s
filters, which dropped short answers ("Yes", "No") and split utterances on
intra-sentence dips.  These tests lock in the new behaviour: adaptive floor,
hysteresis, hangover, pre/post-roll, short-turn support and flush-on-disconnect.
"""

import numpy as np
import pytest

SR = 16000
FRAME_MS = 32


def pcm(seconds, amp, sr=SR):
    """int16 PCM bytes for a constant-amplitude segment."""
    n = max(1, int(seconds * sr))
    return (np.ones(n, dtype=np.int16) * int(amp * 32767)).astype("<i2").tobytes()


def silence(seconds, sr=SR):
    return pcm(seconds, 0.0, sr)


def feed_all(det, chunks):
    turns = []
    for c in chunks:
        turns.extend(det.feed(c))
    return turns


# --- config / helpers -----------------------------------------------------


def test_config_defaults_present(turn_detector):
    cfg = turn_detector.config()
    for key in ("frame_ms", "start_threshold_ratio", "end_threshold_ratio",
                "hangover_ms", "pre_roll_ms", "post_roll_ms", "min_voiced_ms",
                "max_turn_sec", "start_confirm_frames"):
        assert key in cfg
    # Hysteresis: end threshold must sit strictly below the start threshold.
    assert cfg["end_threshold_ratio"] < cfg["start_threshold_ratio"]


def test_frame_size(turn_detector):
    assert turn_detector.frame_size({"frame_ms": 32}) == 512
    assert turn_detector.frame_size({"frame_ms": 20}) == 320


def test_rms(turn_detector):
    assert turn_detector.rms(b"") == 0.0
    assert turn_detector.rms(pcm(0.1, 0.5)) == pytest.approx(0.5, rel=0.01)
    assert turn_detector.rms(pcm(0.1, 0.0)) == 0.0


# --- basic endpointing ----------------------------------------------------


def test_speech_emits_single_turn(turn_detector):
    det = turn_detector.TurnDetector()
    turns = feed_all(det, [silence(0.5), pcm(1.0, 0.2), silence(1.0)])
    assert len(turns) == 1
    assert turns[0].reason == "hangover"
    assert turns[0].duration_sec >= 0.9
    assert det.turns_emitted == 1


def test_short_answer_is_kept(turn_detector):
    # 0.2s "Yes" — the old <0.3s filter dropped it.
    det = turn_detector.TurnDetector()
    turns = feed_all(det, [silence(0.4), pcm(0.2, 0.25), silence(1.0)])
    assert len(turns) == 1
    assert turns[0].duration_sec >= 0.12
    assert det.dropped_turns == 0


def test_intra_utterance_dip_does_not_split(turn_detector):
    # 300ms consonant gap inside a sentence: hangover must bridge it.
    det = turn_detector.TurnDetector()
    turns = feed_all(det, [silence(0.4), pcm(0.6, 0.2), silence(0.3), pcm(0.6, 0.2), silence(1.0)])
    assert len(turns) == 1


def test_two_separate_turns(turn_detector):
    det = turn_detector.TurnDetector()
    turns = feed_all(det, [
        silence(0.4), pcm(0.8, 0.2), silence(1.0), pcm(0.8, 0.2), silence(1.0)])
    assert len(turns) == 2
    assert turns[0].end_sample < turns[1].start_sample


def test_turns_do_not_overlap(turn_detector):
    det = turn_detector.TurnDetector()
    turns = feed_all(det, [
        silence(0.4), pcm(0.5, 0.2), silence(0.9), pcm(0.5, 0.2), silence(0.9)])
    for a, b in zip(turns, turns[1:]):
        assert a.end_sample <= b.start_sample


# --- adaptive floor -------------------------------------------------------


def test_noise_floor_tracks_loud_room_down(turn_detector):
    det = turn_detector.TurnDetector()
    feed_all(det, [silence(0.2), pcm(2.0, 0.02)])
    quiet_floor = det.noise_floor
    # Same absolute level now counts as speech relative to a quiet room...
    turns = feed_all(det, [silence(0.2), pcm(0.3, 0.02), silence(1.0)])
    assert quiet_floor < 0.05
    assert len(turns) == 1


def test_noise_floor_clamped(turn_detector):
    det = turn_detector.TurnDetector()
    feed_all(det, [silence(1.0)])
    assert det.noise_floor >= det.cfg["noise_floor_min"]
    feed_all(det, [pcm(3.0, 0.9)])
    assert det.noise_floor <= det.cfg["noise_floor_max"]


def test_single_transient_does_not_open_turn(turn_detector):
    det = turn_detector.TurnDetector()
    feed_all(det, [silence(0.6)])
    turns = feed_all(det, [pcm(FRAME_MS / 1000.0 * 2, 0.9), silence(1.0)])
    assert turns == []


# --- padding / limits -----------------------------------------------------


def test_preroll_captures_onset(turn_detector):
    det = turn_detector.TurnDetector()
    feed_all(det, [silence(1.0)])
    onset_sec = 1.0
    turns = feed_all(det, [pcm(0.5, 0.2), silence(1.0)])
    start = turns[0].start_sec
    # Pre-roll pulls the boundary back before the true onset, and start
    # confirmation never pushes it more than a couple of frames past it.
    pre = det.cfg["pre_roll_ms"] / 1000.0
    confirm = FRAME_MS / 1000.0 * det.cfg["start_confirm_frames"]
    assert onset_sec - pre - FRAME_MS / 1000.0 <= start <= onset_sec + confirm
    # ...but it is bounded: not the whole preceding silence.
    assert start > onset_sec - 1.0


def test_turn_audio_length_matches_span(turn_detector):
    det = turn_detector.TurnDetector()
    turns = feed_all(det, [silence(0.4), pcm(0.7, 0.2), silence(1.0)])
    t = turns[0]
    assert len(t.audio) / det.sample_rate == pytest.approx(t.duration_sec, abs=1e-6)
    assert t.peak_rms > 0.05
    assert t.mean_rms > 0.0


def test_max_turn_force_split(turn_detector):
    det = turn_detector.TurnDetector({"max_turn_sec": 1.0})
    turns = feed_all(det, [silence(0.4), pcm(2.5, 0.2), silence(1.0)])
    assert len(turns) >= 2
    assert turns[0].reason == "max_turn"
    assert all(t.duration_sec <= 1.2 for t in turns)


def test_min_voiced_drops_click(turn_detector):
    det = turn_detector.TurnDetector({"min_voiced_ms": 300})
    feed_all(det, [silence(0.5)])
    turns = feed_all(det, [pcm(0.05, 0.5), silence(1.0)])
    assert turns == []
    assert det.dropped_turns == 1


# --- flush on disconnect --------------------------------------------------


def test_flush_emits_open_turn(turn_detector):
    det = turn_detector.TurnDetector()
    feed_all(det, [silence(0.4), pcm(0.5, 0.2)])
    assert det.in_speech
    t = det.flush()
    assert t is not None
    assert t.reason == "flush"
    assert t.duration_sec >= 0.3
    # Idempotent: a second flush has nothing left to close.
    assert det.flush() is None
    assert not det.in_speech


def test_flush_without_speech_is_none(turn_detector):
    det = turn_detector.TurnDetector()
    feed_all(det, [silence(0.5)])
    assert det.flush() is None


# --- byte-level plumbing --------------------------------------------------


def test_partial_frames_are_buffered(turn_detector):
    det = turn_detector.TurnDetector()
    data = silence(0.4) + pcm(0.4, 0.2)
    turns = []
    for i in range(0, len(data), 97):  # deliberately unaligned chunks
        turns.extend(det.feed(data[i:i + 97]))
    turns.extend(det.feed(silence(1.0)))
    assert len(turns) == 1
    # No duplicate/partial frames: the span is frame aligned.
    assert turns[0].start_sample % det.frame_samples == 0
    assert (turns[0].end_sample - turns[0].start_sample) % det.frame_samples == 0


def test_empty_feed_is_noop(turn_detector):
    det = turn_detector.TurnDetector()
    assert det.feed(b"") == []
    assert det.stats()["frames"] == 0


def test_stats(turn_detector):
    det = turn_detector.TurnDetector()
    feed_all(det, [silence(0.4), pcm(0.5, 0.2), silence(1.0)])
    st = det.stats()
    assert st["turns"] == 1
    assert st["dropped_turns"] == 0
    assert st["frames"] > 0
    assert st["in_speech"] is False
