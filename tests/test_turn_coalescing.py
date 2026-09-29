"""Turn coalescing: stop submitting 0.4s fragments.

Whisper hallucinates on short non-speech audio ("Thank you." arrives with a
confident avg_logprob) and ECAPA cannot name anyone from it, so every
intra-sentence pause used to cost a wasted decode and an UNKNOWN line. The
coalescer holds a closed turn for at most ``merge_gap_sec`` and merges the
next one into it.

The client mirrors the same class, so the parity tests matter: a turn cut live
must be submitted the same way it would be re-diarized offline.
"""

import pytest

from asr_mcp.streaming.turn_detector import Turn, TurnCoalescer, SAMPLE_RATE


@pytest.fixture
def clock():
    return _Clock()


class _Clock:
    """Deterministic monotonic clock."""

    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


def _t(start_sec, dur_sec, amp=0.5):
    n = int(dur_sec * SAMPLE_RATE)
    import numpy as np

    return Turn(
        start_sample=int(start_sec * SAMPLE_RATE),
        end_sample=int((start_sec + dur_sec) * SAMPLE_RATE),
        audio=np.full(n, amp, dtype=np.float32),
        reason="hangover",
        peak_rms=amp,
    )


def _cfg(**over):
    base = {"merge_gap_sec": 1.0, "max_merge_sec": 15.0}
    base.update(over)
    return base


# ── merging ────────────────────────────────────────────────────────────────

def test_first_turn_is_held_not_released(clock):
    """Releasing immediately would make the coalescer a no-op."""
    c = TurnCoalescer(_cfg(), clock=clock)
    assert c.submit(_t(0, 2.0)) == []
    assert c.pending_sec == pytest.approx(2.0)


def test_short_pause_merges(clock):
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 2.0))
    assert c.submit(_t(2.4, 1.0)) == []          # 0.4s gap
    assert c._merged == 1
    assert c.pending_sec == pytest.approx(3.4)


def test_merged_audio_is_concatenated(clock):
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 1.0, amp=0.1))
    c.submit(_t(1.2, 1.0, amp=0.9))
    merged = c._pending
    assert len(merged.audio) == pytest.approx(2.0 * SAMPLE_RATE, abs=2)
    assert merged.peak_rms == pytest.approx(0.9)
    assert merged.reason == "merged"
    assert merged.start_sample == 0
    assert merged.end_sample == pytest.approx(2.2 * SAMPLE_RATE, abs=2)


def test_long_pause_releases_the_previous_turn(clock):
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 2.0))
    out = c.submit(_t(4.0, 1.0))                 # 2.0s gap > merge_gap_sec
    assert len(out) == 1
    assert out[0].start_sample == 0
    assert c.pending_sec == pytest.approx(1.0)


def test_max_merge_sec_stops_growth(clock):
    c = TurnCoalescer(_cfg(max_merge_sec=5.0), clock=clock)
    c.submit(_t(0, 4.0))
    # 4.0 + 0.2 gap + 2.0 = 6.2s > 5.0s -> release instead of merging
    out = c.submit(_t(4.2, 2.0))
    assert len(out) == 1
    assert c._merged == 0


def test_overlapping_turns_are_not_merged(clock):
    """gap < 0 means the spans overlap; concatenating would double-count audio."""
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 2.0))
    out = c.submit(_t(1.5, 1.0))                 # starts before the first ends
    assert len(out) == 1


def test_turn_is_merged_at_exactly_merge_gap(clock):
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 2.0))
    assert c.submit(_t(3.0, 1.0)) == []          # gap == 1.0 exactly
    assert c._merged == 1


# ── the deadline that makes the added latency bounded ──────────────────────

def test_poll_releases_after_the_merge_window(clock):
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 2.0))
    assert c.poll() == []                        # not yet due
    clock.advance(0.9)
    assert c.poll() == []
    clock.advance(0.2)
    out = c.poll()
    assert len(out) == 1 and out[0].start_sample == 0
    assert c.poll() == []                        # released once only


def test_poll_uses_the_deadline_not_the_pause(clock):
    """A single sentence in a long silence must still go out on time."""
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 2.0))
    clock.advance(1.0)                           # merge_gap_sec
    assert len(c.poll()) == 1


def test_merge_resets_the_deadline(clock):
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 1.0))
    clock.advance(0.8)
    c.submit(_t(1.2, 1.0))                       # merged at t=0.8
    clock.advance(0.8)                           # t=1.6 -- would be due from
    assert c.poll() == []                        # the FIRST turn alone
    clock.advance(0.3)
    assert len(c.poll()) == 1


def test_flush_releases_the_held_turn(clock):
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 2.0))
    out = c.flush()
    assert len(out) == 1 and out[0].start_sample == 0
    assert c.flush() == [] and c.poll() == []


def test_flush_on_an_empty_coalescer(clock):
    assert TurnCoalescer(_cfg(), clock=clock).flush() == []


def test_stats(clock):
    c = TurnCoalescer(_cfg(), clock=clock)
    c.submit(_t(0, 1.0))
    assert c.stats()["pending"] is True
    c.submit(_t(1.2, 1.0))
    c.flush()
    assert c.stats() == {"merged": 1, "released": 1, "pending": False}


def test_submit_ignores_none(clock):
    c = TurnCoalescer(_cfg(), clock=clock)
    assert c.submit(None) == []
    assert c.stats()["pending"] is False


# ── the client's mirror must behave identically ────────────────────────────

def test_client_coalescer_matches_the_server(live):
    import numpy as np

    def t(start_sec, dur_sec, amp=0.5):
        n = int(dur_sec * live.SAMPLE_RATE)
        return live.Turn(
            start_sample=int(start_sec * live.SAMPLE_RATE),
            end_sample=int((start_sec + dur_sec) * live.SAMPLE_RATE),
            pcm=np.full(n * 2, int(amp * 32767), dtype=np.int16).tobytes(),
            reason="hangover", peak_rms=amp,
        )

    clock = _Clock()
    server = TurnCoalescer(_cfg(), clock=clock)
    client = live.TurnCoalescer(_cfg(), clock=clock)

    def drive(c, submit, poll, flush):
        seq = []
        seq += [x.reason for x in submit(c, t(0, 2.0))]
        clock.advance(0.3)
        seq += [x.reason for x in submit(c, t(2.3, 1.0))]
        clock.advance(0.2)
        seq += [x.reason for x in poll(c)]
        clock.advance(1.0)
        seq += [x.reason for x in poll(c)]
        seq += [x.reason for x in flush(c)]
        return seq

    s = drive(server, TurnCoalescer.submit, TurnCoalescer.poll, TurnCoalescer.flush)
    c = drive(client, live.TurnCoalescer.submit, live.TurnCoalescer.poll,
              live.TurnCoalescer.flush)
    # submit x2 are both held (second merges), the first poll is early, the
    # second releases the merged turn, flush has nothing left.
    assert s == ["merged"]
    assert c == s
    assert server.stats()["merged"] == client.stats()["merged"] == 1


def test_client_coalescer_defaults_match_the_server(live):
    from asr_mcp.streaming.turn_detector import config as server_config

    server = server_config()
    for key in ("merge_gap_sec", "max_merge_sec"):
        assert server[key] == live.DETECTOR_DEFAULTS[key]
        assert live.TurnCoalescer().cfg[key] == live.DETECTOR_DEFAULTS[key]
