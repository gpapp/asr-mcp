"""Unit tests for the one boundary engine (``asr_mcp/speaker/boundary.py``).

Before the engine there were three gap-cut implementations that disagreed with
each other on the same gap, ``thresholds.json -> boundary_refine`` was read by
nothing, and ``split_at_energy_dips`` deleted real audio (P0/P1 of
``docs/plans/boundary-and-voice-quality-plan.md``).

The engine is numpy-only (no torch at import time) so this file imports it
directly and runs without the ML stack.  The frozen ``_legacy_*`` functions are
copies of the pre-P1 code, verbatim in arithmetic: they define "today's numbers"
and are the regression lock.

Two configuration flags, easy to confuse:

* ``boundary_refine.acoustic: false`` is the P0 **A/B arm** - methods 1 and 2
  off, energy-only.  It is deliberately *not* today's behaviour, because today's
  behaviour already uses method 1.
* ``boundary_refine.spectral_novelty: false`` (``acoustic`` still true) is the
  pre-P1 chain, and that is what reproduces today's numbers exactly.

Nothing here claims the spectral method is better on real recordings.  The one
test that claims superiority does so on a synthetic pure-tone signal with a known
change point; the real-recording judgement is the P0 A/B measurement, which has
not been run.
"""

import sys
import types

import numpy as np
import pytest

from asr_mcp.speaker import boundary as bd

SR = 16000


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _tone(buf, start, end, freq, amp):
    i0, i1 = int(start * SR), int(end * SR)
    t = np.arange(i1 - i0) / SR
    buf[i0:i1] = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(total=8.0):
    return np.zeros(int(total * SR), dtype=np.float32)


def _unit(vec):
    arr = np.asarray(vec, dtype=np.float64).ravel()
    return arr / np.linalg.norm(arr)


def _lookup_embedder(sections, vectors):
    """A ``refs['embed_sections']`` implementation backed by a lookup table.

    The pipeline batches every candidate section of a boundary into one ONNX call
    and then looks the vectors up by ``(start, end)``; the router embeds them one
    at a time.  Both are exercised in ``test_former_callers_agree``.
    """
    table = {
        (float(s["start"]), float(s["end"])): np.asarray(v, dtype=np.float64)
        for s, v in zip(sections, vectors)
    }

    def _embed(selected):
        return [table.get((float(s["start"]), float(s["end"]))) for s in selected]

    return _embed


def _single_embedder(sections, vectors):
    """A ``refs['embed_sections']`` implementation that walks the list one by one."""
    table = dict(zip(
        [(float(s["start"]), float(s["end"])) for s in sections],
        [np.asarray(v, dtype=np.float64) for v in vectors],
    ))

    def _embed(selected):
        out = []
        for sec in selected:
            key = (float(sec["start"]), float(sec["end"]))
            out.append(table.get(key))
        return out

    return _embed


# --------------------------------------------------------------------------- #
# frozen pre-P1 implementations - the regression lock                         #
# --------------------------------------------------------------------------- #


def _legacy_gap_boundary(audio, gap_start_sec, gap_end_sec, sample_rate=SR,
                         frame_ms=20.0, dip_ratio=0.35, min_dip_sec=0.12):
    """Verbatim copy of the pre-P1 ``asr_router._gap_boundary``."""
    mid = (gap_start_sec + gap_end_sec) / 2.0
    if audio is None or len(audio) == 0:
        return mid
    s = max(0, int(gap_start_sec * sample_rate))
    e = min(len(audio), int(gap_end_sec * sample_rate))
    frame_len = max(1, int(frame_ms / 1000 * sample_rate))
    if e - s < 2 * frame_len:
        return mid
    chunk = audio[s:e].astype(np.float32)
    energies = [
        float(np.sqrt(np.mean(chunk[i:i + frame_len] ** 2)))
        for i in range(0, len(chunk) - frame_len + 1, frame_len)
    ]
    if not energies:
        return mid
    max_e = max(energies)
    if max_e < 1e-8:
        return mid
    thresh = max_e * dip_ratio
    runs = []
    i = 0
    while i < len(energies):
        if energies[i] < thresh:
            j = i
            while j < len(energies) and energies[j] < thresh:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    min_frames = max(1, int(round(min_dip_sec * 1000 / frame_ms)))
    eligible = [r for r in runs if r[1] - r[0] >= min_frames]
    if eligible:
        centre_idx = len(energies) / 2.0
        best = max(
            eligible,
            key=lambda r: (r[1] - r[0], -abs((r[0] + r[1]) / 2.0 - centre_idx)),
        )
        cut_frame = (best[0] + best[1]) // 2
    else:
        cut_frame = min(range(len(energies)), key=lambda k: energies[k])
    cut_sample = s + cut_frame * frame_len + frame_len // 2
    return min(max(cut_sample / sample_rate, gap_start_sec), gap_end_sec)


def _legacy_best_split(owners, weights):
    """Verbatim copy of the pre-P1 ``asr_router._best_split``."""
    n = len(owners)
    if n == 0:
        return 0
    best_k, best_cost = 0, float("inf")
    for k in range(n + 1):
        cost = 0.0
        for i in range(k):
            if owners[i] == 1:
                cost += weights[i]
        for i in range(k, n):
            if owners[i] == 0:
                cost += weights[i]
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_k = k
    return best_k


def _legacy_pipeline_ownership_cut(sections, vectors, left_ref, right_ref,
                                   gap_start, gap_end, left_start, right_end):
    """Verbatim copy of the pre-P1 step-13 cut (pipeline.py, pre-P1 lines 727-764)."""
    owners, weights, valid = [], [], []
    for s, emb in zip(sections, vectors):
        if emb is None:
            continue
        sa = float(np.dot(emb, left_ref))
        sb = float(np.dot(emb, right_ref))
        owners.append(0 if sa >= sb else 1)
        weights.append(abs(sa - sb) + 1e-3)
        valid.append(s)
    if not valid:
        return None
    n = len(owners)
    best_k, best_cost = 0, float("inf")
    for k in range(n + 1):
        cost = 0.0
        for j in range(k):
            if owners[j] == 1:
                cost += weights[j]
        for j in range(k, n):
            if owners[j] == 0:
                cost += weights[j]
        if cost < best_cost:
            best_cost = cost
            best_k = k
    cut = float(valid[best_k]["start"]) if best_k < n else gap_end
    cut = max(float(left_start) + 0.001, min(cut, float(right_end) - 0.001))
    if gap_end > gap_start:
        cut = max(gap_start, min(cut, gap_end))
    return cut


def _legacy_step7_cut(candidates, left_spk, right_spk, search_start, search_end):
    """Verbatim copy of the pre-P1 ``speaker/audio.refine_speaker_boundaries`` tail."""
    last_left_t = search_start
    first_right_t = search_end
    for center_t, spk in candidates:
        if spk == left_spk:
            last_left_t = center_t
    for center_t, spk in candidates:
        if spk == right_spk and center_t > last_left_t:
            first_right_t = center_t
            break
    return round((last_left_t + first_right_t) / 2.0, 4)


# --------------------------------------------------------------------------- #
# P0.1 - the dead config is now read                                         #
# --------------------------------------------------------------------------- #


def test_boundary_refine_config_is_actually_read():
    """P0 step 1: the section must reach the engine, not just the JSON file."""
    from asr_mcp.config import get_config

    section = get_config()["boundary_refine"]
    conf = bd._conf()
    for key in ("enabled", "acoustic", "spectral_novelty", "lookaround_sec",
                "min_section_sec", "dip_ratio", "min_dip_sec", "min_gap_sec"):
        assert key in section, "missing from thresholds.json: %s" % key
        assert conf[key] == section[key], "config key %s is not read" % key
    # The values that used to be hardcoded signature defaults.
    assert conf["dip_ratio"] == 0.35
    assert conf["min_dip_sec"] == 0.12


def test_boundary_refine_file_values_win_over_module_defaults():
    """Mutating the loaded config must move the engine, i.e. the file is the source.

    Asserting ``conf[key] == json[key]`` is not enough on its own: the module
    DEFAULTS happen to mirror the file, so it would pass even if the file were
    never read.  This test mutates the live config instead.
    """
    from asr_mcp.config import get_config

    section = get_config()["boundary_refine"]
    for key, value in (("dip_ratio", 0.77), ("lookaround_sec", 2.5),
                       ("spectral_novelty", False)):
        original = section[key]
        section[key] = value
        try:
            assert bd._conf()[key] == value
        finally:
            section[key] = original
    assert bd._conf()["dip_ratio"] == 0.35


def test_per_call_cfg_overrides_the_file():
    conf = bd._conf({"lookaround_sec": 1.2, "section_cut_rule": "window_midpoint"})
    assert conf["lookaround_sec"] == 1.2
    assert conf["section_cut_rule"] == "window_midpoint"
    assert conf["dip_ratio"] == 0.35  # untouched keys still come from the file


# --------------------------------------------------------------------------- #
# P0.2 - every method is reachable and is selected when it should be          #
# --------------------------------------------------------------------------- #


def test_method_vad_embedding_is_selected():
    audio = _silence()
    sections = [{"start": 2.05, "end": 2.40}, {"start": 2.60, "end": 2.95}]
    vectors = [_unit([1.0, 0.0]), _unit([0.0, 1.0])]
    refs = {
        "sections": sections,
        "embed_sections": _lookup_embedder(sections, vectors),
        "left": _unit([1.0, 0.0]),
        "right": _unit([0.0, 1.0]),
        "left_start": 1.0,
        "right_end": 4.0,
        "sample_rate": SR,
    }
    cut, method = bd.cut_between(2.0, 3.0, refs, audio)
    assert method == "vad_embedding"
    assert cut == pytest.approx(2.60)  # start of the first right-owned section


def test_method_vad_embedding_all_left_owned_keeps_gap_end():
    """The preserved rule: an entirely left-owned gap puts the cut at gap_end."""
    audio = _silence()
    sections = [{"start": 2.05, "end": 2.40}, {"start": 2.60, "end": 2.95}]
    vectors = [_unit([1.0, 0.0]), _unit([0.99, 0.14])]
    refs = {
        "sections": sections,
        "embed_sections": _lookup_embedder(sections, vectors),
        "left": _unit([1.0, 0.0]),
        "right": _unit([0.0, 1.0]),
        "left_start": 1.0,
        "right_end": 4.0,
        "sample_rate": SR,
    }
    cut, method = bd.cut_between(2.0, 3.0, refs, audio)
    assert method == "vad_embedding"
    assert cut == pytest.approx(3.0)


def test_ownership_evidence_requires_both_references():
    """Sections alone are not evidence; without both references method 1 declines."""
    audio = _silence()
    sections = [{"start": 2.05, "end": 2.40}, {"start": 2.60, "end": 2.95}]
    vectors = [_unit([1.0, 0.0]), _unit([0.0, 1.0])]
    for missing in ("left", "right"):
        refs = {
            "sections": sections,
            "embed_sections": _lookup_embedder(sections, vectors),
            "left": _unit([1.0, 0.0]),
            "right": _unit([0.0, 1.0]),
            "sample_rate": SR,
        }
        refs[missing] = None
        _, method = bd.cut_between(2.0, 3.0, refs, audio,
                                   {"spectral_novelty": False})
        assert method != "vad_embedding", missing


def test_acoustic_false_skips_the_ownership_method():
    """The A/B arm must not use method 1 either - it is the energy-only arm."""
    audio = _silence()
    sections = [{"start": 2.05, "end": 2.40}, {"start": 2.60, "end": 2.95}]
    vectors = [_unit([1.0, 0.0]), _unit([0.0, 1.0])]
    refs = {
        "sections": sections,
        "embed_sections": _lookup_embedder(sections, vectors),
        "left": _unit([1.0, 0.0]),
        "right": _unit([0.0, 1.0]),
        "sample_rate": SR,
    }
    cut, method = bd.cut_between(2.0, 3.0, refs, audio, {"acoustic": False})
    assert method != "vad_embedding"
    assert cut == pytest.approx(
        _legacy_gap_boundary(audio, 2.0, 3.0, SR), abs=1e-9)


def test_method_energy_dip_is_selected():
    audio = _silence()
    _tone(audio, 1.0, 2.40, 300, 0.2)
    _tone(audio, 2.95, 4.0, 300, 0.2)
    cut, method = bd.cut_between(2.0, 3.0, {}, audio, {"spectral_novelty": False})
    assert method == "energy_dip"
    assert cut == pytest.approx(2.67, abs=0.02)


def test_method_quietest_frame_is_selected():
    """Continuous speech: no dip run long enough, so the quietest frame decides."""
    audio = _silence()
    _tone(audio, 1.0, 3.0, 300, 0.2)
    cut, method = bd.cut_between(2.0, 3.0, {}, audio, {"spectral_novelty": False})
    assert method == "quietest_frame"
    assert 2.0 <= cut <= 3.0


def test_method_midpoint_is_selected_without_audio():
    cut, method = bd.cut_between(2.0, 3.0, {}, None)
    assert (method, cut) == ("midpoint", 2.5)


def test_method_midpoint_is_selected_when_disabled():
    audio = _silence()
    _tone(audio, 1.0, 2.40, 300, 0.2)
    _tone(audio, 2.95, 4.0, 300, 0.2)
    cut, method = bd.cut_between(2.0, 3.0, {}, audio, {"enabled": False})
    assert (method, cut) == ("midpoint", 2.5)


def test_method_spectral_change_is_selected():
    """Clean spectral step inside the gap, constant amplitude (no energy dip)."""
    audio = _silence()
    _tone(audio, 0.5, 2.35, 400, 0.2)
    _tone(audio, 2.35, 4.0, 1300, 0.2)
    cut, method = bd.cut_between(2.0, 3.0, {}, audio)
    assert method == "spectral_change"
    assert cut == pytest.approx(2.35, abs=0.02)


def test_every_documented_method_is_reachable():
    seen = set()

    dip = _silence()
    _tone(dip, 1.0, 2.40, 300, 0.2)
    _tone(dip, 2.95, 4.0, 300, 0.2)
    seen.add(bd.cut_between(2.0, 3.0, {}, dip, {"spectral_novelty": False})[1])

    cont = _silence()
    _tone(cont, 1.0, 3.0, 300, 0.2)
    seen.add(bd.cut_between(2.0, 3.0, {}, cont, {"spectral_novelty": False})[1])

    seen.add(bd.cut_between(2.0, 3.0, {}, None)[1])

    sections = [{"start": 2.05, "end": 2.40}, {"start": 2.60, "end": 2.95}]
    refs = {
        "sections": sections,
        "embed_sections": _lookup_embedder(
            sections, [_unit([1.0, 0.0]), _unit([0.0, 1.0])]),
        "left": _unit([1.0, 0.0]),
        "right": _unit([0.0, 1.0]),
    }
    seen.add(bd.cut_between(2.0, 3.0, refs, dip)[1])

    spec = _silence()
    _tone(spec, 0.5, 2.35, 400, 0.2)
    _tone(spec, 2.35, 4.0, 1300, 0.2)
    seen.add(bd.cut_between(2.0, 3.0, {}, spec)[1])

    assert seen == set(bd.METHODS)


# --------------------------------------------------------------------------- #
# P0.4 / P1 step 6 - the regression lock against today's numbers               #
# --------------------------------------------------------------------------- #


def _energy_gap_signals():
    """A spread of gaps: deep dip, shallow dip, continuous speech, noise, silence."""
    cases = []

    deep = _silence()
    _tone(deep, 1.0, 2.40, 300, 0.2)
    _tone(deep, 2.95, 4.0, 300, 0.2)
    cases.append((deep, 2.0, 3.0))

    shallow = _silence()
    _tone(shallow, 1.0, 2.10, 300, 0.2)
    _tone(shallow, 2.90, 4.0, 300, 0.2)
    cases.append((shallow, 2.0, 3.0))

    continuous = _silence()
    _tone(continuous, 1.0, 3.5, 300, 0.2)
    cases.append((continuous, 2.0, 3.0))

    two_dips = _silence(9.0)
    _tone(two_dips, 1.0, 2.20, 300, 0.2)
    _tone(two_dips, 2.50, 2.70, 300, 0.2)
    _tone(two_dips, 3.00, 4.0, 300, 0.2)
    cases.append((two_dips, 2.0, 3.0))

    asymmetric = _silence()
    # Two eligible dips in [2.0, 3.0]: 8 frames at 2.04-2.20 (far from the gap
    # centre) and 7 frames at 2.44-2.58 (at the centre).  The LONGEST one wins,
    # so a tie-break that preferred "nearest the centre" would give a different
    # answer.
    for s0, s1 in ((1.0, 2.04), (2.20, 2.44), (2.58, 4.0)):
        _tone(asymmetric, s0, s1, 300, 0.2)
    cases.append((asymmetric, 2.0, 3.0))

    cases.append((_silence(), 2.0, 3.0))

    rng = np.random.default_rng(7)
    cases.append(((rng.standard_normal(8 * SR) * 0.05).astype(np.float32), 2.0, 3.0))
    return cases


def test_acoustic_false_reproduces_legacy_gap_boundary():
    """The A/B 'off' arm is bit-identical to the pre-P1 energy arithmetic."""
    for audio, gs, ge in _energy_gap_signals():
        cut, method = bd.cut_between(gs, ge, {}, audio, {"acoustic": False})
        expected = _legacy_gap_boundary(audio, gs, ge, SR)
        assert cut == pytest.approx(expected, abs=1e-9), (gs, ge)
        assert method in ("energy_dip", "quietest_frame", "midpoint")


def test_spectral_novelty_false_reproduces_legacy_gap_boundary():
    """The pre-P1 chain (method 1 still on, method 2 off) == today's numbers."""
    for audio, gs, ge in _energy_gap_signals():
        cut, _ = bd.cut_between(gs, ge, {}, audio, {"spectral_novelty": False})
        assert cut == pytest.approx(
            _legacy_gap_boundary(audio, gs, ge, SR), abs=1e-9), (gs, ge)


def test_energy_chain_helpers_match_legacy_bitwise():
    for audio, gs, ge in _energy_gap_signals():
        dip = bd.energy_dip_cut(audio, gs, ge, SR, dip_ratio=0.35, min_dip_sec=0.12)
        quiet = bd.quietest_frame_cut(audio, gs, ge, SR)
        legacy = _legacy_gap_boundary(audio, gs, ge, SR)
        chosen = dip if dip is not None else (quiet if quiet is not None
                                              else (gs + ge) / 2.0)
        assert chosen == pytest.approx(legacy, abs=1e-9)


def test_best_split_matches_legacy():
    import random

    rng = random.Random(11)
    for _ in range(200):
        n = rng.randint(1, 7)
        owners = [rng.randint(0, 1) for _ in range(n)]
        weights = [rng.random() + 1e-3 for _ in range(n)]
        assert bd.best_split(owners, weights) == _legacy_best_split(owners, weights)
    # An exact cost tie must resolve to the LOWEST k (the pre-P1 behaviour);
    # `<=` instead of `<` would take the highest.
    # costs are [2, 1, 2, 1, 2] for k = 0..4, so k = 1 and k = 3 tie.
    tie_owners = [0, 1, 0, 1]
    tie_weights = [1.0, 1.0, 1.0, 1.0]
    assert bd.best_split(tie_owners, tie_weights) == 1
    assert _legacy_best_split(tie_owners, tie_weights) == 1


# --------------------------------------------------------------------------- #
# bounds: shift inside the lookaround window, cut inside the gap                #
# --------------------------------------------------------------------------- #


def _ragged_timeline():
    """Segments with a mix of abutting boundaries and gaps of several widths."""
    audio = _silence(40.0)
    rng = np.random.default_rng(3)
    segments = []
    t = 0.4
    widths = [1.0, 0.0, 0.3, 2.0, 0.0, 0.05, 4.0, 1.5]
    i = 0
    while t < 38.0:
        dur = widths[i % len(widths)]
        speaker = "Speaker 1" if (i // 2) % 2 == 0 else "Speaker 2"
        start, end = t, t + dur
        # speech-like noise inside the span, near-silence in the gaps
        n0, n1 = int(start * SR), int(end * SR)
        audio[n0:n1] += (rng.standard_normal(n1 - n0) * 0.05).astype(np.float32)
        segments.append({"start": start, "end": end, "speaker": speaker})
        t = end + (0.0 if i % 3 == 0 else 0.05 * (i % 4))
        i += 1
    return audio, segments


def test_cut_is_always_inside_the_gap():
    audio, segments = _ragged_timeline()
    rng = np.random.default_rng(5)
    for _ in range(120):
        gs = float(rng.uniform(0.5, 36.0))
        ge = gs + float(rng.uniform(0.002, 1.2))
        cut, _method = bd.cut_between(
            gs, ge,
            {"left_start": max(0.0, gs - 3.0), "right_end": ge + 3.0,
             "sample_rate": SR},
            audio,
        )
        assert gs - 1e-9 <= cut <= ge + 1e-9, (gs, ge, cut)


def test_shift_never_leaves_the_lookaround_window():
    """For a gap-less (abutting) boundary the cut stays inside +-lookaround."""
    audio, _ = _ragged_timeline()
    lookaround = float(bd._conf()["lookaround_sec"])
    rng = np.random.default_rng(6)
    for _ in range(120):
        centre = float(rng.uniform(1.0, 35.0))
        refs = {"left_start": centre - 4.0, "right_end": centre + 4.0,
                "sample_rate": SR}
        cut, _ = bd.cut_between(centre, centre, refs, audio)
        assert centre - lookaround - 1e-6 <= cut <= centre + lookaround + 1e-6


def test_cut_from_an_outside_section_is_clamped_into_the_gap():
    """A section inside the lookaround but before the gap must not push the cut out."""
    audio = _silence()
    sections = [{"start": 1.60, "end": 1.95}, {"start": 1.95, "end": 2.40}]
    vectors = [_unit([1.0, 0.0]), _unit([0.0, 1.0])]
    refs = {
        "sections": sections,
        "embed_sections": _lookup_embedder(sections, vectors),
        "left": _unit([1.0, 0.0]),
        "right": _unit([0.0, 1.0]),
        "left_start": 1.0,
        "right_end": 4.0,
        "sample_rate": SR,
    }
    cut, method = bd.cut_between(2.0, 3.0, refs, audio)
    assert method == "vad_embedding"
    # The raw answer is valid[1]["start"] == 1.95, which is BEFORE the gap.
    assert cut == pytest.approx(2.0)


def test_cut_is_rounded_to_four_decimals():
    """The pre-P1 pipeline rounded; the router did not. The engine rounds."""
    audio = _silence()
    _tone(audio, 1.0, 2.40, 300, 0.2)
    _tone(audio, 2.95, 4.0, 300, 0.2)
    # A gap start that is not a multiple of the 20 ms frame grid, so the raw
    # sample-quantised answer carries more than four decimals.
    cut, _ = bd.cut_between(2.0001, 3.0001, {}, audio, {"spectral_novelty": False})
    assert cut == round(cut, 4)
    assert len(str(cut).split(".")[1]) <= 4


def test_min_gap_sec_is_live_configuration():
    """Raising ``min_gap_sec`` reclassifies a gap as an abutting boundary.

    Abutting boundaries search the whole lookaround window and are not clamped
    back into the gap, so the cut is allowed to sit outside it.  That difference
    is what proves the key is read rather than hardcoded.
    """
    audio = _silence()
    # One deep dip at 2.1-2.4 s, which sits OUTSIDE the gap [2.5, 3.5] but INSIDE
    # the +/-0.5 s lookaround window [2.0, 4.0].
    for s0, s1 in ((1.0, 2.1), (2.4, 6.0)):
        _tone(audio, s0, s1, 300, 0.2)
    inside, method = bd.cut_between(2.5, 3.5, {"sample_rate": SR}, audio,
                                    {"spectral_novelty": False})
    assert method == "quietest_frame"
    assert 2.5 - 1e-9 <= inside <= 3.5 + 1e-9
    wide, _ = bd.cut_between(2.5, 3.5, {"sample_rate": SR}, audio,
                             {"spectral_novelty": False, "min_gap_sec": 2.0})
    assert wide < 2.5 - 1e-9


def test_cut_never_eats_a_side_completely():
    audio = _silence()
    cut, _ = bd.cut_between(
        5.0, 5.0, {"left_start": 5.0, "right_end": 5.4, "sample_rate": SR}, audio)
    assert 5.0 + 0.001 - 1e-9 <= cut <= 5.4 - 0.001 + 1e-9


# --------------------------------------------------------------------------- #
# lesson 17 - every second of the timeline is covered by exactly one turn      #
# --------------------------------------------------------------------------- #


def _asr_router():
    """``asr_mcp.api.asr_router`` without the ML stack.

    ``asr_mcp/api/__init__.py`` pulls in the speaker router (torchaudio), so the
    package is stubbed the same way ``conftest.load_module`` stubs packages.
    """
    if "asr_mcp.api" not in sys.modules:
        pkg = types.ModuleType("asr_mcp.api")
        pkg.__path__ = ["asr_mcp/api"]
        sys.modules["asr_mcp.api"] = pkg
    import asr_mcp.api.asr_router as mod
    return mod


def test_prepare_turns_covers_every_second_exactly_once():
    ar = _asr_router()
    audio, segments = _ragged_timeline()
    turns = ar._prepare_turns(segments, audio_duration_sec=40.0,
                              audio=audio, sample_rate=SR)
    assert len(turns) >= 2
    assert turns[0]["start"] == pytest.approx(0.0, abs=1e-6)
    assert turns[-1]["end"] == pytest.approx(40.0, abs=1e-6)
    for left, right in zip(turns, turns[1:]):
        # Abutting turns are the DESIGNED invariant, not a bug.
        assert right["start"] == pytest.approx(left["end"], abs=1e-6)
        assert right["start"] >= left["end"] - 1e-9
    # Total covered duration == audio duration, i.e. no second lost or doubled.
    total = sum(t["end"] - t["start"] for t in turns)
    assert total == pytest.approx(40.0, abs=1e-6)


def test_prepare_turns_covers_every_second_when_refinement_is_disabled():
    ar = _asr_router()
    audio, segments = _ragged_timeline()
    from asr_mcp import config

    original = config.get_config()["boundary_refine"]["enabled"]
    config.get_config()["boundary_refine"]["enabled"] = False
    try:
        turns = ar._prepare_turns(segments, audio_duration_sec=40.0,
                                  audio=audio, sample_rate=SR)
    finally:
        config.get_config()["boundary_refine"]["enabled"] = original
    total = sum(t["end"] - t["start"] for t in turns)
    assert total == pytest.approx(40.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# nothing is deleted                                                           #
# --------------------------------------------------------------------------- #


def test_refine_gap_deletes_nothing():
    audio = _silence()
    left = {"start": 4.98, "end": 5.00, "speaker": "Speaker 1"}
    right = {"start": 5.02, "end": 5.06, "speaker": "Speaker 2"}
    cut, method = bd.refine_gap(
        left, right, audio,
        {"min_side_sec": 0.3, "sample_rate": SR},
    )
    assert left["end"] == right["start"] == cut
    assert right["end"] - right["start"] > 0.0
    assert left["start"] < left["end"] < right["end"]


def test_refine_gap_refuses_a_move_that_would_shrink_a_side():
    """Ownership says 2.05 s, which would leave the right span 0.15 s long.

    The move is refused and both spans keep their nominal extent.  The pre-P1
    step-7 code applied nothing here but then DELETED every span under 0.3 s.
    """
    sections = [{"start": 2.10, "end": 2.45}]
    left = {"start": 1.00, "end": 2.00, "speaker": "Speaker 1"}
    right = {"start": 2.05, "end": 2.60, "speaker": "Speaker 2"}
    cut, method = bd.refine_gap(
        left, right, _silence(),
        {
            "sections": sections,
            "embed_sections": _lookup_embedder(sections, [_unit([0.0, 1.0])]),
            "left": _unit([1.0, 0.0]),
            "right": _unit([0.0, 1.0]),
            "min_side_sec": 0.7,
            "sample_rate": SR,
        },
    )
    # Ownership wants 2.10, which would leave the right span 0.50 s long.
    # Refusing means keeping the nominal boundary (the left span's current end).
    assert method == "midpoint"
    assert cut == pytest.approx(2.00, abs=1e-6)
    assert right["end"] - right["start"] > 0.0


def test_refinement_passes_never_drop_a_segment():
    """Boundary refinement returns exactly as many turns as it was handed.

    The pre-P1 ``speaker/audio.refine_speaker_boundaries`` DELETED every segment
    shorter than 0.3 s, so this count was not preserved anywhere.
    """
    ar = _asr_router()
    audio, segments = _ragged_timeline()
    merged = ar._merge_into_turns(segments)
    pre_refinement = []
    for turn in merged:
        pre_refinement.extend(ar._split_long_turn(turn))
    turns = ar._prepare_turns(segments, audio_duration_sec=40.0,
                              audio=audio, sample_rate=SR)
    assert len(turns) == len(pre_refinement)
    assert all(t["end"] - t["start"] > 0.0 for t in turns)


# --------------------------------------------------------------------------- #
# P1 step 7 - one VAD pass, not two                                           #
# --------------------------------------------------------------------------- #


def _fake_embeddings(monkeypatch, ar):
    """Give the router working speaker references without the ML stack.

    ``_speaker_refs`` and ``_embed_section`` both need torch; stubbing them lets
    the router's VAD-section branch actually run, so the section handoff is
    exercised instead of short-circuiting on an ImportError.
    """
    refs = {"Speaker 1": _unit([1.0, 0.0]), "Speaker 2": _unit([0.0, 1.0])}
    monkeypatch.setattr(ar, "_speaker_refs",
                        lambda *a, **k: dict(refs))
    # Every section is labelled as the right-hand speaker of Speaker 1 <-> 2.
    monkeypatch.setattr(ar, "_embed_section",
                        lambda audio, start, end, sr: _unit([0.0, 1.0]))
    return refs


def test_router_refinement_uses_the_supplied_sections(monkeypatch):
    """The sections the router hands over actually decide a boundary."""
    ar = _asr_router()
    monkeypatch.setattr(ar, "_raw_vad_sections",
                        lambda *a, **k: pytest.fail("second VAD pass"))
    _fake_embeddings(monkeypatch, ar)
    audio = _silence(8.0)
    turns = [
        {"start": 1.00, "end": 2.00, "speaker": "Speaker 1"},
        {"start": 2.60, "end": 4.00, "speaker": "Speaker 2"},
    ]
    ar._refine_boundaries_with_vad(
        turns, audio, SR, None,
        sections=[{"start": 2.20, "end": 2.55}],  # inside the gap, >= min_section_sec
    )
    # The supplied section is right-owned, so the cut is its start, clamped
    # into the gap. The energy-chain fallback would have answered 2.30.
    assert turns[0]["end"] == pytest.approx(2.20)
    assert turns[1]["start"] == pytest.approx(2.20)


def test_router_refinement_reuses_the_diarizers_sections(monkeypatch):
    """``_prepare_turns`` must NOT trigger a second full-file Silero pass."""
    ar = _asr_router()

    def _boom(*args, **kwargs):
        raise AssertionError("a second full-file VAD pass was requested")

    monkeypatch.setattr(ar, "_raw_vad_sections", _boom)
    _fake_embeddings(monkeypatch, ar)
    audio = _silence(8.0)
    segments = [
        {"start": 1.00, "end": 2.00, "speaker": "Speaker 1"},
        {"start": 2.60, "end": 4.00, "speaker": "Speaker 2"},
    ]
    turns = ar._prepare_turns(segments, audio_duration_sec=8.0, audio=audio,
                              sample_rate=SR,
                              raw_vad_sections=[{"start": 2.20, "end": 2.55}])
    assert len(turns) == 2
    # The supplied section decided this boundary (the midpoint would be 2.30), so
    # it was actually used -- and it is >= min_section_sec, unlike a 0.2 s stub.
    assert turns[0]["end"] == pytest.approx(2.20)
    assert turns[1]["start"] == pytest.approx(2.20)
    # ... and the timeline is still fully covered (lesson 17).
    assert sum(t["end"] - t["start"] for t in turns) == pytest.approx(8.0, abs=1e-6)


def test_router_refinement_falls_back_to_its_own_pass_when_absent(monkeypatch):
    """``/attribution`` runs no diarizer, so the router must still produce sections.

    The fallback is the same model over the same audio, so the section list is
    identical; only the number of passes differs.  Here the fallback is stubbed to
    return a known list, which is what the equality below relies on.
    """
    ar = _asr_router()
    calls = []

    def _fake(audio, sample_rate):
        calls.append(sample_rate)
        return [{"start": 1.0, "end": 1.5}, {"start": 2.0, "end": 2.6}]

    monkeypatch.setattr(ar, "_raw_vad_sections", _fake)
    audio, segments = _ragged_timeline()
    turns = ar._prepare_turns(segments, audio_duration_sec=40.0, audio=audio,
                              sample_rate=SR)
    assert calls == [SR]
    assert turns
    assert sum(t["end"] - t["start"] for t in turns) == pytest.approx(40.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# the three former callers now agree with the engine                           #
# --------------------------------------------------------------------------- #


def test_engine_ownership_cut_matches_pre_p1_pipeline_arithmetic():
    """Step 13 on a gap whose sections all sit inside the gap.

    The pre-P1 pipeline and the pre-P1 router used different lookarounds and
    different minimum section lengths and therefore disagreed; on this gap their
    arithmetic is identical, and the engine must reproduce it.
    """
    sections = [
        {"start": 2.00, "end": 2.31},
        {"start": 2.32, "end": 2.63},
        {"start": 2.64, "end": 2.95},
    ]
    vectors = [_unit([1.0, 0.0]), _unit([0.99, 0.15]), _unit([0.05, 1.0])]
    left_ref, right_ref = _unit([1.0, 0.0]), _unit([0.0, 1.0])
    expected = _legacy_pipeline_ownership_cut(
        sections, vectors, left_ref, right_ref,
        2.0, 3.0, 1.5, 3.5,
    )
    refs = {
        "sections": sections,
        "embed_sections": _lookup_embedder(sections, vectors),
        "left": left_ref,
        "right": right_ref,
        "left_start": 1.5,
        "right_end": 3.5,
        "sample_rate": SR,
    }
    cut, method = bd.cut_between(2.0, 3.0, refs, _silence())
    assert method == "vad_embedding"
    assert cut == pytest.approx(expected, abs=1e-9)


def test_former_callers_agree_on_the_same_gap():
    """Pipeline-style (batched lookup) vs router-style (per-section) embedders."""
    audio = _silence()
    sections = [
        {"start": 2.00, "end": 2.31},
        {"start": 2.32, "end": 2.63},
        {"start": 2.64, "end": 2.95},
    ]
    vectors = [_unit([1.0, 0.0]), _unit([0.99, 0.15]), _unit([0.05, 1.0])]
    common = {
        "sections": sections,
        "left": _unit([1.0, 0.0]),
        "right": _unit([0.0, 1.0]),
        "left_start": 1.5,
        "right_end": 3.5,
        "sample_rate": SR,
    }
    pipeline_refs = dict(common,
                         embed_sections=_lookup_embedder(sections, vectors))
    router_refs = dict(common,
                       embed_sections=_single_embedder(sections, vectors))
    assert (bd.cut_between(2.0, 3.0, pipeline_refs, audio)
            == bd.cut_between(2.0, 3.0, router_refs, audio))


def test_window_midpoint_rule_matches_pre_p1_step7_arithmetic():
    """Step 7's overlapping sub-windows use the centre-midpoint rule."""
    search_start, search_end = 4.3, 6.5
    sub = 0.5
    sections, vectors, candidates = [], [], []
    ref_l, ref_r = _unit([1.0, 0.0]), _unit([0.0, 1.0])
    rng = np.random.default_rng(2)
    centres = np.arange(search_start + sub / 2, search_end - sub / 2, 0.2)
    for i, centre in enumerate(centres):
        owner = 0 if i < 6 else 1
        vec = ref_l + 0.05 * rng.standard_normal(2) if owner == 0 \
            else ref_r + 0.05 * rng.standard_normal(2)
        sections.append({"start": float(centre - sub / 2), "end": float(centre + sub / 2)})
        vectors.append(vec)
        candidates.append((float(centre), "Speaker 1" if owner == 0 else "Speaker 2"))
    expected = _legacy_step7_cut(candidates, "Speaker 1", "Speaker 2",
                                 search_start, search_end)
    cut, method = bd.cut_between(
        5.4, 5.4,
        {
            "sections": sections,
            "embed_sections": _lookup_embedder(sections, vectors),
            "left": ref_l,
            "right": ref_r,
            "left_start": 4.5,
            "right_end": 6.5,
            "sample_rate": SR,
        },
        None,
        {"lookaround_sec": 1.2, "section_cut_rule": "window_midpoint"},
    )
    assert method == "vad_embedding"
    assert cut == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# the spectral method: what can and cannot be claimed                           #
# --------------------------------------------------------------------------- #


def test_spectral_change_beats_midpoint_on_a_synthetic_step():
    """SYNTHETIC ONLY.  Claimed: it finds a clean spectral step that midpoint misses.

    This is a capability demonstration on a signal whose change point is known
    exactly.  It is NOT evidence that the method improves boundaries on real
    recordings - that is the P0 A/B measurement, still to be run.
    """
    truth = 2.35
    audio = _silence()
    _tone(audio, 0.5, truth, 400, 0.2)
    _tone(audio, truth, 4.0, 1300, 0.2)
    cut, method = bd.cut_between(2.0, 3.0, {}, audio)
    assert method == "spectral_change"
    midpoint = (2.0 + 3.0) / 2.0
    assert abs(cut - truth) < abs(midpoint - truth)


def test_spectral_change_declines_on_a_flat_spectral_gap():
    """No discontinuity, no answer: the chain must fall through, not guess."""
    audio = _silence()
    _tone(audio, 0.5, 4.0, 500, 0.2)
    assert bd.spectral_change_cut(audio, 1.5, 3.5, 2.0, 3.0) is None


def test_spectral_change_declines_in_pure_silence():
    assert bd.spectral_change_cut(_silence(), 1.5, 3.5, 2.0, 3.0) is None


def test_spectral_change_does_not_leave_the_gap():
    audio = _silence()
    truth = 2.35
    _tone(audio, 0.5, truth, 400, 0.2)
    _tone(audio, truth, 4.0, 1300, 0.2)
    for gs, ge in ((2.0, 3.0), (2.10, 2.90), (1.6, 3.4)):
        cut, _ = bd.cut_between(gs, ge,
                                {"left_start": gs - 2.0, "right_end": ge + 2.0,
                                 "sample_rate": SR}, audio)
        assert gs - 1e-9 <= cut <= ge + 1e-9


# --------------------------------------------------------------------------- #
# split_at_energy_dips must stop deleting audio (P1 step 8)                     #
# --------------------------------------------------------------------------- #


def _load_vad():
    """``speaker/vad.py`` is importable without torch (annotations only)."""
    from asr_mcp.speaker import vad

    return vad


@pytest.mark.parametrize("segment", [
    # Two dips, both pieces long enough: nothing to attach.
    (0.0, 12.0, ((0.0, 4.3), (4.9, 12.0))),
    # The FIRST piece is shorter than min_split_piece and has no predecessor to
    # attach to, so it must be folded into the next one instead of discarded.
    (2.0, 12.0, ((2.0, 2.7), (3.3, 5.7), (6.3, 12.0))),
    # The LAST piece is shorter than min_split_piece.
    (2.0, 12.0, ((2.0, 7.4), (8.0, 9.9), (10.0, 12.0))),
    # A MIDDLE piece is shorter than min_split_piece: the pre-P1 code dropped it,
    # taking 1.1 s of real speech with it.
    (2.0, 12.0, ((2.0, 3.5), (4.0, 6.5), (7.0, 7.6), (8.1, 12.0))),
])
def test_split_at_energy_dips_no_longer_deletes_audio(segment):
    """A sub-``min_split_piece`` piece is ATTACHED to its neighbour, not dropped."""
    split = _load_vad().split_at_energy_dips
    start, end, spans = segment
    audio = np.zeros(30 * SR, dtype=np.float32)
    for s0, s1 in spans:
        i0, i1 = int(s0 * SR), int(s1 * SR)
        t = np.arange(i1 - i0) / SR
        audio[i0:i1] = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    out = split([{"start": start, "end": end}], audio,
                min_segment_dur=3.0, dip_ratio=0.35, min_dip_dur=0.5,
                min_split_piece=2.0)
    assert out, "expected at least one piece"
    # Nothing dropped: the pieces still tile the input segment exactly.
    assert out[0]["start"] == pytest.approx(start, abs=1e-4)
    assert out[-1]["end"] == pytest.approx(end, abs=1e-4)
    for left, right in zip(out, out[1:]):
        assert right["start"] == pytest.approx(left["end"], abs=1e-4)
    # A short piece is ATTACHED to its neighbour, not emitted on its own: the
    # splitter either refuses to split or returns only usable-length pieces.
    for piece in out:
        assert (piece["end"] - piece["start"]) >= 2.0 - 1e-4, piece


def test_split_at_energy_dips_accepts_a_2d_waveform():
    """``(1, N)`` input is squeezed, as the ONNX VAD hands it over."""
    split = _load_vad().split_at_energy_dips
    audio = np.zeros((1, 30 * SR), dtype=np.float32)
    i0, i1 = int(2.0 * SR), int(7.0 * SR)
    t = np.arange(i1 - i0) / SR
    audio[0, i0:i1] = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    i0, i1 = int(7.6 * SR), int(8.1 * SR)
    t = np.arange(i1 - i0) / SR
    audio[0, i0:i1] = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    out = split([{"start": 2.0, "end": 12.0}], audio,
                min_segment_dur=3.0, dip_ratio=0.35, min_dip_dur=0.5,
                min_split_piece=2.0)
    assert len(out) > 1, "the 0.5 s dip at 7.6-8.1 s should split the segment"
    assert out[0]["start"] == pytest.approx(2.0, abs=1e-4)
    assert out[-1]["end"] == pytest.approx(12.0, abs=1e-4)
    # Same answer as the 1-D equivalent.
    flat = np.repeat(audio, 1, axis=0)[0]
    assert split([{"start": 2.0, "end": 12.0}], flat,
                 min_segment_dur=3.0, dip_ratio=0.35, min_dip_dur=0.5,
                 min_split_piece=2.0) == out


def test_split_at_energy_dips_absolute_floor_splits_uniformly_quiet_audio():
    """What counts as a "pause" must not depend on the segment's absolute level.

    The threshold is ``median(segment energies) * dip_ratio``, i.e. relative to
    the segment's OWN median, so a segment whose median is dragged down by one
    loud burst has almost nothing below its threshold and is never split.
    ``abs_floor_ratio`` anchors the comparison to the peak frame instead and does
    split it.
    """
    split = _load_vad().split_at_energy_dips
    audio = np.zeros(30 * SR, dtype=np.float32)
    rng = np.random.default_rng(4)
    i0, i1 = int(0.5 * SR), int(20.5 * SR)
    audio[i0:i1] = (rng.standard_normal(i1 - i0) * 0.002).astype(np.float32)
    # one 1 s loud burst in the middle -> the segment median stays tiny
    j0, j1 = int(10.0 * SR), int(11.0 * SR)
    t = np.arange(j1 - j0) / SR
    audio[j0:j1] = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    seg = [{"start": 0.5, "end": 20.5}]
    relative_only = split(seg, audio, min_segment_dur=3.0, dip_ratio=0.35,
                          min_dip_dur=0.5, min_split_piece=2.0)
    assert relative_only == [{"start": 0.5, "end": 20.5}]
    with_floor = split(seg, audio, min_segment_dur=3.0, dip_ratio=0.35,
                       min_dip_dur=0.5, min_split_piece=2.0,
                       abs_floor_ratio=0.35)
    assert len(with_floor) > 1
    assert with_floor[0]["start"] == pytest.approx(0.5, abs=1e-4)
    assert with_floor[-1]["end"] == pytest.approx(20.5, abs=1e-4)


def test_speaker_vad_module_imports_without_torch():
    """The energy splitter is testable in a lightweight environment.

    ``speaker/vad.py`` used to ``import torch`` at module level purely for
    annotations, which put it behind the ML stack and left the split policy
    untested outside the container.
    """
    import inspect

    module = _load_vad()
    assert "torch" not in getattr(module, "__dict__", {})
    params = inspect.signature(module.split_at_energy_dips).parameters
    assert "abs_floor_ratio" in params
