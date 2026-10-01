"""The single speaker/turn boundary engine (docs/plans/boundary-and-voice-quality-plan.md, P1).

Before this module there were **three** independent gap-cut implementations that
disagreed with each other on the same gap:

* ``diarization/pipeline.py::_refine_turn_boundaries_exact`` (step 13)  — +/-0.5 s
  lookaround, sections >= 0.1 s, ownership min-cost cut, fallback = centre of the
  single quietest 20 ms frame.
* ``api/asr_router.py::_refine_boundaries_with_vad`` (+ ``_best_split``) — no
  lookaround, sections >= 0.3 s (imposed by ``_embed_section``), fallback =
  ``_gap_boundary`` (longest sub-threshold energy run).
* ``speaker/audio.py::refine_speaker_boundaries`` (step 7) — 0.5 s / 0.2 s sliding
  sub-windows +/-1.2 s, cut = midpoint between the last left-owned and first
  right-owned sub-window *centre*, and segments left shorter than 0.3 s were
  **deleted**.

All three are now thin callers of :func:`cut_between`, so the disagreement is gone
by construction.  This module owns the one fallback chain:

1. ``vad_embedding``     — embed each raw VAD section spanning the boundary and
                          attribute it to the better-matching adjacent speaker
                          reference by cosine; minimum-cost ownership split.
2. ``spectral_change``   — NEW.  Frame-to-frame log-spectral-distance peak inside
                          the gap (signal-based snapping; nothing in this repo did
                          this before).
3. ``energy_dip``        — centre of the longest run of 20 ms frames below
                          ``dip_ratio`` of the gap peak, runs shorter than
                          ``min_dip_sec`` are ineligible, ties -> nearest the gap
                          centre (the old ``_gap_boundary``).
4. ``quietest_frame``    — centre of the single quietest 20 ms frame (the old
                          ``_gap_energy_cut`` / ``_gap_boundary`` tail).
5. ``midpoint``          — arithmetic midpoint of the gap (always the answer when
                          there is no audio at all).

The module is dependency-light on purpose: numpy only, **no torch and no
onnxruntime at import time** so ``tests/conftest.py::load_module`` can load it by
path exactly like ``asr_mcp/speaker/uncertainty.py``.  Every torch-dependent step
is injected by the caller through the ``refs`` mapping:

* ``refs["embed_sections"]`` — ``callable(list[{"start","end"}]) -> list[vector|None]``.
  The engine calls it **once per boundary** with the whole selection so the
  implementation can batch (the pipeline embeds all sections of a boundary in one
  batch of 16 with the fbank MD5 cache; the router embeds them individually,
  which is what it did before).
* ``refs["left"] / refs["right"]`` — unit-norm reference embeddings, or ``None``.
* ``refs["sections"]`` — raw (uncollapsed) VAD sections, or ``None``.
* ``refs["left_start"] / refs["right_end"]`` — the spans the cut is clamped into,
  so a cut can never eat the whole of either side (lesson 17: every second of the
  timeline is covered by exactly one turn).
* ``refs["stats"]`` — an optional :class:`BoundaryStats` to accumulate the
  per-boundary record for the P0 measurement.
* ``refs["sample_rate"]`` — defaults to 16000.

Invariants every caller relies on:

* ``cut`` is rounded to 4 decimals and always lies inside the gap when there is a
  gap wider than ``GAP_EPS``; for an abutting boundary it is clamped into
  ``[left_start + EDGE_EPS, right_end - EDGE_EPS]``.
* **Nothing is ever deleted.**  A side that would end up shorter than the caller's
  minimum simply keeps the nominal boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("asr_mcp.speaker.boundary")

# --- method names -------------------------------------------------------------
#: Emitted verbatim in per-boundary log lines and in the per-run histogram, so a
#: grep across the log is enough to reconstruct which method decided a boundary.
METHOD_VAD = "vad_embedding"
METHOD_SPECTRAL = "spectral_change"
METHOD_DIP = "energy_dip"
METHOD_QUIETEST = "quietest_frame"
METHOD_MIDPOINT = "midpoint"

METHODS = (METHOD_VAD, METHOD_SPECTRAL, METHOD_DIP, METHOD_QUIETEST, METHOD_MIDPOINT)

#: A boundary whose gap is narrower than this is *abutting*, not a gap: there is
#: no silence to look at, so the lookaround window is searched instead of the gap.
#: This is the 1e-3 test the router has always used; it is now real configuration
#: (``boundary_refine.min_gap_sec``).
GAP_EPS = 1e-3

#: A cut may eat at most this much off either side of the boundary.
EDGE_EPS = 0.001

#: Frame grid for every energy / spectral computation, in milliseconds.
FRAME_MS = 20.0

#: Defaults for ``thresholds.json -> boundary_refine``.  The JSON overrides these;
#: a caller may pass a further per-call override mapping as ``cfg``.
DEFAULTS: dict[str, Any] = {
    # Master switch.  When false every boundary is closed at the gap midpoint
    # (method ``midpoint``) and no acoustic work is done at all.
    "enabled": True,
    # ``boundary_refine.acoustic`` is the P0 A/B arm switch: false = energy-only,
    # i.e. skip methods 1 and 2.  See the note in ``tests/test_boundary_engine.py``
    # about the difference between this and ``spectral_novelty``.
    "acoustic": True,
    # NEW in P1.  Disable alone (keeping ``acoustic`` true) to reproduce the
    # pre-P1 numbers exactly — this is the regression lock, ``acoustic: false``
    # is a different, deliberately *worse* arm of the A/B.
    "spectral_novelty": True,
    # A boundary whose gap is narrower than this is treated as *abutting*: the
    # lookaround window is searched instead of the gap.  1e-3 is the test the
    # router has always used; below it, nothing changes.
    "min_gap_sec": GAP_EPS,
    # How far either side of the boundary raw VAD sections are considered.
    # Material only for abutting boundaries, because the cut is clamped back into
    # the gap.
    "lookaround_sec": 0.5,
    # A raw VAD section shorter than this is not embeddable reliably and is not
    # used as ownership evidence.
    "min_section_sec": 0.3,
    # ``energy_dip``: eligible frames are below ``dip_ratio`` x the gap PEAK.
    "dip_ratio": 0.35,
    # ``energy_dip``: a dip run must last at least this long to be eligible.
    "min_dip_sec": 0.12,
    # Per-call override of the section ownership cut rule:
    #   "first_right_start"  — the start of the first right-owned section.  Correct
    #                          for disjoint VAD sections.
    #   "window_midpoint"    — midpoint between the last left-owned and first
    #                          right-owned section *centre*.  Required for the
    #                          overlapping 0.5 s sub-windows of step 7, where the
    #                          start of a window sits before the audio it labels.
    "section_cut_rule": "first_right_start",
    # ``spectral_change`` frame grid and FFT size.
    "spectral_frame_ms": FRAME_MS,
    "spectral_n_fft": 512,
    # The log-spectral-distance peak must beat the median peak of the candidate
    # frames by this factor ...
    "spectral_min_ratio": 1.5,
    # ... and must clear this absolute floor (mean |d log| per bin).  The absolute
    # floor is what stops pure digital noise from producing a confident "peak":
    # in a silent window every frame-to-frame log difference is the same size.
    "spectral_min_abs": 0.15,
}


def _conf(cfg: Optional[dict] = None) -> dict[str, Any]:
    """thresholds.json ``boundary_refine`` over :data:`DEFAULTS`, plus ``cfg``."""
    merged = dict(DEFAULTS)
    try:
        from asr_mcp.config import get_config

        raw = get_config() or {}
        section = raw.get("boundary_refine") or {}
        if isinstance(section, dict):
            for key, value in section.items():
                if not key.startswith("_"):
                    merged[key] = value
    except Exception:  # config unavailable — the defaults are authoritative
        pass
    if cfg:
        for key, value in cfg.items():
            if not key.startswith("_"):
                merged[key] = value
    return merged


class BoundaryStats:
    """Per-run accumulator for the P0 measurement (method histogram + shifts).

    One instance per run; pass it as ``refs["stats"]`` and call
    :meth:`summary` once at the end (INFO level).  Per-boundary lines are DEBUG,
    because a 57-minute file produces hundreds of them.
    """

    def __init__(self) -> None:
        self.methods: dict[str, int] = {}
        self.shifts: list[float] = []

    def record(self, nominal: float, final: float, method: str) -> None:
        self.methods[method] = self.methods.get(method, 0) + 1
        try:
            self.shifts.append(abs(float(final) - float(nominal)))
        except (TypeError, ValueError):
            pass

    def reset(self) -> None:
        self.methods = {}
        self.shifts = []

    def __len__(self) -> int:
        return len(self.shifts)

    def summary(self) -> str:
        if not self.shifts:
            return "boundary_refine: no boundaries decided"
        arr = np.asarray(self.shifts, dtype=np.float64)
        hist = ", ".join("%s=%d" % (m, self.methods.get(m, 0)) for m in METHODS)
        return (
            "boundary_refine: n=%d methods[%s] |shift_sec| p50=%.3f p90=%.3f max=%.3f"
            % (
                arr.size,
                hist,
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 90)),
                float(arr.max()),
            )
        )


# --- ownership (method 1) -----------------------------------------------------


def best_split(owners: list[int], weights: list[float]) -> int:
    """Cut index ``k`` (the boundary sits before section ``k``) minimising cost.

    ``owners[i]`` is 0 for the left speaker and 1 for the right speaker; the cost
    of cutting at ``k`` is the total weight of the right-owned sections left of
    ``k`` plus the left-owned sections right of ``k``.  Weights are the embedding
    margin ``|sa - sb| + 1e-3``, so a confident mistake costs more than an
    ambiguous one.  A section list that is entirely one side collapses to ``k == n``
    (nothing to move the cut for), which the caller turns into "gap_end".
    """
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


def _unit(vec: Any) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return arr / norm


def _ownership_cut(
    window_lo: float,
    window_hi: float,
    gap_end: float,
    refs: dict,
    conf: dict,
) -> Optional[float]:
    """Method 1 — VAD-section ownership.  ``None`` when there is no evidence."""
    sections = refs.get("sections")
    embed = refs.get("embed_sections")
    left_ref = _unit(refs["left"]) if refs.get("left") is not None else None
    right_ref = _unit(refs["right"]) if refs.get("right") is not None else None
    if not sections or embed is None or left_ref is None or right_ref is None:
        return None

    min_dur = float(conf["min_section_sec"])
    selected = [
        s for s in sections
        if s["end"] > window_lo and s["start"] < window_hi
        and (float(s["end"]) - float(s["start"])) >= min_dur
    ]
    if not selected:
        return None
    selected.sort(key=lambda s: float(s["start"]))

    try:
        vectors = list(embed(selected))
    except Exception as exc:  # a failed batch must not lose the boundary
        logger.warning("Boundary section embedding failed: %s", exc)
        return None
    if len(vectors) != len(selected):
        vectors = list(vectors) + [None] * (len(selected) - len(vectors))

    owners: list[int] = []
    weights: list[float] = []
    valid: list[dict] = []
    for section, vec in zip(selected, vectors):
        unit = _unit(vec) if vec is not None else None
        if unit is None:
            continue
        sa = float(np.dot(unit, left_ref))
        sb = float(np.dot(unit, right_ref))
        owners.append(0 if sa >= sb else 1)
        weights.append(abs(sa - sb) + 1e-3)
        valid.append(section)
    if not valid:
        return None

    k = best_split(owners, weights)
    if k < len(valid):
        if str(conf.get("section_cut_rule")) == "window_midpoint":
            # Last left-owned / first right-owned sub-window *centre*.  Step 7 uses
            # overlapping sub-windows, so the window's start is before the audio it
            # labels; snapping there would systematically cut too early.
            last_left = window_lo
            first_right = window_hi
            seen_left = False
            for section, owner in zip(valid, owners):
                centre = (float(section["start"]) + float(section["end"])) / 2.0
                if owner == 0:
                    last_left = centre
                    seen_left = True
            if seen_left:
                for section, owner in zip(valid, owners):
                    if owner != 1:
                        continue
                    centre = (float(section["start"]) + float(section["end"])) / 2.0
                    if centre > last_left:
                        first_right = centre
                        break
            if last_left > window_hi or first_right < window_lo:
                return None
            return (last_left + first_right) / 2.0
        return float(valid[k]["start"])
    # Everything in the window belongs to the left speaker: the cut is the gap end
    # (an abutting boundary then keeps its nominal position).
    return float(gap_end)


# --- energy (methods 3 and 4) -------------------------------------------------


def _frame_energies(chunk: np.ndarray, frame_len: int) -> np.ndarray:
    """RMS of every whole 20 ms frame of ``chunk`` (trailing partial frame dropped).

    Vectorised, but frame-for-frame identical to the old per-frame Python loop:
    the frame count is ``(len - frame_len) // frame_len + 1`` and each row is the
    same contiguous slice the loop took.
    """
    if len(chunk) < frame_len:
        return np.zeros(0, dtype=np.float64)
    n = (len(chunk) - frame_len) // frame_len + 1
    usable = n * frame_len
    rows = chunk[:usable].astype(np.float64).reshape(n, frame_len)
    return np.sqrt(np.mean(rows ** 2, axis=1))


def energy_dip_cut(
    audio: Optional[np.ndarray],
    gap_start: float,
    gap_end: float,
    sample_rate: int = 16000,
    dip_ratio: float = 0.35,
    min_dip_sec: float = 0.12,
    frame_ms: float = FRAME_MS,
) -> Optional[float]:
    """Centre of the longest sub-threshold energy run inside a gap.

    ``None`` when there is nothing to decide (no audio, gap narrower than two
    frames, or digitally silent gap).  This is the arithmetic of the old
    ``asr_router._gap_boundary`` and is reproduced exactly so that
    ``acoustic: false`` / ``spectral_novelty: false`` land on today's numbers.
    """
    if audio is None or len(audio) == 0:
        return None
    start = max(0, int(gap_start * sample_rate))
    end = min(len(audio), int(gap_end * sample_rate))
    frame_len = max(1, int(frame_ms / 1000 * sample_rate))
    if end - start < 2 * frame_len:
        return None
    energies = _frame_energies(np.asarray(audio[start:end]), frame_len)
    if energies.size == 0:
        return None
    peak = float(energies.max())
    if peak < 1e-8:
        return None
    thresh = peak * float(dip_ratio)

    runs: list[tuple[int, int]] = []
    i = 0
    size = energies.size
    while i < size:
        if energies[i] < thresh:
            j = i
            while j < size and energies[j] < thresh:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return None
    min_frames = max(1, int(round(float(min_dip_sec) * 1000 / frame_ms)))
    eligible = [r for r in runs if r[1] - r[0] >= min_frames]
    if not eligible:
        return None
    centre = size / 2.0
    best = max(
        eligible,
        key=lambda r: (r[1] - r[0], -abs((r[0] + r[1]) / 2.0 - centre)),
    )
    cut_frame = (best[0] + best[1]) // 2
    cut_sample = start + cut_frame * frame_len + frame_len // 2
    return float(min(max(cut_sample / sample_rate, gap_start), gap_end))


def quietest_frame_cut(
    audio: Optional[np.ndarray],
    gap_start: float,
    gap_end: float,
    sample_rate: int = 16000,
    frame_ms: float = FRAME_MS,
) -> Optional[float]:
    """Centre of the single quietest frame of the window, or ``None``."""
    if audio is None or len(audio) == 0:
        return None
    start = max(0, int(gap_start * sample_rate))
    end = min(len(audio), int(gap_end * sample_rate))
    frame_len = max(1, int(frame_ms / 1000 * sample_rate))
    if end - start < 2 * frame_len:
        return None
    energies = _frame_energies(np.asarray(audio[start:end]), frame_len)
    if energies.size == 0:
        return None
    if float(energies.max()) < 1e-8:
        # A digitally silent window: every frame ties, so "the quietest" is a coin
        # flip with no evidence behind it.  Declining lets the midpoint decide,
        # which is what the old _gap_boundary did for the same input.
        return None
    # No silence requirement beyond the above: the caller has already failed to
    # find an eligible dip run, so the least-bad frame is the remaining
    # signal-based answer. This is the old _gap_boundary tail, kept identical.
    min_idx = int(np.argmin(energies))
    cut = gap_start + (min_idx + 0.5) * frame_len / sample_rate
    return float(min(max(cut, gap_start), gap_end))


# --- spectral novelty (method 2) ----------------------------------------------


def spectral_change_cut(
    audio: Optional[np.ndarray],
    window_lo: float,
    window_hi: float,
    restrict_lo: float,
    restrict_hi: float,
    sample_rate: int = 16000,
    frame_ms: float = FRAME_MS,
    n_fft: int = 512,
    min_ratio: float = 1.5,
    min_abs: float = 0.15,
) -> Optional[float]:
    """NEW: snap to the frame-to-frame log-spectral-distance peak in a gap.

    A speaker change alters the spectral envelope (and usually the mic distance
    and the room), so inside a pause there is one frame boundary where the
    spectrum changes most.  That is a signal-derived cut; every other method in
    this module is either an energy minimum, a VAD edge or arithmetic.

    ``restrict_lo/restrict_hi`` keep the answer inside the real gap (for an
    abutting boundary they are the whole lookaround window).  ``None`` is returned
    whenever the peak is not convincingly above the rest of the window, so a
    continuous or silent passage degrades to the energy chain rather than to a
    confident-looking guess.

    The thresholds are deliberately conservative and are **not** calibrated on
    real recordings (see the plan's phase ordering — this is the capability the P0
    A/B is meant to measure).  ``boundary_refine.spectral_novelty: false``
    disables the method entirely and restores the pre-P1 chain.
    """
    if audio is None or len(audio) == 0:
        return None
    frame = max(1, int(round(float(frame_ms) / 1000.0 * sample_rate)))
    fft = int(n_fft)
    if fft < frame:
        fft = 2 * frame
    start = max(0, int(window_lo * sample_rate))
    end = min(len(audio), int(window_hi * sample_rate))
    if end - start < 4 * frame:
        return None
    n = (end - start) // frame
    if n < 5:
        return None
    rows = np.asarray(audio[start:start + n * frame], dtype=np.float64).reshape(n, frame)

    window = np.hanning(frame)
    spectrum = np.abs(np.fft.rfft(rows * window, n=fft, axis=1))
    logmag = np.log(spectrum + 1e-8)
    # Distance between consecutive frames, averaged over every bin.
    novelty = np.sqrt(np.mean(np.diff(logmag, axis=0) ** 2, axis=1))
    if novelty.size < 3:
        return None
    # novelty[k] describes the boundary between frame k and frame k + 1, whose
    # time is the *start* of frame k + 1.  ``start`` is a sample index here, so
    # the division happens once, at the end.
    times = (start + (np.arange(novelty.size) + 1) * frame) / float(sample_rate)
    eligible = (times >= restrict_lo) & (times <= restrict_hi)
    if not bool(eligible.any()):
        return None

    scores = np.where(eligible, novelty, -1.0)
    k = int(np.argmax(scores))
    best = float(scores[k])
    median = float(np.median(novelty[eligible]))
    if best < max(median * float(min_ratio), float(min_abs)):
        return None
    return float(min(max(times[k], restrict_lo), restrict_hi))


# --- the chain ----------------------------------------------------------------


def cut_between(
    start: float,
    end: float,
    refs: Optional[dict] = None,
    audio: Optional[np.ndarray] = None,
    cfg: Optional[dict] = None,
) -> tuple[float, str]:
    """Decide the boundary between two adjacent spans.  Returns ``(cut, method)``.

    ``start``/``end`` are the nominal gap bounds (``left["end"]`` and
    ``right["start"]``); ``start`` is also the "nominal boundary" reported in the
    per-boundary log line and in the recorded ``shift_sec``.  ``end - start <= min_gap_sec`` means the two spans abut, in
    which case the lookaround window is searched instead of the gap.

    The cut is clamped into the gap when there is one, and otherwise into
    ``[left_start + EDGE_EPS, right_end - EDGE_EPS]``, then rounded to 4 decimals.
    Both sides of a boundary take the same cut (lesson 17).
    """
    refs = refs or {}
    conf = _conf(cfg)
    nominal = float(start)
    gap_start = float(start)
    gap_end = float(end)
    if gap_end < gap_start:
        gap_end = gap_start
    gap_eps = max(float(conf.get("min_gap_sec") or GAP_EPS), 0.0)
    has_gap = (gap_end - gap_start) > gap_eps

    left_start = refs.get("left_start")
    right_end = refs.get("right_end")
    lo = None if left_start is None else float(left_start) + EDGE_EPS
    hi = None if right_end is None else float(right_end) - EDGE_EPS
    if lo is not None and hi is not None and hi < lo:
        lo = hi = None

    sample_rate = int(refs.get("sample_rate") or 16000)

    def _clamp(value: float) -> float:
        cut = float(value)
        if lo is not None:
            cut = min(max(cut, lo), hi)
        if has_gap:
            cut = min(max(cut, gap_start), gap_end)
        cut = round(cut, 4)
        if lo is not None:
            cut = min(max(cut, lo), hi)
        if has_gap:
            cut = min(max(cut, gap_start), gap_end)
        return cut

    def _emit(value: float, method: str) -> tuple[float, str]:
        cut = _clamp(value)
        logger.debug(
            "boundary nominal=%.3f cut=%.3f shift=%+.3f method=%s gap=[%.3f,%.3f]",
            nominal, cut, cut - nominal, method, gap_start, gap_end,
        )
        return cut, method

    if not conf.get("enabled", True):
        return _emit((gap_start + gap_end) / 2.0, METHOD_MIDPOINT)

    # Abutting boundaries have no silence of their own, so the search window is
    # the lookaround; a real gap searches the gap plus its lookaround.
    window_lo = gap_start - float(conf["lookaround_sec"])
    window_hi = gap_end + float(conf["lookaround_sec"])
    if not has_gap:
        window_lo = max(0.0, window_lo)
    restrict_lo, restrict_hi = (gap_start, gap_end) if has_gap else (window_lo, window_hi)

    if conf.get("acoustic", True):
        cut = _ownership_cut(window_lo, window_hi, gap_end, refs, conf)
        if cut is not None:
            return _emit(cut, METHOD_VAD)

        if conf.get("spectral_novelty", True):
            cut = spectral_change_cut(
                audio, window_lo, window_hi, restrict_lo, restrict_hi,
                sample_rate=sample_rate,
                frame_ms=float(conf["spectral_frame_ms"]),
                n_fft=int(conf["spectral_n_fft"]),
                min_ratio=float(conf["spectral_min_ratio"]),
                min_abs=float(conf["spectral_min_abs"]),
            )
            if cut is not None:
                return _emit(cut, METHOD_SPECTRAL)

    # An abutting boundary has no silence of its own, so the energy chain searches
    # the lookaround window (this is what the pre-P1 step-13 fallback did); a real
    # gap searches the gap.
    energy_lo, energy_hi = (gap_start, gap_end) if has_gap else (window_lo, window_hi)

    cut = energy_dip_cut(
        audio, energy_lo, energy_hi, sample_rate,
        dip_ratio=float(conf["dip_ratio"]),
        min_dip_sec=float(conf["min_dip_sec"]),
        frame_ms=float(conf["frame_ms"]) if "frame_ms" in conf else FRAME_MS,
    )
    if cut is not None:
        return _emit(cut, METHOD_DIP)

    cut = quietest_frame_cut(audio, energy_lo, energy_hi, sample_rate)
    if cut is not None:
        return _emit(cut, METHOD_QUIETEST)

    return _emit((gap_start + gap_end) / 2.0, METHOD_MIDPOINT)


def refine_gap(
    left: dict,
    right: dict,
    audio: Optional[np.ndarray] = None,
    refs: Optional[dict] = None,
    cfg: Optional[dict] = None,
) -> tuple[float, str]:
    """Convenience wrapper: refine the ``left``/``right`` pair in place.

    Both dictionaries are updated to share the one cut (lesson 17).  A cut that
    would leave either side shorter than ``refs["min_side_sec"]`` is **not**
    applied — the nominal boundary stays, and nothing is deleted.
    """
    base = dict(refs or {})
    base.setdefault("left_start", left.get("start"))
    base.setdefault("right_end", right.get("end"))
    # The nominal boundary is where the left span currently ends — the same value
    # ``cut_between`` logs — so the per-boundary line, the recorded shift and the
    # histogram all measure movement from the same origin.
    nominal = float(left["end"])
    stats = base.get("stats")
    cut, method = cut_between(
        float(left["end"]), float(right["start"]), base, audio, cfg,
    )
    min_side = float(base.get("min_side_sec") or 0.0)
    min_shift = float(base.get("min_shift_sec") or 0.0)
    too_short_side = min_side > 0.0 and (
        cut - float(left["start"]) < min_side
        or float(right["end"]) - cut < min_side
    )
    trivial_shift = min_shift > 0.0 and abs(cut - nominal) <= min_shift
    if too_short_side or trivial_shift:
        # Refusing to move is not the same as deleting: both spans keep their
        # nominal extent and the timeline stays fully covered.
        cut = float(round(min(max(nominal, float(left["start"])), float(right["end"])), 4))
        method = METHOD_MIDPOINT
        logger.debug(
            "boundary nominal=%.3f cut=%.3f shift=%+.3f method=midpoint (refused: %s)",
            nominal, cut, cut - nominal,
            "a side would drop below %.2fs" % min_side if too_short_side
            else "shift below %.2fs" % min_shift,
        )
    left["end"] = cut
    right["start"] = cut
    if stats is not None:
        stats.record(nominal, cut, method)
    return cut, method
