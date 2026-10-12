import asyncio
import datetime
import functools
import hashlib
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from asr_mcp.db.manager import DatabaseManager, SnippetDB, VoiceprintDB, DEFAULT_USER
from asr_mcp.speaker.uncertainty import (
    is_generic_speaker, is_uncertain_label,
    auto_collect_refusal_reason, learning_refusal_reason,
)
from asr_mcp.speaker.embedding import (
    extract_embedding, batch_embed_files, compute_pitch, compute_energy,
)
from asr_mcp.voiceprint.utils import (
    AudioSegmentSource, load_audio, load_audio_segment,
    generate_segment_hash, format_time_short,
)

logger = logging.getLogger("asr_mcp.voiceprint.service")

SAMPLE_RATE = 16000
AUTO_COLLECT_MIN_DURATION = 3.0
AUTO_COLLECT_MAX_TOTAL_SEC = 1200.0
AUTO_COLLECT_MAX_SEGMENT_SEC = 300.0
AUTO_COLLECT_MIN_SPEAKER_SEGMENTS = 2
MIN_SNIPPET_DURATION = 1.5

# Learning an unknown speaker. A pending profile is created from a generic
# "Speaker N" cluster, collects snippets, and is EXCLUDED from voiceprint
# matching until the user names it -- so learning a colleague can never produce
# a confident misattribution (lesson 30).
PENDING_PREFIX = "Pending"
LEARN_MIN_SPEECH_SEC = 10.0
LEARN_MIN_SEGMENTS = 2

# Refused near-misses are reported per run, not per segment: a 50-minute
# recording can refuse hundreds, and the point of the record is to be read.
# The count in the log is always the true total, so the cap hides rows, never
# the fact that there were more.
LEARN_DIAG_MAX_REFUSALS = 25
LEARN_DIAG_MAX_ALTERNATIVES = 3

# Dead snippet rows removed per speaker per cleanup pass. The audit caps a
# finding list at 50 entries, so a store with hundreds of dead rows needs more
# than one pass to converge -- and a delete driven by a report should have a
# ceiling, not take whatever the list happens to hold.
DEAD_SNIPPET_CAP = 200


def _learning_cfg() -> dict:
    """`learning` section of thresholds.json, with the shipped defaults."""
    defaults = {
        "split_clusters": True,
        "max_intra_cluster_dist": 0.32,
        # P2 (fail closed).  False restores the pre-P2 behaviour of learning the
        # whole cluster whenever the cohesion check could not be completed.
        "fail_closed": True,
        "max_profile_intra_dist": 0.32,
        "min_inter_dist": 0.32,
        "extend_max_dist": 0.32,
        "max_snippet_sec": 30.0,
        "max_profile_total_sec": 900.0,
        "max_refine_weight_ratio": 4.0,
        "consolidation": {
            "outlier_floor": 0.35,
            "outlier_mad_k": 6.0,
            "outlier_max_fraction": 0.2,
            "outlier_min_keep": 3,
            "outlier_mad_rounds": 2,
        },
    }
    try:
        from asr_mcp.config import get_config

        raw = get_config() or {}
        section = raw.get("learning") or {}
        if isinstance(section, dict):
            defaults.update(section)
    except Exception:
        pass
    return defaults


def _consolidation_cfg() -> dict:
    """`learning.consolidation` of thresholds.json, with the shipped defaults.

    Read at call time (not at import) so a threshold edit takes effect without
    a restart, like every other tuning block in this project.
    """
    defaults = {
        "outlier_floor": 0.35,
        "outlier_mad_k": 6.0,
        "outlier_max_fraction": 0.2,
        "outlier_min_keep": 3,
        "outlier_mad_rounds": 2,
    }
    raw = (_learning_cfg() or {}).get("consolidation")
    if isinstance(raw, dict):
        for key, value in raw.items():
            if value is not None:
                defaults[key] = value
    return defaults


def score_outliers(vectors, cfg: dict = None) -> dict:
    """Flag the snippets that do not look like the speaker. Pure, no I/O.

    A snippet is accused only when its distance to the speaker's centre is
    BOTH a robust outlier (``median + k * MAD * 1.4826``) AND beyond an
    absolute floor near the naming bar (``matching.accept_threshold``), because
    "this is not the same person" IS the naming question. Two further caps make
    a runaway detector visible instead of destructive: never more than
    ``outlier_max_fraction`` of the speaker's snippets, and never fewer than
    ``outlier_min_keep`` survivors.

    The centre is recomputed from the currently-kept snippets between rounds
    (``outlier_mad_rounds``). A single centre computed over everything is
    dragged toward the outliers and hides the very thing being looked for --
    which is exactly the case where a profile was fitted to the wrong voice and
    the majority is the mistake.

    Returns ``{"distances", "kept_indices", "flagged_indices", "kept_vectors",
    "reasons", "rounds"}``. Every rejection is explained in
    ``reasons`` ("beyond the floor", "capped at max_fraction", "would go
    under min_keep") so a user can argue with a specific accusation.
    """
    cfg = cfg or _consolidation_cfg()
    floor = float(cfg.get("outlier_floor") or 0.0)
    k = float(cfg.get("outlier_mad_k") or 0.0)
    max_fraction = float(cfg.get("outlier_max_fraction") or 0.0)
    min_keep = int(cfg.get("outlier_min_keep") or 0)
    # An explicit 0 means REPORT ONLY: every snippet is scored and the cut is
    # recorded, but nobody is accused. That is the knob to reach for while a
    # threshold is being calibrated against a real library, so it must not be
    # floored at 1 the way a missing value would be.
    _raw_rounds = int(cfg.get("outlier_mad_rounds", 2))
    report_only = _raw_rounds <= 0
    rounds = 2 if report_only else _raw_rounds

    vectors = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    n = len(vectors)
    if n == 0:
        return {"distances": [], "kept_indices": [], "flagged_indices": [],
                "kept_vectors": [], "reasons": {}, "rounds": []}

    # Nothing to accuse: a single snippet has no internal spread.
    if n < 3:
        return {"distances": [0.0] * n, "kept_indices": list(range(n)),
                "flagged_indices": [], "kept_vectors": list(vectors),
                "reasons": {}, "rounds": []}

    # At least one: a fraction cap that rounds DOWN to zero would silence the
    # detector on every small profile, which is not what "at most 20%" means.
    max_flag = n if max_fraction <= 0 else max(1, int(np.floor(max_fraction * n)))
    kept = list(range(n))
    flagged: list = []
    reasons: dict = {}
    distances = [0.0] * n
    history = []

    for round_no in range(1, rounds + 1):
        centre = _centroid([vectors[i] for i in kept])
        if centre is None:
            break
        dists = np.array([_cosine_distance(v, centre) for v in vectors], dtype=np.float64)
        kept_dists = dists[kept]
        median = float(np.median(kept_dists))
        mad = float(np.median(np.abs(kept_dists - median))) * 1.4826
        cut = max(median + k * mad, floor)
        history.append({"round": round_no, "kept": len(kept),
                        "median": round(median, 4), "mad": round(mad, 4),
                        "cut": round(float(cut), 4)})
        if report_only:
            break
        if mad <= 0 and float(np.max(kept_dists)) <= floor:
            # A perfectly tight profile that also sits inside the floor: there is
            # nothing to separate, and inventing an outlier would be a fiction.
            break

        accuse = [i for i in kept if float(dists[i]) > cut]
        # Cap first, and it is a cap on the WHOLE PASS, not on one round: the
        # budget left by earlier rounds is already spent, so re-deriving the cap
        # from `kept` alone would let a multi-round sweep flag several times
        # the fraction it promises. Worst offenders first.
        budget = max(0, max_flag - len(flagged))
        if budget and len(accuse) > budget:
            accuse.sort(key=lambda i: -float(dists[i]))
            dropped = accuse[budget:]
            for i in dropped:
                reasons[i] = "capped at outlier_max_fraction (%.0f%% of the profile)" % (
                    max_fraction * 100)
            accuse = accuse[:budget]
        elif not budget and accuse:
            for i in accuse:
                reasons.setdefault(
                    i, "capped at outlier_max_fraction (%.0f%% of the profile)" % (
                        max_fraction * 100))
            accuse = []
        # Then the floor on survivors: putting the speaker below min_keep would
        # be a profile nobody can match, so the mildest accusations are dropped.
        if min_keep and len(kept) - len(accuse) < min_keep:
            accuse.sort(key=lambda i: float(dists[i]))
            spare = min_keep - (len(kept) - len(accuse))
            for i in accuse[:spare]:
                reasons[i] = "kept to honour outlier_min_keep (%d)" % min_keep
            accuse = accuse[spare:]
        if not accuse:
            break
        for i in accuse:
            # Overwrite: an earlier round may have parked a "capped" or
            # "min_keep" note on this index, and a snippet that ends up FLAGGED
            # must not report a reason saying it was spared.
            reasons[i] = "beyond the floor (%.3f > %.3f)" % (
                float(dists[i]), float(cut))
            flagged.append(i)
        kept = [i for i in kept if i not in accuse]

    final = _centroid([vectors[i] for i in kept])
    if final is not None:
        distances = [float(_cosine_distance(v, final)) for v in vectors]

    return {
        "distances": distances,
        "kept_indices": kept,
        "flagged_indices": flagged,
        "kept_vectors": [vectors[i] for i in kept],
        "reasons": reasons,
        "rounds": history,
    }


def _cosine_distance(a, b) -> float:
    """1 - cosine similarity of two vectors (0 = identical direction)."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if not np.isfinite(na) or not np.isfinite(nb) or na <= 1e-8 or nb <= 1e-8:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))


def _centroid(vectors) -> Optional[np.ndarray]:
    if not vectors:
        return None
    arr = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return mean / norm


def _pairwise_distances(vectors) -> Optional[np.ndarray]:
    """Within-group cosine distances, upper triangle only, or None for < 2."""
    if not vectors or len(vectors) < 2:
        return None
    arr = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
    arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
    sim = np.clip(arr @ arr.T, -1.0, 1.0)
    iu = np.triu_indices(sim.shape[0], k=1)
    return 1.0 - sim[iu]


def _max_pairwise_distance(vectors) -> Optional[float]:
    """Largest within-group cosine distance, or None for < 2 vectors.

    Diagnostic only. The value scales with group size -- it is a maximum over
    n*(n-1)/2 pairs -- so it must NOT be compared against a threshold that an
    average-linkage clustering produced. See ``_mean_pairwise_distance``.
    """
    dists = _pairwise_distances(vectors)
    return None if dists is None else float(np.max(dists))


def _mean_pairwise_distance(vectors) -> Optional[float]:
    """Mean within-group cosine distance, or None for < 2 vectors.

    This is the statistic the split threshold is expressed in.  ``
    _split_cluster_by_cohesion`` clusters with ``AgglomerativeClustering(
    distance_threshold=..., linkage="average")``, which merges while the AVERAGE
    pairwise distance to the cluster stays under the bar.  Gating the resulting
    group on its MAXIMUM instead is comparing two different quantities against
    the same number, and the mismatch grows with the number of spans: the split
    says nothing about the max, and the max of n*(n-1)/2 samples rises with n.

    Measured on 2026-09-04 12-08-59.mp4 that cost every cluster in the file: a
    22-span group averaged 0.21 (well inside 0.32) but reached a max of 0.379,
    so it was refused as ``impure_group`` while ten one-span fragments around it
    were refused as ``too_few_spans`` -- zero profiles from a 34-minute
    recording with four speakers.
    """
    dists = _pairwise_distances(vectors)
    return None if dists is None else float(np.mean(dists))


def _name_tokens(name: str) -> list[str]:
    """Lowercase alphanumeric tokens of a profile name / cluster label.

    Exact TOKEN matching is what keeps ``Speaker 1`` from being treated as a
    substring of ``Speaker 10``: the label tokens are ``["speaker", "1"]`` and
    ``["speaker", "10"]``, which are not the same.
    """
    return re.findall(r"[a-z0-9]+", str(name or "").lower())


def _tokens_contain(haystack: list[str], needle: list[str]) -> bool:
    """Is ``needle`` a contiguous token sub-list of ``haystack``?"""
    if not needle or len(needle) > len(haystack):
        return False
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return True
    return False


def trim_span_to_vad(start_sec: float, end_sec: float, sections,
                     eps: float = 1e-3):
    """Snap a snippet's edges to the VAD speech sections inside it.

    Why: a learned snippet is embedded verbatim, and that embedding is later
    used as a **reference for cutting boundaries**
    (``diarization/pipeline.py::_get_speaker_ref``,
    ``api/asr_router.py::_speaker_refs``).  An edge that lands inside a speech
    section keeps a fragment of a phoneme in the training audio.

    The operation is deliberately **inward only**.  Two of the three places that
    place a boundary do it arithmetically (a midpoint, or a cut clamped into a
    gap the neighbouring turn also owns), and turns ABUT by design
    (lesson 17), so padding outward would pull the neighbouring speaker's
    speech into this snippet and duplicate it there — trading a cut phoneme for
    a contaminated sample.  So each edge is moved at most onto the outermost
    section that lies inside the span, and never past it.

    Returns ``(start, end)``; unchanged when there is no usable section.
    """
    start = float(start_sec)
    end = float(end_sec)
    if not sections or end <= start:
        return start, end
    ordered = sorted(
        (s for s in sections if s.get("end", 0) > s.get("start", 0)),
        key=lambda s: float(s["start"]),
    )
    overlapping = [
        s for s in ordered
        if float(s["end"]) > start + eps and float(s["start"]) < end - eps
    ]
    if not overlapping:
        return start, end
    new_start = max(start, float(overlapping[0]["start"]))
    new_end = min(end, float(overlapping[-1]["end"]))
    if new_end - new_start < MIN_SNIPPET_DURATION:
        # Trimming must not produce an unusable snippet: keep the untrimmed
        # span and let the existing duration floor decide.
        return start, end
    return new_start, new_end


def is_pending_profile(name: str, voiceprint: dict | None = None) -> bool:
    """Is this an auto-learned profile still waiting for a name?

    Checks the stored ``pending`` flag AND the name prefix. The prefix check
    matters because the flag lives on the voiceprints row, and that row is only
    written by ``_auto_refine`` -- so if embedding was unavailable when a
    speaker was learned, its snippets exist with no row and would otherwise be
    invisible. A directory named "Pending ..." is a pending profile by
    construction, and both signals say "not matchable" anyway.
    """
    if str(name).startswith(PENDING_PREFIX + " "):
        return True
    return bool((voiceprint or {}).get("pending"))


def pending_profile_name(source_id: str, cluster: str = "") -> str:
    """A unique, human-readable placeholder name for an unnamed profile.

    The CLUSTER label is part of the name on purpose. Without it, two different
    speakers in the same recording produce the same name (same minute, same
    source stem) and the second cluster silently extends the first one's
    profile -- merging two people. Including "Speaker 5" keeps one profile per
    cluster per recording, which is also what makes the extend-rather-than-
    create rule below safe.
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [PENDING_PREFIX, stamp]
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", cluster or "").strip("_")
    if label:
        parts.append(label)
    stem = _origin_prefix(source_id)
    if stem:
        parts.append(stem)
    return " ".join(parts)[:120]


def _origin_prefix(audio_path: str) -> str:
    """Filesystem-safe prefix from the source audio filename stem."""
    stem = Path(audio_path).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return (stem or "audio")[:64]


def is_spurious_speaker_name(name: str) -> bool:
    """Check if a speaker name is generic or auto-generated.

    Delegates to the shared uncertainty policy (asr_mcp.speaker.uncertainty) so
    this and the auto-collection gate can never disagree on what counts as an
    identity.
    """
    from asr_mcp.speaker.uncertainty import is_uncertain_label
    return is_uncertain_label(name)


class VoiceprintService:
    def __init__(self, data_dir: Path, db_manager: DatabaseManager, embedding_session=None):
        self._data_dir = data_dir
        self._db = VoiceprintDB(db_manager)
        self._snippets = SnippetDB(db_manager)
        self._db_manager = db_manager
        self._embedding_session = embedding_session
        self._voices_dir = data_dir / "voices"

    def set_embedding_session(self, session):
        self._embedding_session = session

    def _emb_session(self):
        from asr_mcp.core.model_state import state
        return self._embedding_session or state.embedding_session

    async def initialize(self):
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        # Across every owner, not the `default` user's rows: a scoped count
        # reads 0 on a store holding other users' profiles, which made the
        # startup line claim an empty library next to a directory full of
        # snippets.
        users = self._db.user_ids()
        count = sum(self._db.count(user_id=u) for u in users) if users else 0
        logger.info("VoiceprintService initialized — %d voiceprints across %d user(s)",
                    count, len(users))

    @property
    def voices_dir(self) -> Path:
        return self._voices_dir

    def set_voices_dir(self, path: Path):
        self._voices_dir = path
        path.mkdir(parents=True, exist_ok=True)

    # ── Snippet Management ───────────────────────────────────────────

    def add_snippet(
        self,
        speaker_name: str,
        audio_data: np.ndarray,
        user_id: str = DEFAULT_USER,
        sample_rate: int = SAMPLE_RATE,
        source_audio: str = None,
        start_sec: float = None,
        end_sec: float = None,
        replace_source: bool = False,
    ) -> dict:
        user_dir = self._voices_dir / user_id
        speaker_dir = user_dir / speaker_name
        speaker_dir.mkdir(parents=True, exist_ok=True)

        duration = len(audio_data) / sample_rate
        if duration < MIN_SNIPPET_DURATION:
            return {"error": f"Snippet too short ({duration:.1f}s < {MIN_SNIPPET_DURATION}s)"}

        # One file, one snippet SET: the previous spans from the same recording
        # go before the new file is written. Doing it first is required, not
        # tidiness -- the filename below is derived from the source stem, the
        # span and the duration, so re-uploading the same clip lands on the same
        # path and a delete afterwards would unlink what we just wrote.
        # Also the point of the rule: re-running a file must REPLACE its
        # contribution instead of stacking a second copy of the same room and
        # channel on top of the first, which is how a profile ends up
        # overfitted to one recording.
        replaced = self._drop_source_snippets(
            speaker_name, source_audio, user_id) if replace_source else 0

        if source_audio:
            audio_hash = generate_segment_hash(source_audio)
            time_part = format_time_short(start_sec) if start_sec is not None else "00-00"
            dur_part = f"{duration:02.0f}"
            filename = f"{audio_hash}_{time_part}_{dur_part}.flac"
        else:
            audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()[:6].upper()
            filename = f"{audio_hash}_manual.flac"
        file_path = speaker_dir / filename

        sf.write(str(file_path), audio_data.astype(np.float32), sample_rate, format="FLAC")
        try:
            fst = file_path.stat()
            file_mtime, file_size = fst.st_mtime, fst.st_size
        except OSError:
            file_mtime, file_size = None, None

        snippet_id = self._snippets.add(
            speaker_name=speaker_name,
            file_path=str(file_path),
            duration_sec=duration,
            user_id=user_id,
            source_audio=source_audio,
            start_sec=start_sec,
            end_sec=end_sec,
            file_mtime=file_mtime,
            file_size=file_size,
        )

        return {
            "id": snippet_id,
            "speaker_name": speaker_name,
            "file_path": str(file_path),
            "duration_sec": round(duration, 2),
            "replaced_snippets": replaced,
        }

    def _drop_source_snippets(self, speaker_name: str, source_audio: str,
                              user_id: str) -> int:
        """Delete every snippet of ``speaker_name`` recorded from ``source_audio``.

        Scoped to ONE speaker: the same recording may legitimately contribute
        to several profiles, and one profile's re-upload must not strip
        another's audio. Matching is by filename HASH, the same identity
        ``_source_snippet_rows`` uses, so the two rules agree on what "the same
        file" means. Uses ``_delete_snippet_row`` (row + resolved file, no
        per-row refine) -- the caller refines once, after this returns.
        """
        stale = self._source_snippet_rows(speaker_name, source_audio, user_id=user_id)
        removed = 0
        for row in stale:
            if self._delete_snippet_row(row["id"], speaker_name, user_id):
                removed += 1
        if removed:
            logger.info(
                "Replaced %d snippet(s) of %r from %s before re-adding it "
                "(user=%s)", removed, speaker_name, source_audio, user_id)
        return removed

    def _source_snippet_rows(self, speaker_name: str, source: str,
                             user_id: str) -> list[dict]:
        """Every snippet of ``speaker_name`` recorded from the file ``source``.

        A recording is identified by its FILENAME HASH -- ``generate_segment_hash``,
        md5 of the source stem -- because that is the hash ``add_snippet`` names
        its files after, so it is the one identity that survives a reprocess:
        the same audio arrives under another directory, another extension or
        another upload name, and every span and duration moves, because the point
        of reprocessing is that the boundaries got better. Comparing the stored
        ``source_audio`` string, or a span, matches none of that.

        Cost of hash identity: two DIFFERENT recordings sharing a stem (say
        ``/a/meeting.wav`` and ``/b/meeting.wav``) collide -- and they already
        collided when their snippets were named, so the hash is no worse than
        the storage layout it mirrors. Rows with no recorded source never match.
        """
        if not source:
            return []
        wanted = generate_segment_hash(source)
        return [
            row for row in self._snippets.list_by_speaker(speaker_name, user_id=user_id)
            if row.get("source_audio")
            and generate_segment_hash(row["source_audio"]) == wanted
        ]

    def _purge_source_rows(self, rows: list[dict], speaker_name: str,
                           user_id: str, keep_paths: set[str] | None = None) -> dict:
        """Delete the given stale rows of one speaker, and report what went.

        ``keep_paths`` are files the run has just written and that a surviving
        row still points at. The naming scheme makes that possible: a reprocess
        whose span and duration round to the same filename writes over the old
        file, and deleting the old row would unlink what the new row reads. The
        row still goes; only the unlink is skipped, so the two cannot both stay.
        """
        keep = {os.path.normcase(str(p)) for p in (keep_paths or ())}
        removed = 0
        freed = 0.0
        for row in rows:
            path = row.get("file_path")
            if path and os.path.normcase(str(path)) in keep:
                # The new snippet owns this file now; drop only the stale row.
                if self._snippets.delete(row["id"], user_id=user_id):
                    removed += 1
                    freed += float(row.get("duration_sec") or 0.0)
                continue
            if self._delete_snippet_row(row["id"], speaker_name, user_id):
                removed += 1
                freed += float(row.get("duration_sec") or 0.0)
        if removed:
            logger.info(
                "Reprocess: dropped %d stale snippet(s) of %r from this "
                "recording (%.1fs), keeping the freshly bounded ones (user=%s)",
                removed, speaker_name, freed, user_id)
        return {"removed": removed, "seconds": freed}

    def replace_source_contribution(self, speaker_name: str, source: str,
                                    user_id: str, fresh_rows: list[dict] | None = None,
                                    fresh_paths: set[str] | None = None) -> dict:
        """One file, one contribution, for a speaker this run re-refined.

        ``fresh_rows``/``fresh_paths`` are what the run just added from this same
        file; they are never touched. Everything else this file contributed to
        THIS speaker is removed, whatever span or duration it was cut at -- which
        is what makes a reprocess replace the previous take instead of stacking a
        second copy of the same room and channel on the profile, the shape of
        overfitting this rule exists to prevent.

        Clips from other recordings are untouched, and so are other speakers'
        clips from this recording: only a profile this run rebuilt is rewritten.
        Call it AFTER the new snippets are written and BEFORE the single
        ``_auto_refine`` for the speaker, so the embedding is fitted once, over
        the final set.
        """
        rows = self._source_snippet_rows(speaker_name, source, user_id=user_id)
        keep_ids = {r.get("id") for r in (fresh_rows or ())}
        stale = [r for r in rows if r.get("id") not in keep_ids]
        return self._purge_source_rows(stale, speaker_name, user_id,
                                       keep_paths=fresh_paths or set())

    def add_snippet_from_segment(
        self,
        speaker_name: str,
        wav_path: str,
        start_sec: float,
        end_sec: float,
        user_id: str = DEFAULT_USER,
        source_audio: str = None,
        audio_source: AudioSegmentSource = None,
    ) -> dict:
        # ``audio_source`` is the request-wide cached loader (one decode for the
        # whole request).  Without it this falls back to the historical
        # behaviour of decoding the ENTIRE file for this one span.
        if audio_source is not None:
            waveform, sr = audio_source.segment(start_sec, end_sec)
        else:
            waveform, sr = load_audio_segment(wav_path, start_sec, end_sec)
        audio_data = waveform.numpy().squeeze()
        return self.add_snippet(
            speaker_name=speaker_name,
            audio_data=audio_data,
            user_id=user_id,
            sample_rate=sr,
            source_audio=source_audio or wav_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )

    def list_speakers(self, user_id: str = DEFAULT_USER) -> list[dict]:
        snippet_info = self._snippets.all_speakers(user_id=user_id)
        voiceprint_info = self._db.list_all(user_id=user_id)

        all_names = set(list(snippet_info.keys()) + list(voiceprint_info.keys()))
        speakers = []
        for name in sorted(all_names):
            sn = snippet_info.get(name, {"count": 0, "total_duration": 0.0})
            vp = voiceprint_info.get(name, {})
            speakers.append({
                "name": name,
                "snippet_count": sn["count"],
                "total_duration_sec": round(sn["total_duration"], 2),
                "has_voiceprint": bool(vp),
                "pending": is_pending_profile(name, vp),
                "pitch_hz": round(vp.get("pitch_hz", 0), 1),
                "energy_rms": round(vp.get("energy_rms", 0), 4),
                # Only set on auto-learned profiles (see _learn_unknown_speaker):
                # how far the profile's own snippets disagree with each other,
                # and with the other groups it was separated from.
                "purity": vp.get("purity") or None,
            })
        return speakers

    def list_snippets(self, speaker_name: str, user_id: str = DEFAULT_USER) -> list[dict]:
        return self._snippets.list_by_speaker(speaker_name, user_id=user_id)

    def refine_speaker(self, speaker_name: str, user_id: str = DEFAULT_USER) -> dict:
        snippets = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        if not snippets:
            return {"error": f"No snippets found for '{speaker_name}'"}
        self._auto_refine(speaker_name, user_id=user_id)
        vp = self._db.get(speaker_name, user_id=user_id)
        if not vp:
            return {"error": f"Voiceprint could not be built for '{speaker_name}' (embedding session unavailable?)"}
        return {
            "status": "refined",
            "speaker_name": speaker_name,
            "snippet_count": len(snippets),
            "total_duration_sec": round(sum(s.get("duration_sec", 0.0) for s in snippets), 2),
            "pitch_hz": round(vp.get("pitch_hz", 0.0), 1),
            "energy_rms": round(vp.get("energy_rms", 0.0), 4),
        }

    def rebuild_profile_from_scratch(self, speaker_name: str,
                                     user_id: str = DEFAULT_USER,
                                     purge_dead: bool = True) -> dict:
        """Drop a speaker's DERIVED profile row and rebuild it from the snippets.

        The row in ``voiceprints`` is a derived artefact -- an embedding, pitch
        and energy summary of the snippet audio. Dropping it loses nothing,
        because the snippets and their FLAC files are the source of truth and
        are never touched here. That is what makes this safe to run
        automatically off an audit finding (``voiceprint/rebuild.py``).

        With ``purge_dead`` (the default) this is the WHOLE repair, in the order
        the data demands: first delete the snippet rows whose audio is gone,
        then rebuild the profile from the audio that is actually on disk.
        Cleaning up after the rebuild instead would embed 73 dead rows' worth of
        failures and log a warning per one, and the profile would be built from
        whatever loaded -- which is the point, but the store would stay dirty
        and the audit would report the same dead rows on every startup, so the
        queue would re-embed the speaker on every boot forever.

        ``pending`` and ``purity`` are read off the old row first and re-applied.
        Deleting the row destroys both, and ``_auto_refine`` saves with
        ``pending=None`` meaning "preserve" -- so a rebuild that ignored them
        would silently promote a learned-but-unnamed profile to matchable
        (lesson 39) and lose the purity verdict the learn path recorded.

        Returns ``{"status": "rebuilt" | "dropped" | "skipped", ...}``:

        * ``dropped`` -- the speaker has no snippets, so there is nothing to
          rebuild from and the row was simply removed;
        * ``rebuilt`` -- the row exists again;
        * ``skipped`` -- no embedding session (nothing was tried; the caller
          must not count this as a failure).
        """
        old = self._db.get(speaker_name, user_id=user_id) or {}
        was_pending = bool(old.get("pending"))
        purity = old.get("purity") or None

        purged = {"dead_snippets": 0, "seconds": 0.0}
        if purge_dead:
            purged = self._purge_dead_snippet_rows(speaker_name, user_id)

        snippets = self._snippets.list_by_speaker(speaker_name, user_id=user_id)

        if not snippets:
            self._db.delete(speaker_name, user_id=user_id)
            logger.warning(
                "Voiceprint rebuild: %s/%s has no snippets -- dropped the "
                "profile row (nothing to rebuild from)", user_id, speaker_name)
            return {"status": "dropped", "speaker_name": speaker_name,
                    "snippet_count": 0, "user_id": user_id, **purged}

        if self._emb_session() is None:
            # Not an attempt: the row is untouched and the queue keeps the entry.
            # The dead rows are already gone, though, and that was the point.
            return {"status": "skipped", "speaker_name": speaker_name,
                    "user_id": user_id, **purged,
                    "error": "no embedding session (models not loaded)"}

        self._db.delete(speaker_name, user_id=user_id)
        self._auto_refine(speaker_name, user_id=user_id, purity=purity)

        vp = self._db.get(speaker_name, user_id=user_id)
        if not vp:
            logger.error("Voiceprint rebuild: %s/%s produced no profile from %d "
                         "snippet(s) -- the speaker has audio but no voiceprint",
                         user_id, speaker_name, len(snippets))
            return {"status": "failed", "speaker_name": speaker_name,
                    "user_id": user_id, "snippet_count": len(snippets), **purged,
                    "error": "no profile could be built from the snippets"}

        if was_pending:
            self._db.set_pending(speaker_name, True, user_id=user_id)
        logger.warning("Voiceprint rebuild: %s/%s rebuilt from %d snippet(s) "
                       "(%.1fs)%s", user_id, speaker_name, len(snippets),
                       sum(float(s.get("duration_sec") or 0.0) for s in snippets),
                       " [pending]" if was_pending else "")
        return {"status": "rebuilt", "speaker_name": speaker_name,
                "user_id": user_id, "snippet_count": len(snippets),
                **purged,
                "total_duration_sec": round(
                    sum(float(s.get("duration_sec") or 0.0) for s in snippets), 2),
                "pending": was_pending}

    # ------------------------------------------------------------------
    # Outlier review (consolidation)
    # ------------------------------------------------------------------

    def consolidate_speaker(self, speaker_name: str, user_id: str = DEFAULT_USER,
                            progress_callback=None) -> dict:
        """Recompute a speaker's profile from its snippets and flag the outliers.

        One embedding pass does both jobs: every usable snippet is embedded
        once, the profile is fitted from the KEPT snippets only, and the
        snippets that look like a different person are MARKED for review --
        never deleted here. Deleting audio on the strength of a distance is not
        a decision this endpoint may make (see ``remove_flagged_snippets``).

        The centre is recomputed from the kept snippets between rounds
        (``outlier_mad_rounds``): a mean that includes the outliers it is
        meant to find is dragged toward them and hides them.

        Returns ``{"status": "consolidated" | "clean" | "skipped", ...}`` with
        ``flagged`` (the accusations, worst first, each carrying its distance
        and the reason), ``kept``, ``mean_distance_before`` /
        ``mean_distance_after`` for the kept snippets, and the per-round
        statistics behind the cut, so the effect is visible and reversible
        (keep a snippet, re-consolidate).
        """
        if self._emb_session() is None:
            return {"status": "skipped", "speaker_name": speaker_name,
                    "user_id": user_id, "flagged": [], "kept": 0,
                    "error": "no embedding session (models not loaded)"}

        data = self._embed_snippets(speaker_name, user_id=user_id,
                                    progress_callback=progress_callback)
        vectors, rows, waveforms = data["vectors"], data["rows"], data["waveforms"]
        durations = data["durations"]
        if not vectors:
            return {"status": "skipped", "speaker_name": speaker_name,
                    "user_id": user_id, "flagged": [], "kept": 0,
                    "unusable_snippets": data.get("missing") or 0,
                    "error": "no usable snippets to embed"}

        cfg = _consolidation_cfg()
        scored = score_outliers(vectors, cfg)

        # Mean distance of the KEPT snippets, before: to the profile as it
        # stands (the stored embedding when there is one), because "the profile
        # got closer to the audio it claims to describe" is the only useful
        # before/after a user can check.
        vp_before = self._db.get(speaker_name, user_id=user_id) or {}
        before_emb = vp_before.get("embedding")
        if before_emb is None:
            before_emb = self._weighted_embedding(vectors, durations)
        kept_durations = [durations[i] for i in scored["kept_indices"]]
        after_emb = self._weighted_embedding(scored["kept_vectors"], kept_durations)

        def _mean_dist(centre, idxs):
            vals = [_cosine_distance(vectors[i], centre) for i in idxs]
            return round(float(np.mean(vals)), 4) if vals else None

        kept_idx = scored["kept_indices"]
        mean_before = _mean_dist(before_emb, kept_idx) if before_emb is not None else None
        mean_after = _mean_dist(after_emb, kept_idx) if after_emb is not None else None

        # Marks: accused snippets get their score and reason (the evidence the
        # user needs); kept ones have any earlier mark CLEARED, so a stale
        # accusation cannot outlive the run that refuted it.
        for i in kept_idx:
            if rows[i].get("review_flag"):
                self._snippets.set_review(rows[i]["id"], user_id=user_id, flagged=False)
        flagged_out = []
        for i in scored["flagged_indices"]:
            reason = scored["reasons"].get(i) or "distance to the speaker's centre"
            self._snippets.set_review(rows[i]["id"], user_id=user_id, flagged=True,
                                      score=scored["distances"][i], reason=reason)
            row = rows[i]
            flagged_out.append({
                "snippet_id": row["id"], "speaker_name": speaker_name,
                "file_path": row["file_path"], "duration_sec": row.get("duration_sec"),
                "source_audio": row.get("source_audio"),
                "start_sec": row.get("start_sec"), "end_sec": row.get("end_sec"),
                "distance": round(float(scored["distances"][i]), 4),
                "reason": reason,
            })
        flagged_out.sort(key=lambda e: -e["distance"])

        if flagged_out and after_emb is not None and scored["kept_vectors"]:
            self._db.save(
                name=speaker_name, user_id=user_id, embedding=after_emb,
                pitch_hz=0.0, pitch_std=0.0, energy_rms=0.0,
                spectral_centroid=0.0, spectral_rolloff=0.0,
                total_speech_sec=round(sum(kept_durations), 1),
                sample_count=len(scored["kept_vectors"]),
                mfcc=None, pending=None, purity=None,
            )
        elif flagged_out:
            logger.warning("Consolidation: every snippet of %s/%s was flagged -- "
                           "the profile is left untouched rather than fitted on "
                           "a set the user has not approved", user_id, speaker_name)

        logger.warning(
            "Consolidation: %s/%s -- %d kept, %d flagged for review "
            "(mean distance %.4f -> %.4f)", user_id, speaker_name,
            len(kept_idx), len(flagged_out),
            mean_before if mean_before is not None else float("nan"),
            mean_after if mean_after is not None else float("nan"))

        return {
            "status": "consolidated" if flagged_out else "clean",
            "speaker_name": speaker_name, "user_id": user_id,
            "flagged": flagged_out, "flagged_count": len(flagged_out),
            "kept": len(kept_idx),
            "kept_seconds": round(sum(kept_durations), 1),
            "mean_distance_before": mean_before,
            "mean_distance_after": mean_after,
            "unusable_snippets": data.get("missing") or 0,
            "rounds": scored["rounds"],
            "profile_updated": bool(flagged_out and after_emb is not None
                                    and scored["kept_vectors"]),
        }

    def consolidate_library(self, user_id: str = DEFAULT_USER,
                            progress_callback=None) -> dict:
        """Consolidate every speaker that owns snippets, one at a time.

        Iterates ``all_speakers`` then ``list_by_speaker`` per name -- the same
        shape as the audit, because there is no way to list snippets across
        speakers in one query. A failure on one speaker is recorded and the
        sweep continues: one broken profile must not hide the rest.
        """
        speakers = sorted(self._snippets.all_speakers(user_id).keys())
        results = []
        for i, name in enumerate(speakers):
            if progress_callback:
                progress_callback({
                    "stage": f"Consolidating {name} ({i + 1}/{len(speakers)})",
                    "speaker": name, "progress": i / max(1, len(speakers)),
                })
            try:
                res = self.consolidate_speaker(name, user_id=user_id)
            except Exception as e:  # one bad speaker must not stop the sweep
                logger.error("Consolidation failed for %s/%s: %s", user_id, name, e)
                res = {"status": "failed", "speaker_name": name, "user_id": user_id,
                       "flagged": [], "kept": 0, "error": str(e)}
            results.append(res)
        total_flagged = sum(int(r.get("flagged_count") or 0) for r in results)
        logger.warning("Consolidation sweep for %s: %d speaker(s), %d snippet(s) "
                       "flagged for review", user_id, len(speakers), total_flagged)
        return {
            "user_id": user_id, "speakers": len(speakers),
            "flagged_count": total_flagged,
            "results": results,
        }

    def list_flagged_snippets(self, user_id: str = DEFAULT_USER,
                              speaker_name: str = None) -> list[dict]:
        """Every snippet marked for review, worst score first (read-only)."""
        return self._snippets.list_flagged(user_id=user_id, speaker_name=speaker_name)

    def keep_flagged_snippet(self, snippet_id: int, user_id: str = DEFAULT_USER) -> dict:
        """Clear one snippet's review mark -- the snippet was never removed."""
        sn = self._snippets.get(snippet_id, user_id=user_id)
        if not sn:
            return {"error": "Snippet not found"}
        if not sn.get("review_flag"):
            return {"status": "kept", "snippet_id": snippet_id, "changed": False}
        self._snippets.set_review(snippet_id, user_id=user_id, flagged=False)
        return {"status": "kept", "snippet_id": snippet_id, "changed": True,
                "speaker_name": sn.get("speaker_name")}

    def keep_flagged_speaker(self, speaker_name: str, user_id: str = DEFAULT_USER) -> dict:
        """Clear every review mark of one speaker (re-run consolidate after)."""
        cleared = self._snippets.clear_review(speaker_name, user_id=user_id)
        return {"status": "kept", "speaker_name": speaker_name, "cleared": cleared}

    def remove_flagged_snippets(self, user_id: str = DEFAULT_USER,
                                speaker_name: str = None,
                                dry_run: bool = True) -> dict:
        """Delete the snippets marked for review, then refine each speaker ONCE.

        ``dry_run=True`` is the default: the first call must change nothing.
        Deletion goes through ``_delete_snippet_row`` (row + the file the audit's
        resolver finds), never ``delete_snippet``, which re-embeds the speaker on
        every call -- removing sixteen flagged snippets that way would re-embed
        the profile sixteen times. The refine happens once per touched speaker,
        over the final snippet set.
        """
        flagged = self._snippets.list_flagged(user_id=user_id, speaker_name=speaker_name)
        plan = [{"snippet_id": r["id"], "speaker_name": r["speaker_name"],
                 "file_path": r["file_path"], "duration_sec": r.get("duration_sec"),
                 "distance": r.get("review_score"), "reason": r.get("review_reason"),
                 "deleted": False}
                for r in flagged]
        if not plan:
            return {"dry_run": dry_run, "user_id": user_id, "removed": [],
                    "removed_count": 0, "removed_seconds": 0.0, "refined": [],
                    "ok": True}
        if dry_run:
            return {"dry_run": True, "user_id": user_id, "removed": plan,
                    "removed_count": len(plan),
                    "removed_seconds": round(sum(
                        float(p["duration_sec"] or 0.0) for p in plan), 1),
                    "refined": [], "ok": True}

        touched: set = set()
        for entry in plan:
            name = entry["speaker_name"]
            if self._delete_snippet_row(entry["snippet_id"], name, user_id):
                entry["deleted"] = True
                touched.add(name)
        removed = [e for e in plan if e["deleted"]]
        for name in sorted(touched):
            if not self._snippets.list_by_speaker(name, user_id=user_id):
                self._db.delete(name, user_id=user_id)
                self._cleanup_speaker_dir(name, user_id=user_id)
            else:
                self._auto_refine(name, user_id=user_id)
        logger.warning("Removed %d flagged snippet(s) for %s (%d speaker(s) "
                       "re-refined)", len(removed), user_id, len(touched))
        return {"dry_run": False, "user_id": user_id, "removed": removed,
                "removed_count": len(removed),
                "removed_seconds": round(sum(
                    float(e["duration_sec"] or 0.0) for e in removed), 1),
                "refined": sorted(touched), "ok": True}

    def delete_snippet(self, snippet_id: int, user_id: str = DEFAULT_USER) -> dict:
        sn = self._snippets.get(snippet_id, user_id=user_id)
        if not sn:
            return {"error": "Snippet not found"}

        speaker_name = sn["speaker_name"]
        file_path = Path(sn["file_path"])
        if file_path.exists():
            file_path.unlink()

        self._snippets.delete(snippet_id, user_id=user_id)

        remaining = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        if not remaining:
            self._db.delete(speaker_name, user_id=user_id)
            self._cleanup_speaker_dir(speaker_name, user_id)
            return {"status": "deleted", "speaker_removed": True}

        self._auto_refine(speaker_name, user_id=user_id)
        return {"status": "deleted", "speaker_removed": False}

    def delete_speaker_bulk(self, speaker_name: str, user_id: str = DEFAULT_USER) -> dict:
        """Delete a speaker, all snippets, and files without intermediate refine steps."""
        snippets = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        for sn in snippets:
            file_path = Path(sn["file_path"])
            if file_path.exists():
                file_path.unlink()
            self._snippets.delete(sn["id"], user_id=user_id)

        self._db.delete(speaker_name, user_id=user_id)
        self._cleanup_speaker_dir(speaker_name, user_id)

        return {"status": "deleted", "name": speaker_name, "snippets_removed": len(snippets)}

    def rename_speaker(self, old_name: str, new_name: str, user_id: str = DEFAULT_USER) -> dict:
        if new_name == old_name:
            return {"error": "New name is the same as the current name"}
        existing_vp = self._db.get(new_name, user_id=user_id)
        snippet_count = self._snippets.count(new_name, user_id=user_id)
        if existing_vp or snippet_count > 0:
            return {"error": f"Speaker '{new_name}' already exists"}

        count = self._snippets.rename_speaker(old_name, new_name, user_id=user_id)

        vp = self._db.get(old_name, user_id=user_id)
        if vp:
            self._db.save(
                name=new_name, user_id=user_id,
                embedding=vp["embedding"],
                pitch_hz=vp.get("pitch_hz", 0),
                pitch_std=vp.get("pitch_std", 0),
                energy_rms=vp.get("energy_rms", 0),
                spectral_centroid=vp.get("spectral_centroid", 0),
                spectral_rolloff=vp.get("spectral_rolloff", 0),
                total_speech_sec=vp.get("total_speech_sec", 0),
                sample_count=vp.get("sample_count", 0),
            )
            self._db.delete(old_name, user_id=user_id)

        old_dir = self._voices_dir / user_id / old_name
        new_dir = self._voices_dir / user_id / new_name
        if old_dir.exists():
            if new_dir.exists():
                for f in old_dir.iterdir():
                    shutil.move(str(f), str(new_dir / f.name))
                old_dir.rmdir()
            else:
                old_dir.rename(new_dir)

        old_prefix = str(old_dir)
        new_prefix = str(new_dir)
        paths_updated = 0
        for sn in self._snippets.list_by_speaker(new_name, user_id=user_id):
            fp = sn["file_path"]
            try:
                rel = Path(fp).relative_to(old_prefix)
            except ValueError:
                continue
            if self._snippets.update_file_path(sn["id"], str(new_prefix / rel), user_id=user_id):
                paths_updated += 1

        transcripts_updated = 0
        try:
            from asr_mcp.db.manager import TranscriptDB
            transcripts_updated = TranscriptDB(self._db._db).rename_speaker(
                old_name, new_name, user_id=user_id
            )
        except Exception as e:
            logger.warning("Failed to update transcripts for rename %r -> %r: %s",
                           old_name, new_name, e)

        return {
            "status": "renamed",
            "from": old_name,
            "to": new_name,
            "snippets_moved": count,
            "paths_updated": paths_updated,
            "transcripts_updated": transcripts_updated,
        }

    def merge_speakers(self, primary_name: str, secondary_name: str, user_id: str = DEFAULT_USER) -> dict:
        secondary_snippets = self._snippets.list_by_speaker(secondary_name, user_id=user_id)
        if not secondary_snippets:
            return {"error": f"No snippets found for '{secondary_name}'"}

        primary_dir = self._voices_dir / user_id / primary_name
        primary_dir.mkdir(parents=True, exist_ok=True)

        for sn in secondary_snippets:
            old_path = Path(sn["file_path"])
            if old_path.exists():
                new_path = primary_dir / old_path.name
                shutil.move(str(old_path), str(new_path))
                self._snippets.delete(sn["id"], user_id=user_id)
                self._snippets.add(
                    speaker_name=primary_name,
                    file_path=str(new_path),
                    duration_sec=sn["duration_sec"],
                    user_id=user_id,
                    source_audio=sn.get("source_audio"),
                    start_sec=sn.get("start_sec"),
                    end_sec=sn.get("end_sec"),
                )

        self._db.delete(secondary_name, user_id=user_id)
        self._cleanup_speaker_dir(secondary_name, user_id)

        self._auto_refine(primary_name, user_id=user_id)

        return {
            "status": "merged",
            "primary": primary_name,
            "secondary": secondary_name,
            "snippets_moved": len(secondary_snippets),
        }

    def rescan_voices_dir(
        self, user_id: str = DEFAULT_USER, progress_callback=None, force: bool = False,
    ) -> dict:
        """Scan the voices directory; add new snippets and rebuild stale voiceprints.

        Two phases:
          1. Scan: register audio files that are new (or new directories) and
             fingerprint-check every already-registered file (mtime+size) to
             detect CHANGED snippets.
          2. Rebuild: invalidate the voiceprint of any speaker whose snippet
             changed, then recompute the voiceprint for every speaker that is
             new, changed, or missing (has snippets but no voiceprint row).

        ``progress_callback(evt: dict)`` is invoked (optionally) after each
        file, so an SSE stream can reflect live scan/add/refine state.
        """
        user_dir = self._voices_dir / user_id
        if not user_dir.exists():
            return {"scanned": 0, "added": 0, "invalidated": 0, "rebuilt": 0, "speakers": []}

        def _progress(evt: dict):
            if progress_callback is None:
                return
            try:
                progress_callback(evt)
            except Exception as e:
                logger.warning("Progress callback error: %s", e)

        existing = {}
        for speaker_name in self._snippets.all_speakers(user_id=user_id):
            for sn in self._snippets.list_by_speaker(speaker_name, user_id=user_id):
                existing[sn["file_path"]] = sn

        audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mkv", ".mp4", ".webm", ".opus"}
        added = 0
        scanned = 0
        new_speakers = set()
        changed_speakers = set()
        speakers_found = []

        total_files = 0
        for speaker_dir in sorted(user_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for audio_file in speaker_dir.iterdir():
                if audio_file.is_file() and audio_file.suffix.lower() in audio_exts:
                    total_files += 1

        for speaker_dir in sorted(user_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_name = speaker_dir.name
            speakers_found.append(speaker_name)

            for audio_file in sorted(speaker_dir.iterdir()):
                if not audio_file.is_file() or audio_file.suffix.lower() not in audio_exts:
                    continue
                scanned += 1
                _progress({
                    "stage": "Scanning voices directory",
                    "speaker": speaker_name,
                    "scanned": scanned,
                    "added": added,
                    "progress": (scanned / total_files) if total_files else 0.0,
                })

                fp = str(audio_file)
                sn = existing.get(fp)
                try:
                    fst = audio_file.stat()
                    mtime, size = fst.st_mtime, fst.st_size
                except OSError:
                    mtime, size = None, None

                if sn is None:
                    try:
                        waveform, sr = load_audio(fp)
                        duration = waveform.shape[-1] / SAMPLE_RATE
                        if duration < MIN_SNIPPET_DURATION:
                            continue

                        audio_data = waveform.numpy().squeeze()
                        sf.write(fp, audio_data.astype(np.float32), sr)
                        try:
                            fst = audio_file.stat()
                            mtime, size = fst.st_mtime, fst.st_size
                        except OSError:
                            mtime, size = None, None

                        self._snippets.add(
                            speaker_name=speaker_name,
                            file_path=fp,
                            duration_sec=duration,
                            user_id=user_id,
                            file_mtime=mtime,
                            file_size=size,
                        )
                        existing[fp] = {
                            "id": None,
                            "file_mtime": mtime,
                            "file_size": size,
                        }
                        added += 1
                        new_speakers.add(speaker_name)
                        _progress({
                            "stage": f"Added snippet for {speaker_name}",
                            "speaker": speaker_name,
                            "scanned": scanned,
                            "added": added,
                            "progress": (scanned / total_files) if total_files else 0.0,
                        })
                    except Exception as e:
                        logger.warning("Failed to scan %s: %s", audio_file, e)
                else:
                    stored_mtime = sn.get("file_mtime")
                    stored_size = sn.get("file_size")
                    if stored_mtime is None or stored_size is None:
                        if sn.get("id") is not None and (mtime is not None or size is not None):
                            self._snippets.update_fingerprint(
                                sn["id"], mtime, size, user_id=user_id
                            )
                    elif (stored_mtime != mtime or stored_size != size):
                        changed_speakers.add(speaker_name)

        return self._rebuild_after_rescan(
            user_id=user_id,
            speakers_found=speakers_found,
            new_speakers=new_speakers,
            changed_speakers=changed_speakers,
            scanned=scanned,
            added=added,
            progress_callback=_progress,
            force_rebuild=force,
        )

    def _rebuild_after_rescan(
        self,
        user_id: str,
        speakers_found: list,
        new_speakers: set,
        changed_speakers: set,
        scanned: int,
        added: int,
        progress_callback=None,
        force_rebuild: bool = False,
    ) -> dict:
        """Invalidate changed voiceprints and rebuild all missing ones.

        Speakers rebuilt = new dirs with snippets ∪ speakers whose snippet
        changed (voiceprint invalidated first) ∪ speakers that have snippets
        but no voiceprint row (missing), or all speakers when force_rebuild=True.
        """
        invalidated = 0
        rebuilt = 0

        if force_rebuild:
            targets = set(speakers_found)
        else:
            targets = set(new_speakers) | set(changed_speakers)
            for speaker_name in speakers_found:
                if speaker_name in targets:
                    continue
                if self._snippets.count(speaker_name, user_id=user_id) > 0 \
                        and self._db.get(speaker_name, user_id=user_id) is None:
                    targets.add(speaker_name)

        for speaker_name in sorted(targets):
            if speaker_name in changed_speakers:
                if self._db.delete(speaker_name, user_id=user_id):
                    invalidated += 1
                progress_callback and progress_callback({
                    "stage": f"Invalidated voiceprint for {speaker_name}",
                    "speaker": speaker_name,
                    "scanned": scanned,
                    "added": added,
                })
            if progress_callback:
                progress_callback({
                    "stage": f"Rebuilding voiceprint for {speaker_name}",
                    "speaker": speaker_name,
                    "scanned": scanned,
                    "added": added,
                })
            self._auto_refine(speaker_name, user_id=user_id, progress_callback=progress_callback)
            rebuilt += 1

        return {
            "scanned": scanned,
            "added": added,
            "invalidated": invalidated,
            "rebuilt": rebuilt,
            "speakers": speakers_found,
        }

    async def auto_collect_from_diarization_async(self, *args, **kwargs) -> dict:
        """Run :meth:`auto_collect_from_diarization` off the event loop.

        Lesson 19: an SSE producer must not do heavy work inline.  Auto-collect
        decodes audio, embeds every snippet and clusters vectors; all of it is
        blocking, and every caller lives in an ``async def`` request body.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.auto_collect_from_diarization,
                                    *args, **kwargs),
        )

    def auto_collect_from_diarization(
        self,
        audio_path: str,
        segments: list[dict],
        user_id: str = DEFAULT_USER,
        source_id: str | None = None,
        learn_new: bool = False,
        vad_sections: list | None = None,
        source: AudioSegmentSource | None = None,
    ) -> dict:
        """Grow voiceprints from a diarization result.

        Two distinct behaviours, deliberately separated:

        * **collect** -- add snippets to an ALREADY-NAMED voiceprint. Only ever
          touches profiles that are registered and not pending.
        * **learn** (``learn_new=True``) -- create a *pending* profile for a
          generic ``Speaker N`` cluster that spoke long enough to be a person
          rather than a cough. The profile collects snippets immediately but
          is excluded from matching (see ``_load_known_speakers``) until the
          user names it, so this can never invent an identity.

        ``segments`` should be the boundary set the user actually SEES — for
        ``/transcribe/upload`` and ``/attribution/upload`` that is the **turns**
        from ``asr_router._prepare_turns``, not the raw diarization segments
        (phase P2 of docs/plans/boundary-and-voice-quality-plan.md, finding F1).
        A snippet cut on the raw segments is cut at a boundary the transcript
        never shows, and since a learned embedding is later used as a reference
        for cutting boundaries, that compounds run over run.

        ``vad_sections`` (the diarizer's uncollapsed Silero pass) trims every
        snippet edge onto a speech-section edge; ``source`` is a request-wide
        cached loader so the file is decoded once instead of once per segment.

        Returns ``{"collected": [...], "pending_created": [...],
        "pending_extended": [...], "learn_skipped": [...], "diagnostics":
        {...}}``. (Earlier versions returned a bare list; the call sites only
        used ``len()``, and returning the breakdown is what lets the API tell
        the user a new profile is waiting for a name. ``diagnostics`` says
        which speakers were candidates and why each one did or did not become
        material -- it is additive and nothing branches on it.)
        """
        if source_id is None:
            source_id = audio_path
        if source is None:
            source = AudioSegmentSource(audio_path)
        collected = []
        pending_created = []
        pending_extended = []
        learn_skipped: list[dict] = []
        speaker_totals = {}
        for name, info in self._snippets.all_speakers(user_id=user_id).items():
            speaker_totals[name] = info["total_duration"]

        by_speaker: dict[str, list[dict]] = {}
        to_learn: dict[str, list[dict]] = {}
        skipped_uncertain = 0
        # Per-label span counts for the diagnostics block, so "not enough
        # material" can be distinguished from "not eligible" without a
        # re-run. Cheap: incremented in the loop that already visits each span.
        span_speech: dict[str, float] = {}
        #: Refused near-misses, kept for the diagnostics block. A refusal is a
        #: span that carried a NAME and was not believed: the colleague the
        #: pipeline was fairly sure about but declined to store. Without these
        #: records, "skipped 88 segment(s) with uncertain speaker identity" is
        #: the whole story, and the fix is invisible from outside.
        refused: list[dict] = []
        refused_total = 0
        #: Reason histogram over EVERY refusal, not just the recorded ones: the
        #: records are capped for the reader's benefit, but "25 of 240 refused,
        #: 12 for a narrow margin" is only true if the tally is complete.
        refused_reasons: dict[str, int] = {}

        def _refusal(reason: str, seg: dict, dur: float) -> None:
            nonlocal refused_total
            refused_total += 1
            refused_reasons[reason] = refused_reasons.get(reason, 0) + 1
            if len(refused) >= LEARN_DIAG_MAX_REFUSALS:
                return
            try:
                start = round(float(seg.get("start", 0.0)), 2)
                end = round(float(seg.get("end", 0.0)), 2)
            except (TypeError, ValueError):
                start = end = 0.0
            record = {
                "reason": reason,
                # original_speaker is what the cluster was called before the
                # identity was stripped; for a generic label it is the only
                # handle a reader has on the refused span.
                "speaker": seg.get("original_speaker") or seg.get("speaker"),
                "start": start,
                "end": end,
                "speech_sec": round(max(0.0, dur), 2),
            }
            for src, key in (("speaker_confidence", "confidence"),
                             ("speaker_margin", "margin"),
                             ("speaker_match_dist", "match_dist")):
                val = seg.get(src)
                if val is None:
                    continue
                try:
                    record[key] = round(float(val), 3)
                except (TypeError, ValueError):
                    continue
            # The runner-up is the evidence that makes a refusal reviewable: it
            # says who else this span resembled, and how close the call was.
            alts = seg.get("alternatives") or []
            if isinstance(alts, list) and alts:
                record["alternatives"] = [
                    {"speaker": a.get("speaker"),
                     "confidence": a.get("confidence")}
                    for a in alts[:LEARN_DIAG_MAX_ALTERNATIVES]
                    if isinstance(a, dict) and a.get("speaker")
                ]
            refused.append(record)

        for seg in segments:
            sp = seg.get("speaker", "")
            if not sp:
                continue
            dur = seg.get("end", 0) - seg.get("start", 0)
            span_speech[sp] = span_speech.get(sp, 0.0) + max(0.0, dur)
            # Uncertainty policy: never feed UNKNOWN / OVERLAP / generic
            # "Speaker N" / suppressed or low-confidence segments into the DB.
            reason = auto_collect_refusal_reason(seg)
            if reason is not None:
                skipped_uncertain += 1
                # ...but a generic "Speaker N" that survived every other check
                # is a candidate to LEARN, not to collect. A generic label is
                # refused for collection by definition, so it is only a
                # *near-miss* when the learning gate turns it away as well --
                # and then the useful reason is the learning one, not
                # "generic_label".
                if learn_new and is_generic_speaker(sp):
                    learning_reason = learning_refusal_reason(seg)
                    if learning_reason is None:
                        if dur >= AUTO_COLLECT_MIN_DURATION:
                            to_learn.setdefault(sp, []).append(seg)
                        continue
                    _refusal(learning_reason, seg, dur)
                    continue
                _refusal(reason, seg, dur)
                continue
            if dur < AUTO_COLLECT_MIN_DURATION:
                continue
            by_speaker.setdefault(sp, []).append(seg)
        if skipped_uncertain:
            logger.info(
                "Auto-collect: skipped %d segment(s) with uncertain speaker identity",
                skipped_uncertain,
            )
        if refused_total:
            reasons: dict[str, int] = {}
            for r in refused:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            logger.info(
                "Auto-collect: refused %d span(s): %s%s", refused_total,
                ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())),
                "" if refused_total <= len(refused) else
                f" (showing {len(refused)}; see diagnostics.refused_truncated)",
            )

        collect_diag: list[dict] = []
        for speaker_name, segs in by_speaker.items():
            def _diag(skipped=None, snippets=0, secs=0.0):
                collect_diag.append({
                    "speaker": speaker_name,
                    "spans": len(segs),
                    "speech_sec": round(span_speech.get(speaker_name, 0.0), 1),
                    "snippets": snippets,
                    "added_sec": round(float(secs), 1),
                    "skipped": skipped,
                })
            # ONLY auto-collect for genuinely registered known speakers (e.g. "Gergely Papp", "Ismael Capel")
            # NEVER create auto-collected voiceprints for generic/unregistered "Speaker N" or timestamped clusters.
            if is_spurious_speaker_name(speaker_name):
                _diag("generic_label")
                continue
            if not self._db.get(speaker_name, user_id=user_id):
                _diag("not_registered")
                continue

            if len(segs) < AUTO_COLLECT_MIN_SPEAKER_SEGMENTS:
                _diag("too_few_spans")
                continue

            # Reprocessing replaces this file's take, so the cap below has to be
            # judged on what SURVIVES the replacement -- the clips from other
            # recordings. Judged on the pre-purge total instead, a speaker who is
            # at the cap purely because of this recording could never refresh it.
            stale_rows = self._source_snippet_rows(speaker_name, source_id, user_id=user_id)
            stale_secs = sum(float(r.get("duration_sec") or 0.0) for r in stale_rows)
            current_total = max(
                0.0, speaker_totals.get(speaker_name, 0.0) - stale_secs)
            if current_total >= AUTO_COLLECT_MAX_TOTAL_SEC:
                logger.info(
                    "Auto-collect: %s at snippet cap (%.0fs >= %.0fs), skipping",
                    speaker_name, current_total, AUTO_COLLECT_MAX_TOTAL_SEC,
                )
                _diag("at_snippet_cap")
                continue

            added_secs: list[float] = []
            fresh: list[dict] = []

            for seg in segs:
                seg_start, seg_end = self._snippet_span(
                    seg, vad_sections,
                )

                # Per-chunk length is capped by chunk_end below (MAX_SEGMENT_SEC);
                # current_total is the speaker's CUMULATIVE total, so it must only
                # be gated by MAX_TOTAL_SEC — gating it on MAX_SEGMENT_SEC made
                # every speaker with >=300s of snippets silently un-collectable.
                while seg_start < seg_end and current_total < AUTO_COLLECT_MAX_TOTAL_SEC:
                    chunk_end = min(seg_start + AUTO_COLLECT_MAX_SEGMENT_SEC, seg_end)
                    dur = chunk_end - seg_start

                    # Speaker-scoped: a row belonging to ANOTHER profile must not
                    # suppress (or be deleted by) this one's collection.
                    existing = self._snippets.find_duplicate(
                        source_id, seg_start, user_id=user_id,
                        speaker_name=speaker_name,
                    )
                    if existing:
                        if existing["duration_sec"] >= dur:
                            seg_start = chunk_end
                            continue
                        old_path = Path(existing["file_path"])
                        if old_path.exists():
                            old_path.unlink()
                        self._snippets.delete(existing["id"], user_id=user_id)
                        current_total -= existing["duration_sec"]

                    result = self.add_snippet_from_segment(
                        speaker_name=speaker_name,
                        wav_path=audio_path,
                        start_sec=seg_start,
                        end_sec=chunk_end,
                        user_id=user_id,
                        source_audio=source_id,
                        audio_source=source,
                    )
                    if "error" not in result:
                        speaker_totals[speaker_name] = current_total + dur
                        current_total += dur
                        collected.append(result)
                        fresh.append(result)
                        added_secs.append(dur)

                    seg_start = chunk_end

            _diag(snippets=len(added_secs), secs=sum(added_secs))
            # Only now, with the fresh spans on disk: the previous take of THIS
            # file goes, whatever span it was cut at, and the profile is fitted
            # once over the final set. Collecting first is what makes this safe --
            # if every span had been rejected above, the old snippets are still
            # here and nothing is lost.
            self.replace_source_contribution(
                speaker_name, source_id, user_id=user_id,
                fresh_rows=fresh, fresh_paths={c["file_path"] for c in fresh},
            )
            self._auto_refine(speaker_name, user_id=user_id)

        learn_diag: list[dict] = []
        for cluster, segs in to_learn.items():
            speech_sec = sum(max(0.0, s["end"] - s["start"]) for s in segs)

            def _ldiag(skipped=None, name=None, existing=False):
                learn_diag.append({
                    "cluster": cluster, "spans": len(segs),
                    "speech_sec": round(speech_sec, 1),
                    "outcome": (("pending_extended" if existing else "pending_created")
                                if name else "skipped"),
                    "name": name, "skipped": skipped,
                })
            parts, status = self._split_cluster_by_cohesion(
                cluster, segs, source,
            )
            if status.startswith("unverified"):
                if _learning_cfg().get("fail_closed", True):
                    # FAIL CLOSED.  A blend we could not verify must not be
                    # learned as a profile: the profile is a *reference for
                    # cutting future boundaries*, so learning an unverified
                    # blend does not just mislabel one run, it corrupts the
                    # next one (plan F1/F4).  ``learning.fail_closed: false``
                    # is the rollback to the previous fail-open behaviour.
                    logger.info(
                        "Not learning %s: cohesion check could not be verified "
                        "(%s). Nothing is learned rather than learning a blend.",
                        cluster, status,
                    )
                    learn_skipped.append({"cluster": cluster, "reason": status})
                    _ldiag(skipped=status)
                    continue
                # Rollback arm: the pre-P2 behaviour, learn the whole cluster.
                parts = [{"label": cluster, "segs": segs, "vectors": [],
                          "intra_max": None, "inter_min": None}]
            for part in parts:
                outcome = self._learn_unknown_speaker(
                    cluster=part["label"], segs=part["segs"],
                    audio_path=audio_path, user_id=user_id,
                    source_id=source_id, source=source,
                    vad_sections=vad_sections,
                    vectors=part.get("vectors") or None,
                    intra_max=part.get("intra_max"),
                    intra_mean=part.get("intra_mean"),
                    inter_min=part.get("inter_min"),
                    groups=len(parts),
                )
                if outcome.get("skipped"):
                    learn_skipped.append({"cluster": cluster,
                                          "reason": outcome["skipped"]})
                    _ldiag(skipped=outcome["skipped"])
                    continue
                if outcome.get("name"):
                    (pending_extended if outcome.get("existing")
                     else pending_created).append(outcome)
                    _ldiag(name=outcome["name"], existing=outcome.get("existing", False))

        return {
            "collected": collected,
            "pending_created": pending_created,
            "pending_extended": pending_extended,
            "learn_skipped": learn_skipped,
            "audio_loads": source.loads,
            # Why each span did or did not become material. Purely additive:
            # nothing above branches on it. Without it a colleague who never
            # becomes a profile is indistinguishable from one who was never a
            # candidate, and the only way to find out is to read the log.
            "diagnostics": {
                "spans_seen": len(segments),
                "spans_skipped_uncertain": skipped_uncertain,
                "refused": refused,
                "refused_total": refused_total,
                "refused_reasons": refused_reasons,
                "refused_truncated": refused_total > len(refused),
                "collect": collect_diag,
                "learn": learn_diag,
            },
        }

    # -- learning internals ---------------------------------------------------

    def _snippet_span(self, seg: dict, vad_sections) -> tuple[float, float]:
        """The span a snippet is actually cut from, trimmed onto VAD edges."""
        start, end = float(seg["start"]), float(seg["end"])
        if not vad_sections:
            return start, end
        return trim_span_to_vad(start, end, vad_sections)

    def _embed_span(self, source: AudioSegmentSource, seg: dict):
        """One segment embedding, or ``None`` when it is unusable."""
        try:
            # The cached source already resamples to SAMPLE_RATE and returns a
            # (1, n) shaped waveform, so the sample count is shape[-1], never
            # len().  One decode for the whole request, not one per segment.
            audio, sr = source.segment(seg["start"], seg["end"])
            if audio is None or audio.shape[-1] < int(0.5 * SAMPLE_RATE):
                return None
            vec = np.asarray(
                extract_embedding(audio, sr, self._emb_session()),
                dtype=np.float32,
            ).reshape(-1)
            if not np.all(np.isfinite(vec)):
                return None
            return vec
        except Exception as e:
            logger.debug("Cohesion embed failed for %s: %s", seg, e)
            return None

    def _split_cluster_by_cohesion(
        self, cluster: str, segs: list[dict], source: AudioSegmentSource,
    ) -> tuple[list[dict], str]:
        """Split one diarization cluster into cohesive groups before learning it.

        Returns ``(parts, status)``.  ``status`` is ``"ok"``, ``"opted_out"``
        (``learning.split_clusters`` is false — an explicit operator decision,
        so the whole cluster is one part) or ``"unverified:<why>"``.

        Note the granularity the caller now passes: the TURNS, not the raw
        diarization segments (phase P2).  Turns are coarser — ``_merge_into_turns``
        folds same-speaker segments across gaps of up to
        ``uncertainty.turn_merge_gap_sec`` — so a cluster whose segments all sit
        within that gap of each other arrives as ONE span, and one span cannot
        be checked for being a blend.  That case is reported as
        ``unverified:too_few_segments`` and nothing is learned.  It is the
        correct outcome rather than a gap: a single whole-file span has exactly
        the property step 14 exists to prevent (one long snippet dominating a
        profile), and a recording whose every second belongs to one unknown
        speaker is the case where the user should register a voiceprint
        deliberately.

        A cluster is NOT a person.  On a real 2-host podcast with inserted clips
        from several other speakers (ZO249, 3439s) the cluster the pipeline
        called ``Speaker 6`` was a blend of two voices: three of its spans
        ranked one registered colleague nearest (cosine 0.533-0.589) and three
        others ranked the host nearest (0.386-0.531), with the two groups
        0.368-0.448 apart while each group sat at 0.175-0.291 internally.
        Learning that as ONE profile would have asserted a single name for two
        people — and worse, the blended profile is then used by
        ``_refine_boundaries_with_vad`` and the second pass, so it corrupts the
        turn boundaries of every later run, not just the printed label.

        So each contributing segment is embedded on its own and the cluster is
        re-clustered at ``learning.max_intra_cluster_dist``.  Every cohesive
        group becomes its own pending profile, and the user is told how many
        voices the cluster contained.

        **Fail closed (phase P2).**  Every path that previously returned the
        whole cluster — no embedding session, no sklearn, fewer than two usable
        vectors, a clustering exception, or fewer than two segments — now
        returns an empty part list plus an ``unverified:`` status, and the
        caller learns nothing.  A blend within the threshold is undetectable by
        construction, so "cannot tell" must never become "learn it anyway".

        **``min_inter_dist`` re-joins, it does not refuse.**  When the split
        produces groups whose centroids sit closer together than the bar that
        produced them, the division is an artefact.  The honest reading of that
        is ONE voice, so the groups are merged back into a single part and
        handed to the ordinary per-group purity gate.  This used to return
        ``unverified:groups_too_close`` and learn nothing, which in practice
        discarded the largest clusters — on 2026-10-07_17-00-32.mp4 fourteen
        clusters were refused at 0.175-0.281 while the two that were learned had
        intra-group means of 0.170 and 0.179, i.e. the refused "groups" were
        closer together than a learned single speaker is spread internally.  A
        real blend is still refused, now by ``max_profile_intra_dist``.

        Threshold provenance: 0.32 sits between the measured within-group
        maximum (0.291) and the between-group minimum (0.368) on that file.  It
        is one file with six sampled spans, so it is a judgement call, not a
        calibrated value — hence ``learning.split_clusters`` exists to turn the
        whole thing off, and ``learning.fail_closed`` to restore the old
        fail-open behaviour without turning the split off.
        """
        cfg = _learning_cfg()
        if not cfg.get("split_clusters", True):
            return ([{"label": cluster, "segs": segs, "vectors": [],
                      "intra_max": None, "inter_min": None}], "opted_out")
        if self._emb_session() is None:
            return [], "unverified:no_embedding_session"
        if len(segs) < 2:
            return [], "unverified:too_few_segments"
        try:
            from sklearn.cluster import AgglomerativeClustering
        except Exception:
            return [], "unverified:no_sklearn"

        vectors, kept = [], []
        for seg in sorted(segs, key=lambda s: s["start"]):
            vec = self._embed_span(source, seg)
            if vec is None:
                continue
            vectors.append(vec)
            kept.append(seg)

        if len(vectors) < 2:
            return [], "unverified:too_few_embeddings"

        try:
            threshold = float(cfg.get("max_intra_cluster_dist", 0.32))
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold,
                metric="cosine", linkage="average",
            ).fit_predict(np.stack(vectors))
        except Exception as e:
            logger.warning("Cohesion split failed for %s: %s", cluster, e)
            return [], "unverified:clustering_failed"

        groups: dict[int, list[dict]] = {}
        vecs: dict[int, list[np.ndarray]] = {}
        for label, seg, vec in zip(labels, kept, vectors):
            groups.setdefault(int(label), []).append(seg)
            vecs.setdefault(int(label), []).append(vec)

        ordered = sorted(
            groups.items(),
            key=lambda kv: -sum(s["end"] - s["start"] for s in kv[1]),
        )
        # Drop one-span groups BEFORE anything else. A singleton cannot become
        # a profile (LEARN_MIN_SEGMENTS is 2), so keeping it only spends an
        # embed and a log line to be refused as ``too_few_spans`` later -- and on
        # 2026-09-04 12-08-59.mp4 ten of Speaker 8's twelve groups were
        # singletons, which buried the one line that mattered.
        #
        # They are excluded from the inter-group check below too: a single span
        # whose vector is a poor sample drags every centroid comparison toward
        # "these groups are really one voice" and suppresses the whole cluster.
        # A group of one carries no evidence about cohesion either way.
        singletons = [g for _, g in ordered if len(g) < 2]
        ordered = [(k, g) for k, g in ordered if len(g) >= 2]
        if singletons:
            logger.info(
                "Cluster %s: dropped %d one-span group(s) (%d span(s) total) "
                "before learning -- a single span cannot become a profile "
                "(LEARN_MIN_SEGMENTS=%d)",
                cluster, len(singletons), sum(len(g) for g in singletons),
                LEARN_MIN_SEGMENTS,
            )
        if not ordered:
            return [], "unverified:too_few_segments"

        parts = []
        for i, (key, group) in enumerate(ordered):
            parts.append({
                "label": cluster if i == 0 else "%s#%d" % (cluster, i + 1),
                "segs": group,
                "vectors": vecs.get(key, []),
                "intra_max": _max_pairwise_distance(vecs.get(key)),
                "intra_mean": _mean_pairwise_distance(vecs.get(key)),
                "inter_min": None,
            })
        centroids = [_centroid(p["vectors"]) for p in parts]
        if len(parts) > 1:
            min_inter = float(cfg.get("min_inter_dist", 0.32))
            for i, p in enumerate(parts):
                others = [centroids[j] for j in range(len(parts)) if j != i]
                mine = centroids[i]
                if mine is None or any(o is None for o in others):
                    p["inter_min"] = None
                    continue
                p["inter_min"] = min(_cosine_distance(mine, o) for o in others)
            worst = max(
                (p["inter_min"] for p in parts if p["inter_min"] is not None),
                default=None,
            )
            if worst is not None and worst < min_inter:
                # The groups we just made sit CLOSER to each other than the bar
                # that produced them: the split is an artefact, not evidence of
                # two people, and learning it as several profiles would invent
                # a division.
                #
                # So RE-JOIN them into one group instead of refusing the
                # cluster. Refusing was measured to be the wrong answer, and it
                # failed hardest on exactly the clusters worth having: on
                # 2026-10-07_17-00-32.mp4 Speaker 10 split into groups 0.186
                # apart -- closer together than the 0.170-0.179 INTRA-group
                # MEAN of Speaker 1 and Speaker 9, both of which were learned
                # as single speakers from the same run.  Almost every cluster
                # in that file was refused this way (0.175-0.281 across 14
                # clusters), so the largest, richest speakers were precisely
                # the ones dropped.
                #
                # "Too close" is evidence of ONE voice, not of an unknown one,
                # and a single group is the only honest reading of it.  The
                # re-joined group then faces the same gate as any other --
                # ``max_profile_intra_dist`` on the MEAN pairwise distance --
                # so a genuine blend still fails closed as ``impure_group``.
                # The blend that motivated the check is still caught: ZO249's
                # groups sat 0.368 apart, outside this window, so they split into
                # two profiles instead of merging.
                #
                # The one-span groups dropped above come back too: they were
                # only dropped because a singleton cannot become a profile, and
                # inside one group that objection does not apply.
                #
                # The merged group is gated exactly like any other, on the mean
                # alone.  It briefly also had a maximum-based gate, which real
                # data killed -- see the note in ``_learn_unknown_speaker``.
                merged_vectors = [v for key, _g in ordered
                                  for v in vecs.get(key, [])]
                merged = {
                    "label": cluster,
                    "segs": sorted(segs, key=lambda s: s["start"]),
                    "vectors": merged_vectors,
                    "intra_max": _max_pairwise_distance(merged_vectors),
                    "intra_mean": _mean_pairwise_distance(merged_vectors),
                    "inter_min": None,
                }
                logger.info(
                    "Cohesion split of %s not trusted: its groups are only %.3f "
                    "apart (min_inter_dist=%.2f), inside the spread of a single "
                    "voice -- re-joined into ONE group of %d span(s) (mean %.3f, "
                    "max %.3f) instead of refusing the cluster",
                    cluster, worst, min_inter, len(merged["segs"]),
                    -1.0 if merged["intra_mean"] is None else merged["intra_mean"],
                    -1.0 if merged["intra_max"] is None else merged["intra_max"],
                )
                return [merged], "ok"
        if len(parts) > 1:
            logger.info(
                "Cluster %s contains more than one voice: split into %d cohesive "
                "group(s) at max_intra_cluster_dist=%.2f (sizes %s, intra_mean %s, "
                "intra_max %s, inter_min %s) -- each is learned as its own "
                "pending profile",
                cluster, len(parts), threshold,
                [len(p["segs"]) for p in parts],
                [None if p["intra_mean"] is None else round(p["intra_mean"], 3)
                 for p in parts],
                [None if p["intra_max"] is None else round(p["intra_max"], 3)
                 for p in parts],
                [None if p["inter_min"] is None else round(p["inter_min"], 3)
                 for p in parts],
            )
        return parts, "ok"

    def _purity(self, intra_max, inter_min, groups: int,
                intra_mean: float | None = None) -> dict:
        ratio = None
        if intra_max is not None and inter_min:
            ratio = round(float(intra_max) / float(inter_min), 4)
        return {
            "intra_max": None if intra_max is None else round(float(intra_max), 4),
            # The gated statistic, kept next to the diagnostic one so the
            # numbers can be compared after the fact.
            "intra_mean": None if intra_mean is None else round(float(intra_mean), 4),
            "inter_min": None if inter_min is None else round(float(inter_min), 4),
            "purity_ratio": ratio,
            "groups": int(groups or 1),
        }

    def _find_pending_to_extend(
        self, centroid, user_id: str, source_id: str, cluster: str,
    ) -> tuple[str | None, float | None]:
        """Which pending profile, if any, is this cluster the SAME person as?

        The primary key is the **voice**, not the source filename (phase P2,
        finding F4).  Keying on the filename stem meant the same colleague in a
        second recording minted a SECOND pending profile, and the second
        ``confirm`` was then refused because the name was taken -- leaving the
        user with no way forward except the merge dialog.

        One deliberate exception: a pending profile learned from **this same
        recording** under a *different* cluster label is not extended by voice
        alone.  The clustering threshold already decided those are two speakers
        in this file, and overriding that decision on embedding similarity is
        how two people end up merged.  (It also keeps the "two clusters, one
        recording, two profiles" behaviour that
        ``test_the_same_recording_extends_one_pending_profile`` pins.)
        """
        if centroid is None:
            return None, None
        rows = self._db.list_all(user_id=user_id, include_pending=True)
        prefix = _origin_prefix(source_id)
        label_tokens = _name_tokens(cluster)
        thr = float(_learning_cfg().get("extend_max_dist", 0.32))
        ranked = []
        for name, vp in sorted(rows.items()):
            if not is_pending_profile(name, vp):
                continue
            emb = vp.get("embedding")
            if emb is None or len(emb) == 0:
                continue
            ranked.append((_cosine_distance(centroid, emb), name))
        ranked.sort()
        for dist, name in ranked:
            if dist > thr:
                break
            same_source = prefix and prefix in name
            if same_source and label_tokens and not _tokens_contain(
                    _name_tokens(name), label_tokens):
                continue
            logger.info(
                "Pending profile %r is %.3f from cluster %r (extend_max_dist=%.2f)"
                " -- extending it rather than minting a new one",
                name, dist, cluster, thr,
            )
            return name, dist
        return None, (ranked[0][0] if ranked else None)

    def _learn_unknown_speaker(
        self,
        cluster: str,
        segs: list[dict],
        audio_path: str,
        user_id: str = DEFAULT_USER,
        source_id: str | None = None,
        source: AudioSegmentSource | None = None,
        vad_sections: list | None = None,
        vectors: list | None = None,
        intra_max: float | None = None,
        intra_mean: float | None = None,
        inter_min: float | None = None,
        groups: int = 1,
    ) -> dict:
        """Create or extend a PENDING profile for an unidentified cluster.

        Extends an existing pending profile when one already covers this person:
        by embedding distance first (so a colleague in a *second recording*
        accumulates into the first profile), and by the source-filename/label
        name match as the fallback.
        """
        if source_id is None:
            source_id = audio_path
        speech = sum(max(0.0, seg["end"] - seg["start"]) for seg in segs)
        if len(segs) < LEARN_MIN_SEGMENTS or speech < LEARN_MIN_SPEECH_SEC:
            # Reported, not swallowed: "not enough speech" is the single most
            # common reason a colleague never gets a profile, and a bare {}
            # made it indistinguishable from every other silent no-op.
            if len(segs) < LEARN_MIN_SEGMENTS:
                return {"skipped": "too_few_spans", "spans": len(segs), "speech_sec": round(speech, 1)}
            return {"skipped": "insufficient_speech", "spans": len(segs),
                    "speech_sec": round(speech, 1)}
        cfg = _learning_cfg()
        # FAIL CLOSED, per group: a group whose own segments disagree more than
        # ``max_profile_intra_dist`` is not one person, so nothing is learned
        # from it even though its siblings passed.
        #
        # The gate is on the MEAN pairwise distance, because that is the
        # statistic ``max_intra_cluster_dist`` is expressed in: the split is
        # average-linkage agglomerative clustering, which merges while the mean
        # stays under the bar. Gating on intra_max instead measured a maximum
        # over n*(n-1)/2 pairs, which no average-linkage threshold bounds, and
        # the excess grows with span count -- so the biggest, richest group was
        # the one most likely to be refused. intra_max is still computed and
        # logged; it is evidence, not a gate.
        max_intra = float(cfg.get("max_profile_intra_dist", 0.32))
        if intra_mean is not None and intra_mean > max_intra:
            logger.info(
                "Not learning %r: its own %d span(s) average %.3f apart "
                "(max %.3f, max_profile_intra_dist=%.2f) -- that is not one person",
                cluster, len(segs), intra_mean,
                -1.0 if intra_max is None else intra_max, max_intra,
            )
            return {"skipped": "impure_group"}

        # There is deliberately NO maximum-based gate here, on any group.
        #
        # A maximum over n(n-1)/2 pairs grows with the span count, so it rises
        # with the richness of the cluster -- which is why gating every group on
        # it refused every cluster of 2026-09-04 12-08-59.mp4 (22 spans, mean
        # 0.21, max 0.379).  A re-joined group briefly had one of its own
        # (``max_profile_intra_max``), justified by a synthetic blend of two
        # internally-perfect voices: 6 spans, mean 0.252, max 0.42.  Real data
        # killed it.  ``Speaker 9`` on 2026-10-07_17-00-32.mp4 is 20 spans,
        # mean 0.255, max 0.463 -- the maximum of a speaker the user had ALREADY
        # learned (18 spans, max 0.3631) is *higher* than the blend's, and the
        # mean does not separate them either.  No threshold admits one and
        # rejects the other, so the gate could only choose which real speakers
        # to throw away, and it chose the largest.
        #
        # The blend it was meant to catch cannot reach this code anyway: the
        # re-join fires only when every group's nearest neighbour is under
        # ``min_inter_dist``, and ZO249 -- the real blend this function exists
        # for -- had its groups 0.368 apart, so it splits into two profiles and
        # never gets here.  The merge window and the blend window are disjoint.
        #
        # ``intra_max`` is still computed and logged.  It is evidence, not a
        # gate.

        purity = self._purity(intra_max, inter_min, groups, intra_mean)
        # Exact TOKEN match on the sanitised cluster label: "Speaker 1" must not
        # match "Speaker 10" (a plain substring test did exactly that).
        label_tokens = _name_tokens(re.sub(r"[^A-Za-z0-9._-]+", "_", cluster or "").strip("_"))

        rows = self._db.list_all(user_id=user_id, include_pending=True)
        known = set(rows) | set(self._snippets.all_speakers(user_id=user_id))
        prefix = _origin_prefix(source_id)

        target, dist = (None, None)
        if vectors:
            target, dist = self._find_pending_to_extend(
                _centroid(vectors), user_id, source_id, cluster,
            )
        if not target:
            for cand in sorted(known):
                if not is_pending_profile(cand, rows.get(cand)):
                    continue
                if prefix in cand and (not label_tokens or _tokens_contain(
                        _name_tokens(cand), label_tokens)):
                    target = cand
                    break
        existing = bool(target)
        if not target:
            target = pending_profile_name(source_id, cluster)
            n = 2
            # Snippet DIRECTORIES count as taken too, not just the DB row: a
            # profile learned while the embedding session was unavailable has
            # snippets and no row, and the old check minted "... #2" over it.
            while target in known or (self._voices_dir / user_id / target).exists():
                target = f"{pending_profile_name(source_id, cluster)} #{n}"
                n += 1
        added = []
        src_hash = generate_segment_hash(audio_path)
        # Extending must be IDEMPOTENT: the same recording re-processed must not
        # add the same snippet twice, or the profile fills with duplicates and
        # the embedding is computed over doubly-counted audio. The guard covers
        # the OTHER recordings only -- this one's previous take is about to be
        # replaced wholesale (see the purge below), and a span that did not move
        # must still be re-cut, or the guard would veto the refresh and the purge
        # would then delete the only copy.
        stale_rows = self._source_snippet_rows(target, source_id, user_id=user_id)
        stale_ids = {x["id"] for x in stale_rows}
        stale_secs = sum(float(x.get("duration_sec") or 0.0) for x in stale_rows)
        have = {
            os.path.basename(x["file_path"])
            for x in self._snippets.list_by_speaker(target, user_id=user_id)
            if x.get("id") not in stale_ids
        }
        # Per-snippet and per-profile caps on the LEARN path.  The collect-only
        # AUTO_COLLECT_MAX_* pair did not apply here, so one 200s turn could
        # dominate a profile of 5s snippets.
        max_snippet = float(cfg.get("max_snippet_sec") or 0.0)
        max_total = float(cfg.get("max_profile_total_sec") or 0.0)
        # The cap is judged on what survives the replacement, for the same reason
        # as on the collect path: a profile sitting at the cap because of THIS
        # recording must still be able to refresh it.
        current_total = max(0.0, sum(
            float(x.get("duration_sec") or 0.0)
            for x in self._snippets.list_by_speaker(target, user_id=user_id)
        ) - stale_secs)
        if max_total and current_total >= max_total:
            logger.info(
                "Pending profile %r is at its learn cap (%.0fs >= %.0fs)",
                target, current_total, max_total,
            )
        for seg in sorted(segs, key=lambda x: x["start"]):
            seg_start, seg_end = self._snippet_span(seg, vad_sections)
            # Mirror add_snippet()'s filename exactly (minus the duration part):
            # the hash is of the source audio and the time part of the start.
            key = f"{src_hash}_{format_time_short(seg['start'])}"
            if any(n.startswith(key + "_") for n in have):
                continue
            while seg_start < seg_end and not (max_total and current_total >= max_total):
                chunk_end = seg_end if not max_snippet else min(
                    seg_start + max_snippet, seg_end,
                )
                if max_total:
                    chunk_end = min(chunk_end, seg_start + (max_total - current_total))
                have.add(key)
                r = self.add_snippet_from_segment(
                    speaker_name=target, wav_path=audio_path,
                    start_sec=seg_start, end_sec=chunk_end, user_id=user_id,
                    source_audio=source_id, audio_source=source,
                )
                if "error" not in r:
                    added.append(r)
                    current_total += (chunk_end - seg_start)
                seg_start = chunk_end
        if not added:
            # Nothing new (a repeat of the same recording). Still report the
            # profile: the user needs to know this speaker is STILL unnamed,
            # and silently returning nothing hides a pending profile that is
            # accumulating nothing. `snippets: 0` keeps it honest.
            if not existing:
                # A NEW profile that got no snippets at all. Every snippet was
                # rejected (too short after VAD trimming, or the learn cap was
                # already reached), so say which -- otherwise this looks
                # identical to "nothing was a candidate" at the call site.
                return {"skipped": "at_learn_cap" if current_total else "snippets_rejected",
                        "spans": len(segs), "speech_sec": round(speech, 1)}
            return {"name": target, "cluster": cluster, "snippets": 0,
                    "speech_sec": round(speech, 1), "existing": True,
                    "purity": purity}
        # This recording's previous take is replaced, not accumulated: the fresh
        # spans are on disk, so deleting the old rows can no longer lose audio.
        # Profiles built from OTHER recordings keep every clip they had.
        replaced = self.replace_source_contribution(
            target, source_id, user_id=user_id,
            fresh_rows=added, fresh_paths={r["file_path"] for r in added},
        )
        # Build the profile so it is ready the moment the user names it. This
        # writes a voiceprint row flagged pending, which matching ignores.
        try:
            self._auto_refine(target, user_id=user_id, purity=purity)
        except Exception as e:
            logger.warning("Pending profile refine failed for %s: %s", target, e)
        # MUST be set AFTER the refine: _auto_refine CREATES the row, and a new
        # row defaults to pending=False. Without this the profile would be
        # matchable from its very first moment -- the exact false attribution
        # this whole feature exists to avoid.
        self._db.set_pending(target, True, user_id=user_id)
        logger.info(
            "Learned new speaker: %s (%s, %d snippet(s), %.1fs, purity %s, "
            "%d stale from this recording replaced) -- PENDING, excluded from "
            "matching until it is named",
            target, cluster, len(added), speech, purity, replaced["removed"],
        )
        return {"name": target, "cluster": cluster, "snippets": len(added),
                "speech_sec": round(speech, 1), "existing": existing,
                "replaced_snippets": replaced["removed"],
                "purity": purity, "extend_distance": dist}

    def _rename_snippet_dir(self, old_name: str, new_name: str,
                            user_id: str = DEFAULT_USER) -> bool:
        """Move a speaker's snippets + DB rows to a new name."""
        old_dir = self._voices_dir / user_id / old_name
        new_dir = self._voices_dir / user_id / new_name
        if old_dir.exists() and not new_dir.exists():
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            old_dir.rename(new_dir)
        moved = 0
        for sn in self._snippets.list_by_speaker(old_name, user_id=user_id):
            try:
                # Re-point at the NEW directory. Copying file_path verbatim is
                # the obvious thing and it is wrong: the directory just moved,
                # so every later load of that snippet 404s on disk.
                old_path = sn["file_path"]
                new_path = str(new_dir / os.path.basename(old_path)) \
                    if str(old_dir) in old_path else old_path
                self._snippets.add(
                    speaker_name=new_name, file_path=new_path,
                    duration_sec=sn.get("duration_sec", 0.0), user_id=user_id,
                    source_audio=sn.get("source_audio"),
                    start_sec=sn.get("start_sec"), end_sec=sn.get("end_sec"),
                    file_mtime=sn.get("file_mtime"), file_size=sn.get("file_size"))
                self._snippets.delete(sn["id"], user_id=user_id)
                moved += 1
            except Exception as e:
                logger.warning("Could not move snippet %s: %s", sn.get("id"), e)
        return bool(moved or new_dir.exists())

    # ── Integrity audit and cleanup ────────────────────────────────────
    #
    # voiceprint/audit.py is read-only and reports; this is the half that acts
    # on two of its findings. It is a separate, explicit call -- never startup,
    # never automatic -- because every action here is destructive or changes a
    # profile that the matcher reads from.

    def integrity_report(self, user_id: str = DEFAULT_USER) -> dict:
        """Run the audit for one user without changing anything."""
        from asr_mcp.voiceprint import audit as vp_audit
        return vp_audit.audit_user(
            user_id, self._db.list_all(user_id=user_id, include_pending=True),
            self._snippets, self.voices_dir,
        )

    def cleanup_integrity(self, user_id: str = DEFAULT_USER, *,
                          duplicates: bool = True,
                          stale_totals: bool = True,
                          dead_snippets: bool = True,
                          dry_run: bool = True) -> dict:
        """Repair the findings that have one obvious correct answer.

        Deliberately narrow. Three of the audit's eight findings are safe to fix
        here and five are not:

        - ``duplicate_snippet_content``: the two files are byte-identical, so
          one of them carries no information the other lacks. Deleting the
          HIGHER-id copy is provably lossless, and the profile is then
          re-refined over what remains. Fixable.
        - ``missing_snippet_files``: the row names a file that is not there, so
          the row describes audio that does not exist. Deleting the row removes
          a claim that can never be honoured and loses nothing, because the
          audio it referred to is already gone. A rescan does not do this: it
          adopts files it finds, it never removes rows whose file has vanished.
          Fixable -- and, measured on a real store, the finding that actually
          needs cleaning: 73 dead rows were left behind by a wav->flac
          migration, all with ``source_audio IS NULL``, invisible to the
          "one file, one contribution" replacement because a row with no
          recorded source never matches.
        - ``stale_speech_totals``: a number disagrees with the sum of the
          durations it is supposed to be the sum of. The snippet durations are
          measured from the audio; the stored total is a leftover. Recomputing
          it cannot lose anything. Fixable.
        - ``near_duplicate_profiles``, ``name_collisions``: two rows that LOOK
          like one person. Only the user knows whether that is one person
          registered twice or two colleagues, and merging destroys the evidence
          for the wrong answer. NOT fixable here -- use
          ``POST /pending/{name}/merge`` or delete one by name.
        - ``profiles_without_snippets``, ``snippets_without_profile``,
          ``orphan_snippet_files``: each needs a judgement about intent
          (re-register? re-learn? which file is the real one?). NOT fixable
          here -- the first two are what ``voiceprint/rebuild.py`` acts on.

        ``dry_run=True`` is the default and returns the plan without touching
        anything, so the first call is always safe.
        """
        from asr_mcp.voiceprint import audit as vp_audit

        report = self.integrity_report(user_id)
        result = {
            "dry_run": bool(dry_run),
            "user_id": user_id,
            "duplicate_snippets": [],
            "dead_snippets": [],
            "stale_totals": [],
            "refined": [],
            "unfixed": {
                key: report.get(key) or []
                for key in vp_audit.FINDING_KEYS
                if key not in ("duplicate_snippet_content", "stale_speech_totals",
                               "missing_snippet_files")
            },
        }

        speakers_to_refine: set[str] = set()
        #: Snippet ids this run removes (or would remove, when dry_run), per
        #: speaker. The total_speech_sec repair below has to be computed over
        #: the SURVIVING snippet set: the audit ran before the deletions, so its
        #: figure still counts the copies this run just threw away, and writing
        #: it would leave the number stale by exactly the audio that was removed.
        doomed_ids: dict[str, set] = {}

        if duplicates:
            for finding in report.get("duplicate_snippet_content") or []:
                dup_id = finding.get("duplicate_id")
                name = finding.get("speaker")
                if dup_id is None or not name:
                    result["unfixed"]["duplicate_snippet_content"].append(finding)
                    continue
                doomed_ids.setdefault(name, set()).add(dup_id)
                entry = {
                    "speaker": name,
                    "kept_snippet_id": finding.get("first_id"),
                    "kept_file": finding.get("first"),
                    "deleted_snippet_id": dup_id,
                    "deleted_file": finding.get("duplicate"),
                }
                if dry_run:
                    result["duplicate_snippets"].append(entry)
                    continue
                deleted = self._delete_snippet_row(dup_id, name, user_id)
                entry["deleted"] = deleted
                if deleted:
                    speakers_to_refine.add(name)
                else:
                    entry["error"] = "row or file could not be removed"
                    # Nothing went away, so this row still counts towards the
                    # speaker's total: forgetting it would under-report it.
                    doomed_ids[name].discard(dup_id)
                result["duplicate_snippets"].append(entry)

        if dead_snippets:
            # Deleted per speaker, not per finding, and capped: the audit caps a
            # finding list at 50 entries, so a store with hundreds of dead rows
            # needs more than one pass to converge, and a single unbounded
            # delete is not something to do off a report without a limit.
            missing_by_speaker: dict[str, set] = {}
            for finding in report.get("missing_snippet_files") or []:
                name = finding.get("speaker")
                snippet_id = finding.get("snippet_id")
                if not name or snippet_id is None:
                    continue
                missing_by_speaker.setdefault(name, set()).add(snippet_id)
            for name in sorted(missing_by_speaker):
                for snippet_id in sorted(missing_by_speaker[name])[:DEAD_SNIPPET_CAP]:
                    doomed_ids.setdefault(name, set()).add(snippet_id)
                    entry = {"speaker": name, "snippet_id": snippet_id}
                    if dry_run:
                        result["dead_snippets"].append(entry)
                        continue
                    deleted = self._delete_snippet_row(snippet_id, name, user_id)
                    entry["deleted"] = deleted
                    if deleted:
                        speakers_to_refine.add(name)
                    else:
                        # The row survived, so it still counts towards the
                        # speaker's total: forgetting it would under-report it.
                        doomed_ids[name].discard(snippet_id)
                    result["dead_snippets"].append(entry)

        if stale_totals:
            for finding in report.get("stale_speech_totals") or []:
                name = finding.get("speaker")
                # Which speakers are stale comes from the audit; what the number
                # should be is recomputed here, over the rows that will remain.
                seconds = self._surviving_speech_seconds(
                    name, user_id, doomed_ids.get(name or "", set()))
                if not name or seconds <= 0:
                    result["unfixed"]["stale_speech_totals"].append(finding)
                    continue
                entry = {
                    "speaker": name,
                    "was": finding.get("total_speech_sec"),
                    "now": round(seconds, 1),
                }
                if dry_run:
                    result["stale_totals"].append(entry)
                    continue
                # Total only. Touching the embedding here would be a refine,
                # and a refine changes who this profile matches -- which is the
                # user's decision, not a side effect of fixing a number.
                self._db.update_speech_total(name, seconds, user_id=user_id)
                entry["updated"] = True
                result["stale_totals"].append(entry)

        if speakers_to_refine:
            for name in sorted(speakers_to_refine):
                # Snippets went away, so the profile no longer describes the
                # audio it is built from. _auto_refine returns silently without
                # an embedding session, which is why the endpoint uses
                # ensure_gpu=True: a "refined" line here has to be true.
                self._auto_refine(name, user_id=user_id)
                result["refined"].append(name)

        result["ok"] = True
        return result

    def _surviving_speech_seconds(self, speaker_name: str, user_id: str,
                                  excluded_ids) -> float:
        """Seconds of stored snippet audio for one speaker, minus some rows.

        Read from the snippet rows rather than from the audit's figure, because
        the audit is a snapshot taken before this run removed anything. A caller
        that passes the ids it is about to delete therefore gets the number the
        profile *will* describe, in dry-run and applied mode alike.
        """
        if not speaker_name:
            return 0.0
        excluded = set(excluded_ids or ())
        rows = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        return round(sum(
            float(r.get("duration_sec") or 0.0)
            for r in rows if r.get("id") not in excluded), 1)

    def _delete_snippet_row(self, snippet_id: int, speaker_name: str,
                            user_id: str) -> bool:
        """Remove one snippet row and its file. Returns True only if both went.

        Deliberately not ``delete_snippet``: that one re-refines the profile on
        every call, so a run that removes sixteen copies of the same span would
        re-embed the whole speaker sixteen times. The caller refines once, at
        the end, over the final snippet set.

        The file is located with the audit's resolver rather than from the
        stored path, because that path is absolute inside the container
        (``/app/voices/...``) and does not exist when the server runs on the
        host -- unlinking it blindly would delete the row, leave the file, and
        turn a duplicate into an orphan.
        """
        from asr_mcp.voiceprint import audit as vp_audit

        row = self._snippets.get(snippet_id, user_id=user_id)
        if not row:
            logger.warning("Cleanup: snippet %s is already gone", snippet_id)
            return False
        if row.get("speaker_name") != speaker_name:
            # The audit ran a moment ago, but never delete audio for a speaker
            # the caller did not name -- a stale report must not be able to
            # redirect the deletion.
            logger.warning(
                "Cleanup: snippet %s belongs to %r, not %r -- skipped",
                snippet_id, row.get("speaker_name"), speaker_name)
            return False
        resolved = vp_audit.resolve_snippet_path(
            row.get("file_path"), self.voices_dir)
        if resolved is not None and resolved.is_file():
            try:
                resolved.unlink()
            except OSError as e:
                logger.warning("Cleanup: could not unlink %s: %s", resolved, e)
                return False
        elif resolved is None:
            logger.warning(
                "Cleanup: snippet %s path %r does not resolve under %s; "
                "removing the row only", snippet_id, row.get("file_path"),
                self.voices_dir)
        return bool(self._snippets.delete(snippet_id, user_id=user_id))

    def _purge_dead_snippet_rows(self, speaker_name: str,
                                 user_id: str) -> dict:
        """Delete this speaker's snippet rows whose audio is gone.

        A row is dead when its file cannot be resolved under the voices dir. The
        row claims audio that does not exist, so it can never be loaded, never
        embedded and never played -- keeping it only means the audit reports it
        forever and every rebuild wastes time failing to open it.

        What is NOT touched: rows whose file merely fails to load (a corrupt
        FLAC is still a file the user can inspect and re-cut), rows with no
        recorded source, and any file that does resolve -- this deletes rows
        only, never audio.
        """
        from asr_mcp.voiceprint import audit as vp_audit

        rows = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        dead = []
        for row in rows:
            if vp_audit.resolve_snippet_path(row.get("file_path"),
                                             self.voices_dir) is None:
                dead.append(row)
        removed = 0
        freed = 0.0
        for row in dead[:DEAD_SNIPPET_CAP]:
            seconds = float(row.get("duration_sec") or 0.0)
            if self._snippets.delete(row["id"], user_id=user_id):
                removed += 1
                freed += seconds
        if removed:
            logger.warning(
                "Voiceprint cleanup: removed %d dead snippet row(s) of %r/%s "
                "(%.1fs of audio that is no longer on disk); %d row(s) remain",
                removed, user_id, speaker_name, freed,
                len(self._snippets.list_by_speaker(speaker_name, user_id=user_id)))
        return {"dead_snippets": removed, "seconds": round(freed, 1)}

    def pending_profiles(self, user_id: str = DEFAULT_USER) -> list[dict]:
        """Unnamed profiles awaiting a name, newest first."""
        out = []
        vps = self._db.list_all(user_id=user_id, include_pending=True)
        totals = self._snippets.all_speakers(user_id=user_id)
        # Union of rows and snippet dirs: a profile learned while embedding was
        # unavailable has snippets but no row.
        for name in sorted(set(vps) | set(totals)):
            vp = vps.get(name, {})
            if not is_pending_profile(name, vp):
                continue
            sn = totals.get(name, {})
            out.append({
                "name": name,
                "snippet_count": sn.get("count", 0),
                "total_duration_sec": round(sn.get("total_duration", 0.0), 2),
                # A profile learned while embedding was unavailable has no row,
                # so created_at is genuinely absent -- "" (the schema's default),
                # not None, or the whole listing fails validation on one such
                # entry instead of the nine healthy ones.
                "created_at": vp.get("created_at") or "",
                "purity": vp.get("purity") or None,
            })
        return out

    def pending_candidates(self, pending_name: str, user_id: str = DEFAULT_USER,
                           limit: int = 5) -> list[dict]:
        """Which registered speakers does this pending profile resemble?

        A pending profile is very often a person who is ALREADY registered --
        the user themselves picked up by the loopback, or a colleague they
        registered months ago. ``confirm_pending`` rightly refuses a name that
        is already taken, but the correct action there is a merge, not a
        rename, and the user has no way to reach one: the merge dialog only
        lists named speakers.

        Returns every registered profile ranked by combined distance, with the
        live-turn gates applied as advisory flags rather than a filter -- the
        caller decides what to show.  ``likely`` marks the candidates that
        clear ``live_attribution.min_match_confidence`` /
        ``min_match_margin``; a pending profile scored against a *profile* is
        a cleaner comparison than a 3s live turn is, so those bars are
        conservative.

        Returns ``[]`` when the profile has no embedding yet (learned while the
        embedding session was unavailable); there is nothing to compare.
        """
        from asr_mcp.streaming.attribution import config as live_cfg
        from asr_mcp.speaker.matcher import find_best_match

        vp = self._db.get(pending_name, user_id=user_id)
        emb = (vp or {}).get("embedding")
        if emb is None or len(emb) == 0:
            return []
        # Registered profiles only: a pending profile has no identity to lend.
        registered = self._db.list_all(user_id=user_id, include_pending=False)
        if not registered:
            return []
        _, _, _, distances = find_best_match(
            emb, (vp or {}).get("pitch_hz", 0.0), (vp or {}).get("energy_rms", 0.0),
            registered,
        )
        ranked = sorted(distances.items(), key=lambda kv: kv[1]["combined"])
        cfg = live_cfg()
        min_conf = float(cfg["min_match_confidence"])
        min_margin = float(cfg["min_match_margin"])
        out = []
        for i, (name, d) in enumerate(ranked[:max(1, limit)]):
            runner_up = ranked[i + 1][1]["combined"] if i + 1 < len(ranked) else 1.0
            margin = runner_up - d["combined"]
            out.append({
                "name": name,
                "distance": round(d["combined"], 4),
                "confidence": round(d.get("confidence", 0.0), 3),
                "margin": round(margin, 4),
                "snippet_count": self._snippets.all_speakers(
                    user_id=user_id).get(name, {}).get("count", 0),
                "likely": bool(d.get("confidence", 0.0) >= min_conf and margin >= min_margin),
            })
        return out

    def merge_pending_into(self, pending_name: str, target_name: str,
                           user_id: str = DEFAULT_USER) -> dict:
        """Fold a pending profile into an EXISTING speaker.

        The pending profile's snippets become the target's, the pending profile
        ceases to exist, and the target is re-refined over the larger corpus.
        This is the correct resolution when the learner turns out to be someone
        already registered: naming them would create a second profile for one
        person, and the uncertainty policy would then split their speech
        between two names forever.
        """
        pending_name = (pending_name or "").strip()
        target_name = (target_name or "").strip()
        if not pending_name or not target_name:
            return {"error": "Both a pending profile and a target speaker are required"}
        if pending_name == target_name:
            return {"error": "A pending profile cannot be merged into itself"}
        vp = self._db.get(pending_name, user_id=user_id)
        if not vp and pending_name not in self._snippets.all_speakers(user_id=user_id):
            return {"error": f"No pending profile named {pending_name!r}"}
        if not is_pending_profile(pending_name, vp):
            return {"error": f"{pending_name!r} is not a pending profile"}
        if not self._db.get(target_name, user_id=user_id) and \
                target_name not in self._snippets.all_speakers(user_id=user_id):
            return {"error": f"No speaker named {target_name!r} to merge into"}
        result = self.merge_speakers(target_name, pending_name, user_id=user_id)
        if result.get("error"):
            return result
        logger.info("Pending profile %r merged into existing speaker %r (%d snippets)",
                    pending_name, target_name, result.get("snippets_moved", 0))
        return {**result, "status": "merged_into_existing",
                "from": pending_name, "into": target_name}

    def confirm_pending(self, pending_name: str, new_name: str,
                        user_id: str = DEFAULT_USER) -> dict:
        """Give a pending profile a real name, making it matchable.

        Refuses a name that is already taken (unless it is the same pending
        profile being renamed) so confirming can never silently overwrite a
        colleague's existing profile.
        """
        new_name = (new_name or "").strip()
        if not new_name:
            return {"error": "A name is required"}
        vp = self._db.get(pending_name, user_id=user_id)
        sn = self._snippets.all_speakers(user_id=user_id).get(pending_name)
        if not vp and not sn:
            return {"error": f"No pending profile named {pending_name!r}"}
        if not is_pending_profile(pending_name, vp):
            return {"error": f"{pending_name!r} is not a pending profile"}
        if new_name != pending_name and (
                self._db.get(new_name, user_id=user_id)
                or new_name in self._snippets.all_speakers(user_id=user_id)):
            return {"error": f"A voiceprint named {new_name!r} already exists"}
        # The snippets are what make the profile usable, so they move whenever
        # the name changes -- whether or not a voiceprint row existed to rename.
        self._rename_snippet_dir(pending_name, new_name, user_id=user_id)
        if not self._db.rename(pending_name, new_name, user_id=user_id):
            if not (self._db.get(new_name, user_id=user_id)
                    or new_name in self._snippets.all_speakers(user_id=user_id)):
                return {"error": f"Could not rename {pending_name!r}"}
        self._db.set_pending(new_name, False, user_id=user_id)
        self._auto_refine(new_name, user_id=user_id)
        logger.info("Pending profile %r confirmed as %r (now matchable)",
                    pending_name, new_name)
        return {"ok": True, "name": new_name, "renamed_from": pending_name}

    def _embed_snippets(self, speaker_name: str, user_id: str = DEFAULT_USER,
                        progress_callback=None, exclude_ids=None) -> dict:
        """Load and embed every usable snippet of a speaker, ONCE.

        Shared by ``_auto_refine`` and ``consolidate_speaker`` so the profile is
        fitted from exactly the vectors the outlier pass scored -- a second
        embedding pass would cost ~25 min of GPU per 219 snippets for numbers
        that could drift.

        Returns ``{"rows", "vectors", "durations", "waveforms", "missing"}``
        where ``rows`` is the list of snippet dicts (aligned index-for-index
        with the vectors) and ``missing`` counts the rows dropped for a short or
        unloadable file.  An empty ``vectors`` means nothing can be fitted --
        including the case where the embedding session was unloaded mid-pass,
        which the caller must report rather than call a success.
        """
        snippet_files = [
            sn for sn in self._snippets.list_by_speaker(speaker_name, user_id=user_id)
            if not exclude_ids or sn["id"] not in exclude_ids
        ]
        rows: list = []
        waveforms = []
        durations = []
        for sn in snippet_files:
            try:
                waveform, sr = load_audio(sn["file_path"])
                dur = waveform.shape[-1] / SAMPLE_RATE
                if dur >= MIN_SNIPPET_DURATION:
                    rows.append(sn)
                    waveforms.append(waveform)
                    durations.append(dur)
            except Exception as e:
                logger.warning("Failed to load snippet %s: %s", sn["file_path"], e)

        vectors: list = []
        for i, waveform in enumerate(waveforms):
            session = self._emb_session()
            if session is None:
                logger.warning(
                    "Embedding session disappeared mid-pass for %s "
                    "(%d/%d snippets scored)", speaker_name, i, len(waveforms))
                return {"rows": [], "vectors": [], "durations": [], "waveforms": [],
                        "missing": len(snippet_files) - i}
            try:
                emb = extract_embedding(waveform, SAMPLE_RATE, session)
                if emb is None or float(np.linalg.norm(emb)) <= 0.0:
                    # A zero vector means "no usable window", which would score
                    # as maximally distant and accuse an unloaded file of being
                    # a different person.
                    logger.warning("Empty embedding for snippet %s", rows[i]["file_path"])
                    vectors.append(None)
                else:
                    vectors.append(emb)
            except Exception as e:
                logger.warning("Failed to embed snippet: %s", e)
                vectors.append(None)
            if progress_callback:
                progress_callback({
                    "stage": f"Embedding snippet {i + 1}/{len(waveforms)} for {speaker_name}",
                    "speaker": speaker_name,
                })

        keep = [i for i, v in enumerate(vectors) if v is not None]
        return {
            "rows": [rows[i] for i in keep],
            "vectors": [vectors[i] for i in keep],
            "durations": [durations[i] for i in keep],
            "waveforms": [waveforms[i] for i in keep],
            # Everything that could not be used: too short, unloadable, or an
            # empty embedding. A caller reporting "N snippets could not be
            # scored" must count all three, or a run that silently dropped half
            # the corpus reads as a clean bill of health.
            "missing": len(snippet_files) - len(keep),
        }

    @staticmethod
    def _weighted_embedding(vectors, durations) -> Optional[np.ndarray]:
        """Duration-weighted, per-snippet-capped mean embedding, L2-normalised.

        Duration weighting, with the length of any ONE snippet capped. Uncapped,
        one 200s snippet in a profile of 5s snippets carried 95% of the weight
        and the "profile" became that one recording.  The cap is relative to the
        profile's own MEDIAN snippet, so it is a no-op for a profile whose
        snippets are of comparable length: with a ratio of R a single outlier
        can never weigh more than R/(R + n - 1) <= R/R = 1.0, and for R = 4
        with two or more snippets, never more than 0.5.
        """
        if not vectors:
            return None
        weights = np.array(durations, dtype=np.float64)
        ratio = float(_learning_cfg().get("max_refine_weight_ratio") or 0.0)
        if ratio > 0 and weights.size > 1:
            median = float(np.median(weights))
            if median > 0:
                weights = np.minimum(weights, ratio * median)
        weights = weights / weights.sum() if weights.sum() > 0 else \
            np.full(weights.shape, 1.0 / max(1, weights.size))
        new_embedding = np.zeros_like(np.asarray(vectors[0]))
        for emb, w in zip(vectors, weights):
            new_embedding += emb * w
        norm = np.linalg.norm(new_embedding)
        return new_embedding / norm if norm > 0 else new_embedding

    def _auto_refine(self, speaker_name: str, user_id: str = DEFAULT_USER,
                     progress_callback=None, purity: Optional[dict] = None):
        if self._emb_session() is None:
            logger.warning("No embedding session, skipping auto-refine for %s", speaker_name)
            return

        snippet_files = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        if not snippet_files:
            return

        data = self._embed_snippets(speaker_name, user_id=user_id,
                                    progress_callback=progress_callback)
        vectors, durations, waveforms = data["vectors"], data["durations"], data["waveforms"]
        if not waveforms:
            return

        new_embedding = self._weighted_embedding(vectors, durations)
        if new_embedding is None:
            return

        total_dur = sum(durations)

        combined = torch.cat(waveforms, dim=-1)
        pitch_hz, pitch_std = compute_pitch(combined, SAMPLE_RATE)
        energy_rms = compute_energy(combined)

        # pending is left as None here on purpose: VoiceprintDB.save preserves
        # the stored flag, so an auto-refine cannot promote a pending profile.
        # purity is None for every caller but the learner, and save() preserves
        # it the same way -- a plain refine must not erase the measurement taken
        # when the profile was learned.
        self._db.save(
            name=speaker_name, user_id=user_id,
            embedding=new_embedding,
            pitch_hz=pitch_hz, pitch_std=pitch_std, energy_rms=energy_rms,
            spectral_centroid=0.0,
            spectral_rolloff=0.0,
            total_speech_sec=total_dur, sample_count=len(vectors),
            mfcc=None,
            purity=purity,
        )
        logger.info("Auto-refined voiceprint for %s (user=%s): %.1fs, %d snippets",
                     speaker_name, user_id, total_dur, len(vectors))

    def _cleanup_speaker_dir(self, speaker_name: str, user_id: str = DEFAULT_USER):
        speaker_dir = self._voices_dir / user_id / speaker_name
        if speaker_dir.exists():
            remaining = list(speaker_dir.iterdir())
            if not remaining:
                speaker_dir.rmdir()

    # ── Legacy API (backward compat) ─────────────────────────────────

    def register_from_segments(self, name, wav_path, segments, user_id=DEFAULT_USER, min_duration=1.5):
        all_audio = []
        total_duration = 0.0
        for seg in segments:
            dur = seg.get("end", 0) - seg.get("start", 0)
            if dur < min_duration:
                continue
            chunk, _ = load_audio_segment(wav_path, seg["start"], seg["end"])
            all_audio.append(chunk)
            total_duration += dur

        if not all_audio:
            return {"error": "No valid segments"}

        combined = torch.cat(all_audio, dim=-1)
        embedding = extract_embedding(combined, SAMPLE_RATE, self._emb_session())
        pitch_hz, pitch_std = compute_pitch(combined, SAMPLE_RATE)
        energy_rms = compute_energy(combined)

        self._db.save(
            name=name, user_id=user_id, embedding=embedding,
            pitch_hz=pitch_hz, pitch_std=pitch_std, energy_rms=energy_rms,
            total_speech_sec=total_duration, sample_count=len(all_audio),
        )
        return {"name": name, "total_speech_sec": round(total_duration, 2), "sample_count": len(all_audio)}

    def register_from_audio(self, name, wav_path, start_sec, end_sec, user_id=DEFAULT_USER):
        waveform, sr = load_audio_segment(wav_path, start_sec, end_sec)
        embedding = extract_embedding(waveform, SAMPLE_RATE, self._emb_session())
        pitch_hz, pitch_std = compute_pitch(waveform, SAMPLE_RATE)
        energy_rms = compute_energy(waveform)
        total_duration = end_sec - start_sec

        self._db.save(
            name=name, user_id=user_id, embedding=embedding,
            pitch_hz=pitch_hz, pitch_std=pitch_std, energy_rms=energy_rms,
            total_speech_sec=total_duration, sample_count=1,
        )
        return {"name": name, "total_speech_sec": round(total_duration, 2), "sample_count": 1}

    def identify_in_audio(self, wav_path, start_sec=0.0, end_sec=None, top_k=5, user_id=DEFAULT_USER):
        waveform, sr = load_audio_segment(wav_path, start_sec, end_sec or 99999)
        embedding = extract_embedding(waveform, SAMPLE_RATE, self._emb_session())
        return self._db.search(embedding, user_id=user_id, top_k=top_k)

    def get_voiceprint(self, name, user_id=DEFAULT_USER):
        return self._db.get(name, user_id=user_id)

    def delete_voiceprint(self, name, user_id=DEFAULT_USER):
        return self._db.delete(name, user_id=user_id)
