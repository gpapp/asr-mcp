"""Forced-k clustering linkage plumbing (lesson 32).

`Diarizer._cluster_embeddings` takes a different branch when a caller forces
`num_speakers`: a hard `n_clusters=k` agglomerative clustering with NO greedy
merge afterwards.  That branch used average linkage, which on a real 52-minute
podcast produced 182/1180/2 windows (86.5%, 1924s of 2070s in ONE cluster)
where complete linkage produces 762/244/358 (55.9%).  The clusterer, not the
uncertainty policy, was what produced "99.3% one identity" transcripts.

These tests cover the plumbing (which linkage is used, the fallback, config
isolation).  The balance numbers above are measurements on real audio and are
recorded in AGENTS.md lesson 32 -- they are NOT reproducible from synthetic
embeddings, because the pathology depends on the real embedding geometry.  A
synthetic test that claimed otherwise would be asserting a coincidence.

Needs torch + sklearn (pipeline.py imports torch at module level), so it is
skipped outside the container.
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

import asr_mcp.diarization.pipeline as pipeline_mod  # noqa: E402
from asr_mcp.config import get_config  # noqa: E402
from asr_mcp.diarization.pipeline import Diarizer  # noqa: E402

_SENTINEL = object()


class _RecordingClusterer:
    """Stand-in for AgglomerativeClustering that records the kwargs."""

    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).calls.append(kwargs)

    def fit_predict(self, X):  # noqa: N803 - mirrors sklearn's name
        return np.zeros(len(X), dtype=int)


@pytest.fixture
def restore_config():
    """`get_config()` returns ONE cached dict; never let a test leak into another."""
    snapshot = copy.deepcopy(get_config())
    yield
    get_config().clear()
    get_config().update(snapshot)


@pytest.fixture
def record_linkage(monkeypatch):
    _RecordingClusterer.calls = []
    monkeypatch.setattr(pipeline_mod, "AgglomerativeClustering", _RecordingClusterer)
    return _RecordingClusterer


def _cluster(embeddings, num_speakers, cfg=None):
    """Return (labels, centroids) exactly as `_cluster_embeddings` produced them."""
    cfg = get_config() if cfg is None else cfg
    labels, centroids = Diarizer._cluster_embeddings(
        None, embeddings, num_speakers, 0.35, cfg
    )
    return np.asarray(labels, dtype=int), centroids


def _embeddings(n=40, seed=3):
    """Tight, well-separated groups: the invariant that must hold for any linkage."""
    rng = np.random.default_rng(seed)
    centres = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vecs = np.vstack([
        c + rng.normal(0.0, 0.05, size=(n, 3)) for c in centres
    ])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_shipped_default_forced_k_linkage_is_complete():
    cfg = get_config().get("diarization", {})
    assert str(cfg.get("forced_k_linkage", "complete")).lower() == "complete"


def test_forced_k_uses_the_configured_linkage(restore_config, record_linkage):
    get_config()["diarization"]["forced_k_linkage"] = "complete"
    _cluster(_embeddings(), 3)
    assert record_linkage.calls[-1]["linkage"] == "complete"
    assert record_linkage.calls[-1]["n_clusters"] == 3


@pytest.mark.parametrize("linkage", ["complete", "average", "single"])
def test_every_allowed_linkage_is_accepted(restore_config, record_linkage, linkage):
    get_config()["diarization"]["forced_k_linkage"] = linkage
    _cluster(_embeddings(), 3)
    assert record_linkage.calls[-1]["linkage"] == linkage


def test_unknown_linkage_falls_back_to_complete(restore_config, record_linkage):
    get_config()["diarization"]["forced_k_linkage"] = "nonsense"
    _cluster(_embeddings(), 3)
    assert record_linkage.calls[-1]["linkage"] == "complete"


def test_linkage_is_case_insensitive(restore_config, record_linkage):
    get_config()["diarization"]["forced_k_linkage"] = "  CoMpLeTe  "
    _cluster(_embeddings(), 3)
    assert record_linkage.calls[-1]["linkage"] == "complete"


def test_threshold_path_ignores_the_forced_k_setting(restore_config, record_linkage):
    """The default path (num_speakers=None) must not be affected by this knob."""
    get_config()["diarization"]["forced_k_linkage"] = "complete"
    _cluster(_embeddings(), None)
    assert record_linkage.calls[-1]["linkage"] == "average"
    assert record_linkage.calls[-1]["distance_threshold"] == 0.35
    assert record_linkage.calls[-1]["n_clusters"] is None


def test_forced_k_keeps_exactly_k_clusters(restore_config):
    """Forcing a count is honoured: no cap, no greedy merge (that is by design)."""
    embeddings = _embeddings(n=60)
    for linkage in ("complete", "average", "single"):
        get_config()["diarization"]["forced_k_linkage"] = linkage
        labels, _ = _cluster(embeddings, 3)
        assert len(set(labels.tolist())) == 3, linkage
        assert (np.bincount(labels, minlength=3) > 0).all(), linkage


def test_threshold_path_still_reports_centroids(restore_config):
    """Regression guard: centroids are computed on the default path."""
    labels, centroids = _cluster(_embeddings(), None)
    assert len(centroids) == len(set(labels.tolist()))
    for centroid in centroids.values():
        assert abs(float(np.linalg.norm(centroid)) - 1.0) < 1e-5


def test_tight_groups_never_collapse_under_any_linkage(restore_config):
    """Real clustering here -- the linkage-recording fake returns all-zero labels."""
    embeddings = _embeddings(n=80)
    for linkage in ("complete", "average", "single"):
        get_config()["diarization"]["forced_k_linkage"] = linkage
        labels, _ = _cluster(embeddings, 3)
        sizes = np.bincount(labels, minlength=3)
        assert sizes.max() / sizes.sum() < 0.75, (
            "%s linkage collapsed tight groups: %s" % (linkage, sizes.tolist())
        )
