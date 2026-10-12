"""A diarization-only request must refine boundaries and learn, like transcribe.

``tests/test_learning_boundaries.py`` pins that ``/transcribe/upload`` and
``/attribution/upload`` learn from the TURNS rather than from the raw
diarization segments (finding F1). The two ``/diarize`` endpoints were the
odd ones out: they passed the raw segments to the learner, passed no
``vad_sections`` (so snippet edges were never trimmed onto a speech-section
edge) and, on the by-path endpoint, passed no ``source_id`` at all -- so a
reprocessed file could not replace its earlier take.

Two levels of guard, because the helper is where the behaviour lives and the
endpoints are where it can stop being called:

* behavioural -- ``asr_router._refine_and_learn`` is driven with the audio
  stack and the voiceprint service stubbed, so the order and the arguments are
  observable;
* wiring -- the source of each endpoint is checked to keep calling the helper
  with the diarizer's own Silero pass still in hand.
"""

import asyncio
import sys
import types

import numpy as np
import pytest

from conftest import REPO_ROOT


class _FakeTensor:
    def __init__(self, array):
        self.a = np.asarray(array, dtype=np.float32)

    def numpy(self):
        return self.a

    def squeeze(self):
        return _FakeTensor(np.squeeze(self.a))

    def astype(self, dtype):
        return _FakeTensor(self.a.astype(dtype))


# --------------------------------------------------------------------------
# the helper, driven directly
# --------------------------------------------------------------------------


@pytest.fixture
def router():
    pytest.importorskip("fastapi")
    import asr_mcp.api.asr_router as asr_router
    return asr_router


@pytest.fixture
def stub_audio_stack(monkeypatch):
    """``load_audio`` and a ``VoiceprintService`` the helper can build.

    The helper imports both lazily (the router must stay importable without the
    ML stack), so the stubs go into ``sys.modules`` -- ``from X import Y``
    consults it before touching the real module.
    """
    utils = types.ModuleType("asr_mcp.voiceprint.utils")
    state = {"load_calls": [], "audio": None, "raise": False}

    def load_audio(path):
        state["load_calls"].append(path)
        if state["raise"]:
            raise RuntimeError("no decoder here")
        return _FakeTensor(state["audio"]), 16000

    utils.load_audio = load_audio
    monkeypatch.setitem(sys.modules, "asr_mcp.voiceprint.utils", utils)

    dbm = types.ModuleType("asr_mcp.db.manager")
    dbm.DatabaseManager = lambda path: types.SimpleNamespace(path=path)
    monkeypatch.setitem(sys.modules, "asr_mcp.db.manager", dbm)

    svc_mod = types.ModuleType("asr_mcp.voiceprint.service")

    class _VoiceprintService:
        def __init__(self, data_dir, db):
            self.data_dir, self.db = data_dir, db

        def set_voices_dir(self, voices_dir):
            self.voices_dir = voices_dir

    svc_mod.VoiceprintService = _VoiceprintService
    monkeypatch.setitem(sys.modules, "asr_mcp.voiceprint.service", svc_mod)
    return state


@pytest.fixture
def recorder(monkeypatch, router):
    """Capture what ``_refine_and_learn`` hands to each stage."""
    seen: dict = {}

    def fake_prepare(segments, **kw):
        seen["prepare_args"] = (list(segments), kw)
        if seen.get("prepare_raises"):
            raise RuntimeError("boundary engine exploded")
        return seen.get("turns") or []

    async def fake_collect(vp_service, **kw):
        seen["collect_vp"] = vp_service
        seen["collect_kwargs"] = kw
        return {
            "collected": [{"speaker_name": "Bob"}],
            "pending_created": [{"name": "Speaker 5", "cluster": "5",
                                 "speech_sec": 40.0}],
            "pending_extended": [{"name": "Speaker 6", "cluster": "6",
                                  "speech_sec": 12.0, "snippets": 2}],
            "learn_skipped": [],
            "diagnostics": {
                "collect": [{"speaker": "Bob", "skipped": "at_snippet_cap"}],
                "refused_reasons": {"too_short": 3},
                "refused_total": 3,
            },
        }

    monkeypatch.setattr(router, "_prepare_turns", fake_prepare)
    monkeypatch.setattr(router, "_auto_collect", fake_collect)
    return seen


def _settings(tmp_path):
    return types.SimpleNamespace(
        db_path=str(tmp_path / "asr.db"),
        data_dir=str(tmp_path),
        voices_dir=str(tmp_path / "voices"),
    )


_SEGMENTS = [
    {"start": 0.0, "end": 12.0, "speaker": "Speaker 1"},
    {"start": 14.0, "end": 30.0, "speaker": "Speaker 2"},
]
_TURNS = [
    {"start": 0.0, "end": 12.4, "speaker": "Speaker 1"},
    {"start": 12.4, "end": 30.0, "speaker": "Speaker 2"},
]
_VAD = [{"start": 0.0, "end": 11.8}, {"start": 14.2, "end": 29.9}]


def test_it_learns_from_the_turns_and_the_diarizers_own_vad_pass(
        router, recorder, stub_audio_stack, tmp_path):
    """The whole point: turns + VAD sections, not the raw segments."""
    stub_audio_stack["audio"] = np.zeros(480000, dtype=np.float32)  # 30s
    recorder["turns"] = _TURNS

    spans, learning = asyncio.run(router._refine_and_learn(
        settings=_settings(tmp_path),
        audio_path="/data/meeting.wav",
        segments=_SEGMENTS,
        known_speakers={"Bob": {"embedding": [0.1] * 192}},
        raw_vad_sections=_VAD,
        audio_duration_sec=30.0,
        user_id="u",
        source_id="meeting.mp4",
    ))

    # The learner is handed the boundary set that is also displayed.
    assert spans == _TURNS
    assert recorder["collect_kwargs"]["segments"] == _TURNS
    # ... with the diarizer's Silero pass, so every snippet edge lands on a
    # speech-section edge (lesson 41), and with the source id the reprocess
    # replacement keys on (one file, one contribution).
    assert recorder["collect_kwargs"]["vad_sections"] is _VAD
    assert recorder["collect_kwargs"]["source_id"] == "meeting.mp4"
    assert recorder["collect_kwargs"]["audio_path"] == "/data/meeting.wav"
    assert recorder["collect_kwargs"]["user_id"] == "u"
    # Turn preparation got the audio and the known speakers -- without either
    # the exact-boundary pass is skipped and the gaps are only closed by the
    # energy fallback.
    prepare_segments, prepare_kw = recorder["prepare_args"]
    assert prepare_segments == _SEGMENTS
    assert prepare_kw["raw_vad_sections"] is _VAD
    assert prepare_kw["audio_duration_sec"] == 30.0
    assert prepare_kw["audio"] is not None
    assert len(prepare_kw["audio"]) == 480000
    assert prepare_kw["known_speakers"] == {"Bob": {"embedding": [0.1] * 192}}
    # What the caller reports back: the learned-but-unnamed profiles and the
    # bounded learning summary. A learned profile that is not named out loud
    # stays invisible and the next run reads UNKNOWN again.
    assert [p["name"] for p in learning["pending_profiles"]] == ["Speaker 5", "Speaker 6"]
    assert learning["learning"]["snippets_added"] == 1
    assert learning["learning"]["not_collected"] == [
        {"speaker": "Bob", "reason": "at_snippet_cap"}
    ]
    assert learning["learning"]["refused_reasons"] == {"too_short": 3}


def test_a_failed_refinement_still_learns_from_the_raw_segments(
        router, recorder, stub_audio_stack, tmp_path):
    """Degrade to what the pipeline gave us; never skip the learning."""
    stub_audio_stack["audio"] = np.zeros(16000, dtype=np.float32)
    recorder["prepare_raises"] = True

    spans, learning = asyncio.run(router._refine_and_learn(
        settings=_settings(tmp_path), audio_path="m.wav", segments=_SEGMENTS,
        known_speakers=None, raw_vad_sections=None, audio_duration_sec=1.0,
        user_id="u", source_id="m.wav",
    ))
    assert spans == _SEGMENTS
    assert recorder["collect_kwargs"]["segments"] == _SEGMENTS
    assert learning["pending_profiles"]


def test_an_unloadable_file_still_refines_and_learns(
        router, recorder, stub_audio_stack, tmp_path):
    """Without audio the exact-boundary pass cannot run, but the rest can."""
    stub_audio_stack["raise"] = True
    recorder["turns"] = _TURNS

    spans, _ = asyncio.run(router._refine_and_learn(
        settings=_settings(tmp_path), audio_path="gone.wav", segments=_SEGMENTS,
        known_speakers=None, raw_vad_sections=_VAD, audio_duration_sec=30.0,
        user_id="u", source_id="gone.wav",
    ))
    assert spans == _TURNS
    _, prepare_kw = recorder["prepare_args"]
    assert prepare_kw["audio"] is None
    # The VAD pass is still forwarded, so the engine gets the sections even
    # without a waveform to embed them from.
    assert prepare_kw["raw_vad_sections"] is _VAD
    assert recorder["collect_kwargs"]["vad_sections"] is _VAD


def test_no_speech_means_nothing_to_learn(
        router, recorder, stub_audio_stack, tmp_path):
    """An empty diarization must not fabricate a turn or a profile."""
    spans, learning = asyncio.run(router._refine_and_learn(
        settings=_settings(tmp_path), audio_path="silent.wav", segments=[],
        known_speakers=None, raw_vad_sections=[], audio_duration_sec=12.0,
        user_id="u", source_id="silent.wav",
    ))
    assert spans == []
    assert learning == {}
    assert "prepare_args" not in recorder
    assert "collect_kwargs" not in recorder
    assert stub_audio_stack["load_calls"] == []


# --------------------------------------------------------------------------
# wiring: each endpoint still calls the helper, with the Silero pass in hand
# --------------------------------------------------------------------------


def _router_source() -> str:
    return (REPO_ROOT / "asr_mcp" / "api" / "asr_router.py").read_text()


def _body(src: str, start_marker: str, end_marker: str) -> str:
    start = src.index(start_marker)
    return src[start:src.index(end_marker, start)]


@pytest.mark.parametrize("start_marker,end_marker", [
    ("async def diarize_endpoint(", '@router.post("/diarize/upload")'),
    ("    async def run_diarize():", "    asyncio.create_task(run_diarize())"),
])
def test_the_diarize_endpoints_refine_and_learn(start_marker, end_marker):
    body = _body(_router_source(), start_marker, end_marker)
    assert "_refine_and_learn(" in body
    # The sections have to be captured BEFORE the pop that keeps them out of the
    # response; popping first and handing over None silently degrades the
    # boundary pass to the energy fallback and the snippets to raw edges.
    assert 'raw_vad_sections = result.pop("raw_vad_sections", None)' in body
    assert body.index('result.pop("raw_vad_sections", None)') < body.index("_refine_and_learn(")
    assert "raw_vad_sections=raw_vad_sections" in body
    # The refined spans are what the client is shown, not the raw segments.
    assert "_display_spans(" in body


def test_the_by_path_diarize_endpoint_names_its_source():
    """``source_id=None`` fell back to the audio path, which is a different key.

    The fallback hashed the stem too, so the two agreed by luck -- but only when
    the file was reachable. The explicit ``source_id`` is what the one-file,
    one-contribution replacement keys on, so it is stated rather than implied.
    """
    body = _body(_router_source(), "async def diarize_endpoint(",
                 '@router.post("/diarize/upload")')
    assert "source_id=req.wav_path" in body


def test_the_display_projection_is_shared():
    """Three copies of the same projection is how they drift apart."""
    src = _router_source()
    assert src.count("def _display_spans(") == 1
    assert src.count("_display_spans(") == 4  # def + 3 call sites
    assert '"attribution_reason": s.get("attribution_reason")' in src


def test_the_diarize_response_can_report_what_it_learned():
    """A learned-but-unnamed profile is invisible unless the response says so."""
    schemas = (REPO_ROOT / "asr_mcp" / "api" / "schemas.py").read_text()
    body = _body(schemas, "class DiarizeResponse(", "class TranscribePathsRequest(")
    assert "pending_profiles: list[dict] = []" in body
    assert "learning: Optional[dict] = None" in body


def test_a_refused_learn_is_not_silently_dropped():
    """The endpoint must not lose the report if the learner throws."""
    router_src = _router_source()
    body = _body(router_src, "async def _refine_and_learn(",
                 "def _load_known_speakers(")
    # Voiceprint service construction and the learner are both inside the try,
    # and the failure is logged rather than raised: a diarization must not fail
    # because a voiceprint could not be rebuilt.
    assert body.count("except Exception as e:") == 3  # audio, prepare, collect
    assert 'logger.warning("Auto-collect failed: %s", e)' in body