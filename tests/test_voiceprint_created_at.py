"""``created_at`` must be a STRING all the way out of the DB layer.

GET /api/voiceprint/pending (and /speakers) validate their rows through
``VoiceprintSpeakerInfo``, whose ``created_at`` is ``str``. Every other row
converter in ``db/manager.py`` already serialized its timestamp
(``.isoformat()``); ``VoiceprintDB._row_to_dict`` was the one exception and
handed a raw ``datetime`` to the schema, so a single pending profile took down
the whole listing with a 500:

    pydantic_core._pydantic_core.ValidationError: 1 validation error for
    VoiceprintSpeakerInfo / created_at / Input should be a valid string
    [type=string_type, input_value=datetime.datetime(2026, 10, 1, 9, 38, ...)]

The second half of the same bug is ``None``: a profile learned while the
embedding session was unavailable has snippets but no ``voiceprints`` row (see
``VoiceprintService.pending_profiles``), so ``created_at`` is absent rather than
a datetime -- and ``None`` fails a non-Optional ``str`` just as loudly.
"""

import datetime
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sqlalchemy", reason="DB layer needed")

from asr_mcp.db.manager import DatabaseManager, VoiceprintDB  # noqa: E402


def _load_schemas():
    """Import ``api/schemas.py`` by path.

    ``asr_mcp.api.__init__`` pulls in the whole router chain (torchaudio, the
    ONNX loaders), and ``schemas.py`` itself imports nothing from the package --
    so a path load is the only way to reach the model without a GPU stack.
    """
    path = Path(__file__).resolve().parents[1] / "asr_mcp" / "api" / "schemas.py"
    spec = importlib.util.spec_from_file_location("_asr_schemas", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VoiceprintSpeakerInfo = _load_schemas().VoiceprintSpeakerInfo


def _unit() -> np.ndarray:
    v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def db(tmp_path):
    mgr = DatabaseManager(str(tmp_path / "asr.db"))
    yield mgr
    mgr.close()


def _as_listing_row(vp: dict, name: str = "", snippet_count: int = 3,
                    total_duration: float = 42.0) -> dict:
    """Rebuild the shape ``VoiceprintService.pending_profiles`` sends.

    Mirrors ``voiceprint/service.py::pending_profiles`` -- the service itself
    needs torch+soundfile, so the row is assembled here to keep this test
    dependency-light while still exercising the real DB dict and the real
    pydantic model. ``vp`` may be ``{}`` for a profile that has snippets but no
    voiceprints row, hence the name fallback.
    """
    return {
        "name": name or vp.get("name", ""),
        "snippet_count": snippet_count,
        "total_duration_sec": round(total_duration, 2),
        "created_at": vp.get("created_at") or "",
        "purity": vp.get("purity") or None,
    }


def test_voiceprint_created_at_is_a_serialized_string(db):
    vp_db = VoiceprintDB(db)
    vp_db.save("Pending 2026-10-01 09:38 Speaker 5 meeting", _unit(),
               user_id="u", total_speech_sec=30.0, pending=True)

    row = vp_db.get("Pending 2026-10-01 09:38 Speaker 5 meeting", user_id="u")
    assert isinstance(row["created_at"], str)
    # Parses back as a datetime, so the UI's formatDate() still works.
    assert datetime.datetime.fromisoformat(row["created_at"]).year == 2026


def test_pending_listing_row_validates_against_the_schema(db):
    """The regression: this exact call raised a 500 before the fix."""
    vp_db = VoiceprintDB(db)
    vp_db.save("Pending 2026-10-01 09:38 Speaker 5 meeting", _unit(),
               user_id="u", total_speech_sec=30.0, pending=True)

    vp = vp_db.get("Pending 2026-10-01 09:38 Speaker 5 meeting", user_id="u")
    info = VoiceprintSpeakerInfo(**_as_listing_row(vp))
    assert info.created_at
    assert info.name == "Pending 2026-10-01 09:38 Speaker 5 meeting"


def test_a_profile_with_no_voiceprint_row_still_validates():
    """Snippets but no row (learned without an embedding session).

    ``vp`` is empty here, exactly as ``pending_profiles`` builds it for such a
    profile, so ``created_at`` must fall back to "" -- never ``None``.
    """
    info = VoiceprintSpeakerInfo(**_as_listing_row({}))
    assert info.created_at == ""
    assert info.has_voiceprint is False


def test_the_schema_rejects_a_raw_datetime():
    """Pins the contract, so a future un-serialized timestamp fails HERE."""
    with pytest.raises(Exception):
        VoiceprintSpeakerInfo(
            name="Pending x",
            created_at=datetime.datetime(2026, 10, 1, 9, 38, 52),
        )
