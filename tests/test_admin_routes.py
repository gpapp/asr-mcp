"""The administrator routes: who may run the server-wide operations, and what
happens to a user's data when their account goes.

Every one of these is a property of the ROUTES, not of the credential store:

* **Server-wide means one configured name, not "anybody logged in."**
  ``POST /shutdown`` had no role check at all before this phase. Stopping the
  process cancels other users' jobs and drops the GPU models they are queueing
  for, so it belongs to the account in ``TRANSCRIBE_ADMIN_USER`` -- and when
  that is unset, to nobody.
* **A delete keeps the evidence by default.** Removing the login and keeping
  the speakers, snippets and recordings is reversible; the other order is not.
  The purge exists, it is opt-in, and it is the only path that unlinks audio.
* **Two accounts can never be deleted from here**: the one the operator is
  signed in as, and the configured admin. An app whose admin names an account
  that no longer exists has no admin at all, and the only way back is an env
  edit and a restart.

The routes are driven as coroutines against a fake request, the same way the
rest of this suite does it.
"""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the admin routes are FastAPI handlers")
pytest.importorskip("sqlalchemy", reason="the credential tables need the DB layer")

PASSWORD = "correct horse battery"
ADMIN = "root"
MEMBER = "ada"


# ----------------------------------------------------------------------
# Fixtures / fakes
# ----------------------------------------------------------------------

class FakeApp:
    def __init__(self, htpasswd_users=None):
        self.state = type("State", (), {})()
        self.state.htpasswd_users = htpasswd_users


class FakeRequest:
    def __init__(self, app=None, session=None, headers=None, client_host="10.0.0.9",
                 base_url="http://asr.internal:8080/", query=None):
        self.app = app if app is not None else FakeApp()
        self.scope = {"session": {} if session is None else session}
        self.headers = {key.lower(): value for key, value in (headers or {}).items()}
        self.client = type("Client", (), {})()
        self.client.host = client_host
        self.base_url = base_url
        self.query_params = query or {}

    @property
    def session(self):
        return self.scope["session"]


@pytest.fixture
def db(tmp_path):
    from asr_mcp.db.manager import DatabaseManager
    return DatabaseManager(str(tmp_path / "asr.db"))


def _settings(tmp_path, **overrides):
    from asr_mcp.config.settings import Settings

    kwargs = {
        "session_secret": "test-secret",
        "db_path": str(tmp_path / "asr.db"),
        "data_dir": str(tmp_path),
        "voices_dir": str(tmp_path / "voices"),
        # Pinned so the assertions are about the router, not about whatever
        # TRANSCRIBE_PREFIX or htpasswd file the container happens to ship. A
        # test that inherits the ambient store is a test that only passes on
        # the machine that wrote it.
        "prefix": "",
        "htpasswd_path": str(tmp_path / "no-htpasswd"),
        "registration_enabled": True,
        "admin_user": ADMIN,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def _verified(store, user_id, password=PASSWORD):
    """Mark an account confirmed, the way opening the emailed link would.

    ``check_login`` refuses an unverified account even with the right
    password, which is the registration policy working -- so a test about
    disabling and re-enabling has to get past it deliberately rather than
    discover it as a failure.
    """
    token = store.issue_verification_token(user_id)
    assert store.verify_email_token(token) == user_id
    return store


def _route():
    from asr_mcp.api import admin_router
    return admin_router


def _store(db):
    from asr_mcp.auth.store import CredentialStore
    return CredentialStore(db)


def _voices(tmp_path, user_id, speaker="Ada", snippets=2):
    """A believable per-user voices directory: voiceprint audio, on disk."""
    target = tmp_path / "voices" / user_id / speaker
    target.mkdir(parents=True, exist_ok=True)
    for i in range(snippets):
        (target / f"{speaker.lower()}-{i}.flac").write_bytes(b"fake flac payload")
    return target


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------

class TestAdminGate:

    def test_a_non_admin_is_refused_with_a_reason(self, tmp_path, db):
        from asr_mcp.api.security import require_admin
        with pytest.raises(Exception) as excinfo:
            asyncio.run(require_admin(user=MEMBER, settings=_settings(tmp_path)))
        assert "TRANSCRIBE_ADMIN_USER" in str(excinfo.value)

    def test_the_configured_admin_gets_through(self, tmp_path):
        from asr_mcp.api.security import require_admin
        assert asyncio.run(require_admin(user=ADMIN, settings=_settings(tmp_path))) == ADMIN

    def test_nobody_is_admin_when_the_name_is_not_set(self, tmp_path):
        from asr_mcp.api.security import require_admin
        settings = _settings(tmp_path, admin_user="")
        with pytest.raises(Exception):
            asyncio.run(require_admin(user=ADMIN, settings=settings))

    def test_it_is_the_exact_name_not_a_member_of_a_group(self, tmp_path):
        from asr_mcp.api.security import is_admin_user
        settings = _settings(tmp_path)
        assert is_admin_user("admin", settings) is False
        assert is_admin_user("Root", settings) is False
        assert is_admin_user(ADMIN + " ", settings) is False

    def test_every_admin_route_declares_the_dependency(self):
        source = Path(_route().__file__).read_text(encoding="utf-8")
        routes = source.count("Depends(require_admin)")
        assert routes >= 4, f"only {routes} admin routes are gated"


# ----------------------------------------------------------------------
# The account table
# ----------------------------------------------------------------------

class TestUserTable:

    def test_a_password_account_is_listed_with_its_state(self, tmp_path, db):
        store = _store(db)
        store.create(MEMBER, PASSWORD, "ada@example.com")
        request = FakeRequest()
        rows = _route()._user_table(request, _settings(tmp_path))
        row = next(r for r in rows if r["user_id"] == MEMBER)
        assert row["sources"] == ["password"]
        assert row["email"] == "ada@example.com"
        assert row["verified"] is False
        assert row["disabled"] is False

    def test_no_row_ever_carries_a_hash_or_a_token(self, tmp_path, db):
        store = _store(db)
        store.create(MEMBER, PASSWORD, "ada@example.com")
        store.link_google_identity("sub-1", "carol@example.com",
                                   email="carol@example.com", name="Carol")
        rows = _route()._user_table(FakeRequest(), _settings(tmp_path))
        blob = json.dumps(rows)
        assert "password_hash" not in blob
        assert "verification_token" not in blob
        # The stored hash really is in the database, so this is a test of the
        # projection rather than of an empty store.
        assert store.get(MEMBER)["password_hash"].startswith("scrypt$")

    def test_an_htpasswd_account_is_read_only_and_says_so(self, tmp_path, db):
        store = _store(db)
        rows = _route()._user_table(
            FakeRequest(app=FakeApp({"operator": "$apr1$zzz"})), _settings(tmp_path))
        row = next(r for r in rows if r["user_id"] == "operator")
        assert "htpasswd" in row["sources"]
        assert row["can_manage"] is False and row["can_delete"] is False
        assert "htpasswd" in row["manage_note"]

    def test_an_account_that_only_owns_data_is_still_listed(self, tmp_path, db):
        """A static API key is a real owner with no account row anywhere.

        Its ``user_<hash16>`` id holds voiceprints and recordings. An operator
        reviewing who has data on this box has to be able to see it, or the
        table answers "nobody" while the disk says otherwise.
        """
        from asr_mcp.db.manager import VoiceprintDB
        settings = _settings(tmp_path)
        VoiceprintDB(db).save(name="Ada", embedding=_unit(), user_id="user_abc123")
        rows = _route()._user_table(FakeRequest(), settings)
        row = next(r for r in rows if r["user_id"] == "user_abc123")
        assert row["sources"] == []
        assert row["data"]["voiceprints"] == 1

    def test_the_admin_row_is_flagged(self, tmp_path, db):
        _store(db).create(ADMIN, PASSWORD, "root@example.com")
        rows = _route()._user_table(FakeRequest(), _settings(tmp_path))
        row = next(r for r in rows if r["user_id"] == ADMIN)
        assert row["is_admin"] is True
        assert row["can_delete"] is False or row["user_id"] == ADMIN

    def test_the_route_names_the_admin_and_the_signup_state(self, tmp_path, db):
        request = FakeRequest()
        response = asyncio.run(_route().list_users(
            request, admin=ADMIN, settings=_settings(tmp_path)))
        assert response["admin"] == ADMIN
        assert response["registration"]["enabled"] is True


def _unit():
    import numpy as np
    v = np.ones(192, dtype=np.float32)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------
# Disable / enable
# ----------------------------------------------------------------------

class TestDisableEnable:

    def test_a_disabled_account_cannot_sign_in_but_keeps_its_data(self, tmp_path, db):
        store = _store(db)
        store.create(MEMBER, PASSWORD, "ada@example.com")
        settings = _settings(tmp_path)
        response = asyncio.run(_route().disable_user(
            MEMBER, request=FakeRequest(), settings=settings, admin=ADMIN))
        assert response.status_code == 200
        assert store.check_login(MEMBER, PASSWORD)[1], "a disabled account must not sign in"
        assert store.public(MEMBER) is not None, "the row must survive a disable"

    def test_enabling_restores_the_sign_in(self, tmp_path, db):
        store = _store(db)
        store.create(MEMBER, PASSWORD, "ada@example.com")
        _verified(store, MEMBER)
        settings = _settings(tmp_path)
        asyncio.run(_route().disable_user(
            MEMBER, request=FakeRequest(), settings=settings, admin=ADMIN))
        assert store.check_login(MEMBER, PASSWORD)[0] == ""
        asyncio.run(_route().enable_user(
            MEMBER, request=FakeRequest(), settings=settings, admin=ADMIN))
        assert store.check_login(MEMBER, PASSWORD)[0] == MEMBER

    def test_a_missing_account_is_a_404_not_a_crash(self, tmp_path):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(_route().disable_user("ghost", request=FakeRequest(),
                                              settings=_settings(tmp_path), admin=ADMIN))
        assert excinfo.value.status_code == 404


# ----------------------------------------------------------------------
# Delete
# ----------------------------------------------------------------------

class TestDeleteUser:

    def test_it_refuses_to_delete_the_account_you_are_signed_in_as(self, tmp_path, db):
        from fastapi import HTTPException
        _store(db).create(ADMIN, PASSWORD, "root@example.com")
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(_route().delete_user(
                ADMIN, FakeRequest(), purge=False, admin=ADMIN,
                settings=_settings(tmp_path)))
        assert excinfo.value.status_code == 400
        assert _store(db).public(ADMIN) is not None

    def test_it_refuses_to_delete_the_configured_admin(self, tmp_path, db):
        from fastapi import HTTPException
        _store(db).create(ADMIN, PASSWORD, "root@example.com")
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(_route().delete_user(
                ADMIN, FakeRequest(), purge=True, admin="someone-else",
                settings=_settings(tmp_path)))
        assert excinfo.value.status_code == 400
        assert _store(db).public(ADMIN) is not None

    def test_a_plain_delete_keeps_the_recordings_and_says_how_many(self, tmp_path, db):
        """The default is the reversible one, and it reports what it left.

        The counts are the point: an operator who deletes an account has to be
        able to see that the recordings are still there, or "kept" is a promise
        rather than a fact.
        """
        store = _store(db)
        store.create(MEMBER, PASSWORD, "ada@example.com")
        store.link_google_identity("sub-1", MEMBER)
        _voices(tmp_path, MEMBER)
        from asr_mcp.db.manager import SnippetDB, VoiceprintDB
        VoiceprintDB(db).save(name="Ada", embedding=_unit(), user_id=MEMBER)
        SnippetDB(db).add("Ada", str(tmp_path / "voices" / MEMBER / "Ada" / "ada-0.flac"),
                          1.5, user_id=MEMBER)
        SnippetDB(db).add("Ada", str(tmp_path / "voices" / MEMBER / "Ada" / "ada-1.flac"),
                          2.0, user_id=MEMBER)

        response = asyncio.run(_route().delete_user(
            MEMBER, FakeRequest(), purge=False, admin=ADMIN,
            settings=_settings(tmp_path)))
        body = json.loads(response.body)
        assert body["ok"] is True
        assert body["removed"]["credentials"] is True
        assert body["removed"]["google_identities"] == 1
        assert body["data_kept"] is True
        assert body["data"]["voiceprints"] == 1
        assert body["data"]["snippets"] == 2
        # The rows and the audio are all still there.
        assert _store(db).public(MEMBER) is None
        assert VoiceprintDB(db).count(user_id=MEMBER) == 1
        assert (tmp_path / "voices" / MEMBER / "Ada").is_dir()

    def test_the_purge_drops_the_rows_and_the_audio(self, tmp_path, db):
        store = _store(db)
        store.create(MEMBER, PASSWORD, "ada@example.com")
        store.link_google_identity("sub-1", MEMBER)
        _voices(tmp_path, MEMBER)
        from asr_mcp.db.manager import SnippetDB, TranscriptDB, VoiceprintDB
        vp = VoiceprintDB(db)
        snippets = SnippetDB(db)
        transcripts = TranscriptDB(db)
        vp.save(name="Ada", embedding=_unit(), user_id=MEMBER)
        snippets.add("Ada", str(tmp_path / "voices" / MEMBER / "Ada" / "ada-0.flac"),
                     1.5, user_id=MEMBER)
        transcripts.save("rec.wav", {"segments": []}, user_id=MEMBER)

        response = asyncio.run(_route().delete_user(
            MEMBER, FakeRequest(), purge=True, admin=ADMIN,
            settings=_settings(tmp_path)))
        body = json.loads(response.body)
        assert body["purged"]["voiceprints"] == 1
        assert body["purged"]["snippets"] == 1
        assert body["purged"]["transcripts"] == 1
        assert body["purged"]["recordings_unlinked"] is True
        assert body["data_kept"] is False
        assert not (tmp_path / "voices" / MEMBER).exists()
        assert vp.count(user_id=MEMBER) == 0
        assert snippets.count(speaker_name=None, user_id=MEMBER) == 0
        assert transcripts.list_all(user_id=MEMBER) == []
        # The Google link goes too, or the subject would resolve to a deleted vault.
        assert store.list_google_identities(MEMBER) == []

    def test_a_purge_cannot_reach_a_neighbours_recordings(self, tmp_path, db):
        """The user_id comes from a URL path parameter.

        ``voices/<user_id>/`` is a path build, so a ``user_id`` that escapes the
        directory must not delete somebody else's audio. Resolving and checking
        containment is the whole defence; this pins it.
        """
        from asr_mcp.api.admin_router import _voices_dir
        settings = _settings(tmp_path)
        (tmp_path / "voices" / "other").mkdir(parents=True, exist_ok=True)
        (tmp_path / "voices" / "other" / "Ada").mkdir(parents=True, exist_ok=True)
        (tmp_path / "voices" / "other" / "Ada" / "ada.flac").write_bytes(b"keep me")
        assert _voices_dir(settings, "../other") is None
        assert _voices_dir(settings, "../../etc") is None

    def test_an_account_in_no_store_is_a_404(self, tmp_path, db):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(_route().delete_user(
                "ghost", FakeRequest(), purge=False, admin=ADMIN,
                settings=_settings(tmp_path)))
        assert excinfo.value.status_code == 404

    def test_the_delete_route_reports_the_api_token_it_dropped(self, tmp_path, db):
        from asr_mcp.db.manager import TokenDB
        _store(db).create(MEMBER, PASSWORD, "ada@example.com")
        TokenDB(db).create(MEMBER)
        response = asyncio.run(_route().delete_user(
            MEMBER, FakeRequest(), purge=False, admin=ADMIN,
            settings=_settings(tmp_path)))
        body = json.loads(response.body)
        assert body["removed"]["api_tokens"] == 1
        assert TokenDB(db).get_info(MEMBER) is None


# ----------------------------------------------------------------------
# The shutdown route
# ----------------------------------------------------------------------

class TestShutdown:

    def test_the_route_is_no_longer_ungated(self):
        source = Path(__file__).resolve()
        server = (source.parents[1] / "asr_mcp" / "server.py").read_text(encoding="utf-8")
        marker = '@app.post("/shutdown")'
        assert marker in server, "the shutdown route moved"
        after = server[server.index(marker):]
        head = after[:after.index("async def shutdown") + 200]
        assert "require_admin" in head, (
            "POST /shutdown is reachable by any session or any valid API key again"
        )
