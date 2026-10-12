"""Administrator routes: the account table and the operations that are not
per-user.

Every gate in here exists because of what it protects, not because the route is
"admin-ish":

* **``POST /shutdown``** stops a process other users have jobs queued against.
* **the user table** reads every account's email address and confirmation state.
* **``?purge=true``** deletes another user's voiceprints, snippets, History and
  their recordings from disk.

Gating those on "whoever is logged in" would put the whole deployment one
signup away from anybody who could create an account, so they require the single
name in ``TRANSCRIBE_ADMIN_USER`` -- and when that is unset, nobody.

The table shape is ported from mem-mcp's ``gui.py::api_admin_list_users``: one
row per identity, ``sources`` naming every store that knows it, and never a
password hash or a confirmation token. Two things are added for this app, because
an asr-mcp user is not necessarily an account: rows are also emitted for
identities that own data but have no credential row anywhere (a static
``TRANSCRIBE_API_KEYS`` value maps to a synthetic ``user_<hash16>`` id, and a
deleted key's rows would otherwise be invisible while still holding recordings),
and each row reports what a delete would touch.
"""

import logging
import shutil
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from asr_mcp.api.security import get_settings, is_admin_user, require_admin
from asr_mcp.auth.store import (
    GOOGLE_PROVIDER,
    CredentialStore,
    credential_store,
    registration_enabled,
)
from asr_mcp.config.settings import Settings
from asr_mcp.db.manager import (
    DatabaseManager,
    SnippetDB,
    TokenDB,
    TranscriptDB,
    VoiceprintDB,
)

logger = logging.getLogger("asr_mcp.api.admin_router")

router = APIRouter()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _db(settings: Settings) -> DatabaseManager:
    return DatabaseManager(settings.db_path)


def _store(request: Request, settings: Settings) -> CredentialStore:
    users = getattr(request.app.state, "htpasswd_users", None) if request.app else None
    return credential_store(settings, htpasswd_users=users)


def _voices_dir(settings: Settings, user_id: str):
    """``voices/<user_id>``, or None when it does not exist.

    Resolved per call rather than cached, and never followed if a user_id
    escapes the directory: the id comes from a URL path parameter, and a
    ``user_id`` of ``../other`` must not delete a neighbour's recordings.
    """
    base = settings.voices_dir.resolve()
    candidate = (base / user_id).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def _user_data_counts(settings: Settings, user_id: str) -> dict:
    """What this identity owns that a delete would touch. Never raises.

    Reported so the operator sees the consequence before pressing the button,
    and so a *non*-purging delete can say exactly what it left behind. An
    unreadable store reads as zeros: the table still has to render.
    """
    counts = {"voiceprints": 0, "snippets": 0, "transcripts": 0, "api_tokens": 0}
    try:
        db = _db(settings)
        counts["voiceprints"] = VoiceprintDB(db).count(user_id=user_id)
        counts["snippets"] = SnippetDB(db).count(speaker_name=None, user_id=user_id)
        counts["transcripts"] = len(TranscriptDB(db).list_all(user_id=user_id))
        counts["api_tokens"] = 1 if TokenDB(db).get_info(user_id) else 0
    except Exception as e:
        logger.debug("Could not count data for %r: %s", user_id, e)
    return counts


def _row(user_id: str, sources: list, credential: Optional[dict],
         google: Optional[dict], settings: Settings) -> dict:
    """One row of the account table.

    ``email``/``verified``/``disabled`` come from the credentials row alone: a
    Google sign-in is verified by construction (Google said so), and an htpasswd
    account has no confirmation state because there is nothing to confirm. The
    hash and the token are not in either source dict -- ``public`` and
    ``list_all_google_identities`` both dropped them before this function could
    see them, which is the only way to guarantee they cannot leak here.
    """
    email = ""
    verified = False
    disabled = False
    created_at = None
    if credential:
        email = credential.get("email") or ""
        verified = credential.get("email_verified_at") is not None
        disabled = credential.get("disabled_at") is not None
        created_at = credential.get("created_at")
    if not email and google:
        email = google.get("email") or ""

    # can_manage / can_delete follow mem-mcp: an htpasswd account is the
    # operator's own file and this app does not write to it, so it is listed
    # and cannot be edited or removed from here. Without those flags an
    # operator gets a 404 with no way to tell "gone" from "not ours to change".
    can_manage = "htpasswd" not in sources and bool(credential)
    can_delete = "htpasswd" not in sources
    notes = []
    if "htpasswd" in sources and not credential:
        notes.append("from the htpasswd file; edit that file to change it")
    if not credential and "google" in sources:
        notes.append("Google sign-in only; no password, nothing to confirm")
    if not sources:
        notes.append("owns data but has no account in any store")
    return {
        "user_id": user_id,
        "sources": sources,
        "email": email,
        "verified": verified,
        "disabled": disabled,
        "created_at": created_at,
        "is_admin": is_admin_user(user_id, settings),
        "can_manage": can_manage,
        "can_delete": can_delete,
        "manage_note": "; ".join(notes),
        "data": _user_data_counts(settings, user_id),
    }


def _user_table(request: Request, settings: Settings) -> list:
    """Every identity the operator can see, one row each.

    Three credential stores are unioned -- credentials, Google identities and
    the htpasswd map -- and then the data stores are walked for identities none
    of them know about. That last pass is the asr-mcp difference: a static API
    key is a real voiceprint owner with no account row anywhere, and an operator
    reviewing privacy has to be able to see it.
    """
    store = _store(request, settings)
    rows = {}

    for record in store.list_all():
        rows[record["user_id"]] = _row(
            record["user_id"], ["password"], record, None, settings)

    for identity in store.list_all_google_identities():
        key = identity["user_id"]
        if key in rows:
            rows[key]["sources"].append("google")
            rows[key]["email"] = rows[key]["email"] or identity.get("email") or ""
            continue
        rows[key] = _row(key, ["google"], None, identity, settings)

    for key in store._htpasswd_keys():
        if key in rows:
            # htpasswd wins for display: the operator's file is the
            # authority on the name, and a registration row with the same key
            # would mean the password path was taken over underneath it.
            rows[key]["sources"].append("htpasswd")
            rows[key]["can_manage"] = False
            rows[key]["can_delete"] = False
            note = rows[key]["manage_note"]
            rows[key]["manage_note"] = (
                (note + "; " if note else "")
                + "also in the htpasswd file; edit that file to change it")
        else:
            rows[key] = _row(key, ["htpasswd"], None, None, settings)

    try:
        db = _db(settings)
        owners = set(VoiceprintDB(db).user_ids())
        owners.update(SnippetDB(db).user_ids())
        owners.update(TranscriptDB(db).user_ids())
    except Exception as e:
        logger.debug("Could not list data owners: %s", e)
        owners = set()
    for key in owners - set(rows):
        if key:
            rows[key] = _row(key, [], None, None, settings)

    return [rows[key] for key in sorted(rows)]


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@router.get("/admin/users")
async def list_users(
    request: Request,
    admin: str = Depends(require_admin),
    settings: Settings = Depends(get_settings),
):
    """The account table, plus whether admin and signup are even configured.

    The response says so out loud because both answers change what the table
    means: with ``TRANSCRIBE_ADMIN_USER`` unset this route would have 403'd
    before reaching here, and with registration off the only accounts an
    operator can create are in the htpasswd file.
    """
    return {
        "admin": admin,
        "admin_configured": bool(str(settings.admin_user or "").strip()),
        "users": _user_table(request, settings),
        "registration": {
            "enabled": bool(settings.registration_enabled),
            "email": registration_enabled("email", settings),
            "google": registration_enabled(GOOGLE_PROVIDER, settings),
        },
    }


@router.post("/admin/users/{user_id}/disable")
async def disable_user(
    user_id: str,
    admin: str = Depends(require_admin),
    request: Request = None,
    settings: Settings = Depends(get_settings),
):
    """Refuse future sign-ins without releasing the account's data.

    Disabling is the *default* answer to a bad account and deleting is the
    destructive one, which is why this exists separately: ``disable`` keeps the
    vault key occupied, so a name cannot be registered again and come back
    pointing at records that were never removed.
    """
    store = _store(request, settings)
    if not store.disable(user_id):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No registrable account named '{user_id}'. htpasswd accounts "
                "and Google-only sign-ins cannot be disabled from here."
            ),
        )
    logger.info("admin %s disabled account %r", admin, user_id)
    return JSONResponse({"ok": True, "user_id": user_id, "disabled": True})


@router.post("/admin/users/{user_id}/enable")
async def enable_user(
    user_id: str,
    admin: str = Depends(require_admin),
    request: Request = None,
    settings: Settings = Depends(get_settings),
):
    """Undo a disable, so a suspended account can sign in again."""
    store = _store(request, settings)
    if not store.enable(user_id):
        raise HTTPException(
            status_code=404,
            detail=f"No suspended account named '{user_id}'.",
        )
    logger.info("admin %s enabled account %r", admin, user_id)
    return JSONResponse({"ok": True, "user_id": user_id, "disabled": False})


@router.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    purge: bool = False,
    admin: str = Depends(require_admin),
    settings: Settings = Depends(get_settings),
):
    """Remove an account. Keeps the user's recordings unless ``purge=True``.

    Two refusals come first, and both are about the deployment rather than the
    target:

    * **the account the caller is signed in as** -- deleting it would leave the
      operator locked out mid-session with no way to finish the job;
    * **the configured admin** -- an app whose ``TRANSCRIBE_ADMIN_USER`` names
      an account that no longer exists has no admin at all, and the only way
      back is an env edit and a restart.

    What happens to the data is the operator's decision, not this route's. The
    default keeps every voiceprint, snippet, transcript and recording and says
    how many there are, because "remove the login, leave the evidence" is the
    right first move and the only reversible one. ``purge=true`` drops those
    rows and unlinks ``voices/<user_id>/`` -- recordings are not recoverable
    from the database afterwards, which is exactly why it is opt-in and the
    counts are reported either way.
    """
    if user_id == admin:
        raise HTTPException(
            status_code=400,
            detail="This is the account you are signed in as. Sign in as "
                   "another administrator first.",
        )
    if is_admin_user(user_id, settings) and user_id == str(settings.admin_user or "").strip():
        raise HTTPException(
            status_code=400,
            detail="That is the configured TRANSCRIBE_ADMIN_USER account. "
                   "Delete refused; disable it instead, or clear "
                   "TRANSCRIBE_ADMIN_USER first.",
        )

    store = _store(request, settings)
    counts = _user_data_counts(settings, user_id)
    removed = {
        "credentials": store.delete(user_id),
        "google_identities": store.delete_google_identities(user_id),
    }

    db = _db(settings)
    try:
        if TokenDB(db).delete(user_id):
            removed["api_tokens"] = 1
        else:
            removed["api_tokens"] = 0
    except Exception as e:
        logger.debug("Could not drop api token for %r: %s", user_id, e)
        removed["api_tokens"] = 0

    purged = None
    if purge:
        purged = {
            "voiceprints": VoiceprintDB(db).delete_all_for_user(user_id),
            "snippets": SnippetDB(db).delete_all_for_user(user_id),
            "transcripts": TranscriptDB(db).delete_all_for_user(user_id),
        }
        target = _voices_dir(settings, user_id)
        purged["recordings_unlinked"] = bool(
            target and target.is_dir() and _rmtree(target))
        logger.info("admin %s purged account %r: %s", admin, user_id, purged)

    if not any(removed.values()) and not (purged and any(purged.values())):
        raise HTTPException(status_code=404, detail=f"No such account '{user_id}'.")

    logger.info("admin %s deleted account %r (purge=%s)", admin, user_id, purge)
    return JSONResponse({
        "ok": True,
        "user_id": user_id,
        "removed": removed,
        "data": counts,
        # ``purged`` carries the counts, so a caller can tell what was removed
        # from what was only left behind; ``data_kept`` is its inverse in words,
        # because that is the sentence the UI has to print.
        "purged": purged,
        "data_kept": not purge,
    })


def _rmtree(path) -> bool:
    """Unlink a directory tree, reporting success. Never raises."""
    try:
        shutil.rmtree(path)
        return True
    except Exception as e:
        logger.warning("Could not unlink %s: %s", path, e)
        return False
