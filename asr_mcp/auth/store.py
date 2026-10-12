"""The credentials and external-identity tables, behind one object.

Structurally a port of mem-mcp's half of ``sessions.py``, on top of
SQLAlchemy and the pydantic ``Settings`` object instead of stdlib sqlite and
``os.getenv`` -- because this app already has a DB layer and a settings object,
and two of either in one deployment is a bug rather than an abstraction.

The behaviours worth knowing before changing anything here:

* **A username is a vault key.** ``user_id`` is what every voiceprint, snippet
  and transcript row is scoped by and what names the ``voices/<user_id>/``
  directory. That is why ``user_id_taken`` consults three stores rather than
  this table alone: registering over an htpasswd user or an existing Google
  vault would produce two passwords into one set of voiceprints.
* **An address is not a key.** A provider can reassign one, so keying a vault
  on an address would hand the previous occupant's speakers to whoever holds
  it next. The address exists to prove the account and to enforce
  one-per-account.
* **Nothing here is an upsert.** ``create`` refuses a taken name or address,
  because an upsert would silently reset the password of an existing account,
  which is the failure mode a "create" endpoint must not have.
* **A disabled row keeps its key.** ``enable`` is guarded on
  ``disabled_at IS NOT NULL`` so re-enabling an already-enabled account does not
  bump ``updated_at``.
"""

import hashlib
import secrets
import threading
import time
from typing import Optional

from asr_mcp.auth.passwords import (
    MAX_EMAIL_CHARS,
    check_password,
    hash_password,
    normalise_username,
    validate_email,
)
from asr_mcp.db.manager import DatabaseManager
from asr_mcp.db.models import CredentialModel, GoogleIdentityModel

GOOGLE_PROVIDER = "google"
MAX_NAME_CHARS = 120

# Signup attempts per client per window. There is no rate limiting anywhere else
# in this app, and a registration endpoint is the one place where "guess a
# password" is a thing an outsider can attempt at will. This is a plain
# in-process counter, which is honest about what it is: it stops a script, not a
# distributed one. The reverse proxy in front of the app is the real answer.
REGISTRATION_ATTEMPT_LIMIT = 10
REGISTRATION_ATTEMPT_WINDOW = 600.0

_ATTEMPTS_LOCK = threading.Lock()
_ATTEMPTS: dict = {}


def _now(now) -> float:
    return float(now) if now is not None else time.time()


def _normalise_provider(provider) -> str:
    """The provider key. Only one is implemented, so this is also the validator.

    Kept as a function rather than a constant comparison at each call site so a
    second provider has one place to be added, and so an unknown one fails here
    with the name in the message rather than silently writing a row no lookup
    will ever read.
    """
    name = str(provider or "").strip().lower()
    if name != GOOGLE_PROVIDER:
        raise ValueError(f"provider must be {GOOGLE_PROVIDER!r}")
    return name


def allow_registration_attempt(client: str, now=None) -> bool:
    """Count one signup attempt for `client`; False if it is over the limit.

    In-process and therefore per-worker, which is stated plainly rather than
    dressed up: it raises the cost of a script and does nothing about a botnet.

    The window slides rather than resetting on the boundary, so a client cannot
    get 2x the quota by straddling it. Entries are pruned on every call, which is
    what keeps this from being a slow memory leak keyed by attacker-chosen
    strings.
    """
    key = str(client or "").strip() or "unknown"
    moment = _now(now)
    cutoff = moment - REGISTRATION_ATTEMPT_WINDOW
    with _ATTEMPTS_LOCK:
        stamps = [value for value in _ATTEMPTS.get(key, ()) if value > cutoff]
        if len(stamps) >= REGISTRATION_ATTEMPT_LIMIT:
            _ATTEMPTS[key] = stamps
            return False
        stamps.append(moment)
        _ATTEMPTS[key] = stamps
        # Prune clients that are over the window and not the one being asked
        # about, so the dict tracks live traffic rather than every address ever
        # seen. Bounded by the number of concurrent clients, not by history.
        for other in [k for k, v in _ATTEMPTS.items()
                      if k != key and not any(t > cutoff for t in v)]:
            _ATTEMPTS.pop(other, None)
    return True


def reset_registration_attempts() -> None:
    """Drop every recorded attempt. For tests, which share the process counter."""
    with _ATTEMPTS_LOCK:
        _ATTEMPTS.clear()


class CredentialStore:
    """Every row of `credentials` and `google_identities`, plus the policy.

    ``htpasswd_users`` is the parsed ``{name: hash}`` map from
    ``app.state.htpasswd_users``. It is passed in by the router and re-read from
    ``settings.htpasswd_path`` when absent, and it is what makes
    ``user_id_taken`` a pure in-process question: the htpasswd file is already
    parsed in Python at startup, so there is no reason to shell out to the
    ``htpasswd`` binary -- which the container does not ship anyway.
    """

    def __init__(self, db: DatabaseManager,
                 htpasswd_users: Optional[dict] = None,
                 settings=None):
        self._db = db
        self._htpasswd_users = htpasswd_users
        self._settings = settings

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, user_id) -> Optional[dict]:
        """The credential row for a vault key, or None. Includes the password hash.

        The only method here that returns the hash, and every caller checks a
        password immediately. Anything that renders a row to a client uses
        :meth:`public` instead.
        """
        key = str(user_id or "").strip()
        if not key:
            return None
        with self._db.get_session() as session:
            row = session.query(CredentialModel).filter_by(user_id=key).first()
            return self._row_to_dict(row) if row is not None else None

    def public(self, user_id) -> Optional[dict]:
        """The credential row without the hash or the verification token."""
        record = self.get(user_id)
        if record is None:
            return None
        return {
            "user_id": record["user_id"],
            "email": record["email"],
            "email_verified_at": record["email_verified_at"],
            "verification_sent_at": record["verification_sent_at"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "disabled_at": record["disabled_at"],
        }

    def list_all(self) -> list:
        """Every registered account, newest first. Never includes the hash or token."""
        with self._db.get_session() as session:
            rows = (
                session.query(CredentialModel)
                .order_by(CredentialModel.created_at.desc(), CredentialModel.user_id)
                .all()
            )
            return [self._public_row(row) for row in rows]

    def user_ids(self) -> list:
        """Every vault key this store owns, sorted. Used by the admin table."""
        with self._db.get_session() as session:
            rows = session.query(CredentialModel.user_id).all()
        return sorted(row[0] for row in rows)

    def count(self) -> int:
        with self._db.get_session() as session:
            return session.query(CredentialModel).count()

    # ------------------------------------------------------------------
    # Account lifecycle
    # ------------------------------------------------------------------

    def create(self, username, password: str, email: str = "", now=None) -> dict:
        """Register a username/password account and return the public record.

        Three refusals, and each one is a different question: the username must
        be free (across all three stores), the address must be free, and the
        password must satisfy the policy. None of them upserts.

        The row is created **unverified**. That is not an oversight to be tidied
        up later by a flag: the address has not been proven to belong to whoever
        typed it, and until it has, the account cannot sign in. :meth:`check_login`
        is what refuses it.
        """
        key = normalise_username(username)
        address = validate_email(email) if str(email or "").strip() else ""
        if self.user_id_taken(key):
            raise ValueError("that username is taken")
        if address and self.email_taken(address):
            raise ValueError("that email address is already registered")
        moment = _now(now)
        encoded = hash_password(password)
        with self._db.get_session() as session:
            session.add(CredentialModel(
                user_id=key,
                password_hash=encoded,
                email=address,
                created_at=moment,
                updated_at=moment,
            ))
            session.commit()
        return self.public(key)

    def delete(self, user_id, now=None) -> bool:
        """Remove an account row outright. Used only to undo a failed signup.

        This is not how an account is turned off -- :meth:`disable` keeps the
        vault key occupied so the name cannot be registered again and come back
        pointing at a vault whose records are still there. This exists for one
        case: a signup created a row and the confirmation message could not be
        sent, so leaving the row behind would strand a half-account that can
        never sign in and whose name the person cannot reclaim.
        """
        key = str(user_id or "").strip()
        if not key:
            return False
        with self._db.get_session() as session:
            count = (
                session.query(CredentialModel)
                .filter_by(user_id=key)
                .delete()
            )
            session.commit()
            return bool(count)

    def disable(self, user_id, now=None) -> bool:
        """Refuse future logins for this account without releasing its vault key."""
        key = str(user_id or "").strip()
        if not key:
            return False
        moment = _now(now)
        with self._db.get_session() as session:
            count = (
                session.query(CredentialModel)
                .filter(CredentialModel.user_id == key,
                        CredentialModel.disabled_at.is_(None))
                .update({"disabled_at": moment, "updated_at": moment})
            )
            session.commit()
            return bool(count)

    def enable(self, user_id, now=None) -> bool:
        """Undo :meth:`disable`, so a suspended account can sign in again.

        The ``disabled_at IS NOT NULL`` guard is the whole reason this is not a
        bare ``SET disabled_at = NULL``. An enabled account that was never
        disabled would have its `updated_at` bumped on every admin page load, so
        "last changed" would track how often somebody looked at the table rather
        than when the account last actually changed -- the same "updated on
        read" mistake a sliding session makes.
        """
        key = str(user_id or "").strip()
        if not key:
            return False
        moment = _now(now)
        with self._db.get_session() as session:
            count = (
                session.query(CredentialModel)
                .filter(CredentialModel.user_id == key,
                        CredentialModel.disabled_at.is_not(None))
                .update({"disabled_at": None, "updated_at": moment})
            )
            session.commit()
            return bool(count)

    def set_password(self, user_id, password: str, now=None) -> bool:
        """Replace the password for an existing registered account."""
        key = str(user_id or "").strip()
        if not key:
            return False
        moment = _now(now)
        with self._db.get_session() as session:
            count = (
                session.query(CredentialModel)
                .filter(CredentialModel.user_id == key,
                        CredentialModel.disabled_at.is_(None))
                .update({"password_hash": hash_password(password), "updated_at": moment})
            )
            session.commit()
            return bool(count)

    # ------------------------------------------------------------------
    # Signing in
    # ------------------------------------------------------------------

    def check_login(self, user_id, password: str) -> tuple:
        """`(user_id, reason)` -- reason is "" when the password is the right one.

        Three separate refusals with three separate answers, because the
        account holder needs to be told which one happened. A disabled account
        is refused even with the right password, and so is an account whose
        address has never been confirmed; the password can be exactly right and
        the account still not be usable, which is the entire point of
        confirming an address.

        A *missing* account, a wrong password, and a missing key all return
        ``("", "")`` -- the same answer -- and that is deliberate. Distinguishing
        "no such user" from "wrong password" is exactly the oracle a credential
        stuffing script uses to enumerate accounts, and it serialises a whole
        probe as "any data returned ⇒ a mismatch". If the reason is blank here,
        the next store to try is the operator's htpasswd file: this row is
        missing, so the login has to keep looking before anyone is told
        anything, and it will fail the same way if the password is wrong.
        """
        key = str(user_id or "").strip()
        record = self.get(key) if key else None
        if record is None:
            return "", ""
        if record.get("disabled_at") is not None:
            return "", "that account has been disabled"
        if record.get("email_verified_at") is None:
            return "", ("this account has not been confirmed yet — open the link "
                        "we sent to your email address")
        if not check_password(password, record["password_hash"]):
            return "", ""
        return key, ""

    def verified(self, user_id) -> bool:
        """True if this account may sign in, on the address question.

        Only meaningful for a row here. An htpasswd account has no row and is
        never asked -- the operator manages those by hand and there is nothing to
        verify -- so this is called only after a credentials row was found.
        """
        record = self.get(user_id)
        if record is None:
            return False
        return record.get("email_verified_at") is not None

    # ------------------------------------------------------------------
    # Confirmation tokens
    # ------------------------------------------------------------------

    def issue_verification_token(self, user_id, now=None) -> str:
        """Mint and store a fresh confirmation token for a password account.

        Returns the plaintext, which goes in the emailed link and is never
        stored: the row holds the SHA-256, so a copy of the database is not a
        list of working confirmation links. Same rule as an API token, and for
        the same reason -- a link in an inbox is a credential until it is spent.
        """
        key = str(user_id or "").strip()
        if not key:
            raise ValueError("an account is required")
        moment = _now(now)
        token = secrets.token_urlsafe(32)
        with self._db.get_session() as session:
            count = (
                session.query(CredentialModel)
                .filter(CredentialModel.user_id == key,
                        CredentialModel.disabled_at.is_(None),
                        CredentialModel.email != "")
                .update({"verification_token": _token_hash(token),
                         "verification_sent_at": moment,
                         "updated_at": moment})
            )
            session.commit()
            if not count:
                # Either no such account, or one with no address to send to.
                # Both are "cannot confirm this", and the caller's only correct
                # response is the same, so one message covers them.
                raise ValueError("that account cannot be verified")
        return token

    def verify_email_token(self, token, now=None) -> Optional[str]:
        """Spend a confirmation token; return the account it confirmed, or None.

        Single-use by construction: the UPDATE that stamps ``email_verified_at``
        also clears ``verification_token``, so a link clicked twice confirms
        once and the second click is a miss. Doing it in one statement rather
        than a SELECT then an UPDATE is what makes that true under two
        simultaneous clicks -- the race would otherwise leave the second caller
        believing it had confirmed something.
        """
        text = str(token or "").strip()
        if not text:
            return None
        moment = _now(now)
        digest = _token_hash(text)
        with self._db.get_session() as session:
            row = (
                session.query(CredentialModel)
                # disabled_at IS NULL is in the WHERE because a disabled account
                # must not be confirmable by a link that was already in flight:
                # clicking it would otherwise re-open a suspended account
                # through the mail client rather than through an administrator.
                .filter(CredentialModel.verification_token == digest,
                        CredentialModel.disabled_at.is_(None))
                .first()
            )
            if row is None:
                return None
            key = row.user_id
            session.query(CredentialModel).filter(
                CredentialModel.verification_token == digest
            ).update({"email_verified_at": moment, "verification_token": None,
                      "updated_at": moment})
            session.commit()
        return key

    # ------------------------------------------------------------------
    # Name and address availability
    # ------------------------------------------------------------------

    def user_id_taken(self, user_id) -> bool:
        """True if this vault key already exists in *any* of the three stores.

        A registered account's ``user_id`` is its username, and a username can
        already name something: an htpasswd user (the operator's own account, or
        one they added by hand) or the vault a Google account is already signed
        into. Creating a credentials row for either would produce two ways into
        one vault with two independent passwords -- or, worse, silently re-point
        an existing vault at a password the person registering just chose, which
        is an account takeover dressed as a signup.
        """
        key = str(user_id or "").strip().lower()
        if not key:
            return False
        with self._db.get_session() as session:
            row = (
                session.query(CredentialModel.user_id)
                .filter(CredentialModel.user_id == key)
                .first()
            )
            if row is not None:
                return True
            row = (
                session.query(GoogleIdentityModel.user_id)
                .filter(GoogleIdentityModel.user_id == key)
                .first()
            )
        if row is not None:
            return True
        # The htpasswd file is already parsed into a dict at startup, so this is
        # a set lookup rather than a subprocess -- and it is case-insensitive,
        # because `Memories` and `memories` are one operator to a human reader
        # even though the file itself is case-sensitive when verifying a
        # password.
        return key in self._htpasswd_keys()

    def email_taken(self, email) -> bool:
        """True if some account is already registered against this address.

        Separate from ``user_id_taken`` because the two answer different
        questions. One address per account is enforced by a unique index, so
        this is how a second signup for an address is turned into "sign in
        instead" rather than a database error. It deliberately does not consult
        htpasswd or Google: an operator's account has no address here, and
        refusing a signup because an unrelated account shares a name somewhere
        else would be a coincidence mistaken for a policy.
        """
        address = str(email or "").strip().lower()
        if not address:
            return False
        with self._db.get_session() as session:
            row = (
                session.query(CredentialModel.user_id)
                .filter(CredentialModel.email == address)
                .first()
            )
        return row is not None

    def _htpasswd_keys(self) -> set:
        """The lowercased names in the operator's htpasswd file."""
        if self._htpasswd_users is not None:
            return {str(name).strip().lower() for name in self._htpasswd_users}
        from asr_mcp.api.auth import parse_htpasswd

        settings = self._settings
        if settings is None:
            from asr_mcp.config.settings import get_settings
            settings = get_settings()
        return {str(name).strip().lower() for name in parse_htpasswd(settings.htpasswd_path)}

    # ------------------------------------------------------------------
    # External identities (Google)
    # ------------------------------------------------------------------

    def link_google_identity(self, subject, user_id, email="", name="",
                             provider=GOOGLE_PROVIDER, now=None) -> dict:
        """Point an external subject at a vault, and return the record.

        Raises ValueError if the subject is already linked to a *different*
        vault. A subject is one account at one provider, so a second link is a
        contradiction rather than a second grant -- and silently reassigning
        would let whoever holds a working Google login for one address take
        over a different vault, which is a privilege escalation dressed up as a
        convenience.

        Re-linking the same subject to the same vault is an update of the
        display fields, so a mistyped address can be corrected without a
        delete. That is why this is an upsert on the *same* pair and still a
        refusal on a *different* one.

        ``provider`` is a parameter rather than a constant at the call site so
        an unrecognised one is refused here, with its name in the message,
        instead of writing a row no lookup will ever read.
        """
        provider = _normalise_provider(provider)
        subject = str(subject or "").strip()
        key = str(user_id or "").strip()
        if not subject:
            raise ValueError("a subject is required")
        if not key:
            raise ValueError("a linked identity needs a vault")
        address = str(email or "").strip()[:MAX_EMAIL_CHARS]
        display_name = " ".join(str(name or "").split())[:MAX_NAME_CHARS]
        moment = _now(now)

        with self._db.get_session() as session:
            existing = (
                session.query(GoogleIdentityModel)
                .filter_by(provider=provider, subject=subject)
                .first()
            )
            if existing is not None and existing.user_id != key:
                raise ValueError(
                    "that Google account is already linked to a different vault"
                )
            if existing is None:
                session.add(GoogleIdentityModel(
                    provider=provider, subject=subject, user_id=key,
                    email=address, display_name=display_name, created_at=moment,
                ))
            else:
                existing.email = address
                existing.display_name = display_name
            session.commit()
            row = (
                session.query(GoogleIdentityModel)
                .filter_by(provider=provider, subject=subject)
                .first()
            )
            return self._identity_to_dict(row)

    def resolve_google_identity(self, subject, provider=GOOGLE_PROVIDER,
                                now=None, touch: bool = True) -> Optional[dict]:
        """The vault a verified subject is linked to, or None.

        ``last_used_at`` is bumped only on the success path: a subject that is
        not linked must not be able to keep a row alive by being replayed, and
        there is no way to warm a row that does not exist.

        This is only ever called after the OAuth exchange has completed. It is a
        lookup, not a check, and its name says so deliberately.
        """
        if not subject:
            return None
        provider = _normalise_provider(provider)
        moment = _now(now)
        with self._db.get_session() as session:
            row = (
                session.query(GoogleIdentityModel)
                .filter_by(provider=provider, subject=str(subject).strip())
                .first()
            )
            if row is None:
                return None
            if touch:
                row.last_used_at = moment
                session.commit()
            return self._identity_to_dict(row)

    def delete_google_identities(self, user_id) -> int:
        """Unlink every external identity for one vault. Count of rows removed.

        Deleting the credentials row alone leaves the Google link in place, and
        a link whose vault has no account is a way back in:
        ``resolve_google_identity`` would resolve the subject to a user_id that
        can no longer be disabled, because there is nothing left to disable.
        The delete path calls both.
        """
        key = str(user_id or "").strip()
        if not key:
            return 0
        with self._db.get_session() as session:
            count = (
                session.query(GoogleIdentityModel)
                .filter(GoogleIdentityModel.user_id == key)
                .delete()
            )
            session.commit()
            return count

    def list_google_identities(self, user_id) -> list:
        """Every external identity linked to one vault, oldest first."""
        key = str(user_id or "").strip()
        if not key:
            return []
        with self._db.get_session() as session:
            rows = (
                session.query(GoogleIdentityModel)
                .filter(GoogleIdentityModel.user_id == key)
                .order_by(GoogleIdentityModel.created_at, GoogleIdentityModel.subject)
                .all()
            )
            return [self._identity_to_dict(row) for row in rows]

    def list_all_google_identities(self) -> list:
        """Every linked external identity, every vault at once.

        ``list_google_identities`` reads one vault's own rows and is what a user
        calls about themselves. The admin table is the other direction: it has to
        walk the whole table to decide which vaults have a Google sign-in,
        because a vault with no row here is indistinguishable from one that has
        never been listed. The ``subject`` is dropped here rather than in the
        renderer: it is the primary key of an identity, an admin table has no use
        for it, and putting it one function away from a JSON response is how a
        primary key ends up on a page.
        """
        with self._db.get_session() as session:
            rows = (
                session.query(GoogleIdentityModel)
                .order_by(GoogleIdentityModel.created_at,
                          GoogleIdentityModel.provider,
                          GoogleIdentityModel.user_id)
                .all()
            )
            identities = []
            for row in rows:
                record = self._identity_to_dict(row)
                record.pop("subject", None)
                identities.append(record)
            return identities

    def google_user_ids(self) -> dict:
        """``{user_id: [identities]}`` for every vault with a Google sign-in."""
        return {row["user_id"]: [row] for row in self.list_all_google_identities()}

    # ------------------------------------------------------------------
    # Row mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: CredentialModel) -> dict:
        return {
            "user_id": row.user_id,
            "password_hash": row.password_hash,
            "email": row.email,
            "email_verified_at": row.email_verified_at,
            "verification_token": row.verification_token,
            "verification_sent_at": row.verification_sent_at,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "disabled_at": row.disabled_at,
        }

    @staticmethod
    def _public_row(row: CredentialModel) -> dict:
        record = CredentialStore._row_to_dict(row)
        record.pop("password_hash", None)
        record.pop("verification_token", None)
        return record

    @staticmethod
    def _identity_to_dict(row: GoogleIdentityModel) -> dict:
        return {
            "provider": row.provider,
            "subject": row.subject,
            "user_id": row.user_id,
            "email": row.email,
            "display_name": row.display_name,
            "created_at": row.created_at,
            "last_used_at": row.last_used_at,
        }


def _token_hash(token) -> str:
    """The stored form of a confirmation token. Same shape as an API token digest."""
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def registration_enabled(method="email", settings=None) -> bool:
    """Whether this signup *method* may be offered right now.

    Two things are required and they are not the same axis.
    ``TRANSCRIBE_REGISTRATION_ENABLED`` is the operator's decision to have
    signup at all -- off by default, because an endpoint that creates accounts
    should be something an operator turned on rather than something a
    deployment inherits by upgrading. The second half is whether the method
    could actually work:

      * **email** needs somewhere to send the confirmation message. Without
        SMTP settings every account would be created and none could ever be
        confirmed, so the form is not shown at all rather than shown and
        useless.
      * **google** needs a client id *and* a client secret. The secret is not
        optional for the code exchange, so a half-configured deployment can
        start a login and not finish it.

    "Enabled but not configured" is a broken feature rather than a feature, so it
    is not offered. The unknown method raises rather than defaulting, because a
    typo in the caller should be a loud failure and not a silently hidden form.
    """
    if settings is None:
        from asr_mcp.config.settings import get_settings
        settings = get_settings()
    if not settings.registration_enabled:
        return False
    name = str(method or "").strip().lower()
    if name == "email":
        from asr_mcp.auth.mailer import smtp_configured
        return smtp_configured(settings)
    if name != GOOGLE_PROVIDER:
        raise ValueError(f"unknown registration method: {method!r}")
    try:
        from asr_mcp.auth import google_auth
    except ImportError:
        # The Google module arrives separately from the credential store; until
        # it exists, "not configured" is the honest answer rather than a crash.
        return False
    return google_auth.configured(settings)


def credential_store(settings=None, htpasswd_users: Optional[dict] = None) -> CredentialStore:
    """A store on the configured database. The one way the routers get one."""
    if settings is None:
        from asr_mcp.config.settings import get_settings
        settings = get_settings()
    return CredentialStore(DatabaseManager(settings.db_path),
                           htpasswd_users=htpasswd_users,
                           settings=settings)
