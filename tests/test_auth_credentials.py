"""Registration, confirmation and sign-in, on the store rather than over HTTP.

This module holds the three things a credentials store can get wrong and still
look fine, each pinned here:

* **A created account must not be signable until the address is confirmed.**
  That is the entire reason a confirmation link exists; an "unverified" flag
  that is checked nowhere is not safety.
* **A missing account and a wrong password must answer the same.** A refusal
  only when the account exists is the exact oracle a credential-stuffing script
  uses to enumerate who has an account.
* **Nothing user-readable may carry the hash or a token.**

Deliberately NOT here: the route-level orderings (throttle before flag, the
rollback, the session rotation), which live in ``test_auth_routes.py`` and need
the web stack, and the settings-derived policy, which lives in
``test_auth_registration_policy.py`` and needs pydantic. Keeping the store's
own tests free of both is what lets them run against a plain SQLAlchemy
install.

Note the store is exercised through a REAL temporary SQLite file rather than a
faked ORM. A fake cannot answer the questions that matter here -- whether the
``disabled_at IS NULL`` guard is genuinely in the WHERE clause, whether a
partial unique index on the address actually refuses a duplicate -- and a fake
that parrots its own filters back would be a test of the fake.
"""

import hashlib

import pytest

pytest.importorskip("sqlalchemy", reason="the credential tables need the DB layer")
pytest.importorskip("numpy", reason="the DatabaseManager imports numpy transitively")

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def fresh_throttle():
    """The signup throttle is a process-global and tests share a process.

    Without this, one test exhausting its quota would silently fail a later one
    in the same run -- a flake that reads as a bug in the store.
    """
    from asr_mcp.auth.store import reset_registration_attempts

    reset_registration_attempts()
    yield
    reset_registration_attempts()


@pytest.fixture
def db(tmp_path):
    from asr_mcp.db.manager import DatabaseManager

    mgr = DatabaseManager(str(tmp_path / "asr.db"))
    try:
        yield mgr
    finally:
        mgr.close()


def _store(db, htpasswd_users=None):
    from asr_mcp.auth.store import CredentialStore

    return CredentialStore(db, htpasswd_users=htpasswd_users)


# ---------------------------------------------------------------------------
# Hashing and policy
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_a_hash_verifies_against_its_own_password(self):
        from asr_mcp.auth.passwords import check_password, hash_password

        assert check_password(PASSWORD, hash_password(PASSWORD)) is True

    def test_the_same_password_hashes_differently_every_time(self):
        from asr_mcp.auth.passwords import hash_password

        first, second = hash_password(PASSWORD), hash_password(PASSWORD)
        assert first != second
        assert first.split("$")[-2] != second.split("$")[-2]  # different salt

    def test_the_plaintext_is_nowhere_in_the_hash(self):
        from asr_mcp.auth.passwords import hash_password

        assert PASSWORD not in hash_password(PASSWORD)
        assert "spot" not in hash_password(PASSWORD)

    def test_the_cost_parameters_travel_inside_the_string(self):
        from asr_mcp.auth.passwords import SCRYPT_N, hash_password

        assert hash_password(PASSWORD).split("$")[1] == str(SCRYPT_N)

    def test_a_hash_written_with_cheaper_parameters_still_verifies(self):
        """The stored hash carries its own `n`, so re-tightening it cannot lock
        anyone out of an account they already have.
        """
        from asr_mcp.auth.passwords import check_password, hash_password

        cheap = hash_password(PASSWORD, n=2 ** 3)
        assert cheap.split("$")[1] == "8"
        assert check_password(PASSWORD, cheap) is True

    def test_a_corrupt_hash_is_false_and_not_an_exception(self):
        from asr_mcp.auth.passwords import check_password

        for corrupt in ("", "scrypt", "scrypt$nope$r$p$dklen$salt$hash",
                        "scrypt$16$8$1$32$!!!$???", "not even close"):
            assert check_password(PASSWORD, corrupt) is False

    def test_a_digest_that_disagrees_with_its_claimed_length_does_not_verify(self):
        """A truncated hash must not verify against itself.

        If the verifier derived the digest length from what it stored instead of
        from the claimed ``dklen``, it would compute a digest of that same
        length and compare equal -- so anyone who could shorten a stored hash by
        one byte would have made it verify against anything.
        """
        import base64

        from asr_mcp.auth.passwords import check_password, hash_password

        parts = hash_password(PASSWORD).split("$")
        parts[4] = "32"  # the claim
        parts[-1] = base64.b64encode(b"short").decode()  # the body does not match
        assert check_password(PASSWORD, "$".join(parts)) is False

    def test_passwords_outside_the_bounds_are_refused(self):
        from asr_mcp.auth.passwords import (
            MAX_PASSWORD_CHARS,
            MIN_PASSWORD_CHARS,
            hash_password,
        )

        with pytest.raises(ValueError):
            hash_password("x" * (MIN_PASSWORD_CHARS - 1))
        with pytest.raises(ValueError):
            hash_password("x" * (MAX_PASSWORD_CHARS + 1))

    def test_the_minimum_length_is_actually_accepted(self):
        from asr_mcp.auth.passwords import (
            MIN_PASSWORD_CHARS,
            check_password,
            hash_password,
        )

        short = "x" * MIN_PASSWORD_CHARS
        assert check_password(short, hash_password(short)) is True


class TestUsernamePolicy:
    def test_a_reasonable_username_comes_back_normalised_not_reshaped(self):
        from asr_mcp.auth.passwords import normalise_username

        assert normalise_username("  Alice  ") == "alice"
        assert normalise_username("Ada.Lovelace_1") == "ada.lovelace_1"

    def test_the_allowed_set_is_exactly_letters_digits_dot_underscore_dash(self):
        from asr_mcp.auth.passwords import normalise_username

        assert normalise_username("a-b_c.d") == "a-b_c.d"

    def test_anything_outside_that_set_is_refused(self):
        from asr_mcp.auth.passwords import normalise_username

        # Interior characters: each of these would be a second spelling of one
        # name, or a character that needs escaping somewhere downstream.
        for bad in ("alice smith", "alice/../bob", "alice@example.com",
                    "alice#1", "alice!", "álice"):
            with pytest.raises(ValueError):
                normalise_username(bad)

    def test_surrounding_whitespace_is_trimmed_rather_than_refused(self):
        """A pasted name with a trailing newline is one account, not an error.

        Trimming is the mem-mcp behaviour and the right one: a form that
        rejected ``"Bob "`` would be a form that makes people retype. What
        matters is that the interior space -- which WOULD be a second spelling
        -- is still refused, so trimming cannot be used to smuggle one in.
        """
        from asr_mcp.auth.passwords import normalise_username

        assert normalise_username("  Ada \n") == "ada"
        assert normalise_username("\tada\n") == "ada"
        with pytest.raises(ValueError):
            normalise_username("ada smith")

    def test_it_must_not_start_or_end_with_a_separator(self):
        from asr_mcp.auth.passwords import normalise_username

        for bad in ("_alice", "alice_", ".alice", "alice."):
            with pytest.raises(ValueError):
                normalise_username(bad)

    def test_the_length_bounds_are_inclusive(self):
        from asr_mcp.auth.passwords import (
            MAX_USERNAME_CHARS,
            MIN_USERNAME_CHARS,
            normalise_username,
        )

        assert normalise_username("a" * MIN_USERNAME_CHARS) == "a" * MIN_USERNAME_CHARS
        assert normalise_username("a" * MAX_USERNAME_CHARS) == "a" * MAX_USERNAME_CHARS
        with pytest.raises(ValueError):
            normalise_username("a" * (MAX_USERNAME_CHARS + 1))

    def test_unusable_values_are_refused(self):
        from asr_mcp.auth.passwords import normalise_username

        # Each of these stringifies to something that is NOT a legal key, so the
        # coercion is harmless: the output is what gets validated.
        for bad in ("", None, "   ", "a", "ab", 12, [], {}):
            with pytest.raises(ValueError):
                normalise_username(bad)

    def test_a_value_whose_string_form_is_valid_is_not_a_hole(self):
        """Coercion is safe because the RESULT is validated, not the input.

        ``1234`` becomes ``"1234"``, which then has to satisfy every rule --
        allowed set, length, alphanumeric at both ends -- before it comes back.
        Refusing a legal key because of the Python type it arrived in would be
        validating the wrong thing; the property that matters is that whatever
        leaves this function is a usable vault key, and that is enforced on the
        output.
        """
        from asr_mcp.auth.passwords import normalise_username

        assert normalise_username(1234) == "1234"
        assert normalise_username("  ADA  ") == "ada"

    def test_the_username_does_not_become_an_address(self):
        """`alice@example.com` must be refused, not split into `alice`.

        The @ is outside the allowed set, so that is free. The reason to say it
        out loud is that the field is named "username" and a user typing their
        email into it is the most common way an address ends up somewhere it was
        never meant to be.
        """
        from asr_mcp.auth.passwords import normalise_username

        with pytest.raises(ValueError):
            normalise_username("alice@example.com")

    def test_no_username_can_contain_a_path_separator(self):
        """The value becomes `voices/<user_id>/`, so `..` is a traversal.

        Not a theoretical concern for a registration form: a single allowed
        separator would let a crafted name escape the per-user voices tree.
        """
        from asr_mcp.auth.passwords import normalise_username

        for bad in ("../etc/passwd", "a/b", "a\\b", "..", "."):
            with pytest.raises(ValueError):
                normalise_username(bad)


class TestEmailValidation:
    def test_a_legal_address_comes_back_normalised_but_otherwise_unchanged(self):
        from asr_mcp.auth.passwords import validate_email

        assert validate_email("  Alice@Example.COM ") == "alice@example.com"

    def test_unusable_shapes_are_refused(self):
        from asr_mcp.auth.passwords import validate_email

        for bad in ("", "   ", "alice", "alice@", "@example.com",
                    "alice example@com", "alice@example", "alice@@example.com"):
            with pytest.raises(ValueError):
                validate_email(bad)

    def test_an_over_long_address_is_refused(self):
        from asr_mcp.auth.passwords import validate_email

        with pytest.raises(ValueError):
            validate_email("a" * 65 + "@example.com")
        with pytest.raises(ValueError):
            validate_email("alice@" + "b" * 254 + ".com")

    def test_empty_and_non_strings_are_refused(self):
        from asr_mcp.auth.passwords import validate_email

        for bad in (None, 1234):
            with pytest.raises(ValueError):
                validate_email(bad)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class TestAccountLifecycle:
    def test_an_account_is_created_and_can_log_in_once_confirmed(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")

        assert store.check_login("alice", PASSWORD)[0] == ""
        token = store.issue_verification_token("alice")
        assert store.verify_email_token(token) == "alice"
        assert store.check_login("alice", PASSWORD) == ("alice", "")

    def test_a_new_account_is_not_verified_yet(self, db):
        store = _store(db)
        record = store.create("alice", PASSWORD, "alice@example.com")

        assert record["email_verified_at"] is None
        assert store.verified("alice") is False

    def test_an_unverified_account_cannot_log_in_even_with_the_right_password(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")

        key, reason = store.check_login("alice", PASSWORD)
        assert key == ""
        assert "confirmed" in reason

    def test_registering_twice_refuses_and_does_not_reset_the_password(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        store.verify_email_token(store.issue_verification_token("alice"))

        with pytest.raises(ValueError):
            store.create("alice", "a-different-password", "alice@example.com")
        assert store.check_login("alice", PASSWORD) == ("alice", "")

    def test_the_address_is_unique_across_accounts(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        with pytest.raises(ValueError):
            store.create("alice2", PASSWORD, "alice@example.com")

    def test_an_unusable_input_creates_nothing(self, db):
        store = _store(db)

        with pytest.raises(ValueError):
            store.create("al", PASSWORD, "alice@example.com")
        with pytest.raises(ValueError):
            store.create("alice", PASSWORD, "not-an-address")
        with pytest.raises(ValueError):
            store.create("alice", "short", "alice@example.com")
        assert store.count() == 0

    def test_deleting_an_account_frees_the_name(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")

        assert store.delete("alice") is True
        assert store.delete("alice") is False
        assert store.user_id_taken("alice") is False
        assert store.count() == 0

    def test_a_disabled_account_cannot_log_in_but_keeps_its_key(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        store.verify_email_token(store.issue_verification_token("alice"))
        assert store.disable("alice") is True

        key, reason = store.check_login("alice", PASSWORD)
        assert key == ""
        assert "disabled" in reason
        # The vault key stays occupied, so the name cannot be registered again
        # over a vault whose voiceprints are still there.
        assert store.user_id_taken("alice") is True

    def test_a_disabled_account_cannot_be_confirmed_by_a_link_in_flight(self, db):
        """`disabled_at IS NULL` is in the WHERE, not checked afterwards.

        Otherwise an operator suspending an account would be undone by the
        confirmation link that was already sitting in the inbox.
        """
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        token = store.issue_verification_token("alice")
        store.disable("alice")

        assert store.verify_email_token(token) is None
        assert store.check_login("alice", PASSWORD)[0] == ""

    def test_a_suspended_account_can_sign_in_again_after_being_enabled(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        store.verify_email_token(store.issue_verification_token("alice"))
        store.disable("alice")

        assert store.enable("alice") is True
        assert store.check_login("alice", PASSWORD) == ("alice", "")

    def test_re_enabling_an_already_enabled_account_changes_nothing(self, db):
        """`updated_at` is guarded so an admin page load cannot bump it.

        A sliding "last changed" would say the account changed on every visit
        that called enable(), which tracks how often somebody looked at the
        table rather than when the account last actually changed.
        """
        store = _store(db)
        record = store.create("alice", PASSWORD, "alice@example.com")
        before = record["updated_at"]

        assert store.enable("alice") is False
        assert store.public("alice")["updated_at"] == before

    def test_a_password_can_be_rotated_for_a_live_account(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        store.verify_email_token(store.issue_verification_token("alice"))

        assert store.set_password("alice", "a-whole-new-password") is True
        assert store.check_login("alice", "a-whole-new-password") == ("alice", "")
        assert store.check_login("alice", PASSWORD)[0] == ""

    def test_rotating_a_suspended_accounts_password_is_refused(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        store.disable("alice")

        assert store.set_password("alice", "a-whole-new-password") is False

    def test_an_unknown_account_has_no_row(self, db):
        store = _store(db)

        assert store.get("nobody") is None
        assert store.public("nobody") is None
        assert store.verified("nobody") is False
        assert store.check_login("nobody", PASSWORD) == ("", "")
        assert store.check_login("", PASSWORD) == ("", "")
        assert store.check_login(None, PASSWORD) == ("", "")


class TestSecretHygiene:
    def test_a_listing_never_carries_the_hash_or_a_token(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        store.issue_verification_token("alice")

        rows = store.list_all()
        assert rows and rows[0]["user_id"] == "alice"
        for row in rows:
            assert "password_hash" not in row
            assert "verification_token" not in row
        assert "password_hash" not in store.public("alice")
        assert "verification_token" not in store.public("alice")

    def test_a_token_is_stored_only_as_a_hash(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        token = store.issue_verification_token("alice")

        stored = store.get("alice")["verification_token"]
        assert stored is not None
        assert token not in stored
        assert stored == hashlib.sha256(token.encode("utf-8")).hexdigest()

    def test_a_hash_is_stored_verbatim_in_its_row(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")

        assert store.get("alice")["password_hash"].startswith("scrypt$")

    def test_an_issued_token_supersedes_the_previous_one(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        first = store.issue_verification_token("alice")
        second = store.issue_verification_token("alice")

        assert first != second
        assert store.verify_email_token(first) is None
        assert store.verify_email_token(second) == "alice"

    def test_a_token_is_spent_once_and_only_once(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")
        token = store.issue_verification_token("alice")

        assert store.verify_email_token(token) == "alice"
        assert store.verify_email_token(token) is None
        assert store.check_login("alice", PASSWORD) == ("alice", "")

    def test_no_token_is_issued_for_an_unknown_or_suspended_account(self, db):
        store = _store(db)
        with pytest.raises(ValueError):
            store.issue_verification_token("nobody")
        store.create("alice", PASSWORD, "alice@example.com")
        store.disable("alice")
        with pytest.raises(ValueError):
            store.issue_verification_token("alice")

    def test_garbage_and_empty_tokens_are_refused_not_raised(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, "alice@example.com")

        for bad in ("", "   ", "not-a-token", "x" * 43, "🙂 not a token"):
            assert store.verify_email_token(bad) is None


class TestNameAndAddressAvailability:
    def test_an_htpasswd_account_reserves_its_name(self, db):
        store = _store(db, htpasswd_users={"dave": "$2b$..."})

        assert store.user_id_taken("dave") is True
        assert store.user_id_taken("DAVE") is True  # case-insensitively
        with pytest.raises(ValueError):
            store.create("dave", PASSWORD, "dave@example.com")

    def test_a_google_vault_reserves_its_key(self, db):
        store = _store(db)
        store.link_google_identity("g-1", "carol@example.com",
                                   email="carol@example.com")

        # The vault key is reserved, so nobody can register the same name.
        assert store.user_id_taken("carol@example.com") is True
        with pytest.raises(ValueError):
            store.create("carol@example.com", PASSWORD, "other@example.com")

    def test_a_google_vaults_address_does_not_block_a_username_signup(self, db):
        """A username signup for an address a Google vault already uses.

        This is the port's documented decision, and it is safe because identity
        is resolved by SUBJECT, never by address: a second account registering
        the same address gets its own vault under its own username, and the
        Google link still resolves to the vault it was created for. The
        alternative -- refusing the signup -- would leak whether an address is
        in use by some other account, which is the enumeration oracle the
        ``email_taken`` docstring rules out.

        What IS blocked is the reverse: signing up with Google for an address
        a credentials account already holds, because there the vault key and the
        address would be the same string.
        """
        store = _store(db)
        store.link_google_identity("g-1", "carol@example.com",
                                   email="carol@example.com")

        carol = store.create("carol", PASSWORD, "carol@example.com")
        assert carol["user_id"] == "carol"
        assert store.resolve_google_identity("g-1")["user_id"] == "carol@example.com"

    def test_an_htpasswd_user_can_take_a_new_address(self, db):
        """The htpasswd store holds no addresses, so it cannot answer the
        "is this address taken" question -- that one is only about accounts this
        app created.
        """
        store = _store(db, htpasswd_users={"dave": "$2b$..."})
        assert store.email_taken("dave@example.com") is False

        store.create("dave2", PASSWORD, "dave@example.com")
        assert store.email_taken("dave@example.com") is True

    def test_the_vault_key_is_the_username_not_the_address(self, db):
        store = _store(db)
        alice = store.create("alice", PASSWORD, "alice@example.com")

        assert alice["user_id"] == "alice"
        assert store.get("alice")["user_id"] == "alice"
        assert store.get("alice")["email"] == "alice@example.com"

    def test_user_id_taken_answers_for_every_store(self, db):
        store = _store(db, htpasswd_users={"dave": "$2b$..."})

        assert store.user_id_taken("") is False
        assert store.user_id_taken(None) is False
        assert store.user_id_taken("len4") is False

        store.create("alice", PASSWORD, "alice@example.com")
        assert store.user_id_taken("alice") is True
        store.link_google_identity("g-2", "erin@example.com")
        assert store.user_id_taken("erin@example.com") is True
        assert store.user_id_taken("dave") is True


class TestGoogleIdentities:
    def test_a_linked_subject_resolves_to_its_vault(self, db):
        store = _store(db)
        store.link_google_identity("g-1", "carol@example.com",
                                   email="carol@example.com", name="Carol")

        assert store.resolve_google_identity("g-1")["user_id"] == "carol@example.com"
        assert set(store.google_user_ids()) == {"carol@example.com"}

    def test_an_unlinked_subject_is_none_not_an_error(self, db):
        store = _store(db)
        assert store.resolve_google_identity("g-nope") is None

    def test_the_same_subject_into_a_different_vault_is_refused(self, db):
        """This is the account-takeover path, and it is a refusal.

        A subject is one person at one provider. Silently reassigning it would
        let whoever holds a working Google login for that address take over a
        different vault, which is a privilege escalation dressed up as a
        convenience.
        """
        store = _store(db)
        store.link_google_identity("g-1", "carol@example.com")

        with pytest.raises(ValueError):
            store.link_google_identity("g-1", "mallory@example.com")

    def test_relinking_the_same_pair_updates_the_display_fields(self, db):
        store = _store(db)
        store.link_google_identity("g-1", "carol@example.com", name="Carol")
        store.link_google_identity("g-1", "carol@example.com", name="Carol Smith")

        assert store.resolve_google_identity("g-1")["display_name"] == "Carol Smith"

    def test_a_read_does_not_bump_last_used_at(self, db):
        """`touch=False` exists so a display never looks like a sign-in."""
        store = _store(db)
        store.link_google_identity("g-1", "carol@example.com")

        assert store.resolve_google_identity("g-1", touch=False)["last_used_at"] is None
        assert store.resolve_google_identity("g-1")["last_used_at"] is not None

    def test_deleting_the_identities_unlinks_the_vault(self, db):
        """A link whose vault is gone is a way back in.

        The credentials row would no longer exist to disable, so the only thing
        standing between a resolved subject and a live vault is the link itself.
        """
        store = _store(db)
        store.link_google_identity("g-1", "carol@example.com")

        assert store.delete_google_identities("carol@example.com") == 1
        assert store.delete_google_identities("carol@example.com") == 0
        assert store.resolve_google_identity("g-1") is None
        assert store.list_google_identities("carol@example.com") == []

    def test_the_admin_listing_omits_the_subject(self, db):
        store = _store(db)
        store.link_google_identity("g-1", "carol@example.com",
                                   email="carol@example.com")

        rows = store.list_all_google_identities()
        assert rows and rows[0]["user_id"] == "carol@example.com"
        assert "subject" not in rows[0]

    def test_only_the_google_provider_is_accepted(self, db):
        store = _store(db)

        with pytest.raises(ValueError):
            store.link_google_identity("g-1", "carol@example.com", provider="microsoft")
        with pytest.raises(ValueError):
            store.resolve_google_identity("g-1", provider="microsoft")

    def test_a_subject_without_a_vault_is_refused(self, db):
        store = _store(db)

        for bad in ("", None, "   "):
            with pytest.raises(ValueError):
                store.link_google_identity("g-1", bad)
        with pytest.raises(ValueError):
            store.link_google_identity("", "carol@example.com")


# ---------------------------------------------------------------------------
# Signup throttle
# ---------------------------------------------------------------------------


class TestSignupThrottle:
    def test_the_limit_is_reached_and_then_refused(self):
        from asr_mcp.auth.store import (
            REGISTRATION_ATTEMPT_LIMIT,
            allow_registration_attempt,
        )

        client = "203.0.113.7"
        for _ in range(REGISTRATION_ATTEMPT_LIMIT):
            assert allow_registration_attempt(client) is True
        assert allow_registration_attempt(client) is False

    def test_the_window_slides_rather_than_resetting(self):
        from asr_mcp.auth.store import (
            REGISTRATION_ATTEMPT_WINDOW,
            allow_registration_attempt,
        )

        moment = 1000.0
        client = "203.0.113.8"
        for _ in range(10):
            assert allow_registration_attempt(client, now=moment) is True
        assert allow_registration_attempt(client, now=moment + 1) is False

        moment += REGISTRATION_ATTEMPT_WINDOW + 1
        assert allow_registration_attempt(client, now=moment) is True

    def test_one_client_exhausting_its_quota_does_not_affect_another(self):
        from asr_mcp.auth.store import (
            REGISTRATION_ATTEMPT_LIMIT,
            allow_registration_attempt,
        )

        for _ in range(REGISTRATION_ATTEMPT_LIMIT):
            allow_registration_attempt("203.0.113.9")
        assert allow_registration_attempt("198.51.100.4") is True

    def test_idle_clients_are_pruned(self):
        from asr_mcp.auth import store as store_module
        from asr_mcp.auth.store import (
            REGISTRATION_ATTEMPT_WINDOW,
            allow_registration_attempt,
        )

        allow_registration_attempt("old-client", now=1.0)
        assert "old-client" in store_module._ATTEMPTS

        allow_registration_attempt("new-client",
                                   now=1.0 + REGISTRATION_ATTEMPT_WINDOW + 1)
        assert "new-client" in store_module._ATTEMPTS
        assert "old-client" not in store_module._ATTEMPTS

    def test_a_client_with_no_address_is_still_counted(self):
        """`unknown` rather than no key at all: an absent hop must not be free."""
        from asr_mcp.auth.store import (
            REGISTRATION_ATTEMPT_LIMIT,
            allow_registration_attempt,
        )

        for _ in range(REGISTRATION_ATTEMPT_LIMIT):
            allow_registration_attempt("")
        assert allow_registration_attempt("") is False


# ---------------------------------------------------------------------------
# Mail -- see test_auth_registration_policy.py, which has pydantic available
# to build the Settings object the mailer reads.
# ---------------------------------------------------------------------------
