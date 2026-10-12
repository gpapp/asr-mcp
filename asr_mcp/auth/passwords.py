"""Password hashing and identity-field validation. Stdlib only.

``asr_mcp.api.auth`` already owns a ``verify_password(stored_hash, password)``
for the htpasswd file, and its arguments are in the OPPOSITE order to the
functions here. Two functions with the same name and swapped parameters in one
codebase is a call site waiting to be written wrong, so the hash-password pair
is named ``hash_password`` / ``check_password`` and takes the *password* first,
which is also the order ``bcrypt.checkpw`` uses.

Nothing in this module imports a database, a settings object or the web
framework: the cost parameters, the password policy and the shape of a username
are all pure decisions that a test can exercise with the standard library
installed and nothing else.
"""

import base64
import hashlib
import hmac
import os
import string

# scrypt parameters. N=2**14 with r=8 is ~16 MiB of memory per hash and roughly
# 50-100ms, which is the intended cost: it is paid on login and on signup, never
# on a request that has no password to check. Stored inside the encoded string
# rather than read from configuration, so raising N later cannot lock anyone out
# of their own account.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16

# A password is not length-capped by the hash, but an unbounded one is a
# denial-of-service primitive: scrypt's cost is in N, but the input still has to
# be hashed, and a 100 MB "password" is 100 MB of work per unauthenticated
# request on an endpoint that has no rate limit upstream.
MIN_PASSWORD_CHARS = 10
MAX_PASSWORD_CHARS = 1024

# A username becomes the vault key -- `voices/<user_id>/`, the user_id column of
# every voiceprint and snippet row -- so the rules are about one thing: two
# people must never end up with the same key, and a key must never contain
# anything that needs escaping in a URL, a log line, an HTML attribute or a
# filesystem path. Lowercase-only also means `Alice` and `alice` are one account
# rather than two vaults.
MIN_USERNAME_CHARS = 3
MAX_USERNAME_CHARS = 64
_USERNAME_ALLOWED = set(string.ascii_lowercase + string.digits + "._-")

# An address has to fit a header and an SMTP envelope. These are the practical
# maxima, not RFC allowances.
MAX_EMAIL_CHARS = 254


def hash_password(password: str, *, n: int = SCRYPT_N, r: int = SCRYPT_R,
                  p: int = SCRYPT_P, dklen: int = SCRYPT_DKLEN) -> str:
    """Hash a password into the one string this app stores.

    scrypt over PBKDF2 because PBKDF2's only cost knob is iterations and
    iterations are cheap to raise on a GPU -- a hash an attacker can do ten
    billion times a second is a hash they get to try ten billion times a
    second. scrypt's cost is in *memory*, which a GPU does not have to spare,
    so the same work stays expensive on the attacker's hardware and not just
    on the server's.

    The parameters travel inside the string (`scrypt$n$r$p$dklen$salt$hash`) so
    they can be raised later without invalidating anyone's existing password: a
    verifier reads the cost from what it was given rather than from what this
    module was compiled with.
    """
    text = str(password or "")
    if len(text) < MIN_PASSWORD_CHARS:
        raise ValueError(f"the password must be at least {MIN_PASSWORD_CHARS} characters")
    if len(text) > MAX_PASSWORD_CHARS:
        raise ValueError(f"the password must be at most {MAX_PASSWORD_CHARS} characters")
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.scrypt(text.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=dklen)
    # dklen travels in the string as well as n/r/p, and that is not redundancy.
    # A verifier that *derived* the length from the stored digest would compute
    # a digest of that same length and compare equal -- so anyone who could
    # shorten the stored hash by one byte would have made it verify against
    # anything.
    return "$".join([
        "scrypt", str(n), str(r), str(p), str(dklen),
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    ])


def check_password(password: str, encoded: str) -> bool:
    """True if `password` produced `encoded`. Never raises on a malformed hash.

    A stored hash that will not parse is a *false*, not an exception. The caller
    is a login path, and a row with a truncated hash -- restored from a damaged
    savepoint, hand-edited, written by a future version with a different format
    -- must read as "this account cannot log in" rather than as a 500 on every
    request that touches it.
    """
    try:
        parts = str(encoded or "").split("$")
        if len(parts) != 7 or parts[0] != "scrypt":
            return False
        _, raw_n, raw_r, raw_p, raw_dklen, raw_salt, raw_digest = parts
        dklen = int(raw_dklen)
        salt = base64.b64decode(raw_salt, validate=True)
        expected = base64.b64decode(raw_digest, validate=True)
        if len(expected) != dklen:
            # The stored digest disagrees with the length it claims to be, so
            # the row is damaged rather than merely foreign. Refusing here is
            # what stops a truncated hash from verifying against itself.
            return False
        digest = hashlib.scrypt(
            str(password or "").encode("utf-8"), salt=salt,
            n=int(raw_n), r=int(raw_r), p=int(raw_p), dklen=dklen,
        )
    except Exception:
        return False
    # compare_digest, not ==: a byte-by-byte comparison leaks the length of the
    # matching prefix through timing, and this is the one comparison where an
    # attacker gets to run it repeatedly against a value of their choosing.
    return hmac.compare_digest(digest, expected)


def normalise_username(username) -> str:
    """The vault key for a password account: lowercased, trimmed, validated.

    The character set is the whole policy. Letters, digits, dot, underscore and
    hyphen cannot collide with a path separator, cannot be confused with each
    other in most fonts, and cannot smuggle a quote into a log line, a URL or
    an HTML attribute. Anything else is refused rather than transliterated,
    because a transliteration is a second spelling of the same name and two
    spellings are the bug this function exists to prevent.

    Because this value becomes a directory name under ``voices/``, the absolute
    ban on ``/`` and ``..`` is not incidental -- a validator that allowed either
    would be a path traversal reached through a registration form.
    """
    text = str(username or "").strip().lower()
    if not text:
        raise ValueError("a username is required")
    if len(text) < MIN_USERNAME_CHARS:
        raise ValueError(f"a username must be at least {MIN_USERNAME_CHARS} characters")
    if len(text) > MAX_USERNAME_CHARS:
        raise ValueError(f"a username must be at most {MAX_USERNAME_CHARS} characters")
    if not text[0].isalnum() or not text[-1].isalnum():
        # A leading or trailing separator makes two keys that look identical at
        # a glance ("alice" and "alice_") and reads as a typo in every log line.
        raise ValueError("a username must start and end with a letter or digit")
    if any(character not in _USERNAME_ALLOWED for character in text):
        raise ValueError(
            "a username may only contain letters, digits, dots, underscores and hyphens"
        )
    return text


def validate_email(email) -> str:
    """Return the normalised address, or raise ValueError with a usable reason.

    The shape check is deliberately loose -- one `@`, something on each side, a
    dot in the domain, no whitespace. A strict RFC 5322 parser rejects valid
    addresses (`a@b` is legal, `user@localhost` is legal in practice) and the
    only authority on whether an address receives mail is a message to it. What
    is being stopped here is a string that could not be an address at all, and
    a string long enough or shaped oddly enough to be something else.
    """
    text = str(email or "").strip().lower()
    if not text:
        raise ValueError("an email address is required")
    if len(text) > MAX_EMAIL_CHARS:
        raise ValueError(f"that email address is longer than {MAX_EMAIL_CHARS} characters")
    if text.count("@") != 1:
        raise ValueError("that does not look like an email address")
    local, _, domain = text.partition("@")
    if not local or not domain:
        raise ValueError("that does not look like an email address")
    if any(character.isspace() for character in text):
        raise ValueError("an email address cannot contain spaces")
    if "." not in domain:
        raise ValueError("that email address has no domain")
    if len(local) > 64 or len(domain) > 253:
        raise ValueError("that email address is too long")
    return text
