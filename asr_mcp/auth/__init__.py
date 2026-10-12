"""Self-service account registration and sign-in.

Three pieces, each with a reason to be its own module:

  * ``passwords`` -- hashing and identity-field validation. Stdlib only.
  * ``mailer``    -- the one message the app sends, behind an injectable client.
  * ``store``     -- the credentials and google_identities tables.

``api/auth_router.py`` owns the HTTP surface (login, register, verify, the
Google redirects). This package is the decision layer it calls, and it never
imports the web framework.
"""
