"""The one outbound message this app sends: an account confirmation link.

Settings-driven rather than import-time constants (``get_settings()`` is the
convention here, and a test needs to point a settings object at a fake host),
but structurally a port of mem-mcp's mail half: one seam, one message, no
template engine, and a hard distinction between "the account was created" and
"the account was created and the message did not go out".

There is no password-reset flow, and deliberately so -- a reset endpoint is a
second way to take over an account, and the verification link below is the only
credential this app ever puts in an inbox.
"""

import logging
from typing import Optional

from asr_mcp.config.settings import Settings, get_settings

logger = logging.getLogger("asr_mcp.auth.mailer")

SITE_NAME = "ASR-MCP"


def smtp_configured(settings: Optional[Settings] = None) -> bool:
    """True when a confirmation message could actually be sent.

    Host and a From address are the two that make a message meaningful; the
    credentials are optional because a relay on a trusted network may take none.
    """
    if settings is None:
        settings = get_settings()
    return bool(settings.smtp_host and settings.smtp_from)


def verification_email(username: str, link: str, site_name: str = SITE_NAME) -> tuple:
    """The message to send. Returns `(subject, body)` so it is testable as text.

    A function returning strings rather than sending is what lets a test assert
    that the link really is in the body and that nothing else reached a header
    it should not be in.
    """
    subject = f"Confirm your {site_name} account"
    body = "\n".join([
        f"Hello {username},",
        "",
        f"Confirm this address to finish setting up your {site_name} account:",
        "",
        link,
        "",
        "The link works once. If you did not ask for an account, ignore this",
        "message and nothing will happen.",
        "",
    ])
    return subject, body


def send_mail(to_address: str, subject: str, body: str, *,
              client=None, settings: Optional[Settings] = None) -> None:
    """Send one message. Raises on any failure.

    ``client`` is the seam: it is anything with ``send_message(msg)``, and the
    default builds an ``smtplib.SMTP`` over the configured settings. The caller
    is a signup route, which must be able to tell "the account was created"
    from "the account was created and the message did not go out" -- so this
    raises rather than returning a flag nobody checks.
    """
    if settings is None:
        settings = get_settings()
    address = str(to_address or "").strip()
    if not address:
        raise ValueError("an address is required")
    if not smtp_configured(settings):
        raise RuntimeError("outbound mail is not configured on this server")
    if client is not None:
        message = _plain_message(address, subject, body, settings.smtp_from)
        client.send_message(message)
        return

    import smtplib
    from email.message import EmailMessage

    message = EmailMessage()
    message["From"] = settings.smtp_from
    message["To"] = address
    message["Subject"] = subject
    message.set_content(body)
    connection = _smtp_client(settings)
    try:
        connection.send_message(message)
    finally:
        _close_quietly(connection)


def _plain_message(to_address: str, subject: str, body: str, sender: str):
    from email.message import EmailMessage

    message = EmailMessage()
    message["From"] = sender
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(body)
    return message


def _smtp_client(settings: Settings):
    """An authenticated SMTP connection, or an unauthenticated one if no user is set."""
    import smtplib

    connection = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=settings.smtp_timeout)
    if settings.smtp_starttls:
        connection.starttls()
    if settings.smtp_user:
        connection.login(settings.smtp_user, settings.smtp_password)
    return connection


def _close_quietly(client) -> None:
    try:
        client.quit()
    except Exception:
        pass


def log_send_failure(exc: Exception) -> None:
    """Record a failed message without putting the address or the link in the log.

    The address is the one field a caller passed in from a registration form,
    and the link contains a live verification token; neither belongs in a log
    file that ships off-box.
    """
    logger.warning("account confirmation email could not be sent: %s", type(exc).__name__)
