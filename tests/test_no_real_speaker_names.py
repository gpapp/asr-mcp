"""No real speaker names in the test suite.

The tests were written against the live voiceprint library, so they carried real
people's names as fixtures. That is not a naming-style question: a test that
asserts something about ``Ahed Hatter`` pins behaviour to one person's data, and
a name in a diff is a name in a repository other people can read.

One name is allowed, ``Gergely Papp``: several tests predate this rule and use
that fixture in ways that are awkward to rewrite mechanically (profile-merging
and second-pass-claim fixtures that read better with a real-looking name).

This list is deliberately a snapshot, not a live database read -- a test that
opened the user's SQLite file would fail on every machine but this one. New real
names get added here when they are found.
"""

import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()

#: Real profile names that must not appear in any test file.
FORBIDDEN = (
    "Ahed Hatter",
    "Alexander Radyukin",
    "Amit Junankar",
    "Chris Smith",
    "Daniil Khabarov",
    "Darren Lester",
    "David Campbell",
    "David Ryan",
    "Eric Dittrich",
    "Freddie Leigh-Akompi",
    "Gareth Thompson",
    "Gourav Sharma",
    "Hany Shehata",
    "Ibrahim Shabea",
    "Ilze Tiltina",
    "Ismael Capel",
    "Kateryna Bukhtoiarova",
    "Maneesh Mali",
    "Marius Babrauskas",
    "Maximilian Beckers",
    "Nico Zeissig",
    "Norman Weigel",
    "Oleksandr Mashchenko",
    "Petr Travkin",
    "Raphael Douin",
    "Reagan Blevins",
    "Rob Fraser",
    "Roman Pak",
    "Sam Laundy",
)

#: The one permitted name, and the surnames that must not be written bare.
ALLOWED = "Gergely Papp"

# Surnames and given names checked on their own, so a test cannot smuggle a real
# person in as "Ahed" or "Capel". Multi-word names are handled by FORBIDDEN.
BARE_TOKENS = tuple(sorted({
    "Ahed", "Hatter", "Radyukin", "Junankar", "Khabarov", "Lester", "Campbell",
    "Dittrich", "Akompi", "Thompson", "Sharma", "Shehata", "Shabea", "Tiltina",
    "Capel", "Bukhtoiarova", "Mali", "Babrauskas", "Beckers", "Zeissig",
    "Weigel", "Mashchenko", "Travkin", "Douin", "Blevins", "Fraser",
    "Laundy", "Ismael", "Siarhei", "Kateryna",
}))


def _test_files():
    """Every test file except this one, which has to spell the names out."""
    return sorted(p for p in TESTS_DIR.rglob("*.py")
                  if "__pycache__" not in p.parts and p.resolve() != SELF)


@pytest.mark.parametrize("name", FORBIDDEN)
def test_no_test_contains_a_real_profile_name(name):
    """The full name, wherever it appears."""
    hits = [f"{p.name}" for p in _test_files() if name in p.read_text()]
    assert hits == [], f"{name!r} is a real profile name; found in {hits}"


@pytest.mark.parametrize("token", BARE_TOKENS)
def test_no_test_contains_a_real_name_component(token):
    """A bare surname or given name is just as identifying as the full name."""
    pattern = re.compile(rf"(?<![\w]){re.escape(token)}(?![\w])")
    hits = [p.name for p in _test_files() if pattern.search(p.read_text())]
    assert hits == [], f"{token!r} names a real person; found in {hits}"


def test_the_allowed_name_is_not_also_forbidden():
    """The exemption has to stay an exemption, or every file using it fails."""
    assert ALLOWED not in FORBIDDEN
    assert not any(token in ALLOWED for token in BARE_TOKENS)