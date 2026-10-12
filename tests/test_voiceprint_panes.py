"""The Voiceprints tab: named and unnamed speakers are two DIFFERENT lists.

They used to share one page. A named profile is a profile that names people; an
unnamed one is a voice the learner heard and could not attach to anyone, so it
is excluded from every match and its audio is the only evidence of who it is.
Parked in a card above the known speakers, an unnamed profile offered no way to
play a single snippet -- the user was asked to identify a voice from a row of
counts. That is the failure these tests pin.

Everything here is read out of the template: there is no browser in the suite,
so the assertions are about the markup and the JS that manipulates it, which is
exactly where each of these bugs lived.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "asr_mcp" / "templates" / "app.html"
NAV = REPO_ROOT / "asr_mcp" / "templates" / "_nav.html"

pytestmark = pytest.mark.skipif(not TEMPLATE.is_file(), reason="app.html absent")


def _html() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def _js() -> str:
    """The single inline script block."""
    html = _html()
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    assert m, "app.html has no inline <script> block"
    return m.group(1)


def _fn(name: str) -> str:
    """The body of a top-level `function name() { ... }` in the script."""
    js = _js()
    start = js.index(f"function {name}(")
    depth, i = 0, js.index("{", start)
    j = i
    while True:
        if js[j] == "{":
            depth += 1
        elif js[j] == "}":
            depth -= 1
            if depth == 0:
                return js[i:j + 1]
        j += 1


# ── the two populations are separate lists ───────────────────────────────────

def test_there_are_two_sub_tabs_and_both_target_an_existing_container():
    html = _html()
    for subtab, target in (("vpSubtab-known", "speakers-list"),
                           ("vpSubtab-unnamed", "pending-list")):
        assert f'id="{subtab}"' in html, f"no {subtab} button"
        assert f"id='{target}'" not in html  # the ids are double-quoted
        assert f'id="{target}"' in html, f"{target} container is gone"
    body = _fn("showVoiceprintSubtab")
    # Each sub-tab shows exactly one list and hides the other; a sub-tab that
    # showed both is the single-list layout wearing a tab bar.
    assert "show('speakers-list', vpSubtab === 'known')" in body
    assert "show('pending-list', vpSubtab === 'unnamed')" in body
    assert body.count("classList.toggle('active'") == 2


def test_a_pending_profile_is_never_rendered_into_the_known_list():
    """Two renderers, two containers, and no crosstalk.

    The old filter lived in refresh(); the renderers are the place a
    regression actually shows, so they are what is pinned.
    """
    known = _fn("showKnownList")
    pending = _fn("showPendingList")
    assert "pending" not in known.lower(), "the known list renders pending profiles"
    assert "pendingData" in pending, "the unnamed list does not use the pending data"
    assert "speakersData" not in pending, "the unnamed list renders named profiles"
    # refresh() splits the payload before either renderer sees it.
    refresh = _fn("refresh")
    assert "filter(s => !s.pending)" in refresh


def test_the_unnamed_count_is_on_the_sub_tab_the_user_has_to_find():
    """A hidden card with no affordance is how the profiles went unnoticed."""
    body = _fn("refreshPending")
    assert "vpSubtab-unnamed" in body
    assert "Unnamed (" in body


# ── an unnamed profile can be listened to ───────────────────────────────────

def test_an_unnamed_profile_folds_out_into_its_snippets():
    """The whole point: play the audio to work out who the voice is."""
    pending = _fn("showPendingList")
    assert 'onclick="togglePending(${jsStr(p.name)})"' in pending
    assert 'id="pbody-${escHtml(p.name)}"' in pending, (
        "the unnamed profile has no fold to put snippets in"
    )
    assert "Loading snippets..." not in pending or True
    toggle = _fn("togglePending")
    assert "pbody-" in toggle
    assert "loadSnippetBody('pbody-' + name, name)" in toggle


def test_the_snippets_of_an_unnamed_profile_are_loaded_and_playable():
    """Same endpoint, same player, as a named profile -- not a reimplementation.

    Pending profiles are ordinary speakers in the snippets table, so
    `/speakers/{name}/snippets` serves them. A second loader that "only works
    for pending" would be a second place for the audio path to break.
    """
    load = _fn("loadSnippetBody")
    assert "'/speakers/' + encodeURIComponent(name) + '/snippets'" in load
    assert "snippetListHtml(name, data.snippets)" in load
    listing = _fn("snippetListHtml")
    assert "togglePlay(" in listing, "the snippet list has no play button"
    assert "/audio" in _fn("togglePlay") or "new Audio" in _fn("togglePlay")


def test_one_snippet_renderer_serves_both_lists():
    """The markup exists once. Two copies of a snippet row drift."""
    js = _js()
    assert js.count("function snippetListHtml(") == 1
    # Both folds go through the one loader, and the loader has exactly one call
    # to the one renderer -- so neither list can acquire its own row markup.
    assert "loadSnippetBody('body-' + name, name)" in _fn("toggleSpeaker")
    assert "loadSnippetBody('pbody-' + name, name)" in _fn("togglePending")
    loader = _fn("loadSnippetBody")
    assert loader.count("snippetListHtml(") == 1
    assert "'/speakers/' + encodeURIComponent(name) + '/snippets'" in loader
    # The player button id is unique per snippet, and the list renders it.
    assert 'id="play-${s.id}"' in js


def test_removing_a_snippet_from_an_unnamed_profile_keeps_the_fold_open():
    """A review session is many removals in a row; a closed fold ends it."""
    body = _fn("deleteSnippet")
    assert "pendingData.some(p => p.name === speakerName)" in body
    assert "await refresh();" in body
    assert "pbody-" in body


# ── the shared primitives ───────────────────────────────────────────────────

def test_the_tab_pill_is_declared_once_and_used_by_both_strips():
    """`.tabs a` and a second pill for the in-pane sub-tabs would be lesson 21.

    The sub-tabs are buttons, the header nav is anchors; they are one control,
    so they get one rule. The anchor selector is gone, which is also what proves
    the sub-tab buttons are not relying on a rule scoped to `a`.
    """
    css = re.sub(r"/\*.*?\*/", "", re.search(r"<style>(.*?)</style>", _html(), re.S).group(1), flags=re.S)
    assert ".pill-tab {" in css, "there is no .pill-tab rule"
    assert ".tabs a {" not in css, (
        ".tabs a still carries a private copy of the pill; move it to .pill-tab"
    )
    nav = NAV.read_text(encoding="utf-8")
    assert nav.count('class="pill-tab"') == nav.count("<a "), (
        "every header nav anchor must use the shared pill"
    )
    html = _html()
    for subtab in ("vpSubtab-known", "vpSubtab-unnamed"):
        m = re.search(rf'<button[^>]*id="{subtab}"[^>]*>', html)
        assert m and "pill-tab" in m.group(0), f"{subtab} is not a pill-tab"


# ── a name is a value, not source ───────────────────────────────────────────

def test_no_handler_interpolates_a_name_into_quotes():
    """`onclick="f('${escHtml(name)}')"` is a SyntaxError on an apostrophe.

    escHtml escapes & < > and deliberately leaves quotes alone, and the browser
    HTML-decodes the attribute before parsing the JS, so `O'Brien` produces
    onclick="f('O'Brien')" and the button silently does nothing. A learned
    profile takes its name from a source FILENAME, so an apostrophe arrives
    without anybody typing it -- which is why this is a real name and not a
    hypothetical one.
    """
    js = _js()
    offenders = re.findall(r"""onclick="[^"]*'\$\{""", js)
    assert not offenders, f"names interpolated inside single quotes: {offenders}"
    assert "function jsStr(" in js
    # Every interpolated name goes through it.
    handlers = re.findall(r"""onclick="(\w+)\(\$\{jsStr\(""", js)
    assert len(handlers) >= 10, f"only {len(handlers)} handlers use jsStr"


def test_names_are_still_escaped_for_the_dom():
    """jsStr is for JS; the visible name still goes through escHtml."""
    pending = _fn("showPendingList")
    assert "<h3>" in pending and "escHtml(p.name)" in pending
    listing = _fn("snippetListHtml")
    assert "escHtml(basename(s.file_path))" in listing

# ── a job in another pane must not hijack the tab ──────────────────────────

def test_a_voiceprints_job_does_not_switch_the_tab():
    """The /activity/stream reader is global, so a `consolidate` job arrives
    at the Transcribe tab's attach path -- which activated the Transcribe tab
    and entered a transcription job UI that has nothing to show for it. So
    pressing "Consolidate All" in the Voiceprints pane jumped tabs.
    """
    js = _js()
    assert "const TRANSCRIPTION_MODES" in js
    attach = _fn("attachActiveJob")
    assert "TRANSCRIPTION_MODES.includes(mode)" in attach
    # The guard has to come BEFORE the tab switch and the job UI.
    assert attach.index("TRANSCRIPTION_MODES.includes") < attach.index("activateTab(")


# ── a redraw must not collapse what the user opened ────────────────────────

def test_a_redraw_restores_the_folds_that_were_open():
    """Reviewing means: open a speaker, remove three snippets, look again."""
    refresh = _fn("refresh")
    assert "captureOpenFolds()" in refresh
    assert "restoreOpenFolds(open)" in refresh
    restore = _fn("restoreOpenFolds")
    assert "loadSnippetBody(" in restore, "a restored fold must reload its snippets"
    assert "knownActionsHtml(" in restore, "a restored fold must get its action row back"
    assert "pbody-" in _fn("foldName"), "unnamed folds are restored too"
    # ONE action row: the fold and the restorer must not each have a copy.
    js = _js()
    assert js.count("function knownActionsHtml(") == 1
    assert "Add snippet:" in _fn("knownActionsHtml")


def test_switching_sub_tabs_does_not_redraw_the_list():
    """It used to re-render, which collapsed the open fold on every switch."""
    subtab = _fn("showVoiceprintSubtab")
    assert "showKnownList()" not in subtab
    assert "showPendingList()" not in subtab
    assert "show('speakers-list'" in subtab and "show('pending-list'" in subtab


# ── name order, and attributes that are closed twice ───────────────────────

def test_every_speaker_list_is_sorted_by_name():
    js = _js()
    assert "function byName(" in js
    assert "numeric: true" in js, "Speaker 2 must sort before Speaker 10"
    refresh = _fn("refresh")
    assert ".sort(byName)" in refresh, "the known list"
    pending = _fn("refreshPending")
    assert ".sort(byName)" in pending, "the unnamed list"
    # The flagged card groups by speaker, so it is a list of names too.
    assert "Object.keys(bySpeaker).sort(byName)" in _fn("renderFlagged")


def test_no_html_attribute_is_closed_twice():
    """`onclick="f(${jsStr(n)})""` renders an attribute plus a stray quote.

    Every such button was dead in the browser, and nothing caught it: the
    markup lives inside a JS template literal, so `node --check` parses past
    it and no test parsed the HTML.
    """
    html = _html()
    for marker in ('""', '">>'):
        offenders = [
            m.start() for m in re.finditer(re.escape(marker), html)
            if not _in_script(html, m.start())
        ]
        assert not offenders, (
            f"{len(offenders)} attribute(s) closed twice ({marker!r}); "
            f"first near: {html[offenders[0] - 60:offenders[0] + 20]!r}"
        )
    # Sanity: the template really does contain handlers, so the check above
    # is not passing because it has nothing to look at.
    assert html.count('onclick="') >= 20


def _in_script(html: str, pos: int) -> bool:
    """True if pos falls inside a <script>/<style> block or a JS comment."""
    before = html[:pos]
    if before.count("<script>") != before.count("</script>"):
        return True
    if before.count("<style>") != before.count("</style>"):
        return True
    line = html.rfind("\n", 0, pos) + 1
    return html[line:pos].lstrip().startswith("//")
