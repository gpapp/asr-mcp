"""The SPA stylesheet is a design system, and the panes are supposed to use it.

Two failure modes cost real time on the Live tab, and both are silent in a
browser -- the page still renders, it is just wrong:

1. **A pane invents its own classes.** The Live tab arrived with
   `.live-status` / `.live-meters` / `.live-meter-fill` / `.live-text` /
   `.live-note` / `.live-downloads`, a second copy of the transcript surface
   and a second copy of the level meter that the rest of the UI already had.
   Two implementations of one thing drift, and only the newer one gets fixed.
2. **CSS that no element matches.** Lesson 21: a rule scoped under a class the
   JS never adds is dead, and the thing it was written to style silently keeps
   its default. The inverse -- a class the JS adds with no rule at all -- is
   the same bug pointing the other way.

These tests pin both directions, plus the responsive contract: the phone
breakpoint has to keep existing, because it is the only thing that makes the
tab usable on a handset and nothing else will tell you it was dropped.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "asr_mcp" / "templates" / "app.html"
VIEWER_JS = REPO_ROOT / "asr_mcp" / "static" / "viewer.js"
LIVE_JS = REPO_ROOT / "asr_mcp" / "static" / "live.js"

pytestmark = pytest.mark.skipif(not TEMPLATE.is_file(), reason="app.html absent")

# Class names that exist only as JS state or as template hooks, and are styled
# through a descendant selector or a bare attribute rather than a rule of their
# own. Listing them is the point -- adding a name here is a decision, not an
# omission.
STATE_ONLY = {
    "show", "error", "open", "active", "selected", "playing", "dragover",
    "ok", "bad", "warn", "italic", "hot", "hidden", "visible", "loading",
    "tall", "ts-panel",  # a console.log guard, not a class
}


def _template() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def _style_block() -> str:
    """The inline stylesheet with its comments removed.

    Comments matter here: several of these tests assert that a class no longer
    has a rule, and the rules were retired into explanatory comments that name
    them. Matching prose would fail forever.
    """
    m = re.search(r"<style>(.*?)</style>", _template(), re.S)
    assert m, "app.html has no inline <style> block"
    return re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)


def _css_classes() -> set:
    """Every class that appears as a selector target in the stylesheet."""
    css = _style_block()
    names = set()
    for selector in re.findall(r"(?:^|\}|\*/)\s*([^{}@]+)\{", css, re.M):
        for part in selector.split(","):
            part = part.strip()
            if not part:
                continue
            for name in re.findall(r"\.([A-Za-z_][\w-]*)", part):
                names.add(name)
    return names


def _js_classes() -> set:
    """Every class name the JS can put on an element."""
    found = set()
    for path in (TEMPLATE, VIEWER_JS, LIVE_JS):
        src = path.read_text(encoding="utf-8")
        # className = 'a b'  /  className += ' a'  /  classList.add('a')
        for m in re.finditer(r"className\s*=\s*'([^']*)'", src):
            found.update(m.group(1).split())
        for m in re.finditer(r"classList\.(?:add|remove|toggle)\('([^']*)'", src):
            found.update(m.group(1).split())
        # static markup: class="a b"
        for m in re.finditer(r'class="([^"{}]+)"', src):
            found.update(m.group(1).split())
    return found - STATE_ONLY


# ── No orphan classes in either direction ───────────────────────────────────

def test_every_class_the_js_uses_has_a_rule():
    """A class the JS adds with no CSS behind it is a silent no-op."""
    known = _css_classes()
    orphans = sorted(_js_classes() - known)
    assert not orphans, (
        "these classes are applied by JS or markup but no rule styles them: "
        f"{orphans}"
    )


def test_no_pane_keeps_a_private_copy_of_a_shared_primitive():
    """The Live tab must compose the shared classes, not re-declare them.

    Each of these used to exist as a `.live-*` rule duplicating something the
    SPA already had. A private copy is not a style choice, it is a second
    implementation that will drift.
    """
    css = _style_block()
    retired = [
        "live-status", "live-meters", "live-meter", "live-meter-name",
        "live-meter-track", "live-meter-fill", "live-meter-label",
        "live-text", "live-note", "live-downloads",
    ]
    left = [c for c in retired if f".{c}" in css]
    assert not left, (
        f"the Live pane re-declared shared primitives: {left}. Use .pill, "
        ".note, .meter-row + .mini-bar, .text-pane and .btn-row instead."
    )
    template = _template()
    assert 'class="live-text"' not in template, "#liveText must use .text-pane"


def test_the_transcript_surface_is_defined_once():
    """Three scrollable transcript panes, one set of rules.

    `.tt-text-pane` (SSE), `.tv-text` (History) and the live pane each used to
    restate border, radius, padding, background and scroll behaviour. They now
    share `.text-pane` and differ only in `--pane-h`.
    """
    css = _style_block()
    assert ".text-pane {" in css
    assert "--pane-h" in css, "the shared pane has no height variable"
    for height in ("pane-short", "pane-tall", "pane-live"):
        assert f".{height} {{" in css, f".{height} does not set a pane height"
    # The bare .tt-text-pane / .tv-text rules must not restate the chrome.
    for selector in (".tt-text-pane {", ".tv-text {"):
        if selector in css:
            rule = css[css.index(selector):css.index("}", css.index(selector))]
            for prop in ("border:", "background:", "overflow-y:", "padding:"):
                assert prop not in rule, (
                    f"{selector} restates {prop} -- that belongs on .text-pane"
                )


def test_the_level_meter_has_exactly_one_implementation():
    """.mini-bar is the meter. The Live pane's second track is gone.

    The fill must be a block box: `width` does not apply to a non-replaced
    inline element, and the original <span> fill made a working microphone look
    like a dead capture with no error anywhere.
    """
    css = _style_block()
    assert ".mini-bar > div {" in css, "the meter fill rule is gone"
    rule = css[css.index(".mini-bar > div {"):css.index("}", css.index(".mini-bar > div {"))]
    assert "display: block" in rule, (
        f"the meter fill must be a block box or its width is ignored: {rule}"
    )
    template = _template()
    for fill in ("liveMeterMic", "liveMeterSpk"):
        m = re.search(rf'id="{fill}"', template)
        assert m, f"#{fill} is missing"
        tag = template[max(0, m.start() - 60):m.start()]
        assert "<div" in tag.split(">")[-1], (
            f"#{fill} must be a <div> inside .mini-bar, not a <span>"
        )
    # No second track element: the meter is a .mini-bar, not a bespoke box.
    assert 'class="mini-bar meter-bar"' in template
    assert "meter-track" not in template


# ── Visibility is a class, not an inline style ──────────────────────────────

def test_visibility_is_toggled_by_class_not_by_inline_style():
    """`.hidden` plus a `show()` helper, not `style.display` writes.

    Inline display writes made every call site hardcode the element's layout
    mode (a flex row restored as `''`, a div restored as `'block'`), which is
    exactly the knowledge the stylesheet should hold.
    """
    for path in (TEMPLATE, LIVE_JS, VIEWER_JS):
        src = path.read_text(encoding="utf-8")
        offenders = [
            line.strip() for line in src.splitlines()
            if re.search(r"\.style\.display\s*=", line)
        ]
        assert not offenders, f"{path.name} still writes style.display: {offenders}"
    assert ".hidden { display: none !important; }" in _style_block()
    assert "function show(elOrId, on)" in _template(), (
        "app.html has no show() helper -- the class toggle needs a name"
    )
    assert "function _show(id, on)" in LIVE_JS.read_text(encoding="utf-8")


def test_the_template_carries_no_inline_styles():
    """No `style="..."` in the markup.

    The Live pane had fifteen of them, and every one of them was a rule that
    belonged in the stylesheet. What remains is escaped text inside JS template
    literals, which this regex skips.
    """
    template = _template()
    # Drop <script> bodies and comments so JS string literals do not count.
    markup = re.sub(r"<script>.*?</script>", "", template, flags=re.S)
    markup = re.sub(r"<!--.*?-->", "", markup, flags=re.S)
    leftovers = re.findall(r'\sstyle="[^"]*"', markup)
    assert not leftovers, f"inline styles left in the markup: {leftovers}"


# ── Design tokens are the only colours ──────────────────────────────────────

def test_status_colours_come_from_tokens():
    """A second literal green means the token and the ad-hoc rule disagree.

    `.live-status.ok` and `.badge-green` were both hardcoded greens. All status
    surfaces now draw from --ok-ink/--bad-ink/--warn-ink.
    """
    css = _style_block()
    start = css.index(":root {")
    root = css[start:css.index("}", start)]
    tokens = set(re.findall(r"(--[\w-]+):", root))
    for token in ("--ok", "--bad", "--warn", "--ok-soft", "--ok-ink",
                  "--bad-soft", "--bad-ink", "--warn-ink", "--pane-h"):
        assert token in tokens, f"{token} is not declared in :root"
    assert ".pill.ok {" in _style_block() and "var(--ok-soft)" in _style_block()
    # The literals live in :root and nowhere else, so "green" cannot drift.
    body = css[:css.index(":root {")] + css[css.index("}", css.index(":root {")):]
    for literal in ("#0f8a47", "#b4232a", "#92400e", "#e7f8ee"):
        assert literal not in body, (
            f"{literal} is hardcoded outside :root -- use the token"
        )


# ── The responsive contract ─────────────────────────────────────────────────

PHONE_RULES = [
    ".tabs", ".pane-inner", ".card-body", ".btn", ".options", ".toast",
    ".meter-row", ".tv-bars", ".status-grid", ".list-pane",
]


def test_there_are_two_breakpoints_and_both_are_max_width():
    """Base styles are the desktop layout; every override is a max-width step.

    A `min-width` query would mean the desktop rules are the exception, and the
    next person to add a component would have to guess which side of the
    stylesheet it belongs on.
    """
    css = _style_block()
    widths = sorted(int(m) for m in re.findall(
        r"@media \(max-width: (\d+)px\)", css))
    assert widths == [560, 860], f"unexpected breakpoints: {widths}"
    assert "@media (min-width" not in css, (
        "a min-width query appeared -- the base rules must stay the desktop layout"
    )


def test_the_phone_breakpoint_covers_the_layout_landmines():
    """Each of these is a layout that only breaks on a narrow viewport."""
    css = _style_block()
    start = css.index("@media (max-width: 560px)")
    end = css.index("@media (prefers-reduced-motion")
    phone = css[start:end]
    for selector in PHONE_RULES:
        assert selector + " {" in phone or selector + ":" in phone, (
            f"the phone breakpoint no longer styles {selector}"
        )
    # A 16px minimum stops iOS Safari zooming on focus, which is unrecoverable
    # in a scrollable pane.
    assert "font-size: 16px" in phone, "the phone input font floor is gone"
    # Notches and the home indicator.
    assert "env(safe-area-inset-top)" in phone
    assert "env(safe-area-inset-bottom)" in phone


def test_the_viewport_allows_the_notch():
    """`viewport-fit=cover` is what makes env(safe-area-inset-*) non-zero."""
    template = _template()
    m = re.search(r'<meta name="viewport" content="([^"]*)"', template)
    assert m, "no viewport meta"
    assert "viewport-fit=cover" in m.group(1), (
        "the safe-area padding in the stylesheet is inert without this"
    )
    assert "width=device-width" in m.group(1)


def test_motion_can_be_switched_off():
    """The live caret and the pane fade are the two perpetual animations."""
    assert "@media (prefers-reduced-motion: reduce)" in _style_block()


# ── The meter row reflows, it does not overflow ────────────────────────────

def test_the_meter_is_a_grid_with_areas():
    """A fixed-width flex row cannot reflow; grid areas can.

    The name and the readout had fixed widths (6.5rem / 9.5rem), which is 16rem
    of a 360px phone before the bar gets anything.
    """
    css = _style_block()
    rule = css[css.index(".meter-row {"):css.index("}", css.index(".meter-row {"))]
    assert "grid-template-areas" in rule, f".meter-row is not a grid: {rule}"
    assert "px" not in rule, f".meter-row has a fixed width: {rule}"
    phone = css[css.index("@media (max-width: 560px)"):
                css.index("@media (prefers-reduced-motion")]
    assert '"name label" "bar bar"' in phone, (
        "the phone layout does not reflow the meter row"
    )
