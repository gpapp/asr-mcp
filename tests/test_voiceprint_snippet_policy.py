"""The snippet budget: ten per file, none under five seconds, best fit first.

The overfitting shape this guards against is ONE RECORDING BECOMING A PROFILE.
A fifty-minute meeting attributed to one colleague for forty of those minutes
yields forty clips from a single file, and the profile then describes that room,
that microphone and that afternoon rather than the person. So the collection
budget is per file and per profile: ``AUTO_COLLECT_MAX_PER_SOURCE`` snippets,
spent in rank order, on clips that clear ``AUTO_COLLECT_MIN_DURATION``.

A reprocess only ever re-cuts the file it re-runs, so the same policy has to be
applied BACKWARDS over a library built before it -- which is what the
``trim_snippets`` step of ``cleanup_integrity`` does and what the pure
``plan_source_trim`` decides. That half is the one that matters for the existing
store, so the ranking is pinned here on plain rows with no database, no audio and
no GPU.

The harness is the sibling module's: ``tests/test_learning_boundaries.py``
already installs a torch/soundfile stub, a dict-backed DB and one counted decode,
and its docstring says it is meant to run everywhere. A second copy of those 200
lines would be a second thing to keep in step with service.py.
"""

import sys

import pytest

from conftest import load_module


@pytest.fixture(scope="module")
def lb():
    """The sibling harness, imported under a private dotted name.

    Loaded by path rather than imported by its collection name, so that the
    sibling's own module-scoped teardown does not run as a side effect of this
    import. Only its functions are used; none of its tests.
    """
    return load_module("_t_learning_boundaries",
                       "tests/test_learning_boundaries.py")


@pytest.fixture(scope="module", autouse=True)
def _restore_module_table():
    """Undo every ``sys.modules`` entry the stub installation injects.

    ``tests/test_speaker_learning.py`` needs the REAL torch and a real
    SQLAlchemy database; a fake ``torch`` left behind here would make its
    ``importorskip("torch")`` pass and run every one of its DB tests against the
    dict-backed stub. Taken from the sibling for the same reason, and scoped to
    this module because the snapshot has to be taken before the stubs exist.
    """
    before = dict(sys.modules)
    yield
    for name in [n for n in sys.modules if n not in before]:
        del sys.modules[name]
    for name, mod in before.items():
        if sys.modules.get(name) is not mod:
            sys.modules[name] = mod


@pytest.fixture(scope="module")
def svc(lb):
    """The service module, loaded against the sibling's dict-backed stack.

    Two things have to be forced here, both because this module runs AFTER the
    sibling and after modules that install the real stack:

    * ``_install_stubs`` returns early when a stub ``torch`` is already in
      ``sys.modules`` -- which is what a previous module leaves behind, so the
      guard would fire before the dict-backed DB stub was installed. Clearing
      the marker makes it install them for THIS load.
    * ``load_module`` returns whatever is cached under a name, and another
      module may already have loaded ``service.py`` against the real SQLAlchemy
      manager. A private name forces a second execution of the file.

    Neither is needed on the host, where the sibling runs first and this module
    is the only one asking. Both are needed in the image, where the whole suite
    shares one interpreter.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        torch._p2_stub = False
    lb._install_stubs()
    service = load_module("_t_voiceprint_service",
                          "asr_mcp/voiceprint/service.py",
                          package="asr_mcp.voiceprint")
    # Guard against the exact silent-wrong-result this module is exposed to: the
    # service module binding a db layer the harness never installed. In the image
    # the whole suite shares one interpreter, so ``service.py`` is already
    # cached under its canonical name -- bound to the REAL SQLAlchemy manager,
    # whose ``save``/``add`` need a ``DatabaseManager`` this harness cannot give
    # (it passes None). The symptom is an AttributeError three frames deep in
    # db/manager.py, which reads like a broken fixture and not like a wrong
    # layer. Assert the binding instead of discovering it later.
    dbm = sys.modules["asr_mcp.db.manager"]
    assert service.VoiceprintDB is dbm.VoiceprintDB \
        and service.SnippetDB is dbm.SnippetDB, (
            "service.py bound a db layer the harness did not install; the "
            "stubbed load did not take and this module would test the wrong "
            "stack")
    # ``_emb_session`` does ``from asr_mcp.core.model_state import state`` on
    # EVERY call, before the ``or`` that would make the imported value
    # unnecessary -- so the collect path cannot run where onnxruntime is absent.
    # The import resolves from sys.modules without the parent package (and
    # therefore without onnxruntime) ever being executed.
    import types

    core = types.ModuleType("asr_mcp.core.model_state")

    class _State:
        embedding_session = object()

    core.state = _State()
    sys.modules["asr_mcp.core.model_state"] = core
    return service


@pytest.fixture
def make_service(lb, svc, tmp_path):
    """The sibling's service factory, with its two dependencies supplied.

    ``lb.make_service`` is a pytest fixture, and a fixture refuses to be called
    directly. ``__wrapped__`` is the function it wraps; supplying its two
    arguments here is what the sibling's own fixture injection would have done.
    """
    return lb.make_service.__wrapped__(svc, tmp_path)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _row(snippet_id, duration, source="meeting.wav", start=0.0, path=None):
    return {
        "id": snippet_id, "speaker_name": "Ana",
        "file_path": path or f"/voices/u/Ana/{snippet_id}.flac",
        "duration_sec": float(duration), "source_audio": source,
        "start_sec": float(start), "end_sec": float(start + duration),
    }


# --------------------------------------------------------------------------
# the cleanup's decision -- pure, and the half that matters for a live store
# --------------------------------------------------------------------------

def test_ten_per_file_and_then_no_more(svc):
    """One recording may contribute ten clips and then its budget is spent."""
    rows = [_row(i + 1, 6.0, start=float(i * 10)) for i in range(14)]

    plan = svc.plan_source_trim(rows)

    assert len(plan["groups"]) == 1
    group = plan["groups"][0]
    assert group["kept"] == 10
    # Ranked by duration, and every clip is the same length, so the ranking is
    # the input order: the FIRST FOUR IN are the first four out.
    assert group["dropped"] == [11, 12, 13, 14]
    assert group["reasons"] == {
        11: "over_per_file_cap", 12: "over_per_file_cap",
        13: "over_per_file_cap", 14: "over_per_file_cap",
    }
    assert plan["unsourced"] == []


def test_a_clip_under_the_floor_is_rejected_before_the_ranking(svc):
    """Too-short clips never earn a slot, so they cannot crowd out a good one.

    Decided first and independently of rank on purpose: judged after the budget
    was spent, a 3s clip that happens to rank well would survive while a 6s one
    behind it was dropped for being over the cap.
    """
    rows = [_row(i + 1, 6.0, start=float(i * 10)) for i in range(10)]
    rows.append(_row(11, 3.0, start=100.0))

    group = svc.plan_source_trim(rows)["groups"][0]

    assert group["kept"] == 10
    assert group["dropped"] == [11]
    assert group["reasons"][11] == "too_short"


def test_the_floor_and_the_budget_are_counted_separately(svc):
    rows = [_row(i + 1, 6.0, start=float(i * 10)) for i in range(14)]
    rows.append(_row(15, 3.0, start=140.0))

    group = svc.plan_source_trim(rows)["groups"][0]

    assert group["kept"] == 10
    assert group["dropped"] == [11, 12, 13, 14, 15]
    assert group["reasons"][15] == "too_short"
    assert {v for k, v in group["reasons"].items() if k != 15} == {
        "over_per_file_cap"}


def test_the_same_recording_is_one_file_across_a_renamed_upload(svc):
    """Identity is the filename-hash rule ``_source_snippet_rows`` uses: the
    emitting stem, whatever the directory, the extension or the upload name."""
    spellings = ["meeting.wav", "/data/2026/meeting.mp4", "call/meeting.wav"]
    rows = [_row(i + 1, 6.0, source=spellings[i % 3], start=float(i * 10))
            for i in range(12)]
    rows.append(_row(13, 6.0, source="interview.wav", start=0.0))

    plan = svc.plan_source_trim(rows)

    # Twelve across three spellings is ONE over-budget file; the second
    # recording is inside its own budget and is not reported at all.
    assert len(plan["groups"]) == 1
    assert plan["groups"][0]["kept"] == 10
    assert plan["groups"][0]["dropped"] == [11, 12]


def test_a_snippet_with_no_recorded_source_is_never_trimmed(svc):
    """A NULL source belongs to no file, so no per-file budget can be counted
    against it -- and deleting audio that cannot be attributed to anything is
    exactly the judgement this cleanup refuses to make. Thirty of them, every
    one also under the duration floor, and not one is dropped.
    """
    rows = [_row(i + 1, 3.0, source=None, start=float(i)) for i in range(30)]

    plan = svc.plan_source_trim(rows)

    assert plan["groups"] == []
    assert plan["unsourced"] == list(range(1, 31))


# --------------------------------------------------------------------------
# a row that lost its source_audio still knows which file it came from
# --------------------------------------------------------------------------

def test_a_row_with_no_source_is_trimmed_by_the_hash_in_its_filename(svc):
    """The wav->flac migration left rows with no source, and their audio is still
    this recording's audio. The filename is ``add_snippet``'s own naming --
    ``<generate_segment_hash(source)>_<MM-SS>_<nn>.flac`` -- so the identity is
    recoverable, not guessed. Twelve recovered rows and two stored rows from the
    same file must be ONE group capped at ten, not an exempt pile beside it.
    """
    h = svc.generate_segment_hash("meeting.wav")
    rows = [_row(i + 1, 6.0, source=None,
                 path=f"/voices/u/Ana/{h}_{i:02d}-00_06.flac")
            for i in range(12)]
    rows += [_row(101, 6.0, source="meeting.wav", start=120.0),
             _row(102, 6.0, source="/data/2026/meeting.mp4", start=130.0)]

    plan = svc.plan_source_trim(rows)

    assert len(plan["groups"]) == 1
    g = plan["groups"][0]
    assert g["kept"] == 10
    assert len(g["dropped"]) == 4
    assert plan["unsourced"] == []
    # The report says how many identities were recovered rather than stored: a
    # reader deciding whether to trust this trim is entitled to that.
    assert g["derived_from_filename"] == 12


def test_the_recovered_hash_ranks_in_the_same_group_as_the_stored_rows(svc):
    """A recovered identity is the SAME identity ``generate_segment_hash``
    returns, so the rows it recovers are ranked with the rows that kept their
    path and can lose their slot to one. Giving the recovered row the shortest
    clip and the latest start puts it last under the ranking key, so on a
    13-row group it is one of the drops -- which is only true if it was in the
    group at all. An exempt row would survive.
    """
    h = svc.generate_segment_hash("meeting.wav")
    rows = [_row(i + 1, 6.0, source="meeting.wav", start=float(i * 10))
            for i in range(12)]
    rows.append(_row(100, 5.5, source=None, start=500.0,
                     path=f"/voices/u/Ana/{h}_50-00_06.flac"))

    plan = svc.plan_source_trim(rows)

    assert plan["unsourced"] == []
    assert len(plan["groups"]) == 1
    g = plan["groups"][0]
    assert g["kept"] == 10
    assert 100 in g["dropped"]
    assert g["derived_from_filename"] == 1


def test_a_manual_upload_names_no_recording(svc):
    """``<hash>_manual.flac`` hashes the AUDIO BYTES, not a file, so that number
    identifies no recording. Treating it as one would silently make a manual
    clip the entire "budget" of a recording it was never cut from.
    """
    rows = [_row(1, 6.0, source=None, path="/voices/u/Ana/9F2A1B_manual.flac")]
    plan = svc.plan_source_trim(rows)
    assert plan["groups"] == []
    assert plan["unsourced"] == [1]


def test_a_filename_without_a_hash_token_is_still_exempt(svc):
    """``Speaker_1.wav`` was typed by an operator. It carries no evidence of a
    recording, so it stays exempt for the same reason the NULL source did.
    """
    rows = [_row(1, 6.0, source=None, path="/voices/u/Ana/Speaker_1.wav")]
    plan = svc.plan_source_trim(rows)
    assert plan["groups"] == []
    assert plan["unsourced"] == [1]
    assert svc.snippet_source_identity(rows[0]) == ("", True)


def test_the_ranking_is_longest_first_then_earliest_then_lowest_id(svc):
    rows = [_row(7, 5.0, start=10.0), _row(8, 5.0, start=0.0),
            _row(9, 9.0, start=3.0)]

    assert [r["id"] for r in svc.rank_snippet_rows(rows)] == [9, 8, 7]


# --------------------------------------------------------------------------
# the ranking a RUN uses -- the attribution's own verdict, no GPU
# --------------------------------------------------------------------------

def test_confidence_outranks_duration_when_spending_the_budget(svc):
    confident = {"speaker": "Ana", "start": 0.0, "end": 6.0,
                 "speaker_confidence": 0.9}
    long_but_shaky = {"speaker": "Ana", "start": 0.0, "end": 60.0,
                      "speaker_confidence": 0.4}

    assert svc.span_fit_key(confident) > svc.span_fit_key(long_but_shaky)


def test_duration_breaks_a_tie_between_two_equally_confident_spans(svc):
    short = {"speaker": "Ana", "start": 0.0, "end": 6.0,
             "speaker_confidence": 0.8}
    long = {"speaker": "Ana", "start": 0.0, "end": 30.0,
            "speaker_confidence": 0.8}

    assert svc.span_fit_key(long) > svc.span_fit_key(short)


def test_a_generic_cluster_sorts_above_a_rejected_match_not_below_it(svc):
    """No recorded verdict is a *generic* label, not a confident mismatch.

    ``learning_refusal_reason`` does not require a confidence on a generic
    cluster, so those spans arrive with none. Ranking them at 0.0 would put
    every one of them below every matched span and starve the learn path of the
    per-file budget it has to spend.
    """
    generic = {"speaker": "Speaker 5", "start": 0.0, "end": 6.0}
    rejected = {"speaker": "Ana", "start": 0.0, "end": 6.0,
                "speaker_confidence": 0.0}

    assert svc.span_fit_key(generic)[0] == 1.0
    assert svc.span_fit_key(generic) > svc.span_fit_key(rejected)


# --------------------------------------------------------------------------
# the collection path -- what a processed file actually leaves behind
# --------------------------------------------------------------------------

def _collect(make_service, lb, segments):
    s = make_service(lb._one_voice(), register={
        "Ana": lb._voice(1.0, 0.0, seed=3, jitter=0.01)})
    return s, s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=segments,
        user_id="u", source_id="meeting.wav",
    )


def test_one_long_file_contributes_ten_snippets_and_not_fifteen(
        make_service, lb):
    s, out = _collect(make_service, lb, lb._segments(
        "Ana", [(float(i * 10), float(i * 10 + 6.0)) for i in range(15)]))

    rows = s._snippets.list_by_speaker("Ana", user_id="u")
    assert len(rows) == 10
    diag = out["diagnostics"]["collect"][0]
    assert diag["snippets"] == 10
    # Everything left over is reported, not just the first casualty.
    assert diag["over_per_file_cap"] == 5
    assert diag["too_short_after_trim"] == 0


def test_the_budget_is_spent_on_the_best_fitting_spans(make_service, lb):
    """A marginal long turn must not crowd out a confident short one.

    Ten spans the matcher was indifferent about and two it was sure about: the
    two confident ones take the first two slots, and only eight of the rest
    survive. Ranked by confidence, so which eight is the input order.
    """
    spans = [(float(i * 10), float(i * 10 + 6.0)) for i in range(12)]
    segments = [dict(seg, speaker_confidence=0.4)
                for seg in lb._segments("Ana", spans)]
    for seg in segments[-2:]:
        seg["speaker_confidence"] = 0.99

    s, _ = _collect(make_service, lb, segments)

    kept = sorted(r["start_sec"] for r in
                  s._snippets.list_by_speaker("Ana", user_id="u"))
    assert kept == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 100.0, 110.0]


def test_a_clip_that_trims_below_the_floor_is_not_stored(make_service, lb):
    """The floor is on the STORED clip, after the VAD trim has pulled its edges
    inward. A ten-second turn whose only speech is a two-second section is a
    two-second snippet, and it spends a slot for audio that is mostly silence.
    """
    spans = [(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)]
    sections = [{"start": 2.0, "end": 4.0}, {"start": 22.0, "end": 29.0},
                {"start": 42.0, "end": 49.0}]

    s = make_service(lb._one_voice(), register={
        "Ana": lb._voice(1.0, 0.0, seed=3, jitter=0.01)})
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=lb._segments("Ana", spans),
        user_id="u", source_id="meeting.wav", vad_sections=sections,
    )

    assert len(s._snippets.list_by_speaker("Ana", user_id="u")) == 2
    diag = out["diagnostics"]["collect"][0]
    assert diag["too_short_after_trim"] == 1
    assert diag["over_per_file_cap"] == 0


def test_reprocessing_a_file_replaces_its_whole_take(make_service, lb):
    """The file's previous clips go, whatever span they were cut at, and only
    the new (budget-capped) take survives -- so re-running the recording does
    not stack a second copy of the same room onto the profile."""
    s = make_service(lb._one_voice(), register={
        "Ana": lb._voice(1.0, 0.0, seed=3, jitter=0.01)})
    first = [(float(i * 10), float(i * 10 + 6.0)) for i in range(15)]
    s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=lb._segments("Ana", first),
        user_id="u", source_id="meeting.wav",
    )
    assert len(s._snippets.list_by_speaker("Ana", user_id="u")) == 10

    second = [(float(i * 10 + 1), float(i * 10 + 7.0)) for i in range(15)]
    s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=lb._segments("Ana", second),
        user_id="u", source_id="meeting.wav",
    )

    rows = s._snippets.list_by_speaker("Ana", user_id="u")
    # Ten, all from the new take -- not ten old plus ten new.
    assert len(rows) == 10
    assert {round(r["start_sec"], 3) for r in rows} == {
        float(i * 10 + 1) for i in range(10)}


def test_a_reprocess_also_replaces_the_rows_that_lost_their_source(make_service, lb, svc):
    """A row whose ``source_audio`` is NULL still belongs to this recording: the
    hash in its filename says so. Matching only stored paths is what let a
    migration-era row survive a reprocess of the file it was cut from, so the
    "remove ALL old snippets" promise quietly did not cover the rows that had
    already lost the field -- which are exactly the ones an old library has.
    """
    s = make_service(lb._one_voice(), register={
        "Ana": lb._voice(1.0, 0.0, seed=3, jitter=0.01)})
    h = svc.generate_segment_hash("meeting.wav")
    # Left behind by the wav->flac migration: audio present, path rewritten,
    # source never carried over.
    s._snippets.add(
        speaker_name="Ana", file_path=f"/voices/u/Ana/{h}_00-00_06.flac",
        duration_sec=6.0, user_id="u", source_audio=None,
        start_sec=0.0, end_sec=6.0,
    )
    assert len(s._snippets.list_by_speaker("Ana", user_id="u")) == 1

    s.auto_collect_from_diarization(
        audio_path="meeting.wav",
        segments=lb._segments("Ana", [(float(i * 10), float(i * 10 + 6.0))
                                      for i in range(15)]),
        user_id="u", source_id="meeting.wav",
    )

    rows = s._snippets.list_by_speaker("Ana", user_id="u")
    assert len(rows) == 10
    assert all(r["source_audio"] for r in rows)
    assert all(r["id"] != 1 for r in rows), "the migration-era row survived"


# --------------------------------------------------------------------------
# the cleanup step -- the same policy, applied to a library that predates it
# --------------------------------------------------------------------------

def _trim_only():
    """Isolate the trim step from the other three.

    The sibling harness's ``sf.write`` stub writes the same ten bytes to every
    snippet path, so the audit's ``duplicate_snippet_content`` finding matches
    every pair and the duplicates step would strip the rest of the profile --
    an artefact of the stub, not of the trim. ``dead_snippets`` and
    ``stale_totals`` have their own module; this one is about the per-file
    budget.
    """
    return {"duplicates": False, "dead_snippets": False, "stale_totals": False}


def _older_print(s, count=30, span=30.0):
    """One recording as a print from before the per-file budget existed.

    ``count`` clips of steadily decreasing length: the shape a run with no cap
    leaves, where a long turn was cut into many pieces and a short one into one.
    Decreasing durations mean the RANKING decides the survivors, so the assertion
    can be about which ten and not merely how many.
    """
    for i in range(count):
        duration = max(1.0, 25.0 - 1.0 * i)
        s._snippets.add(
            speaker_name="Ana", file_path=f"/tmp/old_{i}.flac",
            duration_sec=duration, user_id="u", source_audio="meeting.wav",
            start_sec=float(i * span), end_sec=float(i * span + duration))


def test_the_cleanup_collapses_an_over_collected_file_to_ten(make_service, lb):
    """A file processed before the budget existed can hold thirty clips.

    The ten LONGEST survive: the most audio evidence there is about this speaker
    is what the profile should be fitted to. Note what the rule is NOT --
    "newest wins" would mean ranking on the row id, which records when a row was
    written rather than how much of the speaker it holds. A re-processed file
    never reaches this branch anyway, because its previous take was already
    replaced.
    """
    s = make_service(lb._one_voice())
    _older_print(s, count=30)

    finding = s._plan_source_trim("u")[0]

    assert finding["speaker"] == "Ana"
    assert finding["kept"] == 10
    # The twenty SHORTEST go, in one file, from one recording.
    assert finding["removed_ids"] == list(range(11, 31))
    assert finding["reasons"][11] == "over_per_file_cap"


def test_the_cleanup_plans_the_whole_store_not_just_the_file_it_ran_on(
        make_service, lb):
    """A reprocess applies the policy to the file it re-runs. This is what
    applies it to every OTHER recording -- and to the speaker that never spoke
    in this run at all, who is the one carrying the pre-policy print."""
    s = make_service(lb._one_voice(), register={
        "Ana": lb._voice(1.0, 0.0, seed=3, jitter=0.01)})
    s.auto_collect_from_diarization(
        audio_path="meeting.wav",
        segments=lb._segments(
            "Ana", [(float(i * 10), float(i * 10 + 6.0)) for i in range(3)]),
        user_id="u", source_id="meeting.wav",
    )
    _older_print(s, count=30)

    findings = s._plan_source_trim("u")

    # The re-run's own three clips are inside the budget and appear in no
    # finding; the over-collected recording from the same stem is one group.
    assert len(findings) == 1
    assert findings[0]["kept"] == 10


def test_cleanup_integrity_reports_its_trim_as_a_plan_and_keeps_the_audio(
        make_service, lb):
    """``dry_run`` is the default and the first call has to change nothing."""
    s = make_service(lb._one_voice())
    _older_print(s, count=30)
    before = len(s._snippets.list_by_speaker("Ana", user_id="u"))

    result = s.cleanup_integrity("u", dry_run=True, **_trim_only())

    assert result["dry_run"] is True
    assert len(result["trimmed_snippets"]) == 1
    assert result["trimmed_snippets"][0]["removed"] == 20
    assert len(s._snippets.list_by_speaker("Ana", user_id="u")) == before


def test_cleanup_integrity_applies_the_trim_and_refines_once(make_service, lb):
    s = make_service(lb._one_voice())
    _older_print(s, count=30)

    result = s.cleanup_integrity("u", dry_run=False, **_trim_only())

    assert len(s._snippets.list_by_speaker("Ana", user_id="u")) == 10
    assert result["trimmed_snippets"][0]["removed"] == 20
    # One refine over the final set, not one per removed row.
    assert result["refined"] == ["Ana"]


def test_a_file_inside_its_budget_is_never_touched(make_service, lb):
    s = make_service(lb._one_voice(), register={
        "Ana": lb._voice(1.0, 0.0, seed=3, jitter=0.01)})
    s.auto_collect_from_diarization(
        audio_path="meeting.wav",
        segments=lb._segments(
            "Ana", [(float(i * 10), float(i * 10 + 6.0)) for i in range(3)]),
        user_id="u", source_id="meeting.wav",
    )

    result = s.cleanup_integrity("u", dry_run=False, **_trim_only())

    assert result["trimmed_snippets"] == []
    assert len(s._snippets.list_by_speaker("Ana", user_id="u")) == 3
