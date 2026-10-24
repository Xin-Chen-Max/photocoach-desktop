"""Editing invariants: processing, transactions, persistence, and coach context."""
import json

import numpy as np
import pytest

from photocoach.adjustments import render_edit
from photocoach.coach import MockCoach, build_request
from photocoach.imaging import RawOptions, render_srgb
from photocoach.project import read_project, write_project, snapshot_project, validate_project, SourceMissing
from photocoach.safe_io import sha256_file
from photocoach.state import EditValues, StateStore


@pytest.fixture
def store(tmp_path):
    source = tmp_path / "sample.ARW"
    source.write_bytes(b"synthetic source for persistence tests, not a decodable RAW")
    value = StateStore()
    value.activate(source, sha256_file(source))
    return value


def ten_edits(store):
    for key, value in [("exposure_ev", .25), ("contrast", .2), ("saturation", .9),
                       ("rgb_curve", ((0, .03), (.5, .55), (1, 1))),
                       ("raw_options", RawOptions("custom", (1.1, 1., .95))),
                       ("goal", "Natural light"), ("exposure_ev", .4),
                       ("contrast", .1), ("saturation", 1.05), ("exposure_ev", .5)]:
        store.set_parameter(key, value)


def test_ten_edits_undo_redo_restore_exact_values_and_pixels(store):
    linear = np.random.default_rng(42).random((20, 30, 3), dtype=np.float32)
    initial = render_edit(linear, store.current.values)
    ten_edits(store)
    final = store.current.values
    pixels = render_edit(linear, final)
    assert len(store.undo_stack) == 10
    for _ in range(10):
        store.undo()
    assert store.current.values == EditValues()
    np.testing.assert_array_equal(render_edit(linear, store.current.values), initial)
    for _ in range(10):
        store.redo()
    assert store.current.values == final
    np.testing.assert_array_equal(render_edit(linear, final), pixels)
    assert store.current.revision == 30
    validate_project(snapshot_project(store))


def test_drag_cancel_branch_reset_and_actual_before_state(store):
    store.begin_drag("contrast")
    for value in (.1, .2, .3):
        store.set_parameter("contrast", value)
    store.end_drag()
    assert len(store.undo_stack) == 1
    assert store.before_last_edit().revision == 0
    store.undo()
    assert store.before_last_edit().contrast == .3
    store.begin_drag("saturation")
    store.set_parameter("saturation", .7)
    store.end_drag(cancel=True)
    assert len(store.redo_stack) == 1
    store.redo()
    store.set_parameter("goal", "Keep natural light")
    store.reset()
    assert store.current.values == EditValues(goal="Keep natural light")
    store.undo()
    assert store.current.contrast == .3
    store.set_parameter("saturation", 1.1)
    assert not store.redo_stack
    count = len(store.undo_stack)
    store.set_compare(True)
    store.set_compare(False)
    assert len(store.undo_stack) == count
    validate_project(snapshot_project(store))


def test_processing_defaults_precision_and_curve():
    linear = np.random.default_rng(2).random((8, 9, 3), dtype=np.float32)
    original = linear.copy()
    np.testing.assert_array_equal(render_edit(linear, EditValues()), render_srgb(linear, 0))
    gray = render_edit(linear, EditValues(saturation=0))
    np.testing.assert_allclose(gray[..., 0], gray[..., 1], atol=1e-7)
    np.testing.assert_allclose(gray[..., 1], gray[..., 2], atol=1e-7)
    curve = render_edit(linear, EditValues(rgb_curve=((0, .1), (1, .9))))
    np.testing.assert_allclose(curve, .1 + .8 * render_srgb(linear, 0), atol=1e-7)
    assert curve.dtype == np.float32
    assert np.any(curve * 255 != np.round(curve * 255))
    np.testing.assert_array_equal(linear, original)


@pytest.mark.parametrize("kwargs", [{"contrast": float("nan")}, {"saturation": 3},
    {"exposure_ev": True}, {"rgb_curve": ((0, 0), (.5, .5), (.5, .6), (1, 1))},
    {"rgb_curve": ((.1, 0), (1, 1))}, {"rgb_curve": ((0, 0), (1, float("inf")))}])
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ValueError):
        EditValues(**kwargs)


def test_project_roundtrip_redo_and_repeated_save(store, tmp_path):
    ten_edits(store)
    store.undo()
    store.set_compare(True)
    store.mark_preview(store.current)
    path = tmp_path / "work.photocoach"
    original_hash = sha256_file(store.current.source_path)
    write_project(path, snapshot_project(store))
    restored = read_project(path)
    assert restored.current.values == store.current.values
    assert restored.current.compare_baseline
    assert restored.undo_stack == store.undo_stack
    assert restored.redo_stack == store.redo_stack
    assert restored.current.session_id != store.current.session_id
    restored.redo()
    with pytest.raises(FileExistsError):
        write_project(path, snapshot_project(restored))
    write_project(path, snapshot_project(restored), overwrite=True)
    again = read_project(path)
    assert again.current.exposure_ev == .5
    assert sha256_file(store.current.source_path) == original_hash


@pytest.mark.parametrize("damage", ["state", "event", "undo", "sequence", "pipeline", "schema"])
def test_tampered_project_rejected(store, damage):
    ten_edits(store)
    data = snapshot_project(store)
    if damage == "state":
        data["state"]["parameters"]["exposure_ev"] = 2
    elif damage == "event":
        data["history"]["events"][1]["after"] = 2
    elif damage == "undo":
        data["history"]["undo"].pop()
    elif damage == "sequence":
        data["history"]["events"][2]["sequence"] += 1
    else:
        data[damage + "_version"] = 999
    with pytest.raises(ValueError):
        validate_project(data)


def test_relocation_requires_identical_source(store, tmp_path):
    path = tmp_path / "work.photocoach"
    write_project(path, snapshot_project(store))
    relocated = tmp_path / "relocated.ARW"
    store.current.source_path.rename(relocated)
    with pytest.raises(SourceMissing):
        read_project(path)
    assert read_project(path, source_override=relocated).current.source_path == relocated
    wrong = tmp_path / "wrong.ARW"
    wrong.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash"):
        read_project(path, source_override=wrong)


def test_project_after_switch_uses_continuous_sequence(store, tmp_path):
    ten_edits(store)
    store.activate(store.current.source_path, store.current.photo_id)
    store.set_exposure(.3)
    path = tmp_path / "switched.photocoach"
    write_project(path, snapshot_project(store))
    restored = read_project(path)
    validate_project(snapshot_project(restored))
    assert restored.events[0]["sequence"] > 1


def test_coach_context_exact_bounded_private_and_mock_limits(store):
    for n in range(20):
        store.set_exposure(n / 10)
    store.set_parameter("saturation", .7)
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    request = build_request(store.current, store.before_last_edit(), store.events, image, image, "Why gray?")
    assert request.parameters == store.current.values.to_dict()
    assert request.before_parameters["saturation"] == 1
    assert request.before_state_version == store.current.revision - 1
    assert len(request.recent_operations) <= 12
    assert "source_path" not in repr(request) and str(store.current.source_path) not in repr(request)
    assert not request.current_preview.flags.writeable
    mock = MockCoach()
    mock.limit = 1
    feedback = mock.analyze(request)
    assert "0.70" in feedback.reason and "gray" in feedback.reason
    assert store.current.values.to_dict() == request.parameters
    with pytest.raises(RuntimeError, match="limit"):
        mock.analyze(request)
    mock.paused = True
    with pytest.raises(RuntimeError, match="paused"):
        mock.analyze(request)
    assert mock.calls == 1


def test_invalid_source_save_leaves_previous_project_intact(store, tmp_path):
    path = tmp_path / "work.photocoach"
    write_project(path, snapshot_project(store))
    content = path.read_bytes()
    store.current.source_path.write_bytes(b"changed test fixture")
    with pytest.raises(ValueError, match="changed"):
        write_project(path, snapshot_project(store), overwrite=True)
    assert path.read_bytes() == content


def test_dependency_change_warns_without_silently_changing_parameters(store, tmp_path):
    path = tmp_path / "versions.photocoach"
    write_project(path, snapshot_project(store))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["provenance"]["dependencies"]["rawpy"] = "old-test-version"
    path.write_text(json.dumps(data), encoding="utf-8")
    restored = read_project(path)
    assert "rawpy" in restored.load_warnings[0]
    assert restored.current.values == store.current.values


def test_confirmed_overwrite_writer_failure_preserves_existing_project(tmp_path):
    from photocoach.safe_io import atomic_write
    path = tmp_path / "existing.photocoach"
    path.write_bytes(b"keep previous project")
    def fail(temporary):
        temporary.write_bytes(b"partial")
        raise OSError("disk failure")
    with pytest.raises(OSError, match="disk failure"):
        atomic_write(path, fail, overwrite=True)
    assert path.read_bytes() == b"keep previous project"
    assert list(tmp_path.glob(".photocoach-*")) == []
