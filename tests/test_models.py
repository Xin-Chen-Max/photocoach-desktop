from pathlib import Path

import pytest

from photocoach.imaging import RawOptions
from photocoach.models import ProbeStore


def test_ten_edits_replay_matches_current_and_source_is_not_mutated():
    store = ProbeStore()
    initial = store.activate(Path('example.ARW'), 'digest-a')
    for step in range(1, 11):
        store.set_exposure(step / 10)
    edits = [e for e in store.events if e['type'] == 'parameter']
    reconstructed = 0.0
    for edit in edits:
        assert edit['before'] == reconstructed
        reconstructed = edit['after']
    assert reconstructed == store.current.exposure_ev == 1
    assert store.current.revision == 10
    assert initial.exposure_ev == 0
    assert [e['sequence'] for e in store.events] == list(range(1, 12))


def test_drag_event_boundaries_share_transaction_and_end_without_extra_edit():
    store = ProbeStore()
    store.activate(Path('example.ARW'), 'digest-a')
    store.begin_drag()
    for value in (.1, .2, .3):
        store.set_exposure(value)
    store.end_drag()
    events = store.events[1:]
    assert [e['phase'] for e in events] == ['begin', 'update', 'update', 'update', 'end']
    assert len({e['transaction_id'] for e in events}) == 1
    assert store.current.revision == 3
    store.set_exposure(.4)
    assert store.events[-1]['transaction_id'] is None


def test_old_preview_rejected_after_edit_and_reopening_same_file():
    store = ProbeStore()
    old = store.activate(Path('example.ARW'), 'digest-a')
    assert store.mark_preview(old)
    store.set_exposure(.25)
    assert not store.mark_preview(old)
    assert store.preview_revision == 0
    edited = store.current
    assert store.mark_preview(edited)
    store.activate(Path('example.ARW'), 'digest-a')
    assert not store.mark_preview(old)
    assert not store.mark_preview(edited)
    assert store.preview_revision is None


def test_comparison_does_not_change_export_parameters():
    store = ProbeStore()
    store.activate(Path('example.ARW'), 'digest-a')
    store.set_exposure(.7)
    wb = RawOptions('custom', (1.2, 1, .9))
    store.set_whitebalance(wb)
    captured = store.current
    store.set_compare(True)
    assert store.current.exposure_ev == captured.exposure_ev
    assert store.current.raw_options == wb
    assert not captured.compare_baseline
    assert store.events[-1]['type'] == 'compare'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 6, -6])
def test_invalid_exposure_does_not_change_state(value):
    store = ProbeStore()
    initial = store.activate(Path('example.ARW'), 'digest-a')
    with pytest.raises(ValueError):
        store.set_exposure(value)
    assert store.current == initial
    assert len(store.events) == 1
