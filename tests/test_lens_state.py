"""Manual lens state and strict, read-only v1-v7 history migration."""
import copy
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from photocoach.coach import build_request
from photocoach.lens import LensCorrections
from photocoach.project import read_project, snapshot_project, upgrade_project, validate_project, write_project
from photocoach.safe_io import sha256_file
from photocoach.state import EditValues, StateStore


@pytest.mark.parametrize('name,limit', [('distortion', 100), ('vignetting_ev', 2)])
@pytest.mark.parametrize('value', [True, None, '0', float('nan'), float('inf'), 1.001])
def test_lens_invalid_values(name, limit, value):
    if value == 1.001:
        value *= limit
    with pytest.raises(ValueError):
        LensCorrections(**{name: value})


def test_lens_immutable_strict_roundtrip_and_limits():
    for sign in (-1, 1):
        lens = LensCorrections(sign * 100, sign * 2)
        assert LensCorrections.from_dict(lens.to_dict()) == lens
    with pytest.raises(FrozenInstanceError):
        lens.distortion = 0
    for bad in ({}, {'distortion': 0}, dict(lens.to_dict(), profile='automatic')):
        with pytest.raises(ValueError): LensCorrections.from_dict(bad)
    with pytest.raises(ValueError): EditValues(lens_corrections={})


def make_store(tmp_path):
    source = tmp_path/'fixture.ARW'
    source.write_bytes(b'unchanged source fixture')
    store = StateStore()
    store.activate(source, sha256_file(source))
    return store


def test_lens_ten_grouped_edits_resets_restore_and_ai(tmp_path):
    store = make_store(tmp_path)
    before = store.current
    for i in range(1, 11):
        store.begin_drag('lens_corrections')
        for j in range(3):
            store.set_parameter('lens_corrections', LensCorrections(i * 3 + j, i / 10))
        store.end_drag()
    assert len(store.undo_stack) == 10
    final = store.current.values
    store.undo(); store.redo()
    assert store.current.values == final
    store.set_parameter('lens_corrections', LensCorrections())
    assert len(store.undo_stack) == 11
    store.undo()
    assert store.current.values == final
    store.set_compare(True)
    assert store.current.values == final
    store.set_compare(False)
    target = tmp_path/'session.photocoach'
    write_project(target, snapshot_project(store))
    loaded = read_project(target)
    assert loaded.current.values == final
    assert loaded.undo_stack == store.undo_stack and loaded.redo_stack == store.redo_stack
    pixels = np.zeros((3, 4, 3), dtype=np.uint8)
    request = build_request(store.current, before, store.events, pixels, pixels)
    assert request.parameters['lens_corrections'] == final.lens_corrections.to_dict()
    store.reset()
    assert store.current.lens_corrections == LensCorrections()
    store.undo()
    assert store.current.values == final


def legacy_shape(data, version):
    removed = {'lens_corrections', 'base_rendering', 'color_grading', 'spots'}
    if version <= 5: removed.add('local_gradients')
    if version <= 4: removed.update(('whites', 'blacks', 'vibrance'))
    if version <= 3: removed.add('color_mixer')
    if version <= 2: removed.add('composition')
    if version == 1: removed.update(('highlights', 'shadows'))
    def walk(value):
        if isinstance(value, dict):
            value.pop('compare_snapshot_id', None)
            value.pop('snapshots', None)
            if 'white_balance' in value:
                for name in removed: value.pop(name)
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(data)
    data['schema_version'] = version
    data['pipeline_version'] = f'phase1-sdr-srgb-v{version}'
    return data


@pytest.mark.parametrize('version', range(1, 8))
def test_all_legacy_snapshots_migrate_neutrally_without_writing(tmp_path, version):
    store = make_store(tmp_path)
    store.begin_drag('exposure_ev')
    store.set_exposure(.2); store.set_exposure(.4); store.end_drag()
    store.set_parameter('contrast', .25)
    store.reset(); store.undo()
    data = legacy_shape(snapshot_project(store), version)
    data['source']['byte_length'] = store.current.source_path.stat().st_size
    original = copy.deepcopy(data)
    up = upgrade_project(data)
    assert data == original and up['schema_version'] == 14
    assert up['pipeline_version'] == 'phase1-sdr-srgb-v14'
    undo, redo = validate_project(up)
    assert undo == store.undo_stack and redo == store.redo_stack
    assert EditValues.from_dict(up['state']['parameters']) == store.current.values
    path = tmp_path/f'v{version}.photocoach'
    path.write_text(json.dumps(data), encoding='utf-8')
    before_hash = sha256_file(path)
    loaded = read_project(path)
    assert sha256_file(path) == before_hash
    assert loaded.current.values == store.current.values
    assert loaded.undo_stack == undo and loaded.redo_stack == redo
    assert 'format v14' in loaded.load_warnings[0]
    bad = copy.deepcopy(data)
    bad['state']['parameters']['lens_corrections'] = LensCorrections().to_dict()
    with pytest.raises(ValueError): upgrade_project(bad)


@pytest.mark.parametrize('damage', ['missing', 'unknown', 'invalid', 'history', 'pipeline'])
def test_v8_missing_or_inconsistent_lens_data_is_never_repaired(tmp_path, damage):
    store = make_store(tmp_path)
    store.set_parameter('lens_corrections', LensCorrections(30, .2))
    data = snapshot_project(store)
    p = data['state']['parameters']
    if damage == 'missing': p.pop('lens_corrections')
    elif damage == 'unknown': p['lens_corrections']['profile'] = 'automatic'
    elif damage == 'invalid': p['lens_corrections']['distortion'] = True
    elif damage == 'history': data['history']['events'][-1]['edit_after']['lens_corrections']['distortion'] = 25
    else: data['pipeline_version'] = 'phase1-sdr-srgb-v7'
    unchanged = copy.deepcopy(data)
    assert upgrade_project(data) is data
    with pytest.raises(ValueError): validate_project(data)
    assert data == unchanged
