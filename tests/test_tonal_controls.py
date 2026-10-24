"""Tonal processing, version migration and whole-photo SDR statistics contracts."""
import copy
import json

import numpy as np
import pytest

from photocoach.adjustments import render_edit
from photocoach.imaging import render_srgb, quantize_srgb
from photocoach.histogram import sampled_histogram
from photocoach.state import EditValues, StateStore, EDIT_PIPELINE_VERSION
from photocoach.project import snapshot_project, write_project, read_project, validate_project
from photocoach.safe_io import sha256_file
from photocoach.preview import render_region, DetailRequest
from photocoach.coach import build_request


@pytest.mark.parametrize("highlights", [-1, -.5, 0, .5, 1])
@pytest.mark.parametrize("shadows", [-1, -.5, 0, .5, 1])
def test_gray_ramp_remains_monotonic_and_finite(highlights, shadows):
    gray = np.geomspace(1e-6, 32, 16000).astype(np.float32)
    source = np.repeat(gray[None,:,None], 3, axis=2)
    source.setflags(write=False)
    result = render_edit(source, EditValues(highlights=highlights, shadows=shadows))
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    assert np.diff(result[0,:,0]).min() >= -2e-7
    np.testing.assert_array_equal(source[0,:,0], gray)


def test_zero_controls_preserve_existing_render_and_do_not_invent_lost_detail():
    source = np.random.default_rng(32).random((300,27,3), dtype=np.float32)
    for exposure in (-5, -.7, 0, .25, 5):
        np.testing.assert_array_equal(render_edit(source, EditValues(exposure_ev=exposure)), render_srgb(source, exposure))
    flat = np.ones((3,17,3), np.float32)
    lowered = render_edit(flat, EditValues(highlights=-1))
    assert lowered.max() < 1 and np.unique(lowered).size == 1  # Darker flat white, no recovered texture.
    assert not render_edit(flat*0, EditValues(shadows=1)).any()


def test_zone_controls_target_different_luminances_and_precede_display_clipping():
    linear = np.repeat(np.array([.02,.8],np.float32)[None,:,None],3,axis=2)
    original = render_srgb(linear)
    shadows = render_edit(linear, EditValues(shadows=1))
    highlights = render_edit(linear, EditValues(highlights=-1))
    assert (shadows-original)[0,0,0] > (shadows-original)[0,1,0]*4
    assert (original-highlights)[0,1,0] > (original-highlights)[0,0,0]*4
    # Display clipping induced by an edit can be reduced; this is not RAW recovery.
    assert quantize_srgb(render_edit(linear, EditValues(exposure_ev=1)))[0,1,0] == 255
    assert quantize_srgb(render_edit(linear, EditValues(exposure_ev=1, highlights=-1)))[0,1,0] < 250


def test_shared_linear_gain_preserves_rgb_ratios_until_display_operations():
    rgb = np.array([[[.04,.08,.12],[.2,.3,.4]]],np.float32)
    out = render_edit(rgb, EditValues(highlights=-.7, shadows=.8))
    linear = np.where(out <= .04045, out/12.92, ((out+.055)/1.055)**2.4)
    np.testing.assert_allclose(linear[...,0]/linear[...,1], rgb[...,0]/rgb[...,1], rtol=1e-5)
    np.testing.assert_allclose(linear[...,2]/linear[...,1], rgb[...,2]/rgb[...,1], rtol=1e-5)


@pytest.mark.parametrize("name", ["highlights", "shadows"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 1.01, -1.01, "0"])
def test_zone_parameter_validation(name, value):
    with pytest.raises(ValueError):
        EditValues(**{name:value})


def test_detail_matches_full_render_with_zone_controls():
    source = np.random.default_rng(5).random((400,700,3), dtype=np.float32)
    values = EditValues(exposure_ev=.6, shadows=.55, highlights=-.8, contrast=.1, saturation=.9,
                        rgb_curve=((0,0),(.5,.53),(1,1)))
    roi = DetailRequest(1,(700,400),(109,83,509,333),1)
    np.testing.assert_array_equal(render_region(source,values,roi), quantize_srgb(render_edit(source,values))[83:333,109:509])


def test_display_histogram_counts_limits_not_raw_recoverability():
    image = np.array([[[0,0,0],[1,1,1],[1,.1,.1],[.18,.18,.18]]],np.float32)
    stats = sampled_histogram(image, EditValues())
    assert stats["samples"] == 4 and stats["black_percent"] == 25 and stats["white_percent"] == 50
    assert len(stats["rgb"]) == 3 and all(len(x)==256 and sum(x)==4 for x in stats["rgb"])
    darker = sampled_histogram(image, EditValues(highlights=-1))
    assert darker["white_percent"] < 50


def test_histogram_work_is_bounded_and_reads_without_modifying_source():
    image = np.full((1800,2700,3), .18, np.float32)
    image.setflags(write=False)
    stats = sampled_histogram(image, EditValues(shadows=.3))
    assert stats["samples"] <= 512*512 and stats["stride"] == 6
    assert np.all(image == np.float32(.18))


def make_store(tmp_path):
    source = tmp_path/"synthetic.ARW"
    source.write_bytes(b"Synthetic persistence fixture, not a decodable RAW")
    store = StateStore()
    store.activate(source,sha256_file(source))
    return store


def test_ten_zone_edits_gestures_undo_save_and_ai_context(tmp_path):
    store = make_store(tmp_path)
    for i in range(1,11):
        store.begin_drag("shadows")
        store.set_parameter("shadows", i/10-.04)
        store.set_parameter("shadows", i/10)
        store.end_drag()
    assert len(store.undo_stack)==10
    before = store.current
    store.set_parameter("highlights", -.6)
    current = store.current
    store.undo()
    assert store.current.values == before.values
    store.redo()
    assert store.current.values == current.values
    pixels = np.zeros((3,4,3),np.uint8)
    request = build_request(store.current, before, store.events, pixels,pixels)
    assert request.parameters["highlights"] == -.6 and request.parameters["shadows"] == 1
    assert request.pipeline_version == EDIT_PIPELINE_VERSION
    target = tmp_path/"v2.photocoach"
    write_project(target,snapshot_project(store))
    restored = read_project(target)
    assert restored.current.values == store.current.values
    assert restored.undo_stack == store.undo_stack
    restored.reset()
    assert restored.current.highlights == restored.current.shadows == 0


@pytest.mark.parametrize("protocol",["responses","chat_completions","gemini"])
def test_zone_values_are_preserved_in_provider_context(tmp_path,protocol):
    from photocoach.api_coach import ApiConfig, GEMINI_ENDPOINT, build_payload, request_context
    store=make_store(tmp_path)
    before=store.current
    store.set_parameter("highlights",-.7)
    store.set_parameter("shadows",.6)
    pixels=np.zeros((3,4,3),np.uint8)
    request=build_request(store.current,before,store.events,pixels,pixels,"Keep the bright sky")
    context=request_context(request)
    assert context["parameters"]["highlights"]==-.7 and context["parameters"]["shadows"]==.6
    config=ApiConfig(GEMINI_ENDPOINT if protocol=="gemini" else "https://example.invalid/api",
                     "gemini-test" if protocol=="gemini" else "test",protocol)
    payload=json.dumps(build_payload(config,request))
    assert "highlights" in payload and "shadows" in payload
    assert "display_histogram" not in payload and str(tmp_path) not in payload


def legacy_document(data):
    """Produce the original exact parameter shape for legacy editing operations."""
    data = copy.deepcopy(data)
    def walk(value):
        if isinstance(value,dict):
            if "white_balance" in value and "rgb_curve" in value:
                value.pop("highlights")
                value.pop("shadows")
                value.pop("composition")
                value.pop("color_mixer")
                value.pop("local_gradients")
                for name in ("whites", "blacks", "vibrance"): value.pop(name)
            for item in value.values(): walk(item)
        elif isinstance(value,list):
            for item in value: walk(item)
    walk(data)
    data["schema_version"] = 1
    data["pipeline_version"] = "phase1-sdr-srgb-v1"
    return data


def test_legacy_project_migration_keeps_history_pixels_and_original_file(tmp_path):
    store = make_store(tmp_path)
    store.begin_drag("exposure_ev")
    store.set_exposure(.4)
    store.end_drag()
    store.set_parameter("contrast", .25)
    store.reset()
    store.undo()
    data = legacy_document(snapshot_project(store))
    data["source"]["byte_length"] = store.current.source_path.stat().st_size
    old = tmp_path/"v1.photocoach"
    old.write_text(json.dumps(data),encoding="utf-8")
    digest = sha256_file(old)
    restored = read_project(old)
    assert restored.current.values == store.current.values and restored.load_warnings
    assert restored.redo_stack == store.redo_stack
    restored.redo()
    restored.undo()
    restored.set_parameter("shadows", .2)
    write_project(tmp_path/"v2.photocoach", snapshot_project(restored))
    assert sha256_file(old)==digest and data["schema_version"]==1
    assert json.loads((tmp_path/"v2.photocoach").read_text())["schema_version"]==7


def test_project_migration_is_strict_for_missing_tampered_or_wrong_version_data(tmp_path):
    store = make_store(tmp_path)
    data = snapshot_project(store)
    del data["state"]["parameters"]["shadows"]
    with pytest.raises(ValueError): validate_project(data)
    legacy = legacy_document(snapshot_project(store))
    legacy["state"]["parameters"]["highlights"] = .5
    with pytest.raises(ValueError): validate_project(legacy)
    legacy = legacy_document(snapshot_project(store))
    legacy["pipeline_version"] = "unknown"
    with pytest.raises(ValueError): validate_project(legacy)
