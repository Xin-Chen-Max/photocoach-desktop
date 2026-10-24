"""Verification reports must distinguish technical checks from acceptance."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from photocoach import cli


@pytest.fixture
def verification_fixture(tmp_path, monkeypatch):
    source = tmp_path / "private-original.ARW"
    original = b"synthetic RAW source, never edited"
    source.write_bytes(original)
    calls = []

    def decode(path, options, *, half_size):
        calls.append((path, options, half_size))
        shape = (4, 6, 3) if half_size else (8, 12, 3)
        # Independent fixtures: a full export accidentally using preview
        # pixels would fail both its size and value assertions.
        level = 0.1 if half_size else 0.2
        linear = np.full(shape, level, dtype=np.float32)
        if options.wb_mode == "custom":
            linear[..., 0] *= options.wb_gains[0]
        linear.setflags(write=False)
        metadata = {
            "visible_size": [12, 8], "output_size": [shape[1], shape[0]],
            "orientation_flip": 0, "effective_whitebalance": [2, 1, 1.5, 1],
            "model": "synthetic camera fixture", "half_size": half_size,
        }
        return SimpleNamespace(linear_rgb=linear, metadata=metadata)

    monkeypatch.setattr(cli, "decode_raw", decode)
    monkeypatch.setattr(cli, "dependency_versions", lambda: {"rawpy": "synthetic test"})
    return SimpleNamespace(source=source, original=original, calls=calls, output=tmp_path / "verification")


def test_verify_exports_latest_full_state_and_keeps_manual_gate_pending(verification_fixture, monkeypatch) -> None:
    fixture = verification_fixture
    recorded = {}
    real_export = cli.export_image

    def observe_export(path, pixels, **kwargs):
        recorded[path.name] = pixels.copy()
        return real_export(path, pixels, **kwargs)

    monkeypatch.setattr(cli, "export_image", observe_export)
    assert cli.verify(fixture.source, fixture.output, exposure=1.0, tiff=True, wb_checks=True) == 0
    report = json.loads((fixture.output / "report.json").read_text(encoding="utf-8"))
    assert report["technical_probe_completed"] is True
    assert report["stage0_gate"].startswith("NOT_PASSED")
    assert all(value.startswith("pending_") for value in report["manual_checks"].values())
    assert report["source_unchanged"] is True
    digest = hashlib.sha256(fixture.original).hexdigest()
    assert report["source_sha256_before"] == report["source_sha256_after"] == digest
    assert fixture.source.read_bytes() == fixture.original
    assert [item[2] for item in fixture.calls] == [True, True, True, False]
    assert report["custom_unity_wb_check"]["bit_exact_linear_match_to_camera"] is True
    assert len(report["exports"]) == 6
    assert all(item["roundtrip_check_passed"] for item in report["exports"])
    assert any(item["bits_per_channel"] == 16 for item in report["exports"])
    final = recorded["current-full-16bit.tiff"]
    assert final.shape == (8, 12, 3)
    expected = 1.055 * (0.2 * 2.0) ** (1.0 / 2.4) - 0.055
    np.testing.assert_allclose(final, expected, rtol=1e-6)
    np.testing.assert_array_equal(final, recorded["current-full.jpg"])
    assert np.mean(recorded["preview-current.jpg"]) > np.mean(recorded["preview-baseline.jpg"])
    assert report["orientation_dimension_check"]["dimensions_match_orientation"] is True
    assert not list(fixture.output.glob(".photocoach-*"))


def test_existing_output_directory_is_rejected_before_hashing_or_decoding(verification_fixture, monkeypatch) -> None:
    fixture = verification_fixture
    fixture.output.mkdir()
    earlier = fixture.output / "keep.txt"
    earlier.write_text("previous result", encoding="utf-8")
    monkeypatch.setattr(cli, "sha256_file", lambda _: pytest.fail("should reject destination first"))
    with pytest.raises(FileExistsError):
        cli.verify(fixture.source, fixture.output, exposure=0, tiff=False, wb_checks=False)
    assert fixture.calls == []
    assert earlier.read_text(encoding="utf-8") == "previous result"
    assert list(fixture.output.iterdir()) == [earlier]


def test_decoder_failure_still_rechecks_source_hash_and_sanitizes_report(verification_fixture, monkeypatch, capsys) -> None:
    fixture = verification_fixture
    private_detail = "C:/private/user/GPS-secret-and-api-key-value.ARW"
    hashes = []
    real_hash = cli.sha256_file

    def observed_hash(path):
        hashes.append(path)
        return real_hash(path)

    def failed_decode(*args, **kwargs):
        raise RuntimeError(private_detail)

    monkeypatch.setattr(cli, "decode_raw", failed_decode)
    monkeypatch.setattr(cli, "sha256_file", observed_hash)
    assert cli.verify(fixture.source, fixture.output, exposure=0, tiff=False, wb_checks=False) == 1
    serialized = (fixture.output / "failure-report.json").read_text(encoding="utf-8")
    report = json.loads(serialized)
    assert hashes == [fixture.source, fixture.source]
    assert report["technical_probe_completed"] is False
    assert report["source_unchanged"] is True
    assert report["error"]["step"] == "decode_camera_half"
    assert report["error"]["type"] == "RuntimeError"
    assert private_detail not in serialized
    assert str(fixture.source) not in serialized
    assert "GPS-secret" not in serialized
    assert private_detail not in capsys.readouterr().err
    assert list(fixture.output.iterdir()) == [fixture.output / "failure-report.json"]


def test_source_hash_change_fails_even_if_all_image_checks_pass(verification_fixture, monkeypatch) -> None:
    fixture = verification_fixture
    hashes = iter(["original-digest", "different-digest"])
    monkeypatch.setattr(cli, "sha256_file", lambda _: next(hashes))
    assert cli.verify(fixture.source, fixture.output, exposure=0, tiff=False, wb_checks=False) == 1
    report = json.loads((fixture.output / "failure-report.json").read_text(encoding="utf-8"))
    assert report["source_unchanged"] is False
    assert report["technical_probe_completed"] is False
    assert "integrity_error" in report
    assert not (fixture.output / "report.json").exists()
    assert fixture.source.read_bytes() == fixture.original


def test_final_hash_failure_is_not_reported_as_unchanged(verification_fixture, monkeypatch) -> None:
    fixture = verification_fixture
    calls = 0

    def hash_then_fail(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("temporary source read failure")
        return "original-digest"

    monkeypatch.setattr(cli, "sha256_file", hash_then_fail)
    assert cli.verify(fixture.source, fixture.output, exposure=0, tiff=False, wb_checks=False) == 1
    report = json.loads((fixture.output / "failure-report.json").read_text(encoding="utf-8"))
    assert report["source_unchanged"] is None
    assert report["technical_probe_completed"] is False


@pytest.mark.parametrize("flip,expected", [(0, [12, 8]), (3, [12, 8]), (5, [8, 12]), (6, [8, 12])])
def test_orientation_dimensions_use_libraw_flip_not_exif_numbers(flip, expected) -> None:
    result = cli._orientation_check({"visible_size": [12, 8], "output_size": expected, "orientation_flip": flip})
    assert result["dimensions_match_orientation"] is True
    assert result["visual_rotation_check"] == "pending_manual"


def test_unknown_orientation_does_not_claim_verified() -> None:
    result = cli._orientation_check({"visible_size": [12, 8], "output_size": [12, 8], "orientation_flip": 99})
    assert result["dimensions_match_orientation"] is None
    assert result["expected_full_size_from_orientation"] is None


def test_difference_reports_known_math_without_quality_verdict() -> None:
    first = np.zeros((1, 1, 3), dtype=np.float32)
    second = np.array([[[0.0, 0.25, 0.5]]], dtype=np.float32)
    result = cli._difference(first, second)
    assert result["mean_absolute_error_srgb_0_to_1"] == pytest.approx(0.25)
    assert result["max_absolute_error_srgb_0_to_1"] == 0.5
    assert result["rmse_srgb_0_to_1"] == pytest.approx((0.3125 / 3) ** 0.5)
    assert "passed" not in result
    assert cli._difference(first, np.zeros((2, 1, 3), dtype=np.float32))["comparable"] is False


def test_versions_mode_does_not_open_or_process_a_photo(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "dependency_versions", lambda: {"rawpy": "test-version"})
    monkeypatch.setattr(cli, "verify", lambda *a, **k: pytest.fail("versions should not verify"))
    assert cli.main(["--versions"]) == 0
    assert json.loads(capsys.readouterr().out)["rawpy"] == "test-version"


@pytest.mark.parametrize("exposure", [float("nan"), float("inf"), -5.1, 5.1])
def test_bad_exposure_creates_no_output_directory(verification_fixture, exposure) -> None:
    fixture = verification_fixture
    with pytest.raises(ValueError):
        cli.verify(fixture.source, fixture.output, exposure=exposure, tiff=False, wb_checks=False)
    assert not fixture.output.exists()
    assert fixture.calls == []
