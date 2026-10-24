"""No-clobber publication and original-source protection, including races."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from photocoach import safe_io
from photocoach.safe_io import atomic_write, sha256_file, validate_destination


def test_sha256_is_exact_and_opens_source_readonly(tmp_path, monkeypatch) -> None:
    source = tmp_path / "photo.ARW"
    data = b"source RAW bytes" * 100_000
    source.write_bytes(data)
    modes = []
    original_open = Path.open

    def observed_open(path, mode="r", *args, **kwargs):
        if path == source:
            modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    assert sha256_file(source) == hashlib.sha256(data).hexdigest()
    assert modes == ["rb"]
    with original_open(source, "rb") as stream:
        assert stream.read() == data


@pytest.mark.parametrize("filename", ["output.arw", "output.ARW", "output.ArW",
                                     "photo.jpg:private", "photo.jpg.", "photo.jpg "])
def test_destination_rejects_raw_and_windows_normalization_aliases(tmp_path, filename) -> None:
    with pytest.raises(ValueError):
        validate_destination(tmp_path / filename)
    assert list(tmp_path.iterdir()) == []


def test_destination_requires_an_existing_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_destination(tmp_path / "missing" / "output.jpg")
    existing = tmp_path / "a-file"
    existing.write_bytes(b"preserve")
    with pytest.raises(NotADirectoryError):
        validate_destination(existing / "output.jpg")


def test_existing_file_is_never_replaced_and_writer_is_not_called(tmp_path) -> None:
    destination = tmp_path / "photo.jpg"
    destination.write_bytes(b"earlier export")
    calls = []
    with pytest.raises(FileExistsError):
        atomic_write(destination, lambda path: calls.append(path))
    assert calls == []
    assert destination.read_bytes() == b"earlier export"
    assert not list(tmp_path.glob(".photocoach-*"))


def test_forbidden_source_path_and_hardlink_alias_are_rejected(tmp_path) -> None:
    # A forbidden source remains protected even when its extension is unusual.
    source = tmp_path / "source.data"
    source.write_bytes(b"private RAW bytes")
    with pytest.raises(ValueError, match="original photo"):
        validate_destination(source, forbidden_paths=(source,))
    alias = tmp_path / "alias.jpg"
    os.link(source, alias)
    with pytest.raises(ValueError, match="original photo"):
        validate_destination(alias, forbidden_paths=(source,))
    assert source.read_bytes() == b"private RAW bytes"
    assert alias.read_bytes() == b"private RAW bytes"


def test_broken_symlink_is_not_replaced(tmp_path) -> None:
    target = tmp_path / "missing-original.jpg"
    link = tmp_path / "output.jpg"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"This Windows account does not permit test symlinks: {exc}")
    with pytest.raises(FileExistsError):
        atomic_write(link, lambda path: path.write_bytes(b"new"))
    assert link.is_symlink()
    assert not target.exists()


def test_successful_atomic_write_publishes_only_completed_bytes(tmp_path) -> None:
    destination = tmp_path / "result.tiff"
    observations = []

    def writer(temporary):
        observations.append(temporary)
        assert temporary.parent == destination.parent
        assert temporary.name.startswith(".photocoach-")
        assert temporary.suffix == ".tiff"
        assert not destination.exists()
        temporary.write_bytes(b"complete export")
        assert not destination.exists()

    assert atomic_write(destination, writer) == destination
    assert destination.read_bytes() == b"complete export"
    assert not observations[0].exists()
    assert list(tmp_path.iterdir()) == [destination]


def test_writer_failure_preserves_source_and_removes_only_owned_temp(tmp_path) -> None:
    destination = tmp_path / "output.jpg"
    source = tmp_path / "photo.ARW"
    source.write_bytes(b"unchanged source")
    other_temp = tmp_path / ".photocoach-somebody-elses.jpg"
    other_temp.write_bytes(b"another operation")

    def broken_writer(temporary):
        temporary.write_bytes(b"partial export")
        raise RuntimeError("encoding failed")

    with pytest.raises(RuntimeError, match="encoding failed"):
        atomic_write(destination, broken_writer, forbidden_paths=(source,))
    assert not destination.exists()
    assert source.read_bytes() == b"unchanged source"
    assert other_temp.read_bytes() == b"another operation"
    assert set(tmp_path.iterdir()) == {source, other_temp}


def test_racing_new_destination_is_not_clobbered_and_temp_is_removed(tmp_path) -> None:
    destination = tmp_path / "result.jpg"

    def racing_writer(temporary):
        temporary.write_bytes(b"our complete export")
        # Another process creates the chosen target after the initial check.
        destination.write_bytes(b"the other process won")

    with pytest.raises(FileExistsError):
        atomic_write(destination, racing_writer)
    assert destination.read_bytes() == b"the other process won"
    assert list(tmp_path.iterdir()) == [destination]


def test_publication_failure_has_no_overwrite_fallback_and_cleans_temp(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "result.jpg"

    def unsupported(*args, **kwargs):
        raise OSError("filesystem refused atomic publication")

    monkeypatch.setattr(safe_io.os, "rename" if os.name == "nt" else "link", unsupported)
    with pytest.raises(OSError, match="refused atomic"):
        atomic_write(destination, lambda temporary: temporary.write_bytes(b"complete"))
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
