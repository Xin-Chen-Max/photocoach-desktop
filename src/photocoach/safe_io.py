"""Read-only source access and no-clobber publication of completed output files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable


def sha256_file(path: str | Path) -> str:
    """Hash source bytes using an explicitly read-only handle."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_destination(
    path: str | Path, *, forbidden_paths: Iterable[str | Path] = (), allow_existing: bool = False
) -> Path:
    """Validate output; existing regular files require explicit caller consent."""
    requested = Path(path).expanduser()
    if ":" in requested.name or requested.name.endswith((".", " ")):
        # Reject NTFS alternate data streams and Win32 path normalization aliases.
        raise ValueError("The destination filename cannot contain a colon or end with a space or period.")
    if requested.suffix.lower() == ".arw":
        raise ValueError("Original ARW files are read-only and cannot be output destinations.")
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(str(parent))
    destination = parent / requested.name
    for forbidden in forbidden_paths:
        if destination.resolve() == Path(forbidden).expanduser().resolve():
            raise ValueError("The output destination cannot be the original photo.")
        if destination.exists() and Path(forbidden).exists():
            if os.path.samefile(destination, forbidden):
                raise ValueError("The output destination cannot point to the original photo.")
    # lexists includes broken symlinks, which must not be replaced either.
    if os.path.lexists(destination):
        if not allow_existing:
            raise FileExistsError(f"The file already exists. Choose a new filename: {destination.name}")
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("Only a regular file can be replaced after confirmation.")
    return destination


def atomic_write(
    path: str | Path,
    writer: Callable[[Path], object],
    *,
    forbidden_paths: Iterable[str | Path] = (),
    overwrite: bool = False,
) -> Path:
    """Write a sibling temporary file, then publish atomically.

    On Windows os.rename fails if the destination exists (unlike POSIX rename).
    On POSIX an atomic hard-link publication provides the same no-clobber rule.
    The temporary file and destination are on the same filesystem. Unsupported
    filesystem operations fail safely; there is no copy fallback. By default
    existing files are refused. overwrite=True is reserved for confirmed project
    replacement and revalidates source protection before os.replace.
    Only this function's own temporary file is ever removed.
    """
    forbidden_paths = tuple(forbidden_paths)
    destination = validate_destination(path, forbidden_paths=forbidden_paths, allow_existing=overwrite)
    descriptor, name = tempfile.mkstemp(
        prefix=".photocoach-", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        # Windows _commit requires a writable handle, even after the writer
        # has closed its handle. This is only our newly created temp output.
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            validate_destination(destination, forbidden_paths=forbidden_paths, allow_existing=True)
            os.replace(temporary, destination)
        elif os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()
