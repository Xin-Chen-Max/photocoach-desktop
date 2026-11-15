"""Build a standalone AMD64 folder/ZIP from explicit application inputs only.

Run with the project environment on Windows. Install requirements-build.txt.
PyInstaller reference: https://pyinstaller.org/en/stable/usage.html
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if sys.platform != 'win32' or platform.machine().upper() != 'AMD64':
        parser.error('Build on 64-bit Windows using 64-bit CPython')
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    version = project['version']
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onedir', '--windowed',
        '--name', 'PhotoCoach', '--noupx', '--paths', str(root / 'src'),
        '--icon', str(root / 'src/photocoach/assets/PhotoCoach.ico'),
        '--distpath', str(output / 'bundle'), '--workpath', str(output / 'work'),
        '--specpath', str(output / 'spec'), '--collect-submodules', 'photocoach',
        '--add-data', str(root / 'src/photocoach/assets') + ';photocoach/assets',
        '--exclude-module', 'PySide6.QtWebEngineWidgets', '--exclude-module', 'PySide6.QtPdf',
        '--exclude-module', 'pytest', '--exclude-module', 'tkinter']
    packages = ['numpy', 'rawpy', 'pillow', 'tifffile', 'PySide6', 'PySide6_Essentials', 'shiboken6']
    for package in packages:
        command.extend(['--copy-metadata', package])
    # Assets are enumerated below; the private source image must never enter.
    assets = output / 'public-assets'
    assets.mkdir()
    for asset in ('PhotoCoach.ico', 'PhotoCoach.png', 'chevron-down.svg', 'chevron-up.svg', 'check.svg'):
        shutil.copy2(root / 'src/photocoach/assets' / asset, assets / asset)
    command[command.index('--add-data') + 1] = str(assets) + ';photocoach/assets'
    command.append(str(root / 'packaging/desktop_entry.py'))
    # Do not let unrelated tools on this machine's PATH supply same-named DLLs.
    # In particular Poppler's ICU DLL has different exports from Windows ICU.
    environment = dict(os.environ)
    environment['PATH'] = os.pathsep.join([str(Path(sys.executable).parent), sys.base_prefix,
        str(Path(sys.base_prefix) / 'DLLs'), str(Path(os.environ['SystemRoot']) / 'System32'),
        os.environ['SystemRoot']])
    for name in ('PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        environment.pop(name, None)
    with (output / 'build.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    bundle = output / 'bundle/PhotoCoach'
    assert not list(bundle.rglob('icuuc.dll')), 'Windows ICU must not be shadowed by a toolchain DLL'
    licenses = bundle / 'licenses'
    licenses.mkdir()
    for package in [*packages, 'PySide6_Addons', 'pyinstaller']:
        distribution = importlib.metadata.distribution(package)
        for file in distribution.files or ():
            if any(part.lower() in ('licenses', 'license', 'copying', 'notices') for part in file.parts) or file.name.lower().startswith(('license', 'copying', 'notice')):
                source = Path(distribution.locate_file(file))
                if source.is_file():
                    target = licenses / package / Path(*file.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    if python_license.is_file():
        shutil.copy2(python_license, licenses / 'Python-LICENSE.txt')
    shutil.copy2(root / 'packaging/README-Windows.txt', bundle / 'README-Windows.txt')
    shutil.copy2(root / 'packaging/THIRD_PARTY_NOTICES.txt', bundle / 'THIRD_PARTY_NOTICES.txt')
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
        if (root / name).is_file():
            shutil.copy2(root / name, bundle / name)
    info = dict(version=version, architecture='win_amd64', python=platform.python_version(),
                builder=importlib.metadata.version('pyinstaller'),
                dependencies={p: importlib.metadata.version(p) for p in packages},
                source_hash_format='sha256-lf',
                source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for p in sorted((root / 'src/photocoach').rglob('*.py'))})
    (bundle / 'build-info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    blocked_suffixes = {'.arw', '.nef', '.photocoach', '.key', '.pem'}
    for path in bundle.rglob('*'):
        if path.is_file():
            assert path.suffix.lower() not in blocked_suffixes, path.name
            assert path.name not in ('settings.json', '.env', 'PROJECT_AUDIT_REPORT.md', 'PhotoCoach-source.png'), path.name
    archive = output / f'PhotoCoach-{version}-win64.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                zipped.write(path, path.relative_to(bundle.parent))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / 'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=digest), indent=2))


if __name__ == '__main__':
    main()
