"""Extract and run the Win64 ZIP without a checkout/Python on the executable PATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pefile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--raw', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    extracted = output / 'Portable test - 解压验证'
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert path.parts[0] == 'PhotoCoach'
            assert path.suffix.lower() not in ('.arw', '.nef', '.photocoach', '.pem', '.key')
            assert path.name not in ('settings.json', '.env', 'PROJECT_AUDIT_REPORT.md', 'PhotoCoach-source.png')
        archive.extractall(extracted)
    application = extracted / 'PhotoCoach'
    executable = application / 'PhotoCoach.exe'
    with pefile.PE(str(executable)) as pe:
        assert pe.FILE_HEADER.Machine == 0x8664
    manifest = json.loads((application / 'build-info.json').read_text(encoding='utf-8'))
    assert manifest['source_hash_format'] == 'sha256-lf'
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest, name + ': rebuild required'
    assert list((application / 'licenses/rawpy').rglob('*'))
    assert list((application / 'licenses/PySide6_Essentials').rglob('*'))
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORM'):
        env.pop(name, None)
    env['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32') + os.pathsep + os.environ['SystemRoot']
    env['LOCALAPPDATA'] = str(output / 'isolated-user')
    temporary = output / 'temporary'
    temporary.mkdir()
    env['TEMP'] = env['TMP'] = str(temporary)
    results = output / 'self-check'
    command = [str(executable), '--self-check', str(results)]
    if args.raw:
        command.append(str(args.raw.resolve()))
    with (output / 'process.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=application, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            code = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
            raise
    assert code == 0, f'Bundled executable exit: {code}; inspect process.log'
    report = json.loads((results / 'report.json').read_text(encoding='utf-8'))
    assert report['frozen'] and report['version'] == manifest['version']
    if args.raw:
        assert report['raw_checked'] and report['source_unchanged'] and report['sharpening_undo']
    report.update(architecture='win_amd64', exit_code=code, isolated_path=True,
                  archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest())
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
