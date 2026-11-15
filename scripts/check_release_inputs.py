"""Check Git publication candidates without printing credential-like contents."""
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'google-key': re.compile(rb'AIza[0-9A-Za-z_-]{30,}'),
    'github-token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'openai-key': re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{40,}'),
    'private-key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}


def main():
    process = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                             cwd=ROOT, capture_output=True, check=True)
    files = sorted(set(process.stdout.decode('utf-8').strip('\0').split('\0')))
    errors = []
    for name in files:
        path = ROOT / name
        if not path.is_file():
            continue
        if path.suffix.lower() in ('.arw', '.nef', '.photocoach', '.pem', '.key') or path.name in (
                '.env', 'settings.json', 'PROJECT_AUDIT_REPORT.md', 'PhotoCoach-source.png'):
            errors.append(f'{name}: private/source material must not be published')
            continue
        data = path.read_bytes()
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(data):
                line = data[:match.start()].count(b'\n') + 1
                errors.append(f'{name}:{line}: {label} candidate (content withheld)')
    for error in errors:
        print(error)
    print(f'Checked {len(files)} publication candidates; {len(errors)} findings. No credential contents printed.')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
