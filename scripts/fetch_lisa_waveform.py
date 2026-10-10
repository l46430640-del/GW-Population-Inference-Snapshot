"""Fetch the pinned public waveform implementation for rebuilding the SNR asset."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '0a008e776fe6f92aeca18ebba0ba6ec7216b6b16'


def main():
    destination = ROOT / 'external_data/literature/PyIMRPhenomD'
    url = f'https://codeload.github.com/XGI-MSU/PyIMRPhenomD/zip/{COMMIT}'
    if not (destination / 'IMRPhenomD_internals.py').exists():
        response = requests.get(url, timeout=(15, 120))
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for name in archive.namelist():
                relative = Path(*Path(name).parts[1:])
                path = (destination / relative).resolve()
                if not path.is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe waveform archive path')
                if name.endswith('/'):
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
    dependencies = ROOT / 'external_data/literature/python_deps'
    if not (dependencies / 'interpolation').exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--target', str(dependencies), 'interpolation==2.2.7'], check=True)
    manifest = {'repository': 'https://github.com/XGI-MSU/PyIMRPhenomD',
                'commit': COMMIT, 'archive_url': url, 'license': 'GPL-2.0-or-later',
                'source_sha256': hashlib.sha256((destination / 'IMRPhenomD_internals.py').read_bytes()).hexdigest(),
                'additional_dependency': 'interpolation==2.2.7'}
    (ROOT / 'docs/analysis/waveform_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
