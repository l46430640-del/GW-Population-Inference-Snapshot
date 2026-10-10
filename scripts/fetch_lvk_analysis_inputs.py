"""Retrieve two published input files without downloading the 2.7 GB archive."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
RECORD = 'https://zenodo.org/api/records/8087858'
NAMES = ('sampleDict_FAR_1_in_1_yr.pickle', 'injectionDict_FAR_1_in_1.pickle',
         'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')


class RemoteZip(io.RawIOBase):
    """A seekable HTTP range reader for Python's ZIP parser."""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position + size, self.size)
        if end <= self.position:
            return b''
        response = self.session.get(self.url, headers={'Range': f'bytes={self.position}-{end - 1}'},
                                    timeout=180)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != end - self.position:
            raise RuntimeError('Server did not honor the requested byte range')
        self.position = end
        return response.content


def main():
    record = requests.get(RECORD, timeout=60)
    record.raise_for_status()
    item = record.json()['files'][0]
    output = ROOT / 'external_data/lvk/analysis_inputs'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'record': 'https://doi.org/10.5281/zenodo.8087858',
                'archive_checksum': item['checksum'], 'files': []}
    with zipfile.ZipFile(RemoteZip(item['links']['self'], item['size'])) as archive:
        for name in NAMES:
            matches = [x for x in archive.infolist()
                       if Path(x.filename).name == name and not x.filename.startswith('__MACOSX/')]
            if len(matches) != 1:
                raise ValueError(f'Ambiguous ZIP member: {name}')
            info = matches[0]
            target = output / name
            if not target.exists():
                target.write_bytes(archive.read(info))
            manifest['files'].append({'name': name, 'zip_member': info.filename,
                                      'bytes': target.stat().st_size,
                                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
            print(name, target.stat().st_size, flush=True)
    (ROOT / 'docs/analysis/lvk_input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
