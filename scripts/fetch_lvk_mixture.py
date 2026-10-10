"""Fetch the published O1--O3 mixture injection campaign, including high masses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
NAME = 'o1+o2+o3_mixture_real+semianalytic-LIGO-T2100377-v2.hdf5'
URL = 'https://zenodo.org/records/5636816/files/'+NAME+'?download=1'
EXPECTED_MD5 = 'd9c14e817dac05c83c66e83fa8a211a0'


def digest(path,algorithm):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,algorithm).hexdigest()


def main():
    path = ROOT/'external_data/lvk/analysis_inputs'/NAME
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        partial = path.with_suffix('.partial')
        with requests.get(URL,stream=True,timeout=180) as response:
            response.raise_for_status()
            with partial.open('wb') as handle:
                for chunk in response.iter_content(4*1024*1024):
                    handle.write(chunk)
        if digest(partial,'md5') != EXPECTED_MD5:
            raise ValueError('Injection download does not match the published checksum')
        partial.rename(path)
    if digest(path,'md5') != EXPECTED_MD5:
        raise ValueError('Cached injection campaign checksum mismatch')
    result = {'url':URL,'doi':'10.5281/zenodo.5636816','file':path.relative_to(ROOT).as_posix(),
              'bytes':path.stat().st_size,'md5':EXPECTED_MD5,'sha256':digest(path,'sha256'),
              'documentation':'https://zenodo.org/records/5636816/files/o1%2Bo2%2Bo3-sensitivity-estimates.md',
              'reason':'The separate BBH campaign has m1 <= 100 Msun; a wider population requires mixture injections.'}
    (ROOT/'docs/analysis/lvk_mixture_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
