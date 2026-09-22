"""Returned downloads span multiple directories; resolve without changing data."""
import hashlib
import json
from pathlib import Path
import pytest
from src.control.study_analysis import load_raw


def split(tmp_path):
    one = tmp_path / 'part-001/results_tiered/sample'
    two = tmp_path / 'part-002/results_tiered/sample'
    one.mkdir(parents=True)
    (two / 'units').mkdir(parents=True)
    raw = b'{"success":false}'
    (two / 'units/u.json').write_bytes(raw)
    (one / 'manifest.json').write_text(json.dumps({'actual': {'u': {
        'path': 'units/u.json', 'sha256': hashlib.sha256(raw).hexdigest()}}}))
    return one, two


def test_loads_verified_unit_from_sibling_part(tmp_path):
    one, _ = split(tmp_path)
    assert load_raw(one, 'u') == {'success': False}


def test_corrupt_sibling_part_is_rejected(tmp_path):
    one, two = split(tmp_path)
    (two / 'units/u.json').write_text('{}')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        load_raw(one, 'u')


def test_conflicting_duplicate_is_rejected(tmp_path):
    one, _ = split(tmp_path)
    (one / 'units').mkdir()
    (one / 'units/u.json').write_text('{}')
    with pytest.raises(ValueError, match='Conflicting'):
        load_raw(one, 'u')


def test_split_lookup_rejects_path_escape(tmp_path):
    one, _ = split(tmp_path)
    m = json.loads((one / 'manifest.json').read_text())
    m['actual']['u']['path'] = '../outside.json'
    (one / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='escapes bundle'):
        load_raw(one, 'u')
