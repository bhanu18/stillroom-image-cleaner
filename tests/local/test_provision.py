import json

import pytest

from cleaner.domain import Problem
from cleaner.models import hash_file
from cleaner.provision import SOURCES, provision


def test_existing_installation_is_verified_and_preserved(tmp_path):
    target = tmp_path / 'models/siglip'
    target.mkdir(parents=True)
    weights = target / 'model.safetensors'
    weights.write_bytes(b'test artifact')
    revision = 'a' * 40
    manifest = {
        'source': SOURCES['siglip'][0], 'revision': revision,
        'adapter': 'siglip_v1', 'approved': True,
        'files': {'model.safetensors': hash_file(weights)},
    }
    path = target / 'manifest.json'
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    assert provision(tmp_path, 'siglip', revision) == manifest
    assert path.read_bytes() == original
    with pytest.raises(Problem, match='differs'):
        provision(tmp_path, 'siglip', 'b' * 40)
    weights.write_bytes(b'corrupt')
    with pytest.raises(Problem, match='hash mismatch'):
        provision(tmp_path, 'siglip', revision)
    assert path.read_bytes() == original


def test_incomplete_installation_is_not_overwritten(tmp_path):
    target = tmp_path / 'models/siglip'
    target.mkdir(parents=True)
    with pytest.raises(Problem, match='not provisioned'):
        provision(tmp_path, 'siglip', 'a' * 40)
    assert list(target.iterdir()) == []
