"""Retained experiment-data integrity, qualifications and matched inputs."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_data_hashes():
    manifest = json.loads((ROOT/'data/evidence-sha256.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name


def test_all_positive_and_negative_attempts_and_L5_are_retained():
    from hf_fas.experiments import evidence as m
    data = json.loads((ROOT/'data/vgf-current-results.json').read_text())
    assert len(data['records']) == 114
    assert sum(r['qualified'] for r in data['records']) == 108
    for r in data['records']:
        m.validate_row(r)
    assert len(data['required_L5']['records']) == 24
    l5 = m.required_l5_panels(data)
    assert set(l5) == set(m.SYSTEMS)
    for system in m.SYSTEMS:
        assert set(l5[system]) == set(m.METHODS)
    for system in m.SYSTEMS:
        for level in (3, 4):
            group = {r['method']: r for r in data['records'] if r['case'] == f'{system}-L{level}'}
            m.matched([group[method] for method in m.METHODS])
    assert len(data['probes']) == 6 and len(data['L5']) == 8
    published = json.loads((ROOT/'data/vgf-cost-ratios-current.json').read_text())
    assert published['snapshot_sha256'] == hashlib.sha256((ROOT/'data/vgf-current-results.json').read_bytes()).hexdigest()
    ratios = published['rows']
    assert len(ratios) == 24
    assert {(r['system'], r['level']) for r in ratios} == {
        (system, level) for system in m.SYSTEMS for level in (3, 4, 5)}
    assert published['excluded_l5'] == []
    for case, host in (('Li2-L5', 'bigMem0'), ('BH-L5', 'bigMem4')):
        evidence = published['added_panel_evidence'][case]
        assert set(evidence) == set(m.METHODS)
        assert {r['host'] for r in evidence.values()} == {host}
        assert all(r['qualified'] and r['precision']['qualified'] for r in evidence.values())
        for key in ('initial_sha256', 'reference_sha256'):
            assert len({r[key] for r in evidence.values()}) == 1
        row = next(r for r in ratios if r['case'] == case)
        for metric in ('numerical_seconds', 'fft_work'):
            for method in ('SCF', 'Newton-horizontal'):
                assert row['ratios'][metric][method] == pytest.approx(
                    evidence[method][metric]/evidence['FAS-H'][metric], rel=1e-14)
