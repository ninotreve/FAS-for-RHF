"""Qualification and identity checks for retained experiment data."""
import math

SYSTEMS = ('LiH', 'Li2', 'BeHplus', 'BeH2', 'BH', 'H2O', 'N2', 'CH4')
METHODS = ('FAS-H', 'Newton-horizontal', 'SCF')

def validate_row(row):
    if not row['qualified']:
        return
    e, p = row['endpoint'], row['precision']
    if not row['complete'] or not row['converged'] or row['status'] != 'qualified':
        raise ValueError('Incomplete result claimed as qualified')
    for key, bound in (('residual', 1e-8), ('energy_difference', 1e-7),
                       ('projector_distance', 1e-6), ('relative_density', 1e-6), ('electron_count_error', 1e-9)):
        if not math.isfinite(e[key]) or e[key] >= bound:
            raise ValueError('Invalid endpoint ' + key)
    if not e['qualified'] or not p['qualified'] or p['raw_residual']['total'] >= 1e-8 or p['rebuilt_residual']['total'] >= 1e-8:
        raise ValueError('Independent residual check failed')
    if p['energy_difference'] >= 1e-9 or p['vgf_precompute'] != 'fft':
        raise ValueError('Independent kernel check failed')
    if row['iterations'] != len(row['history']) - 1:
        raise ValueError('History length differs')

def matched(rows):
    for key in ('initial_sha256', 'reference_sha256', 'host', 'affinity', 'job_id', 'manifest_sha256'):
        if any(r[key] is None or r[key] != rows[0][key] for r in rows):
            raise ValueError('Unmatched panel: ' + key)
    if any(not r['qualified'] for r in rows):
        raise ValueError('Unqualified solve cannot enter a successful cost ratio')
    return rows

def required_l5_panels(snapshot):
    panels = {system:{} for system in SYSTEMS}
    for row in snapshot.get('required_L5', {}).get('records', []):
        system, method = row['system'], row['method']
        if system not in panels or method not in METHODS or row['level'] != 5 or row['case'] != f'{system}-L5':
            raise ValueError('Invalid required-L5 case')
        if row['source_label'] != ('C-response' if method == 'FAS-H' else method):
            raise ValueError('Superseded L5 algorithm cannot be relabelled')
        if not row['qualified'] or method in panels[system]:
            raise ValueError('Unqualified or duplicate L5 result')
        validate_row(row)
        for other in panels[system].values():
            for key in ('initial_sha256', 'reference_sha256', 'manifest_sha256'):
                if row[key] != other[key]:
                    raise ValueError('L5 common input differs: ' + key)
        panels[system][method] = row
    return panels
