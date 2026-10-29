"""Read-only audit of completed TRAIN/VAL experiments; never reads TEST."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

KIT = Path('/workspace/HELMET2020_training_kit_v1')
OUT = Path('/workspace/runs/gq5_extend24_20261028')
PARENT = Path('/workspace/runs/gq5_long12_20261027')

def read(p):
    return json.loads(p.read_text())

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    result = read(OUT / 'comparison_result.json')
    checks, metrics = [], {}
    for arm in result['arms']:
        name = arm['arm']
        metrics[name] = {}
        for rec in arm['evaluations']:
            epoch = rec['epoch']
            pairs = [
                (f'{name}/epoch{epoch:02d}.pt', 'checkpoint_sha256'),
                (f'{name}/val_epoch{epoch:02d}/metrics.json', 'metrics_sha256'),
                (f'{name}/val_epoch{epoch:02d}/predictions.jsonl', 'predictions_sha256'),
            ]
            for rel, key in pairs:
                actual = digest(OUT / rel)
                checks.append({'path': rel, 'expected_sha256': rec[key],
                               'actual_sha256': actual, 'ok': actual == rec[key]})
            m = read(OUT / name / f'val_epoch{epoch:02d}' / 'metrics.json')
            metrics[name][str(epoch)] = m
            ok = m['macro_mAP'] == rec['macro_AP50'] and m['weighted_mAP'] == rec['weighted_AP50']
            checks.append({'path': f'{name}/val_epoch{epoch:02d}/metrics.json',
                           'metric_values_match': ok, 'ok': ok})
        metrics[name]['12'] = read(PARENT / name / 'val_epoch12/metrics.json')
    labels = read(KIT / 'manifests/classes.json')
    counts = {}
    for split in ('train', 'val'):
        class_counts = collections.Counter()
        clips, sites, frames = set(), set(), 0
        with (KIT / 'manifests' / f'{split}.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                frames += 1
                clips.add(row['clip_id'])
                sites.add(str(row.get('site_id', row.get('site', 'unknown'))))
                class_counts.update(o['class_id'] for o in row['objects'])
        role_states = [[0, 0, 0] for _ in range(5)]
        for cls in labels['classes']:
            for role, state in enumerate(cls['states']):
                role_states[role][state] += class_counts[cls['id']]
        counts[split] = {'frames': frames, 'crops': sum(class_counts.values()),
                         'clips': len(clips), 'sites': sorted(sites),
                         'class_counts': dict(class_counts), 'role_states': role_states}
    files = ['helmetlab/kit_models.py', 'helmetlab/role_queries.py',
             'helmetlab/role_query_grouping.py', 'helmetlab/positional.py',
             'helmetlab/labels.py', 'helmetlab/metrics.py']
    evidence = {
        'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'remote_root': str(OUT), 'status': read(OUT / 'status.json'),
        'progress': read(OUT / 'progress.json'), 'selection': read(OUT / 'selection.json'),
        'comparison': result, 'parent_comparison': read(PARENT / 'comparison_result.json'),
        'checks': checks, 'all_checks_passed': all(c['ok'] for c in checks),
        'study': read(OUT / 'study.json'), 'metrics': metrics,
        'histories': {a: read(OUT / a / 'history.json') for a in ('G', 'Q5')},
        'labels': labels, 'counts': counts,
        'sources': {name: digest(KIT / name) for name in files}, 'test_accessed': False,
    }
    print(json.dumps(evidence, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
