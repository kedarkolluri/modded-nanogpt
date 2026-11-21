"""Attribute actual CUDA kernels using launch correlations in a Chrome trace.

GPU user annotations are ranges, not kernels; exclude them to avoid double
counting. CPU range nesting is used at launch time, not GPU execution time.
"""
from collections import defaultdict
import gzip
import json
from pathlib import Path


def analyze_trace(path, steps):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        events = json.load(f)['traceEvents']
    cpu = [e for e in events if e.get('cat') in ('cpu_op', 'user_annotation') and e.get('ph') == 'X']
    ranges = [e for e in cpu if e['name'].startswith(('phase/', 'optimizer/', 'module/', 'native/'))]
    by_external = {e.get('args', {}).get('External id'): e for e in cpu if 'External id' in e.get('args', {})}
    launches = {e['args']['correlation']: e for e in events
                if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in e.get('args', {})}
    totals = defaultdict(lambda: {'count': 0, 'cuda_us': 0.})
    range_totals = defaultdict(lambda: {'kernel_count': 0, 'cuda_us': 0., 'cpu_range_us': 0.})
    for r in ranges:
        range_totals[r['name']]['cpu_range_us'] += r['dur']
    shape_totals = defaultdict(lambda: {'count': 0, 'cuda_us': 0.})
    phase_ranges = defaultdict(lambda: {'kernel_count': 0, 'cuda_us': 0.})
    unmatched = 0
    for event in events:
        if event.get('cat') != 'kernel' or event.get('ph') != 'X':
            continue
        name, duration = event['name'], event['dur']
        totals[name]['count'] += 1
        totals[name]['cuda_us'] += duration
        args = event.get('args', {})
        launch = launches.get(args.get('correlation'))
        op = by_external.get(args.get('External id'))
        if launch is None:
            launch = op
        if launch is None:
            unmatched += 1
            continue
        ts = launch['ts']
        containing = [r for r in ranges if r['ts'] <= ts <= r['ts'] + r['dur']]
        for r in containing:
            range_totals[r['name']]['cuda_us'] += duration
            range_totals[r['name']]['kernel_count'] += 1
        phase = next((r['name'] for r in containing if r['name'].startswith('phase/')), 'unassigned')
        for r in containing:
            key = (phase, r['name'])
            phase_ranges[key]['kernel_count'] += 1
            phase_ranges[key]['cuda_us'] += duration
        if op:
            shape = json.dumps(op.get('args', {}).get('Input Dims', []))
            key = (phase, op['name'], shape)
            shape_totals[key]['count'] += 1
            shape_totals[key]['cuda_us'] += duration
    kernels = [{'name': k, **v, 'cuda_ms_per_step': v['cuda_us']/1000/steps} for k,v in totals.items()]
    attributed = [{'name': k, **v, 'cuda_ms_per_step': v['cuda_us']/1000/steps,
                   'cpu_range_ms_per_step': v['cpu_range_us']/1000/steps} for k,v in range_totals.items()]
    shapes = [{'phase': k[0], 'operator': k[1], 'input_shapes': json.loads(k[2]), **v,
               'cuda_ms_per_step': v['cuda_us']/1000/steps} for k,v in shape_totals.items()]
    return {'kernels': sorted(kernels, key=lambda r:r['cuda_us'], reverse=True),
            'attributed_ranges': sorted(attributed, key=lambda r:r['cuda_us'], reverse=True),
            'ranges_by_phase': [{'phase': key[0], 'name': key[1], **value,
                                  'cuda_ms_per_step': value['cuda_us']/1000/steps}
                                 for key, value in phase_ranges.items()],
            'operator_shapes': sorted(shapes, key=lambda r:r['cuda_us'], reverse=True),
            'unmatched_kernels': unmatched,
            'kernel_ms_per_step': sum(v['cuda_us'] for v in totals.values())/1000/steps,
            'note': 'Actual kernel categories only. Nested ranges overlap; do not add parent and child ranges.'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('trace', type=Path)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(analyze_trace(args.trace, args.steps), indent=2)+'\n')
