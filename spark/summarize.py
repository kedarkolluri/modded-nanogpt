"""Summarize completed matched diagnostic runs from JSONL logs."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('logs', type=Path, nargs='+')
args = p.parse_args()
print('| Run | Final validation loss | Steady tokens/s | Peak allocated GiB |')
print('|---|---:|---:|---:|')
for path in args.logs:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    final = rows[-1]
    if not final.get('finished'):
        print(f'| {path.stem} | incomplete | — | — |')
        continue
    loss = [r['val_loss'] for r in rows if 'val_loss' in r][-1]
    rate = final['steady_tokens_per_second']
    rate_text = f'{rate:,.0f}' if rate else '—'
    print(f'| {path.stem} | {loss:.4f} | {rate_text} | {final["peak_allocated_bytes"]/2**30:.2f} |')
