"""Exhaustive phase/kernel accounting; fused kernels stay grouped together."""
import collections
import csv
import json
from pathlib import Path
import sqlite3
import sys

root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/operation-budget/nsys')


def classify(name,phase):
    if phase=='phase/optimizer':
        return 'Optimizer (Muon + AdamW)'
    if name=='_activation_kernel':
        return 'Activation quantization + first-projection GEMM'
    if 'block_scaled' in name:
        return 'Native MXFP8 GEMMs'
    if name=='_fp8_transpose':
        return 'MXFP8 transpose packing'
    if name=='_fp8':
        return 'MXFP8 row packing (includes fused activation packing)'
    if name=='_relu_square_grad':
        return 'MLP activation derivative'
    if name=='_relu_square':
        return 'MLP activation forward'
    if 'flash_' in name:
        return 'Flash attention'
    if 'rms_norm' in name:
        return 'Fused normalization / residual / rotary / related ops'
    if 'softmax' in name or 'nll_loss' in name:
        return 'Cross entropy / softmax'
    if 'gemm' in name.lower() or name.startswith('nvjet_'):
        return 'Other GEMMs (BF16 MLP, attention projections, head)'
    return 'Other fused pointwise / casts / copies / embedding / RNG'


output={}
for mode in ('bf16','current','reuse'):
    db=sqlite3.connect(root/f'{mode}.sqlite')
    phases=list(db.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'phase/%' AND end IS NOT NULL ORDER BY start"))
    steps=sum(name=='phase/forward' for a,b,name in phases)
    launch=dict(db.execute('SELECT correlationId,start FROM CUPTI_ACTIVITY_KIND_RUNTIME'))
    rows=list(db.execute('SELECT k.start,k.end,k.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start'))
    assert steps>0, (mode,'no captured steps')
    assert len(rows)==db.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0], (mode,'kernel-name join dropped rows')
    totals=collections.Counter()
    counts=collections.Counter()
    raw=collections.Counter()
    raw_counts=collections.Counter()
    unmatched=0
    for start,end,correlation,name in rows:
        cpu=launch.get(correlation)
        phase=next((label for a,b,label in phases if cpu is not None and a<=cpu<=b),None)
        if phase is None:
            unmatched+=1
            phase='unmatched'
        category=classify(name,phase)
        totals[category]+=(end-start)/steps/1e6
        counts[category]+=1/steps
        raw[(phase,category,name)]+=(end-start)/steps/1e6
        raw_counts[(phase,category,name)]+=1/steps
    assert not unmatched,(mode,unmatched)
    expected=sum(end-start for start,end,_,_ in rows)/steps/1e6
    assert abs(sum(totals.values())-expected)<1e-7
    output[mode]={'captured_steps':steps,'unmatched_kernels':unmatched,'kernel_ms_per_step':dict(totals),'kernel_count_per_step':dict(counts),'total_kernel_ms_per_step':expected}
    with (root/f'{mode}-all-kernels.csv').open('w') as handle:
        writer=csv.writer(handle)
        writer.writerow(['phase','category','kernel','ms_per_step','calls_per_step'])
        for key,value in sorted(raw.items(),key=lambda row:-row[1]):
            writer.writerow([*key,value,raw_counts[key]])
(root/'operation-budget.json').write_text(json.dumps(output,indent=2)+'\n')
categories=list(dict.fromkeys(name for mode in output.values() for name in mode['kernel_ms_per_step']))
with (root/'operation-budget.csv').open('w') as handle:
    writer=csv.writer(handle)
    writer.writerow(['category','bf16_ms','current_mxfp8_ms','reuse_mxfp8_ms'])
    for category in categories:
        writer.writerow([category,*[output[m]['kernel_ms_per_step'].get(category,0) for m in output]])
print(json.dumps(output,indent=2))
