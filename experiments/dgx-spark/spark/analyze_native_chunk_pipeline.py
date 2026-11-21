"""Summarize repeated timings and measured pack/GEMM interval intersections."""
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
root = Path('results/native-chunk-pipeline')
rows = defaultdict(list)
for line in (root/'paired.jsonl').read_text().splitlines():
    x = json.loads(line)
    rows[x['path']].append(x['ms'])
summary = {'median_ms': {k: statistics.median(v) for k,v in rows.items()}}
c = sqlite3.connect(root/'trace.sqlite')
kernels = c.execute('select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id').fetchall()
summary['trace'] = {}
for start,end,name in c.execute('select start,end,text from NVTX_EVENTS where text in ("native","rows2048-serial","rows2048-overlap")'):
    selected = [k for k in kernels if k[0]>=start and k[1]<=end]
    packs = [k for k in selected if k[3]=='_fp8']
    gemms = [k for k in selected if 'cutlass' in k[3]]
    overlap = sum(max(0,min(p[1],g[1])-max(p[0],g[0])) for p in packs for g in gemms if p[2]!=g[2])
    summary['trace'][name] = {'pack_count':len(packs),'gemm_count':len(gemms),'cross_stream_pack_gemm_overlap_ms':overlap/1e6,'pack_gpu_ms':sum(k[1]-k[0] for k in packs)/1e6,'gemm_gpu_ms':sum(k[1]-k[0] for k in gemms)/1e6,'other_gpu_ms':sum(k[1]-k[0] for k in selected if k not in packs and k not in gemms)/1e6}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
