"""Attribute Nsight kernels across autograd threads through CPU launch times.

Main-thread NVTX backward ranges do not contain autograd worker-thread IDs,
so Nsight's default NVTX GPU projection misses these launches. Training phases
are serial here; associate each correlated runtime launch with its enclosing
phase time window across threads. Report unmatched kernels explicitly.
"""
import argparse
import collections
import json
import sqlite3

p = argparse.ArgumentParser()
p.add_argument('database')
args = p.parse_args()
c = sqlite3.connect(args.database)
phases = list(c.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'phase/%' AND end IS NOT NULL ORDER BY start"))
launch = dict(c.execute('SELECT correlationId,start FROM CUPTI_ACTIVITY_KIND_RUNTIME'))
kernels = list(c.execute('SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'))
counts = collections.Counter()
times = collections.Counter()
unmatched = 0
for start, end, correlation in kernels:
    cpu = launch.get(correlation)
    phase = next((name for a,b,name in phases if cpu is not None and a <= cpu <= b), None)
    if phase is None:
        unmatched += 1
        continue
    times[phase] += end-start
    counts[phase] += 1
# Union reports device busy time without double counting simultaneous kernels.
intervals = sorted((a,b) for a,b,_ in kernels)
busy = 0
left, right = intervals[0]
for a,b in intervals[1:]:
    if a > right:
        busy += right-left
        left,right = a,b
    else:
        right = max(right,b)
busy += right-left
steps = sum(name == 'phase/forward' for _,_,name in phases)
span = max(b for _,b in intervals)-intervals[0][0]
result = dict(captured_steps=steps, kernel_count=len(kernels), unmatched_kernels=unmatched,
    phase_kernel_ms_per_step={k:v/steps/1e6 for k,v in times.items()},
    phase_kernel_count_per_step={k:v/steps for k,v in counts.items()},
    kernel_sum_ms_per_step=sum(b-a for a,b,_ in kernels)/steps/1e6,
    kernel_busy_union_ms_per_step=busy/steps/1e6,
    first_to_last_kernel_span_ms_per_step=span/steps/1e6,
    no_kernel_running_ms_per_step=(span-busy)/steps/1e6,
    note='No-kernel intervals are not necessarily CPU launch stalls; includes synchronization, memory operations, and step boundaries. Profiling adds overhead.')
print(json.dumps(result, indent=2))
