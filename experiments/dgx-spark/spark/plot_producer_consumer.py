"""Plot measured single-CTA clock64 work windows with standard plotting tools."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root=Path('results/producer-consumer')
records=json.loads((root/'concurrency-traces.json').read_text())
fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True,layout='constrained')
max_x=0
for axis,serial in zip(axes,(True,False)):
    record=next(r for r in records if r['shape']==[16,512,768] and r['config']==2 and r['serial']==serial)
    rows=record['tile_events_clock64'][:8]
    base=min(row[0] for row in rows)
    for tile,(p0,p1,c0,c1,first,last,quant) in enumerate(rows):
        axis.broken_barh([((p0-base)/1000,(p1-p0)/1000)],(.73,.45),facecolors='#277da8')
        axis.broken_barh([((c0-base)/1000,(c1-c0)/1000)],(-.02,.45),facecolors='#e9a23b')
        axis.plot([(first-base)/1000,(last-base)/1000],[.20,.20],color='#6b3400',linewidth=3)
        axis.scatter([(quant-base)/1000],[.95],marker='D',s=14,color='#142c45',zorder=3)
        axis.text(((p0+p1)/2-base)/1000,.95,str(tile),ha='center',va='center',color='white',fontsize=8)
        axis.text(((c0+c1)/2-base)/1000,.20,str(tile),ha='center',va='center',color='#142c45',fontsize=8)
        max_x=max(max_x,(c1-base)/1000)
    axis.set_yticks([.20,.95],['Consumer warps','Producer warps'])
    axis.set_ylim(-.23,1.45)
    axis.set_title('Serial control: next conversion waits for current matmul' if serial else 'Two shared buffers: next conversion runs during current matmul',loc='left',fontsize=11)
    axis.grid(axis='x',alpha=.2)
    axis.spines[['top','right']].set_visible(False)
axes[-1].set_xlim(0,max_x*1.02)
axes[-1].set_xlabel('Time (thousand SM clock cycles; instrumented single CTA, first eight tiles)')
fig.suptitle('Explicit producer/consumer pipeline on GB10 — BN512, BK32, K768',fontsize=13)
fig.legend(handles=[Patch(color='#277da8',label='Activation quantization region'),Patch(color='#e9a23b',label='Consumer matmul region (includes weight loads)'),Patch(color='#6b3400',label='First-to-last MMA issue window')],loc='outside lower center',ncols=3,fontsize=9)
fig.savefig(root/'concurrency-timeline.png',dpi=180)
