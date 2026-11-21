"""Clock64 concurrency evidence for one CTA; probes are not throughput timings."""
import json
from pathlib import Path
import torch
import native
from producer_consumer_mxfp8 import activation_mm,CONFIGS
native.set_pack_backend('triton-tiled')
torch.manual_seed(42)
root=Path('results/producer-consumer')
root.mkdir(parents=True,exist_ok=True)
records=[]
for k in (768,4096):
    for config in (2,4,5,7,9):
        bn,bk,pw=CONFIGS[config]
        a=torch.randn(16,k,device='cuda',dtype=torch.float32)
        w=torch.randn(bn,k,device='cuda',dtype=torch.float32)*.03
        wq,sw,_=native.fp8_pack(w,False,True)
        for serial in (True,False):
            for _ in range(5):
                activation_mm(a,wq,sw,config=config,serial=serial)
            out,ad,events=activation_mm(a,wq,sw,config=config,serial=serial,trace=True)
            values=events.cpu().tolist()
            assert all(0<p0<p1 and 0<c0<c1 for p0,p1,c0,c1,*_ in values)
            overlap=0
            pairs=[]
            for i,(_,_,c0,c1,*_) in enumerate(values):
                for j,(p0,p1,*_) in enumerate(values):
                    cycles=max(0,min(p1,c1)-max(p0,c0))
                    if cycles:
                        assert j>i,(i,j)
                        overlap+=cycles
                        pairs.append({'consumer_tile':i,'producer_tile':j,'cycles':cycles})
            if serial:
                assert overlap==0,overlap
            else:
                assert overlap>0,(k,config)
            checkpoints=[{'consumer_tile':i,'producer_tile':j,'quant_done_clock':v[6]}
                         for i,row in enumerate(values) for j,v in enumerate(values)
                         if j>i and row[4]<v[6]<row[5]]
            if serial:
                assert not checkpoints
            else:
                assert checkpoints,(k,config)
            record={'shape':[16,bn,k],'config':config,'serial':serial,'producer_warps':pw,'consumer_warps':4,
                    'producer_region_cycles':sum(p1-p0 for p0,p1,c0,c1,*_ in values),
                    'consumer_region_cycles':sum(c1-c0 for p0,p1,c0,c1,*_ in values),
                    'overlap_cycles':overlap,'quant_checkpoints_inside_mma_window':checkpoints,'tile_events_clock64':values,'overlapping_tiles':pairs,
                    'limitation':'Instrumented single CTA. Regions include role-local loads, computation and synchronization; this is not tensor/scalar hardware-unit simultaneous-active percentage.'}
            records.append(record)
            print(json.dumps({key:value for key,value in record.items() if key not in ('tile_events_clock64','overlapping_tiles')}),flush=True)
(root/'concurrency-traces.json').write_text(json.dumps(records,indent=2)+'\n')
