"""Save disassembly excerpts and lowered loop signatures, not overlap estimates."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/buffered-mxfp8')
summary={}
for name,buffered in [('serial',False),('buffered',True)]:
    stem=f'8192-3072-768-16-64-128-4-1-{buffered}-1'
    binary=root/f'{stem}.cubin'
    sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True)
    (root/f'{name}.sass').write_text(sass)
    instructions=[line.strip() for line in sass.splitlines() if re.match(r'\s*/\*[0-9a-f]+\*/',line)]
    mma=[i for i,line in enumerate(instructions) if 'QMMA.' in line]
    ir=(root/f'{stem}.ttgir').read_text()
    summary[name]={
        'cubin_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'static_instructions':len(instructions),
        'static_qmma':len(mma),
        'fp8_pack_instructions':sum('F2FP.SATFINITE.E4M3' in line for line in instructions),
        'lowered_loop_signatures':[line.strip() for line in ir.splitlines() if 'scf.for' in line],
        'instructions_between_first_8_qmma':instructions[mma[0]:mma[7]+1],
    }
summary['limitation']='Static scheduling evidence does not measure simultaneous execution or percent overlap.'
(root/'schedule-evidence.json').write_text(json.dumps(summary,indent=2)+'\n')
