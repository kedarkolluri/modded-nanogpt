"""Build the standalone SM121a CUDA library, without changing the Python env."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent.parent
source=root/'spark/cuda/producer_consumer_mxfp8.cu'
output=source.with_suffix('.so')
results=root/'results/producer-consumer'
results.mkdir(parents=True,exist_ok=True)
command=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-gencode','arch=compute_121a,code=sm_121a',
         '-Xcompiler=-fPIC','--shared','--ptxas-options=-v',str(source),'-o',str(output)]
# --metadata-only records a build already performed with this exact command.
if '--metadata-only' not in sys.argv:
    with (results/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
metadata={'command':command,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'library_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
          'nvcc_version':subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True)}
(results/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata))
