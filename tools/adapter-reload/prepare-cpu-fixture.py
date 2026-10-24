#!/usr/bin/env python3
import hashlib,json,os,urllib.request
from pathlib import Path
out=Path('/run/media/me/storage/builds/jasonelle-bake/cpu-test'); out.mkdir(exist_ok=True)
repo='ggml-org/tiny-llamas'
revision='99dd1a73db5a37100bd4ae633f4cfce6560e1567'
target=out/'stories260K.gguf'
if not target.exists(): urllib.request.urlretrieve(f'https://huggingface.co/{repo}/resolve/{revision}/stories260K.gguf',target)
assert hashlib.sha256(target.read_bytes()).hexdigest()=='047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b'
(out/'source.json').write_text(json.dumps({'repo':repo,'revision':revision,'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2)+'\n')
print(target, target.stat().st_size)
# Controlled rank-1 LoRAs: same base, distinct perturbations; not persona adapters.
import sys
sys.path.insert(0,'/home/me/worktrees/jasonelle-bake-llama/gguf-py')
import gguf,numpy as np
for name,seed in [('adapter-a',17),('adapter-b',29)]:
 rng=np.random.default_rng(seed)
 w=gguf.GGUFWriter(str(out/(name+'.gguf')),'llama')
 w.add_string('general.type','adapter'); w.add_string('adapter.type','lora'); w.add_float32('adapter.lora.alpha',1.0)
 w.add_tensor('output.weight.lora_a',rng.normal(0,0.5,(1,64)).astype(np.float32))
 w.add_tensor('output.weight.lora_b',rng.normal(0,0.5,(512,1)).astype(np.float32))
 w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print('Created two rank-1 output-projection LoRAs, 2304 bytes of tensor data each.')
