#!/usr/bin/env python3
"""Real CPU router integration test; all child processes owned and reaped here."""
import hashlib,json,os,shutil,socket,subprocess,time
from pathlib import Path
from swap_adapter import request,fingerprint,digest
ROOT=Path(__file__).resolve().parent
OUT=Path('/run/media/me/storage/builds/jasonelle-bake/cpu-test')
BIN=OUT.parent/'llama-cpu/bin/llama-server'
def port():
 with socket.socket() as s: s.bind(('127.0.0.1',0)); return s.getsockname()[1]
def ready(url,process):
 for _ in range(240):
  if process.poll() is not None: raise RuntimeError('Server exited; inspect CPU test logs')
  try: request(url,'/health'); return
  except Exception: time.sleep(.25)
 raise TimeoutError(url)
def main():
 os.umask(0o077); processes=[]; logs=[]; router_port=port(); ref_port=port()
 env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_'))}; env['CUDA_VISIBLE_DEVICES']=''
 target=OUT/'active.gguf'; shutil.copyfile(OUT/'adapter-a.gguf',target)
 probe={'prompt':'Once upon a time','n_predict':12,'n_probs':5,'temperature':0,'seed':42,'cache_prompt':False,'lora':[{'id':0,'scale':1.0}]}
 (OUT/'probe.json').write_text(json.dumps(probe,indent=2)+'\n')
 (OUT/'models.ini').write_text(f'version = 1\n[tiny]\nmodel = {OUT}/stories260K.gguf\nlora = {target}\nlora-init-without-apply = true\nn-gpu-layers = 0\nctx-size = 256\nthreads = 2\nparallel = 1\n')
 def start(name,args):
  log=open(OUT/(name+'.log'),'w'); logs.append(log)
  p=subprocess.Popen([str(BIN),*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); processes.append(p); return p
 try:
  reference=start('reference',['--host','127.0.0.1','--port',str(ref_port),'-m',str(OUT/'stories260K.gguf'),'--lora',str(OUT/'adapter-b.gguf'),'--lora-init-without-apply','-ngl','0','-c','256','-t','2','-np','1'])
  refurl=f'http://127.0.0.1:{ref_port}'; ready(refurl,reference)
  expected=request(refurl,'/completion',probe); expected_sha=fingerprint(expected)
  (OUT/'expected-b.json').write_text(json.dumps(expected,indent=2)+'\n')
  reference.terminate(); reference.wait(timeout=20)
  router=start('router',['--host','127.0.0.1','--port',str(router_port),'--models-preset',str(OUT/'models.ini'),'-ngl','0','-t','2','-c','256','-np','1'])
  url=f'http://127.0.0.1:{router_port}'; ready(url,router)
  request(url,'/models/load',{'model':'tiny'})
  before=request(url,'/completion',dict(probe,model='tiny'))
  (OUT/'before-a.json').write_text(json.dumps(before,indent=2)+'\n')
  assert fingerprint(before)!=expected_sha
  common=[str(ROOT/'swap-adapter.sh'),'--url',url,'--model','tiny','--adapter',str(target),'--replacement',str(OUT/'adapter-b.gguf'),'--probe',str(OUT/'probe.json'),'--expected-file-sha',digest(OUT/'adapter-b.gguf')]
  # Incorrect expected behavior must roll back both bytes and running behavior.
  failed=subprocess.run(common+['--expected-probe-sha','0'*64,'--receipt',str(OUT/'rollback-receipt.json')],capture_output=True,text=True)
  assert failed.returncode!=0
  assert json.loads((OUT/'rollback-receipt.json').read_text())['rollback_verified']
  success=subprocess.run(common+['--expected-probe-sha',expected_sha,'--receipt',str(OUT/'swap-receipt.json')],capture_output=True,text=True)
  (OUT/'swap-command.log').write_text(success.stdout+success.stderr)
  assert success.returncode==0, success.stderr
  after=request(url,'/completion',dict(probe,model='tiny')); assert fingerprint(after)==expected_sha
  disabled=request(url,'/completion',dict(probe,model='tiny',lora=[{'id':0,'scale':0.0}]))
  assert fingerprint(disabled)!=expected_sha
  assert router.poll() is None
  receipt=json.loads((OUT/'swap-receipt.json').read_text())
  summary={'result':'PASS','router_pid':router.pid,'router_survived':True,'rollback_test':'PASS','adapter_disabled_differs':True,'before_content':before['content'],'after_content':after['content'],'disabled_content':disabled['content'],'swap_seconds':receipt['elapsed_seconds'],'expected_probe_sha':expected_sha,'base_sha':digest(OUT/'stories260K.gguf')}
  (OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
 finally:
  for p in reversed(processes):
   if p.poll() is None:
    p.terminate()
    try: p.wait(timeout=20)
    except subprocess.TimeoutExpired:
     # This process group contains only this test's router and its children.
     import signal
     os.killpg(p.pid,signal.SIGKILL); p.wait(timeout=10)
  for f in logs: f.close()
if __name__=='__main__': main()
