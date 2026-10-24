#!/usr/bin/env python3
"""Local-router adapter swap with independent behavioral fingerprint and rollback."""
import argparse,fcntl,hashlib,json,os,shutil,tempfile,time,urllib.parse,urllib.request
from pathlib import Path

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def request(base,path,body=None):
 data=None if body is None else json.dumps(body).encode()
 req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
 # Never forward local operations through HTTP(S)_PROXY or follow a redirect.
 class NoRedirect(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*a,**k): return None
 with urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect()).open(req,timeout=120) as r: return json.load(r)
def fingerprint(response):
 fields={k:response[k] for k in ('content','tokens','completion_probabilities') if k in response}
 if not fields: raise RuntimeError('Probe returned no comparable output')
 return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic_copy(src,dst):
 fd,name=tempfile.mkstemp(prefix='.'+dst.name+'.',dir=dst.parent)
 try:
  with os.fdopen(fd,'wb') as out,src.open('rb') as inp:
   shutil.copyfileobj(inp,out); out.flush(); os.fsync(out.fileno())
  os.chmod(name,dst.stat().st_mode & 0o777); os.replace(name,dst)
  d=os.open(dst.parent,os.O_DIRECTORY)
  try: os.fsync(d)
  finally: os.close(d)
 finally:
  if os.path.exists(name): os.unlink(name)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for a in ('url','model','adapter','replacement','probe','expected-probe-sha','expected-file-sha','receipt'): p.add_argument('--'+a,required=True)
 a=p.parse_args(); u=urllib.parse.urlsplit(a.url)
 if u.scheme!='http' or u.hostname not in ('127.0.0.1','::1') or not u.port or u.username or u.password or u.path or u.query or u.fragment: p.error('Only explicit loopback HTTP router URLs are allowed')
 target=Path(a.adapter).absolute(); source=Path(a.replacement).resolve()
 if target.is_symlink() or not target.is_file() or not source.is_file() or source==target: p.error('Need two distinct regular files; target cannot be a symlink')
 if source.read_bytes()[:4]!=b'GGUF' or digest(source)!=a.expected_file_sha: p.error('Replacement magic/hash mismatch')
 probe=json.loads(Path(a.probe).read_text()); probe.update(model=a.model,temperature=0,seed=42,cache_prompt=False)
 if probe.get('lora')!=[{'id':0,'scale':1.0}]: p.error('Probe must explicitly select adapter id 0 at scale 1')
 query='?'+urllib.parse.urlencode({'model':a.model})
 def check_adapter():
  adapters=request(a.url,'/lora-adapters'+query)
  if len(adapters)!=1 or adapters[0]['id']!=0 or Path(adapters[0]['path']).resolve()!=target.resolve(): raise RuntimeError('Router adapter registry does not match single target adapter')
  return adapters
 def run_probe(): return fingerprint(request(a.url,'/completion',probe))
 def reload():
  request(a.url,'/models/unload',{'model':a.model})
  request(a.url,'/models/load',{'model':a.model})
  end=time.monotonic()+120
  while True:
   try: return check_adapter()
   except Exception:
    if time.monotonic()>=end: raise
    time.sleep(.25)
 with open(str(target)+'.swap.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  before=check_adapter(); old_hash=digest(target); old_probe=run_probe()
  if old_hash==a.expected_file_sha: raise RuntimeError('Replacement is identical')
  if old_probe==a.expected_probe_sha: raise RuntimeError('Probe does not distinguish old and expected replacement')
  fd,backup_name=tempfile.mkstemp(prefix='.'+target.name+'.backup.',dir=target.parent); os.close(fd); backup=Path(backup_name)
  shutil.copyfile(target,backup)
  with backup.open('rb') as saved: os.fsync(saved.fileno())
  started=time.monotonic(); changed=False
  receipt={'before_file_sha':old_hash,'before_probe_sha':old_probe,'before_registry':before,'model':a.model,'url':a.url}
  try:
   atomic_copy(source,target); changed=True
   if digest(target)!=a.expected_file_sha: raise RuntimeError('Installed file hash mismatch')
   after=reload()
   request(a.url,'/lora-adapters'+query,[{'id':x['id'],'scale':x['scale']} for x in before])
   after=check_adapter(); actual=run_probe()
   if actual!=a.expected_probe_sha: raise RuntimeError('Reloaded adapter behavioral fingerprint mismatch')
   receipt.update(status='passed',after_file_sha=digest(target),after_probe_sha=actual,after_registry=after)
  except Exception as e:
   receipt.update(status='failed',error=str(e))
   if changed:
    try:
     atomic_copy(backup,target)
     # Unload can fail if a bad candidate prevented startup. Load still needed.
     try: request(a.url,'/models/unload',{'model':a.model})
     except Exception: pass
     request(a.url,'/models/load',{'model':a.model}); check_adapter()
     request(a.url,'/lora-adapters'+query,[{'id':x['id'],'scale':x['scale']} for x in before])
     receipt['rollback_verified']=digest(target)==old_hash and run_probe()==old_probe
    except Exception as rollback_error: receipt['rollback_error']=str(rollback_error)
   raise
  finally:
   receipt['elapsed_seconds']=round(time.monotonic()-started,3)
   Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
   if receipt.get('status')=='passed' or receipt.get('rollback_verified'): backup.unlink()
   else: print('Recovery backup retained:',backup)
  print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()
