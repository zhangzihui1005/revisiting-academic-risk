"""Restore and check the frozen November 2026 reviewer-revision evidence.

No training or bootstrap is performed by this launcher. Training entry points
are explicit Python files in the restored code directory.
"""
import argparse,hashlib,json,tarfile,subprocess,sys,shutil
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['restore','verify'])
 p.add_argument('--workdir',type=Path,default=Path('_reviewer'))
 a=p.parse_args();out=a.workdir.resolve()
 if out==HERE or HERE.is_relative_to(out):raise SystemExit('Choose a separate working directory.')
 spec=json.loads((HERE/'reviewer_bundle.json').read_text())
 if not out.exists() or not any(out.iterdir()):
  out.mkdir(parents=True,exist_ok=True)
  archive=out/'_transport.tar.xz'
  with archive.open('wb') as f:
   for part in spec['parts']:
    q=HERE/part['name']
    if q.stat().st_size!=part['bytes'] or sha(q)!=part['sha256']:raise SystemExit('Transport hash failed: '+part['name'])
    with q.open('rb') as src:shutil.copyfileobj(src,f)
  if sha(archive)!=spec['archive_sha256']:raise SystemExit('Combined transport hash failed.')
  with tarfile.open(archive,'r:xz') as t:
   members=t.getmembers()
   for m in members:
    target=(out/m.name).resolve()
    if not m.isfile() or not target.is_relative_to(out) or target==archive:raise SystemExit('Unsafe archive member.')
   for m in members:
    target=out/m.name;target.parent.mkdir(parents=True,exist_ok=True)
    with t.extractfile(m) as src,target.open('wb') as f:shutil.copyfileobj(src,f)
  archive.unlink()
 idx=json.loads((out/'SOURCE_INDEX.json').read_text())
 for name,v in idx.items():
  q=out/name
  if not q.is_file() or q.stat().st_size!=v['bytes'] or sha(q)!=v['sha256']:raise SystemExit('Frozen input check failed: '+name)
 print('REVIEWER PAYLOAD VERIFIED:',len(idx),'files; no scientific files modified.',flush=True)
 if a.action=='verify':subprocess.run([sys.executable,'-B',str(out/'code/verify_frozen.py')],check=True)
 print('Source, full-precision tables, membership IDs, predictions and saved draws:',out,flush=True)
if __name__=='__main__':main()
