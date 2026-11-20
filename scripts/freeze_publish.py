"""Publish audited LIFE-002 refs and assets without changing existing refs."""
import hashlib,io,json,re,subprocess,urllib.request,urllib.parse
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED,ZIP_DEFLATED
ROOT=Path('C:/AGI/PQT_QCAT_AUDIT'); REPO=ROOT/'PQT_QCat_LIFE_002_FREEZE_001'; META=REPO/'archive/life-002/2026-11-20'; ASSETS=ROOT/'LIFE_002_FREEZE_001_ASSETS'; ASSETS.mkdir(exist_ok=True)
TAG='life-002-freeze-2026-11-20'; BRANCH='archive/life-002-freeze-2026-11-20'
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def api(path,data=None,method=None,raw=None):
 cred=subprocess.run(['git','credential','fill'],cwd=REPO,input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,check=True)
 token=dict(l.split('=',1) for l in cred.stdout.splitlines() if '=' in l)['password']
 url=path if path.startswith('https:') else 'https://api.github.com/repos/tocataskyrimova-lang/PQT-QCAT'+path
 body=raw if raw is not None else json.dumps(data).encode() if data is not None else None
 req=urllib.request.Request(url,data=body,method=method,headers={'Authorization':'Bearer '+token,'User-Agent':'LIFE002-FREEZE','Content-Type':'application/octet-stream' if raw is not None else 'application/json'})
 with urllib.request.urlopen(req,timeout=1200) as r:return json.load(r)
def main():
 scan=json.loads((META/'SECURITY_SCAN.json').read_text());assert not scan['findings'], 'Publication blocked by security findings'
 # Select original ZIPs not already embedded byte-for-byte in the F0B delivery.
 rows=json.loads((META/'ARCHIVE_CONTAINMENT.json').read_text())
 selected=[r for r in rows if not r['nested_route'] and r['name']!='PQT_QCAT_DELIVERY_20261111.zip']
 artifacts=ASSETS/'ARTIFACTS.zip'
 with ZipFile(artifacts,'x',compression=ZIP_STORED) as z:
  for row in selected:
   assert sha(ROOT/row['name'])==row['sha256'];z.write(ROOT/row['name'],row['name'])
 extra=ASSETS/'SUPPLEMENTAL.zip'
 extra_manifest={}
 roots=[ROOT/'PQT_QCat_LIFE_002'/'results', ROOT/'CODEX_MEMORY_LOCATION_001_RUN_A', ROOT/'CODEX_MEMORY_LOCATION_001_RUN_B',ROOT/'results']
 with ZipFile(extra,'x',compression=ZIP_DEFLATED) as z:
  for folder in roots:
   if not folder.exists():continue
   for path in sorted(folder.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts:continue
    data=path.read_bytes()
    for pattern in (rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{50,}',rb'AKIA[0-9A-Z]{16}'):
     assert not re.search(pattern,data),'Supplemental secret signature requires review'
    name=path.relative_to(ROOT).as_posix();z.writestr(name,data)
    extra_manifest[name]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data)}
 (META/'SUPPLEMENTAL_MANIFEST.json').write_text(json.dumps(extra_manifest,indent=2))
 historical=ASSETS/'LIFE002_HISTORY.bundle' 
 refs=json.loads((META/'SOURCE_REFS.json').read_text())
 subprocess.run(['git','bundle','create',str(historical),*refs.keys()],cwd=REPO,check=True)
 subprocess.run(['git','bundle','verify',str(historical)],cwd=REPO,check=True,capture_output=True)
 hashes={p.name:{'sha256':sha(p),'size':p.stat().st_size} for p in ASSETS.iterdir() if p.is_file()}
 (META/'RELEASE_ASSETS.json').write_text(json.dumps(hashes,indent=2)+'\n')
 sums=''.join(f"{v['sha256']}  {n}\n" for n,v in hashes.items())
 (ASSETS/'SHA256SUMS.txt').write_text(sums);(META/'SHA256SUMS.txt').write_text(sums)
 (META/'README.md').write_text('# LIFE-002 research freeze\n\nStatus RESTORE_PARTIAL until independent GitHub restore succeeds. See RESEARCH_LEDGER.md, SOURCE_REFS.json, CHECKPOINT_MANIFEST.json, RESTORE.md and release assets. Original results are immutable.\n')
 subprocess.run(['git','add','archive','scripts/freeze_inventory.py','scripts/freeze_publish.py'],cwd=REPO,check=True)
 subprocess.run(['git','commit','-m','Archive LIFE-002 research provenance and audited restoration assets'],cwd=REPO,check=True)
 commit=git('rev-parse','HEAD')
 subprocess.run(['git','tag','-a',TAG,'-m','LIFE-002 reproducible research freeze; scientific limits retained'],cwd=REPO,check=True)
 before=json.loads((META/'REMOTE_BEFORE.json').read_text())
 current=dict(l.split('\t')[::-1] for l in git('ls-remote','origin').splitlines())
 assert all(current[k]==v for k,v in before.items())
 pushrefs=[r+':'+r for r in refs if r not in current]
 for r in refs:
  if r in current: assert current[r]==refs[r]
 pushrefs += ['refs/heads/'+BRANCH+':refs/heads/'+BRANCH,'refs/tags/'+TAG+':refs/tags/'+TAG]
 # The archive ref was inventoried before its final commit and is handled above.
 pushrefs=list(dict.fromkeys(pushrefs))
 subprocess.run(['git','push','--atomic','origin',*pushrefs],cwd=REPO,check=True)
 release=api('/releases',{'tag_name':TAG,'name':'LIFE-002 Research Freeze — 2026-11-20','body':'Archival checkpoint. Restoration verification pending. Negative C0/D4/F0/F0B results preserved.','draft':True,'prerelease':False,'make_latest':'false'})
 (ASSETS/'UPLOAD_STATE.json').write_text(json.dumps(release,indent=2))
 url=release['upload_url'].split('{')[0]
 for path in sorted(ASSETS.iterdir()):
  if path.name=='UPLOAD_STATE.json':continue
  print('Upload',path.name,path.stat().st_size,flush=True)
  result=api(url+'?name='+urllib.parse.quote(path.name),raw=path.read_bytes(),method='POST')
  assert result['size']==path.stat().st_size
 print('DRAFT_RELEASE_READY',release['id'],commit,git('rev-parse',TAG),flush=True)
if __name__=='__main__':main()
