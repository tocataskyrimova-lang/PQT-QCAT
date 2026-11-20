"""Read-only research freeze inventory and credential signature audit."""
import hashlib,io,json,re,subprocess,time
from pathlib import Path
from zipfile import ZipFile
ROOT=Path('C:/AGI/PQT_QCAT_AUDIT')
REPO=ROOT/'PQT_QCat_LIFE_002_FREEZE_001'
OUT=REPO/'archive/life-002/2026-11-20'
OUT.mkdir(parents=True,exist_ok=True)
def git(*a): return subprocess.check_output(['git',*a],cwd=REPO)
def sha(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v): (OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
refs=dict(l.split(' ',1) for l in git('for-each-ref','--format=%(refname) %(objectname)','refs/heads','refs/tags').decode().splitlines())
save('SOURCE_REFS.json',refs)
save('REMOTE_BEFORE.json',dict(l.split('\t')[::-1] for l in git('ls-remote','origin').decode().splitlines()))
(OUT/'WORKTREES.txt').write_bytes(git('worktree','list','--porcelain'))
(OUT/'GIT_FSCK.txt').write_bytes(git('fsck','--no-reflogs'))
(OUT/'HISTORY.txt').write_bytes(git('log','--all','--oneline','--decorate','--graph','-n','160'))
patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{50,}',rb'AKIA[0-9A-Z]{16}',rb'sk-(?:proj-)?[A-Za-z0-9_-]{40,}']
findings=[]; examined=set(); objects=0; members=0; snapshots=[]; archives=[]
def check(data,name):
 global members
 h=hashlib.sha256(data).hexdigest()
 if h in examined:return
 examined.add(h);members+=1
 for pattern in patterns:
  if re.search(pattern,data): findings.append({'path':name,'signature':pattern.decode()})
 if data[:4]==b'PK\x03\x04':
  try:
   with ZipFile(io.BytesIO(data)) as z:
    for item in z.infolist():
     if item.is_dir():continue
     if item.filename.endswith('.npz'): snapshots.append({'path':name+'!'+item.filename,'size':item.file_size})
     check(z.read(item),name+'!'+item.filename)
  except Exception as e: findings.append({'path':name,'archive_error':type(e).__name__})
for line in git('rev-list','--objects',*refs.values()).decode().splitlines():
 oid,_,name=line.partition(' ')
 if git('cat-file','-t',oid).strip()!=b'blob':continue
 objects+=1;check(git('cat-file','blob',oid),'git:'+oid+':'+name)
for p in sorted(ROOT.glob('*.zip')):
 if p.name=='PQT_QCAT_DELIVERY_20261111.zip':continue
 print('Audit',p.name,flush=True)
 archives.append({'name':p.name,'size':p.stat().st_size,'sha256':sha(p)})
 check(p.read_bytes(),p.name)
save('ARCHIVES.json',archives);save('CHECKPOINT_MANIFEST.json',snapshots)
save('SECURITY_SCAN.json',{'git_blobs':objects,'unique_payloads':members,'findings':findings,'scope':'All reachable local branch/tag blobs and recursive ZIP payloads. Signature-based audit, no entropy claim.'})
print('DONE',objects,members,len(snapshots),'findings',len(findings),flush=True)
