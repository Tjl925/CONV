#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python3 - "$root" "${1:?Usage: bash tools/scores.sh v00}" <<'PY'
from pathlib import Path
import sys,re,math,hashlib,fcntl,os
root=Path(sys.argv[1]); version=sys.argv[2]
if not re.fullmatch(r'v\d+',version): sys.exit('Invalid version')
rid=(root/'logs'/(version+'.latest')).read_text().strip()
if not re.fullmatch(re.escape(version)+r'-[0-9]{8}T[0-9]{6}-[0-9]+',rid): sys.exit('Invalid run ID')
run=root/'logs'/rid
text=(run/'run.log').read_text() if (run/'run.log').exists() else ''
pattern=r'^\s*(\d+)\s+x\s*(\d+)\s+(\d+)\s+x\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(PASS|FAIL)\s*$'
rows=re.findall(pattern,text,re.M)
cases=[(4096,6144,39,39),(6144,4096,41,41),(4256,6390,55,55),(6390,4256,81,81)]
good=[]
for row in rows:
    try:
        dims=tuple(map(int,row[:4])); ms,gf,err=map(float,row[4:7])
        if row[7]=='PASS' and all(map(math.isfinite,(ms,gf,err))) and ms>0 and gf>0 and 0<=err<=1e-5:
            good.append((dims,ms,gf,err))
    except ValueError: pass
rc=(run/'exitcode').read_text().strip() if (run/'exitcode').exists() else None
ok=rc=='0' and len(rows)==4 and len(good)==4 and [r[0] for r in good]==cases and 'FAIL' not in text
status='PASS' if ok else ('FAIL' if rc is not None or 'FAIL' in text else 'INCOMPLETE')
digest=hashlib.sha256((run/'CONV/conv2d.c').read_bytes()).hexdigest()
cells=[version,rid,digest[:12]]
if ok:
    ms=sum(r[1] for r in good); total_gflops=sum(r[2] for r in good)
    cells+=['%.2f / %.4f'%(r[1],r[2]) for r in good]+['%.2f'%ms,'%.4f'%total_gflops,status]
else: cells+=['—']*6+[status]
line='| '+' | '.join(cells)+' |'
table=root/'scores.md'
header='# CONV 成绩表\n\n自测；非官方榜单分数。每项：ms / GFLOPS。GCC -O3，38 线程，NUMA 1，OMP_PROC_BIND=true，每项计时 1 次。\n总 GFLOPS 为四个 case 的 GFLOPS 之和；失败/未完成不填写性能。\n\n| Version | Run | SHA256 (12) | Case1 | Case2 | Case3 | Case4 | Total(ms) | GFLOPS | Status |\n|---|---|---|---|---|---|---|---:|---:|---|\n'
with (root/'logs/.scores.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    lines=table.read_text().splitlines() if table.exists() else header.splitlines()
    lines=[s for s in lines if ('| '+rid+' |') not in s]
    lines.append(line)
    tmp=root/'logs/scores.tmp'; tmp.write_text('\n'.join(lines)+'\n'); os.replace(str(tmp),str(table))
(run/'score-status').write_text(status+'\n')
print(line)
if not ok: sys.exit('Not a complete passing run; see log and djob (timeout may appear INCOMPLETE).')
PY
