"""Record measured results and create a private, complete CONV backup ZIP."""
import datetime
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = [(4096, 6144, 39, 39), (6144, 4096, 41, 41),
         (4256, 6390, 55, 55), (6390, 4256, 81, 81)]
ROW = re.compile(r'^\s*(\d+)\s+x\s*(\d+)\s+(\d+)\s+x\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(PASS|FAIL)\s*$', re.M)


def checked(case_no, path):
    text = Path(path).read_text(encoding='utf-8')
    rows = ROW.findall(text)
    if len(rows) != 1 or 'FAIL' in text:
        raise ValueError('Expected exactly one PASS row in ' + str(path))
    row = rows[0]
    if tuple(map(int, row[:4])) != CASES[case_no - 1]:
        raise ValueError('Wrong case dimensions in ' + str(path))
    ms, gf, error = map(float, row[4:7])
    if not all(map(math.isfinite, [ms, gf, error])) or ms <= 0 or gf <= 0 or not 0 <= error <= 1e-5:
        raise ValueError('Invalid timing or correctness result in ' + str(path))
    return {'ms': ms, 'gflops': gf, 'max_error': error}


def record(version, directory):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', version):
        raise ValueError('Invalid version label')
    directory = Path(directory).resolve()
    cases = [checked(i, directory / ('case%d.log' % i)) for i in range(1, 5)]
    source = directory / 'conv2d.c'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    total_ms = sum(c['ms'] for c in cases)
    flops = sum(2 * (h-kh+1) * (w-kw+1) * kh * kw for h, w, kh, kw in CASES)
    entry = {'version': version, 'run_id': directory.name, 'source_sha256': digest,
             'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'cases': cases, 'total_ms': total_ms, 'aggregate_gflops': flops / total_ms / 1e6,
             'settings': 'gcc -O3; threads=38; OMP_PROC_BIND=true; NUMA=1; runs=1'}
    # Both the record and Markdown are generated while holding the same lock.
    lock_path = ROOT / '.scores.lock'
    with lock_path.open('a') as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX)
        database = ROOT / 'scores.json'
        entries = json.loads(database.read_text()) if database.exists() else []
        old = next((e for e in entries if e['run_id'] == entry['run_id']), None)
        if old:
            if old['source_sha256'] != digest or old['cases'] != cases or old['version'] != version:
                raise ValueError('Run ID already has different recorded results')
        else:
            entries.append(entry)
        temp = ROOT / 'scores.json.tmp'
        temp.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(str(temp), str(database))
        lines = ['# CONV 成绩表', '',
                 '自测成绩；非官方排行榜分数。每项为 Time(ms) / GFLOPS。',
                 '固定设置：GCC -O3，38 线程，OMP_PROC_BIND=true，NUMA 1，每项计时 1 次。',
                 'Total 是四项时间之和；汇总 GFLOPS = 四项总 FLOP / 总时间，使用打印时间计算。',
                 '只有四项均 PASS 才入表；失败或超时的日志保留在 runs/，不伪造成绩。', '',
                 '| Version | Run | 源码 SHA256（前12位） | Case1 | Case2 | Case3 | Case4 | Total(ms) | 汇总 GFLOPS | PASS |',
                 '|---|---|---|---|---|---|---|---:|---:|---|']
        for e in entries:
            cells = [e['version'], e['run_id'], e['source_sha256'][:12]]
            cells += ['%.2f / %.4f' % (c['ms'], c['gflops']) for c in e['cases']]
            cells += ['%.2f' % e['total_ms'], '%.4f' % e['aggregate_gflops'], 'PASS']
            lines.append('| ' + ' | '.join(cells) + ' |')
        temp = ROOT / 'SCORES.md.tmp'
        temp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        os.replace(str(temp), str(ROOT / 'SCORES.md'))
    print('Recorded: %s; total %.2f ms; %.4f GFLOPS' % (version, total_ms, entry['aggregate_gflops']))


def pack(label):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', label):
        raise ValueError('Invalid archive label')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = ROOT.parent / ('CONV-%s-%s.zip' % (label, stamp))
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name != '.scores.lock')
    if any(p.is_symlink() or ROOT not in p.resolve().parents for p in files):
        raise ValueError('Refusing to follow a file outside CONV')
    with zipfile.ZipFile(str(archive), 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(str(p), str(Path('CONV') / p.relative_to(ROOT)))
    with zipfile.ZipFile(str(archive)) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
    print('PRIVATE FULL BACKUP (includes logs and .git; do not publish without review):')
    print(archive)
    print('SHA256=' + hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == '__main__':
    try:
        action = sys.argv[1]
        if action == 'check':
            checked(int(sys.argv[2]), sys.argv[3])
        elif action == 'record':
            record(sys.argv[2], sys.argv[3])
        elif action == 'pack':
            pack(sys.argv[2])
        else:
            raise ValueError('Use check, record or pack')
    except (ValueError, OSError, IndexError) as exc:
        sys.exit('ERROR: ' + str(exc))
