# CONV 成绩表

自测；非官方榜单分数。每项：ms / GFLOPS。GCC -O3，38 线程，NUMA 1，OMP_PROC_BIND=true，每项计时 1 次。
仅保留本人跑出的 v00 baseline。总 GFLOPS 为四个 case 的 GFLOPS 之和；Total(ms) 为四项耗时之和。

| Version | Run | SHA256 (12) | Case1 | Case2 | Case3 | Case4 | Total(ms) | GFLOPS | Status |
|---|---|---|---|---|---|---|---:|---:|---|
| v00 | 20260912-user-baseline | bb34cc60053b | 1901.86 / 39.6324 | 2062.73 / 40.3522 | 3885.70 / 41.4531 | 8380.06 / 41.2613 | 16230.35 | 162.6990 | PASS |
