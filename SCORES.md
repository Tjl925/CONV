# CONV 成绩表

自测成绩；非官方排行榜分数。每项为 Time(ms) / GFLOPS。
固定设置：GCC -O3，38 线程，OMP_PROC_BIND=true，NUMA 1，每项计时 1 次。
Total 是四项时间之和；汇总 GFLOPS = 四项总 FLOP / 总时间，使用打印时间计算。
只有四项均 PASS 才入表；失败或超时的日志保留在 runs/，不伪造成绩。

| Version | Run | 源码 SHA256（前12位） | Case1 | Case2 | Case3 | Case4 | Total(ms) | 汇总 GFLOPS | PASS |
|---|---|---|---|---|---|---|---:|---:|---|
| baseline_original_bind | 20260912-user-baseline | bb34cc60053b | 1901.86 / 39.6324 | 2062.73 / 40.3522 | 3885.70 / 41.4531 | 8380.06 / 41.2613 | 16230.35 | 41.0008 | PASS |
| workflow_check | 20260912T114530-workflow_check-2190134 | bb34cc60053b | 1901.79 / 39.6337 | 2064.37 / 40.3202 | 3885.98 / 41.4501 | 8377.78 / 41.2725 | 16229.92 | 41.0019 | PASS |
