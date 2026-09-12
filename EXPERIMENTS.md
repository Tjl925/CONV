# CONV 实验记录

后续自动成绩表见 [SCORES.md](SCORES.md)，标准命令流程见 [WORKFLOW.md](WORKFLOW.md)。
下表保留早期实验，SCORES.md 单独记录用户手动 baseline 及后续脚本运行，不覆盖历史。

## 原始版本

- 2026-09-10：原始源码提交 `57ecc4c`，已在超算计算节点完成四个正式 case；源码未修改，运行时增加 `OMP_PROC_BIND=true`。
- 保留文件：`conv2d.c`、`bench_conv.c`、`README.md`。
- 目录原有 `conv2d_test` 的构建来源未知，不能用它代表当前源码性能。
- 官方网页本次未能读取；赛制暂以用户说明及本地 README 为依据。

## 源码核查

- 校验采用逐元素绝对误差 `fabs(result-reference) <= 1e-5`，并拒绝 NaN。
- benchmark 顺序：参考计算 → 待测实现 → 校验 → 一次预热 → 正式计时。
- 计时包含 conv2d 内部清零与 OpenMP 开销；外部 malloc/memcpy/free 不计入时间。
- FAIL 或内存分配失败后，main 仍返回 0。必须检查输出中的 PASS/FAIL，不能仅依赖退出码。
- 随机数种子使用线程编号，改变线程数可能改变输入数据。
- validate_result 中有未使用的可疑声明：inputHeight 的推导不正确，inputWidth 的初始化引用自身。原始文件暂不修改；编译时记录诊断，必要时单独处理并保留官方校验口径。
- 重复相同输入并不保证避免缓存影响；先按官方流程测试，后续比较需保持相同流程。

## 结果表

Case 单元格填写 `Time(ms) / GFLOPS`。Total 为四个 case 时间之和。
汇总 GFLOPS = 四个 case 的 FLOP 总和 / 四个 case 时间总和（秒）/ 1e9，
仅用于内部比较，不假定它等于官方总分。未运行填“待测”，不得填 0 或 PASS。

| Version | Case1 | Case2 | Case3 | Case4 | Total(ms) | GFLOPS（汇总） | PASS |
|---|---|---|---|---|---|---|---|
| 原始命令，无显式绑定 | 中止，未取得成绩 | 未运行 | 未运行 | 未运行 | — | — | 仅小规模 PASS |
| B1：`57ecc4c` + `OMP_PROC_BIND=true` | 1900.91 / 39.6521 | 2064.79 / 40.3121 | 3895.36 / 41.3503 | 8379.90 / 41.2621 | 16240.96 | 40.9740 | 全部 PASS，误差 0 |

四个正式 case 此轮各计时一次，尚不代表多轮稳定性统计或官方排行榜成绩。
完整环境、执行脚本、编译日志及原始结果保留在本地 results/2026-09-10，不公开上传。
实验解释见 [首次超算实验](docs/2026-09-10-first-kunpeng-run.md)。

## 正式命令

```sh
gcc -O3 bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp
export OMP_PROC_BIND=true # B1 唯一运行配置变化；原始官方命令不含此项
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 4096 6144 39 39 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 6144 4096 41 41 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 4256 6390 55 55 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 6390 4256 81 81 1
```

先确认调度/资源使用要求并进入获分配计算节点，再运行正式 case。
`numactl -N 1` 选择 NUMA 节点的 CPU，并不等同于内存绑定或每线程固定核。
未绑定原始运行已作为诊断记录保留：38 个线程采样时集中在一个 CPU，合计约 100% CPU。
B1 增加 `OMP_PROC_BIND=true` 后，采样确认线程分别运行在 CPU 38–75。
以后源码优化与 B1 比较时固定此环境变量；不能将环境变化和源码变化混为一次实验。

## 环境核查

正式性能测试在通过调度系统分配的 ARM 计算节点上运行，使用 GCC 10.3.1。
编译参数与官方一致，没有额外添加 march、mcpu 或 fast-math 选项。
已核实实际线程分布及 perf 可用性；硬件拓扑、内部节点和作业信息保留在本地记录中。
没有额外添加内存绑定。作业已经结束并释放资源。

### 采集范围

- uname -a、lscpu（含 CPU/核/NUMA 映射）、nproc、numactl -H。
- /proc/self/status 中 Cpus_allowed_list 与 Mems_allowed_list；调度作业信息。
- sysfs CPU cache 层级、大小、共享 CPU 列表。
- gcc/clang 版本、目标三元组、ARM target 参数、预定义特性宏。
- CPU 宣告的指令集与编译器启用的指令集分别记录；不能根据 ARM 名称假定 NEON/SVE 能力。
- perf 是否安装、perf_event_paranoid、实际计数器是否有使用权限。
- 编译命令、源码提交与哈希、线程数、绑定方式、原始日志。

## 迭代约定

baseline → 单一优化 → 编译 → correctness → benchmark → 记录比较。
保留稳定提升的版本并提交 Git；失败/变慢时仅还原本次明确修改的文件，保留日志。
先在同一配置复测确认提升，再考虑提交排行榜。
