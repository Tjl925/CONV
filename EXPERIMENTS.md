# CONV 实验记录

## 原始版本

- 2026-09-10：本地原始源码提交 `57ecc4c`，尚未在鲲鹏上编译或测试。
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
| baseline `57ecc4c` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 未验证 |

## 正式命令

```sh
gcc -O3 bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 4096 6144 39 39 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 6144 4096 41 41 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 4256 6390 55 55 1
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 6390 4256 81 81 1
```

先确认调度/资源使用要求并进入获分配计算节点，再运行正式 case。
`numactl -N 1` 选择 NUMA 节点的 CPU，并不等同于内存绑定或每线程固定核。
第一轮不额外添加绑定环境变量，记录继承的 OMP/GOMP 环境和实际 CPU/内存允许列表。

## 鲲鹏环境待采集

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
