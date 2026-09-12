# CONV 标准流程

本地工作目录是这个 Git 仓库。超算工作目录是你的个人目录 `~`。两个地方都把提交文件放在 `CONV/`，工具放在旁边。

```text
工作目录/
├── CONV/       conv2d.c、bench_conv.c、run.sh、conv2d_test
├── tools/      测试、统计、打包、上传、下载工具
├── logs/       每轮独立日志、源码快照（不进 Git）
├── packages/   用于官网下载选择的 ZIP（不进 Git）
├── scores.md
└── WORKFLOW.md
```

## 1. 本地修改、上传（Windows PowerShell）

只修改 `CONV/conv2d.c`。版本用 `v00`（原始 baseline）、`v01`、`v02`。一次只改一个优化。
运行编号使用超算系统本地时间（当前 CST，UTC+08:00），格式 `v01-20260912T220202-13587`，不再使用 UTC 或 Z 后缀。

```powershell
cd C:\Users\86173\Desktop\conv
.\tools\upload.ps1
```

此脚本只上传一个文件 conv2d.c，不上传 Windows 二进制，也不触碰服务器的其他文件。
远端账号与路径保存在 `tools/remote.local.json`，该文件不进 Git。新机器需要自行配置 Remote 和 RemoteRoot。

## 2. 申请计算资源并测试（超算 SSH 终端）

```bash
cd ~
dlogin
bash tools/test.sh v01
```

认证仍有效时可省略 dlogin。不要在登录节点直接运行 CONV/run.sh 的重计算。
test.sh 申请整节点独占，以获得 NUMA 1 的完整 38 核；它仍只启动 38 个 OpenMP 线程。
运行设置固定为 OMP_PROC_BIND=true、OMP_NUM_THREADS=38、numactl -N 1。此绑定设置应与官方规则允许范围核对。
提交任务前，工具在 logs/版本-时间-随机数/CONV 保存本轮源码快照，然后执行快照中的 run.sh。
编译出的 ARM 可执行文件只有在当前源文件仍与快照一致时才复制回主 CONV。

run.sh 无需任何参数，编译并按顺序跑四个 case，原始结果直接输出到 stdout。
每个 case 内部先校验，再预热和计时；若 FAIL 或缺失 PASS，停止后续 case 并返回非零。
临时校验文本放在系统临时目录，退出时清理，不污染 CONV。
工具保存完整 stdout/stderr 到 logs/本轮/run.log；显示和保存可以同时进行。

Ctrl+C 只停止前台等待，任务仍在后台运行，不要重复提交。查看：

```bash
djob 作业编号
cat logs/v01.latest
tail -n 30 logs/本轮目录/run.log
```

结束自己某一个任务时：`dkill -y 作业编号`。

## 3. 统计（超算）

任务结束后：

```bash
bash tools/scores.sh v01
cat scores.md
```

v01 自动定位到本版本最近一次提交的日志，不用手输时间戳。
同版本复测保留多行，同一轮重复统计更新同一行。
四项尺寸、数据和 PASS 都正确且脚本正常结束才记 PASS；失败/未完成不填性能。
调度器强制终止可能来不及写退出状态，这时记 INCOMPLETE，并以 djob 判断是否超时。
成绩表是自测，不是官方榜单分数。总 GFLOPS 是四个 case 的 GFLOPS 直接相加；Total(ms) 是四项耗时相加。

## 4. 核对并用 Linux zip 打包（超算）

```bash
bash tools/package.sh v01
```

脚本检查当前四个文件与本轮通过测试的 SHA-256 一致，且 CONV 只有这四个普通文件，
然后实际执行 `zip -r packages/CONV-本轮标识.zip CONV`，验证 ZIP 并生成校验文件。
修改源码后未经重测不能打包。包名带版本和时间，避免旧 ZIP 残留文件，不覆盖旧包。
手动 zip 的等效方式是从工作目录执行 `zip -r packages/CONV-唯一名称.zip CONV`，
但不推荐跳过核对；下载工具使用 package.sh 生成的版本指针。

## 5. 下载（Windows PowerShell）

```powershell
cd C:\Users\86173\Desktop\conv
.\tools\download.ps1 v01
```

自动下载该版本最近打包的 ZIP、对应的本轮日志快照和服务器 scores.md，并验证 ZIP SHA-256。
文件位于 packages/、logs/ 和工作目录根部，不会用远端快照覆盖你正在编辑的本地源码。
下载工具只建立一次 SSH 连接，正常每轮只需输入一次密码；不保存密码。ZIP、对应日志和成绩表通过同一连接传回并校验。
已经完整下载的同名日志目录不会被覆盖。

## 6. 官网提交与 GitHub

你在官网选择 packages/ 下本轮 ZIP 上传。ZIP 只有一个 CONV/ 顶层目录，内部为四个文件。
run.sh 可独立编译测试、无参数、无 Python 或调度工具依赖，不打印成绩汇总供判题器混淆。
尚未实际提交官网或验证判题系统解析；首次上传仍由你操作，等待它约十五分钟一轮的评测。

本地保存源码和成绩到 GitHub：

```powershell
git diff -- CONV/conv2d.c CONV/run.sh scores.md
git add CONV/conv2d.c CONV/run.sh scores.md
git commit -m "v01: describe the single optimization"
git push origin master
```

不提交 logs、packages、二进制、私有连接配置。Git 历史位于外层工作目录的 .git，ZIP 内没有 .git。
如果同一版本对应多份不同源码，请新命名 v02，或在提交说明中明确变化。

## 三条超算命令速查

```bash
bash tools/test.sh v01
bash tools/scores.sh v01
bash tools/package.sh v01
```

等第一条真正结束后再执行后两条；日志、成绩表和代码哈希可以互相核对。
