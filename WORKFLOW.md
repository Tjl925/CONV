# CONV 标准流程

## 文件职责

- `conv2d.c`：后续唯一需要修改的计算实现。
- `bench_conv.c`：保持官方原始内容；脚本检查其 SHA-256。
- `submit.sh`：在登录节点申请资源并启动 run.sh。
- `run.sh`：在计算节点编译源码快照，逐项运行四个正式尺寸，先校验再计时。
- `SCORES.md` / `scores.json`：自动维护的人类可读成绩表和结构化数据。
- `pack.sh`：完整目录备份 ZIP，存到 CONV 的父目录。

依赖：Bash、GCC/OpenMP、numactl、Python 3、Donau 命令。代码和脚本使用 LF 换行。

## 每次实验（超算终端）

先修改 conv2d.c，一次只改一个优化。用有意义的英文版本名：

```bash
cd ~/CONV
dlogin
bash submit.sh v01_local_sum
```

如果调度认证仍有效，dlogin 可省略。编译器参数固定为官方 `gcc -O3 ... -lm -fopenmp`。
资源申请沿用本账号已验证的整节点独占方式；卷积只使用 NUMA 1 的 38 核。
这不是申请资源最省的方案，后续资源策略单独比较。运行配置明确增加 OMP_PROC_BIND=true。

脚本显示独立的 runs/时间戳-版本目录。先看所有 case 都 PASS，再看性能。
四个 case 全部通过后才更新 SCORES.md；失败时停止，保留已产生的日志和源码快照。
编译失败也会停止。不要只看调度器 SUCCEEDED，因为官方测试程序可能打印 FAIL 仍返回 0。

```bash
cat SCORES.md
```

同一版本可以重复提交测试，每次是独立一行，不覆盖历史记录。
同一个运行目录再次导入成绩不会重复入表。
不要在打包或同步期间修改源码，也不要在任务仍运行时打包，以免快照不一致。
发现变慢或错误时，先保存日志，再仅还原本次的 conv2d.c 改动。

Ctrl+C 仅停止前台等待，作业仍继续。不要立即再提交一个：

```bash
djob 作业编号
tail -n 30 runs/本次运行目录/job.log
```

需要取消自己这一个任务时使用 `dkill -y 作业编号`。

## 打包与下载

确认作业已结束、成绩已入表后：

```bash
bash pack.sh v01_local_sum
```

命令打印 ZIP 的完整路径与 SHA-256。ZIP 含整个 CONV（源码、脚本、日志、二进制及 .git），
是个人完整备份，不应公开推送 GitHub。尚未核实官方上传格式，不能称其为已合规的官方提交包。
隐藏的 .scores.lock 是锁文件，不进备份。

切到 Windows PowerShell（不是 SSH 内的远端 shell），在希望保存 ZIP 的目录运行：

```powershell
scp 用户名@入口地址:CONV-版本-时间.zip .
Get-FileHash .\CONV-版本-时间.zip -Algorithm SHA256
```

替换为 pack.sh 实际打印的文件名。远程路径也可以使用打印的完整路径。
确认本地与远端哈希一致。备份不需要覆盖解压到已有的 Windows Git 仓库。

## 更新 GitHub（Windows 本地 Git 仓库）

超算目前不能直接解析 GitHub，所以由 Windows 同步再推送。
把本轮的 conv2d.c、SCORES.md 和 scores.json 用 scp 下载到本地仓库；先确认本地没有未保存的独立源码修改。

```powershell
scp 用户名@入口地址:CONV/conv2d.c .
scp 用户名@入口地址:CONV/SCORES.md .
scp 用户名@入口地址:CONV/scores.json .
git diff -- conv2d.c SCORES.md scores.json
git add conv2d.c SCORES.md scores.json
git commit -m "v01: local accumulator experiment"
git push origin master
```

公开仓库只包含源码、流程和性能摘要；runs/、原始日志、ZIP 和内部环境记录被忽略。
不使用 git add . 把未经检查的材料一起提交。

## 成绩口径

表中每项是毫秒 / GFLOPS。汇总 GFLOPS 按四项总 FLOP 除以总时间计算，非官方分数。
当前每 case 正式计时 1 次；要评估稳定性，可完整重复同一版本，保留每次记录。
自动记录当前源文件快照的 SHA-256，用来识别不同代码版本。
原始无绑定超时记录与增加绑定后的成绩必须分开。

## 官方提交

这套脚本是超算自测流程，不自动向比赛平台提交，也不读取或伪造排名。
官方是否接收自定义 run.sh、是否允许运行环境变量、ZIP 目录层级及应包含哪些文件，
须在实际提交页核实后再制作正式提交包。首次提交仍由参赛者操作。
