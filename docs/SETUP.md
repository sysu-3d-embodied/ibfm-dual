# 复现指南

## 1. 官方依赖与检查点

按 `repo/README.md` 准备官方环境。上游要求 Python 3.10。
只运行本仓库的推理实验时，可先按路径获取检查点 LFS 分片，避免下载训练动作库：

```bash
git -C repo lfs install --local
git -C repo lfs pull --include='checkpoints/ibfm-166m/model/model.safetensors.part-*'
bash repo/restore_checkpoint.sh
cd repo/ibfm
uv sync
cd ../..
# 如根目录还没有 .venv，可将默认解释器指向官方环境：
export IBFM_PYTHON="$PWD/repo/ibfm/.venv/bin/python"
bash runtime.sh "$IBFM_PYTHON" -m pip install -r "$PWD/requirements-extra.txt"
```

服务器既有验证环境使用 PyTorch 2.8.0+cu128、MuJoCo 3.8.1 和 Python 3.10。上游依赖声明是 PyTorch 2.7 系列，两者不能混为同一环境锁文件。新机器应根据 GPU 选择兼容 PyTorch 并重新做运行检查。本发布整理没有在一台全新机器上完成依赖安装测试。

仓库 `runtime.sh` 把临时文件与缓存放在克隆目录下 `.runtime/`，运行时应将整个项目放在数据盘。
运行脚本前若直接执行 `uv` 安装，可自行将 `UV_CACHE_DIR`、`TMPDIR` 指到数据盘。

## 2. G1/Revo2 资产

取得项目使用的 `g1_revo2_bottle_sim_model_assets_20260915.zip`，放到：

```text
dual_carry/g1_revo2_assets.zip
```

然后导入：

```bash
bash runtime.sh "$IBFM_PYTHON" "$PWD/dual_carry/prepare_revo2.py"
```

生成 `model_assets/`、`g1_revo2_robot.xml`、`revo2_import.json`，这些都被 Git 忽略。
脚本只解析模型，不执行压缩包中的 Python 脚本。生成 XML 包含当前机器的绝对网格路径，移动项目后重新导入。
资产包未随本仓库分发，需要项目成员自行提供，不存在自动公开下载链接。

## 3. 回放状态库

来源：[官方 166.560M replay](https://huggingface.co/datasets/HanSj77/I-BFM-166560M-replay-buffer)。
可从可信的现有部署复制 `replay_bank_67974_90655a91.pt` 到 `dual_carry/`，保留 SHA-256 及来源记录。
也可重新抽样并缓存后向特征：

```bash
bash runtime.sh "$IBFM_PYTHON" "$PWD/dual_carry/sample_replay.py" --ranks 8 --times 32
bash runtime.sh "$IBFM_PYTHON" "$PWD/dual_carry/build_replay_bank.py"
```

抽样器验证 HTTP Range 和数据长度，默认访问 Hugging Face，可显式设置 `HF_ENDPOINT`。网络需要支持范围请求。
这是一种重建流程，并不承诺重建文件的字节哈希与历史库一致。以输出报告中的 `bank_file` 为准，运行时显式指定 `--state-bank`。
不要下载或加载来源不明的 pickle / PyTorch 序列化文件。

## 4. 放置参考库（完整任务可选）

参考库由包含成功抓取/抬升的本地轨迹构建，不是本仓库随附的官方权重：

```bash
bash runtime.sh "$IBFM_PYTHON" "$PWD/dual_carry/build_placement_guide.py"   "$PWD/dual_carry/outputs/你的已检查运行"   --output "$PWD/dual_carry/placement_guide_custom.pt"
bash dual_carry/run_goal_carry.sh --tag carry_with_guide --box-mass 1.4   --placement-guide placement_guide_custom.pt --state-bank 你的状态库.pt
```

历史默认名称 `placement_guide_v6_static.pt` 不随仓库提供。新生成参考必须重新检查放置表现。

## 5. 输出与验收

`dual_carry/outputs/<tag>/` 保存配置、代码快照、状态、奖励、潜变量、轨迹和指标。
标签应使用新的名称。`success=false` 必须如实保留，不能只凭末帧、到达阶段或末段速度判断成功。
最新已记录的单次速度实验仍出现超速、倾斜和停车时接触丢失。这里发布代码，不把实验原型描述成已完成的稳定控制系统。
