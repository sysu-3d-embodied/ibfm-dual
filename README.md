# ibfm-dual

基于 **I-BFM** 的双 G1 + Revo2 共享箱体搬运研究原型。两台机器人在同一个 MuJoCo 场景中，通过共享实时上下文、动态奖励潜变量推断及可选接触反馈执行动作。

**当前状态：仍在验证。** 最新 1.4 kg 速度跟随实验完成了速度序列，但速度精度、倾斜、队形与手部负载检查未全部通过。不能将阶段完成等同于稳定搬运成功。这里没有重新训练 I-BFM，也不包含真机控制程序。

## 整体方法

1. 读取两台机器人、共享箱体和真实接触信息，构造实时上下文 `c_t`。
2. 生成共同箱子目标及各自身体、手部目标，分别计算 `r_k(s_n, c_t)`。
3. 在符合原 I-BFM 格式的固定状态库上评分，聚合离线缓存的 `B(s_n)`，分别推断 `z_1`、`z_2`。
4. 冻结策略 `pi(o_k, z_k)` 输出各自关节目标。
5. 最新速度实验可加入倾斜驱动的手位、支撑力分配和有界手臂残差反馈。

速度验证采用协调层 10 Hz、策略 50 Hz、物理与底层反馈 200 Hz。没有箱体焊接、根部位姿改写或直接向箱体加外力。目标支撑力有界不代表实际接触冲击力一定有界。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| `repo/` | 官方 I-BFM 子模块，固定到服务器验证过的提交 |
| `dual_carry/environment.py` | 双机器人、共享物体与原模型观测接口 |
| `dual_carry/coordinator.py` | 阶段状态机、目标、奖励、潜变量与转向混合 |
| `dual_carry/level_speed_coordination.py` | 共同箱子速度目标与倾斜/支撑力纠偏 |
| `dual_carry/speed_follow_probe.py` | A 变速、B 跟随，含同状态反事实检查 |
| `dual_carry/standing_height_probe.py` | 单侧高度变化协同验证 |
| `dual_carry/mission.py` | 完整搬运任务的实验入口 |
| `dual_carry/sample_replay.py` / `build_replay_bank.py` | 官方回放抽样与 B 缓存 |
| `dual_carry/render_*.py` | 保存轨迹的离线渲染 |
| `docs/` | 复现说明、资源边界和版本来源 |

## 获取源码

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/sysu-3d-embodied/ibfm-dual.git
cd ibfm-dual
```

项目面向 Linux、Python 3.10、CUDA GPU 和 MuJoCo EGL。Windows 可查看代码，但上游符号链接与运行流程按 Linux 设计。

## 准备环境与资源

详细步骤见 [复现指南](docs/SETUP.md)。需要分别准备：

- 官方 I-BFM 环境、166.560M-step 检查点及配套 latent。
- 项目使用的 G1/Revo2 资产包，在本地运行 `prepare_revo2.py` 导入。
- 官方回放抽样生成的状态库。历史实验使用 `replay_bank_67974_90655a91.pt`，不会上传到 Git。

有现成服务器环境时，可用 `IBFM_ROOT` 指向已有 I-BFM 根目录，用 `IBFM_PYTHON` 指向现有解释器。自定义代码和输出始终使用本仓库 `dual_carry/`，模型和官方源码从 `IBFM_ROOT/repo/ibfm` 读取。

## 运行最新速度跟随实验

```bash
CUDA_VISIBLE_DEVICES=0 bash dual_carry/run_speed_follow_pair.sh trial_001 --level-coordination
bash dual_carry/render_speed_follow_pair.sh trial_001
```

这会顺序运行实时速度组和冻结速度线索组，各自产生独立输出。新状态库可追加 `--state-bank 你的库文件.pt`，文件放在 `dual_carry/` 或使用绝对路径。

不带 `--level-coordination` 可以运行修改前的速度验证路径。冻结组仍保留队形、共同箱子目标（启用新方案时）和物理耦合，并非完全无信息交流的基线。

速度序列：0.05、0.10、0.20、0.10、0 m/s，每档 6 秒，前 2 秒平滑过渡。**本验证不包含终点放置**。

## 其他入口

```bash
bash dual_carry/run_height_probe_pair.sh height_001
bash dual_carry/render_height_probe_pair.sh height_001
bash dual_carry/run_revo2.sh --tag carry_001 --box-mass 1.4
```

`config.json` 保留现有完整任务的 5 kg 配置，速度/高度诊断显式覆盖为 1.4 kg。5 kg 稳定搬运尚未验证通过。
`run_goal_carry.sh` 还需要本地放置参考库，见复现指南。最新倾斜反馈目前只接入速度验证，不会自动启用到完整任务。

## 测试

环境准备完成后：

```bash
bash runtime.sh "${IBFM_PYTHON:-$PWD/.venv/bin/python}" -m unittest discover -s "$PWD/dual_carry" -p 'test_*.py'
bash runtime.sh "${IBFM_PYTHON:-$PWD/.venv/bin/python}" "$PWD/dual_carry/test_level_speed_coordination.py"
```

若使用外部环境，把 `.venv/bin/python` 替换成解释器的绝对路径。
单元测试检查符号、约束和接口，不能替代完整动力学验收。

## 许可与引用

保留上游 CC BY-NC 4.0，见 [LICENSE](LICENSE) 和 [第三方来源](THIRD_PARTY_NOTICES.md)。
官方模型说明与论文引用见 [I-BFM](https://github.com/Roboparty/I-BFM-Code)。本仓库保留官方网络接口，不把双方联合状态直接送入原后向映射。
