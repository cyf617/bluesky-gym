# 在自己电脑上训练与评估

本文件对应分支 `cursor/training-dashboard-b718`。所有命令都在仓库根目录运行。

## 1. 安装

需要 Python ≥ 3.10（云端用的是 3.12）和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone <仓库地址> bluesky-gym && cd bluesky-gym
git checkout cursor/training-dashboard-b718
uv sync                      # 按 uv.lock 装好 torch / stable-baselines3 / bluesky-simulator 等
```

- 第一次运行时 BlueSky 会在用户目录下建 `~/bluesky/`（配置、导航数据缓存），会多花十几秒。
- 无界面训练要关掉 pygame 窗口：Linux/macOS 用 `export SDL_VIDEODRIVER=dummy`，Windows PowerShell 用 `$env:SDL_VIDEODRIVER="dummy"`（脚本里也会默认设置）。
- 只用 CPU 即可。策略是两层 256 的 MLP，PPO 在 CPU 上比 GPU 快；`--device` 默认就是 `cpu`。
- 先跑冒烟测试确认环境正常：

```bash
uv run python -m scripts.smoke_mdp_v1
uv run --with pytest pytest -q competitor/tests dashboard/tests
```

## 2. 训练

BlueSky 每个进程只能有一个仿真，所以并行靠 `--n-envs` 个子进程（SubprocVecEnv）。`--n-envs` 建议设成物理核数（多个训练同时跑时平分核数）。
参考速度：云端 4 核机器同时跑 4 个训练、每个 4 个环境时，每个约 110 步/秒（含评估），即 100 万步约 2.5 小时。

exp3 中 A1（SAFE_V2 + 过滤器）的完整命令：

```bash
uv run python -m scripts.train_sa_sac --algo ppo --action-mode goal_relative \
    --timesteps 2000000 --eval-freq 100000 --eval-episodes 100 --eval-seed 123 --seed 0 --n-envs 4 \
    --stop-after 500000 --stop-min-arrival 0.95 --stop-patience 2 --snapshot-steps 1000000 \
    --set SHIELD=1 \
    --set RESTRICTED_EVENT_PENALTY=20 --set RESTRICTED_TIME_WEIGHT=0.2 \
    --set SECTOR_EXIT_EVENT_PENALTY=25 --set SECTOR_TIME_WEIGHT=0.2 \
    --set INTRUSION_EVENT_PENALTY=12 --set INTRUSION_TIME_WEIGHT=0.25 --set CONFLICT_WEIGHT=0.1 \
    --set DCPA_LIMIT_NM=7 --set TCPA_HORIZON_S=180 --set PROXIMITY_HORIZON_KM=10 --set PROXIMITY_WEIGHT=0.1 \
    --run-name my_safe_v2
```

- 结果写在 `competitor/results/<run-name>/`：`config.json`、`progress.csv`、`episodes.csv`、`eval_log.csv`、`selection.json`、`status.json`，以及 `latest.zip`、`best_*.zip`、`selected.zip`、`final.zip`、`replays/`。
- 可调参数全部在 `competitor/config.py` 的 `TUNABLE_KEYS` 里，用 `--set KEY=VALUE` 覆盖，不用改代码。
- `selected.zip` 按“冲突时间最小（到达率 ≥ 90%）”选；开了过滤器时它常常选到第 0 步的模型，这种情况下用 `best_safety_total.zip` 或 `latest.zip`。

### 从云端的模型接着训练

仓库里带了 exp3 四个运行在 90 万步时的 `latest.zip`。`--mdp-from` 复用原运行的奖励/动作参数，`--init-from` 载入权重（步数和 VecNormalize 的奖励统计从 0 开始，原运行的 PPO 超参数要自己带上，例如 `--gamma 0.995`）：

```bash
uv run python -m scripts.train_sa_sac --algo ppo --action-mode goal_relative \
    --mdp-from competitor/results/exp3_ppo_safe_v2_half_shield \
    --init-from competitor/results/exp3_ppo_safe_v2_half_shield/latest.zip \
    --timesteps 1000000 --eval-freq 100000 --eval-episodes 100 --n-envs 4 \
    --run-name exp3_half_continue
```

## 3. 看板

```bash
uv run python -m dashboard            # 打开 http://127.0.0.1:8050
```

训练过程中每 10 秒自动刷新；“评估回放”标签页需要 `replays/`（训练结束自动录制，或手动 `uv run python -m scripts.record_replays --run-dir competitor/results/<run> --model latest`）。

## 4. 官方评估（seed 42，1000 局）

```bash
# 中性基线（只需跑一次）
uv run python -m scripts.evaluate_competition --env sa --out competitor/results/official_seed42_1000_neutral.csv
# 某个运行的某个检查点（自动套用该运行的参数，包括 SHIELD）
uv run python -m scripts.evaluate_competition --env sa --run-dir competitor/results/<run> \
    --model competitor/results/<run>/latest --out competitor/results/<run>/official_seed42_1000_latest.csv
# 与中性逐局配对比较（均值差 ± 95% 置信区间）
uv run python -m scripts.compare_official --neutral competitor/results/official_seed42_1000_neutral.csv \
    mine=competitor/results/<run>/official_seed42_1000_latest.csv
```

一个 1000 局评估单进程约 15–35 分钟（开过滤器时更久）。多个评估可以并行，每个进程只用 1 个 torch 线程。

## 5. 其他脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/run_neutral_baseline.py --seed S --episodes N` | 中性策略在验证 seed 上的基线（看板按 seed + 回合数匹配） |
| `scripts/sweep_shield.py` | 在验证 seed 上扫过滤器参数 |
| `scripts/reward_term_shares.py <run>` | 训练早期/后期各奖励项占比 |
| `scripts/record_replays.py` | 录制评估回放 |
