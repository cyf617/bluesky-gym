"""Step 2: single-agent SAC / PPO on MDP v1, checkpoints from official metrics.

BlueSky allows one simulation per process, so this uses SubprocVecEnv
(not DummyVecEnv). The policy is a flattened-obs MLP (default [256, 256]).

Checkpointing ignores SB3 episode return. Every ``--eval-freq`` steps the
callback runs ``--eval-episodes`` on a validation seed and keeps the model
with high arrival rate and low intrusion time. See competitor/callbacks.py.

Each run directory gets ``config.json`` (resolved MDP config + hyperparameters),
``progress.csv`` (SB3 logger), ``episodes.csv`` / ``status.json`` (telemetry)
and ``eval_log.csv`` / ``selection.json`` (official-metric evals). The
dashboard (``python -m dashboard``) reads all of them.

    SDL_VIDEODRIVER=dummy python -m scripts.train_sa_sac
    SDL_VIDEODRIVER=dummy python -m scripts.train_sa_sac --timesteps 500000
    SDL_VIDEODRIVER=dummy python -m scripts.train_sa_sac --algo ppo --action-mode goal_relative
    SDL_VIDEODRIVER=dummy python -m scripts.train_sa_sac --set CONFLICT_WEIGHT=0.1 --run-name sac_cw01
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from competitor.callbacks import OfficialMetricCheckpointCallback
from competitor.checkpoint_rule import ArrivalStopRule
from competitor.config import ACTION_MODES, parse_overrides, resolve_mdp_config
from competitor.telemetry import TrainingTelemetryCallback, write_json_atomic
from competitor.vec_env import make_flattened_sa_env


def build_vec_env(n_envs, seed, env_kwargs, norm_reward, gamma):
    venv = make_vec_env(
        make_flattened_sa_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        # forkserver is unavailable on Windows; spawn works everywhere.
        vec_env_kwargs=dict(start_method="forkserver" if "forkserver" in mp.get_all_start_methods()
                            else "spawn"),
        monitor_dir=None,
    )
    if norm_reward:
        # Observations are already hand-normalised; only rescale the return.
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=gamma)
    return venv


def build_model(args, venv):
    policy_kwargs = dict(net_arch=list(args.net_arch))
    if args.algo == "sac":
        return SAC(
            "MlpPolicy",
            venv,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=args.gamma,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
    policy_kwargs["log_std_init"] = args.log_std_init
    return PPO(
        "MlpPolicy",
        venv,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gae_lambda=0.95,
        gamma=args.gamma,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        clip_range=0.2,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="total env steps across all workers (default 500k)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="SubprocVecEnv workers; one BlueSky sim each")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="run official-metric eval every N timesteps")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=123,
                        help="validation seed (not the official seed 42)")
    parser.add_argument("--min-arrival", type=float, default=0.90,
                        help="safety checkpoints must keep at least this arrival rate")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--run-name", default=None,
                        help="directory name under competitor/results (default <algo>_sa_v1)")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="enable the arrival stop rule after this many steps")
    parser.add_argument("--stop-min-arrival", type=float, default=0.95)
    parser.add_argument("--stop-patience", type=int, default=2,
                        help="consecutive evals below --stop-min-arrival before stopping")
    parser.add_argument("--snapshot-steps", type=int, nargs="*", default=[],
                        help="also keep step_<N>.zip at the first eval at or after N steps")
    parser.add_argument("--replay-episodes", type=int, default=5,
                        help="after training, record selected.zip on the first N seed-42 "
                             "episodes for the dashboard replay tab (0 = skip)")

    mdp = parser.add_argument_group("MDP (competitor/config.py overrides)")
    mdp.add_argument("--action-mode", choices=ACTION_MODES, default=None,
                     help="shortcut for --set ACTION_MODE=...")
    mdp.add_argument("--mdp-from", default=None, metavar="RUN_DIR",
                     help="start from that run's config.json mdp_overrides (then apply --set)")
    mdp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="override a TUNABLE_KEYS constant, e.g. --set CONFLICT_WEIGHT=0.1")

    hp = parser.add_argument_group("algorithm hyperparameters")
    hp.add_argument("--lr", type=float, default=3e-4)
    hp.add_argument("--gamma", type=float, default=0.99)
    hp.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    hp.add_argument("--buffer-size", type=int, default=200_000, help="SAC replay buffer")
    hp.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length per env")
    hp.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size")
    hp.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per rollout")
    hp.add_argument("--ent-coef", type=float, default=0.0, help="PPO entropy bonus")
    hp.add_argument("--target-kl", type=float, default=0.03, help="PPO early-stop KL")
    hp.add_argument("--log-std-init", type=float, default=-0.5,
                    help="PPO initial log std (-0.5 ≈ std 0.6)")
    hp.add_argument("--device", default="cpu",
                    help="torch device; small MLP policies train faster on cpu than cuda")
    hp.add_argument("--init-from", default=None,
                    help="start from this checkpoint's weights (same algo, obs and net-arch); "
                         "timesteps and VecNormalize stats start fresh")
    hp.add_argument("--norm-reward", action=argparse.BooleanOptionalAction, default=None,
                    help="VecNormalize the reward (default: on for PPO, off for SAC)")
    args = parser.parse_args()

    overrides = {}
    if args.mdp_from:
        import json
        with open(os.path.join(args.mdp_from, "config.json")) as f:
            overrides.update(json.load(f).get("mdp_overrides") or {})
    overrides.update(parse_overrides(args.set))
    if args.action_mode:
        overrides["ACTION_MODE"] = args.action_mode
    mdp_cfg = resolve_mdp_config(overrides)
    env_kwargs = {"mdp_overrides": overrides} if overrides else {}
    norm_reward = args.norm_reward if args.norm_reward is not None else args.algo == "ppo"

    run_name = args.run_name or f"{args.algo}_sa_v1"
    save_dir = args.save_dir or os.path.join("competitor", "results", run_name)
    os.makedirs(save_dir, exist_ok=True)

    hyperparams = {k: getattr(args, k) for k in ("lr", "gamma", "net_arch", "n_envs", "seed")}
    if args.algo == "sac":
        hyperparams.update(buffer_size=args.buffer_size, batch_size=256, learning_starts=5_000,
                           tau=0.005, train_freq=1, gradient_steps=1, ent_coef="auto")
    else:
        hyperparams.update(n_steps=args.n_steps, batch_size=args.batch_size,
                           n_epochs=args.n_epochs, ent_coef=args.ent_coef,
                           target_kl=args.target_kl, log_std_init=args.log_std_init,
                           gae_lambda=0.95, clip_range=0.2)
    hyperparams["norm_reward"] = norm_reward
    hyperparams["init_from"] = args.init_from
    write_json_atomic(os.path.join(save_dir, "config.json"), {
        "run_name": os.path.basename(os.path.normpath(save_dir)),
        "algo": args.algo,
        "timesteps": args.timesteps,
        "mdp": mdp_cfg,
        "mdp_overrides": overrides,
        "hyperparams": hyperparams,
        "eval": {"freq": args.eval_freq, "episodes": args.eval_episodes,
                 "seed": args.eval_seed, "min_arrival": args.min_arrival},
        "stop_rule": ({"after": args.stop_after, "min_arrival": args.stop_min_arrival,
                       "patience": args.stop_patience} if args.stop_after is not None else None),
        "snapshot_steps": args.snapshot_steps,
        "git_commit": git_commit(),
        "command": " ".join([os.path.basename(sys.executable), "-m", "scripts.train_sa_sac"]
                            + sys.argv[1:]),
        "created_at": time.time(),
    })

    print(
        f"Step 2 train: algo={args.algo}  timesteps={args.timesteps}  "
        f"n_envs={args.n_envs}  net_arch={args.net_arch}  flatten=Dict->Box  "
        f"action_mode={mdp_cfg['ACTION_MODE']}  norm_reward={norm_reward}\n"
        f"MDP overrides: {overrides or 'none'}\n"
        f"Checkpoint on arrival_rate + intrusion_time "
        f"(min_arrival={args.min_arrival}); SB3 return is logged but ignored.\n"
        f"Eval: {args.eval_episodes} eps, seed {args.eval_seed}, every "
        f"{args.eval_freq} steps → {save_dir}"
    )

    torch.set_num_threads(1)
    venv = build_vec_env(args.n_envs, args.seed, env_kwargs, norm_reward, args.gamma)
    model = build_model(args, venv)
    if args.init_from:
        model.set_parameters(args.init_from, exact_match=True, device=args.device)
        print(f"initialised weights from {args.init_from}")
    model.set_logger(configure(save_dir, ["stdout", "csv"]))
    stop_rule = ArrivalStopRule(args.stop_after, args.stop_min_arrival, args.stop_patience)
    eval_cb = OfficialMetricCheckpointCallback(
        save_dir=save_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        min_arrival=args.min_arrival,
        env_kwargs=env_kwargs,
        stop_rule=stop_rule,
        snapshot_steps=args.snapshot_steps,
        verbose=1,
    )
    # Telemetry goes last so its final heartbeat follows the final eval.
    callback = CallbackList([
        eval_cb,
        TrainingTelemetryCallback(save_dir=save_dir, total_timesteps=args.timesteps,
                                  stop_rule=stop_rule),
    ])
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    if stop_rule.reason:
        print(f"stopped early: {stop_rule.reason}")
    model.save(os.path.join(save_dir, "final"))
    print(f"training finished. selected checkpoint in {save_dir}/selected.zip")
    print(f"selection summary: {os.path.join(save_dir, 'selection.json')}")

    selected = os.path.join(save_dir, "selected")
    if args.replay_episodes > 0 and os.path.exists(selected + ".zip"):
        from competitor.replay import load_policy, record_replays
        index = record_replays(
            os.path.join(save_dir, "replays"), load_policy(selected, args.algo),
            env_kwargs=env_kwargs, seed=42, n_episodes=args.replay_episodes,
            label="selected", meta={"model": "selected"})
        print(f"recorded {len(index)} replay episodes → {save_dir}/replays")


if __name__ == "__main__":
    main()
