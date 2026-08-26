"""Step 2: single-agent SAC on MDP v1, checkpoints from official metrics.

BlueSky allows one simulation per process, so this uses SubprocVecEnv
(not DummyVecEnv). The policy is a flattened-obs MLP [256, 256].

Checkpointing ignores SB3 episode return. Every ``--eval-freq`` steps the
callback runs ``--eval-episodes`` on a validation seed and keeps the model
with high arrival rate and low intrusion time. See competitor/callbacks.py.

    SDL_VIDEODRIVER=dummy python -m scripts.train_sa_sac
    SDL_VIDEODRIVER=dummy python -m scripts.train_sa_sac --timesteps 500000
    SDL_VIDEODRIVER=dummy python -m scripts.train_sa_sac --algo ppo
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from competitor.callbacks import OfficialMetricCheckpointCallback
from competitor.vec_env import make_flattened_sa_env


def build_vec_env(n_envs, seed):
    return make_vec_env(
        make_flattened_sa_env,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="forkserver"),
        monitor_dir=None,
    )


def build_model(algo, venv, seed):
    policy_kwargs = dict(net_arch=[256, 256])
    if algo == "sac":
        return SAC(
            "MlpPolicy",
            venv,
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device="auto",
        )
    return PPO(
        "MlpPolicy",
        venv,
        n_steps=512,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(
        "competitor", "results", f"{args.algo}_sa_v1")
    os.makedirs(save_dir, exist_ok=True)

    print(
        f"Step 2 train: algo={args.algo}  timesteps={args.timesteps}  "
        f"n_envs={args.n_envs}  net_arch=[256,256]  flatten=Dict->Box\n"
        f"Checkpoint on arrival_rate + intrusion_time "
        f"(min_arrival={args.min_arrival}); SB3 return is logged but ignored.\n"
        f"Eval: {args.eval_episodes} eps, seed {args.eval_seed}, every "
        f"{args.eval_freq} steps → {save_dir}"
    )

    torch.set_num_threads(1)
    venv = build_vec_env(args.n_envs, args.seed)
    model = build_model(args.algo, venv, args.seed)
    callback = OfficialMetricCheckpointCallback(
        save_dir=save_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        min_arrival=args.min_arrival,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(os.path.join(save_dir, "final"))
    print(f"training finished. selected checkpoint in {save_dir}/selected.zip")
    print(f"selection summary: {os.path.join(save_dir, 'selection.json')}")


if __name__ == "__main__":
    main()
