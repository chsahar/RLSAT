from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from saharsat import SATBranchingEnv
from saharsat.model_io import load_ppo_model


def run_episode(model: PPO, env: SATBranchingEnv, seed: int) -> dict:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    final_info = info

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, final_info = env.step(int(action))
        total_reward += float(reward)

    return {
        "solved": bool(final_info["all_satisfied"]),
        "satisfied_clauses": int(final_info["satisfied_clauses"]),
        "assigned_variables": int(final_info["assigned_variables"]),
        "steps": env.steps,
        "total_reward": total_reward,
        "instance_path": final_info["instance_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", default="models/ppo_uf20_91.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = SATBranchingEnv(args.data_dir, max_steps=args.max_steps, seed=args.seed)
    model = load_ppo_model(args.model)

    results = [
        run_episode(model, env, seed=args.seed + episode)
        for episode in range(args.episodes)
    ]

    solved = sum(result["solved"] for result in results)
    satisfied = np.asarray([result["satisfied_clauses"] for result in results])
    steps = np.asarray([result["steps"] for result in results])

    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Solved: {solved}/{args.episodes} ({solved / args.episodes:.1%})")
    print(
        "Satisfied clauses: "
        f"mean={satisfied.mean():.2f}, min={satisfied.min()}, max={satisfied.max()}"
    )
    print(f"Steps: mean={steps.mean():.2f}, min={steps.min()}, max={steps.max()}")

    unsolved = [result for result in results if not result["solved"]]
    if unsolved:
        worst = min(unsolved, key=lambda result: result["satisfied_clauses"])
        print(
            "Worst unsolved instance: "
            f"{worst['instance_path']} "
            f"({worst['satisfied_clauses']}/{env.num_clauses} clauses)"
        )


if __name__ == "__main__":
    main()
