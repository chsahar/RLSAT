from __future__ import annotations

import argparse

from saharsat.training import TrainingConfig, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--invalid-action-penalty", type=float, default=-1.0)
    parser.add_argument("--solved-bonus", type=float, default=100.0)
    parser.add_argument("--failed-penalty", type=float, default=-100.0)
    parser.add_argument("--model-out", default=None)
    args = parser.parse_args()

    model_path = train_ppo(
        TrainingConfig(
            data_dir=args.data_dir,
            timesteps=args.timesteps,
            seed=args.seed,
            max_steps=args.max_steps,
            invalid_action_penalty=args.invalid_action_penalty,
            solved_bonus=args.solved_bonus,
            failed_penalty=args.failed_penalty,
            model_out=args.model_out,
        )
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
