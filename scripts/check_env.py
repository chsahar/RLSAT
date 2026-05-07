from __future__ import annotations

import argparse

from gymnasium.utils.env_checker import check_env

from saharsat import SATBranchingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    env = SATBranchingEnv(args.data_dir)
    check_env(env)
    observation, info = env.reset(seed=0)
    print(f"Observation shape: {observation.shape}")
    print(f"Sample instance: {info['instance_path']}")
    print("Environment check passed.")


if __name__ == "__main__":
    main()
