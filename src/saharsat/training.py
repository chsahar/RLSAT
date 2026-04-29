from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from saharsat import SATBranchingEnv


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_timesteps(timesteps: int) -> str:
    if timesteps >= 1_000_000 and timesteps % 1_000_000 == 0:
        return f"{timesteps // 1_000_000}m"
    if timesteps >= 1_000 and timesteps % 1_000 == 0:
        return f"{timesteps // 1_000}k"
    return str(timesteps)


@dataclass
class TrainingConfig:
    data_dir: str = "data/uf20-91"
    timesteps: int = 100_000
    seed: int = 0
    max_steps: int = 40
    invalid_action_penalty: float = -1.0
    solved_bonus: float = 100.0
    failed_penalty: float = -100.0
    n_steps: int = 512
    batch_size: int = 64
    gamma: float = 0.99
    model_out: str | None = None


@dataclass
class TrainingProgress:
    total_timesteps: int
    current_timesteps: int = 0
    last_mean_reward: float | None = None
    last_mean_episode_length: float | None = None
    updated_at: str = field(default_factory=current_timestamp)


class ProgressCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        on_progress: Callable[[TrainingProgress], None],
    ) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps
        self.on_progress = on_progress

    def _on_step(self) -> bool:
        if self.n_calls % 100 != 0:
            return True

        self.on_progress(self._progress())
        return True

    def _on_rollout_end(self) -> None:
        self.on_progress(self._progress())

    def _on_training_end(self) -> None:
        self.on_progress(self._progress())

    def _progress(self) -> TrainingProgress:
        return TrainingProgress(
            total_timesteps=self.total_timesteps,
            current_timesteps=self.num_timesteps,
            last_mean_reward=self._mean_episode_info("r")
            or self._safe_float("rollout/ep_rew_mean"),
            last_mean_episode_length=self._mean_episode_info("l")
            or self._safe_float("rollout/ep_len_mean"),
        )

    def _mean_episode_info(self, key: str) -> float | None:
        values = [
            float(episode[key])
            for episode in self.model.ep_info_buffer
            if key in episode
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def _safe_float(self, name: str) -> float | None:
        value = self.logger.name_to_value.get(name)
        if value is None:
            return None
        return float(value)


def default_model_path(timesteps: int) -> str:
    return f"models/ppo_uf20_91_{format_timesteps(timesteps)}_{current_timestamp()}"


def train_ppo(
    config: TrainingConfig,
    on_progress: Callable[[TrainingProgress], None] | None = None,
) -> Path:
    env = Monitor(
        SATBranchingEnv(
            config.data_dir,
            max_steps=config.max_steps,
            invalid_action_penalty=config.invalid_action_penalty,
            solved_bonus=config.solved_bonus,
            failed_penalty=config.failed_penalty,
            seed=config.seed,
        )
    )
    check_env(env.unwrapped)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
    )

    callback = None
    if on_progress is not None:
        callback = ProgressCallback(config.timesteps, on_progress)

    model.learn(total_timesteps=config.timesteps, callback=callback)

    model_path = normalize_model_path(Path(config.model_out or default_model_path(config.timesteps)))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{model_path.stem}_",
        suffix=".zip",
        dir=model_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        model.save(temp_path.with_suffix(""))
        temp_path.replace(model_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return model_path


def normalize_model_path(path: Path) -> Path:
    if path.suffix == ".zip":
        return path
    return path.with_suffix(".zip")
