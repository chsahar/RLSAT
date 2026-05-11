from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from saharsat import SATBranchingEnv

POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    activation_fn=__import__("torch").nn.Tanh,
)

DEFAULT_N_ENVS = min(os.cpu_count() or 1, 8)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_timesteps(timesteps: int) -> str:
    if timesteps >= 1_000_000 and timesteps % 1_000_000 == 0:
        return f"{timesteps // 1_000_000}m"
    if timesteps >= 1_000 and timesteps % 1_000 == 0:
        return f"{timesteps // 1_000}k"
    return str(timesteps)


def linear_schedule(initial_lr: float) -> Callable[[float], float]:
    """Linear LR decay from initial_lr to 0 over training."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return schedule


@dataclass
class TrainingConfig:
    data_dir: str = "data/uf20-91"
    timesteps: int = 100_000
    seed: int = 0
    max_steps: int = 40
    invalid_action_penalty: float = -1.0
    solved_bonus: float = 10.0
    failed_penalty: float = -10.0
    falsified_clause_penalty: float = -0.5
    n_steps: int = 2048
    batch_size: int = 128
    gamma: float = 1.0
    learning_rate: float = 3e-4
    ent_coef: float = 0.02
    n_envs: int = DEFAULT_N_ENVS
    lr_decay: bool = True
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
        mean_reward = self._mean_episode_info("r")
        if mean_reward is None:
            mean_reward = self._safe_float("rollout/ep_rew_mean")
        mean_length = self._mean_episode_info("l")
        if mean_length is None:
            mean_length = self._safe_float("rollout/ep_len_mean")
        return TrainingProgress(
            total_timesteps=self.total_timesteps,
            current_timesteps=self.num_timesteps,
            last_mean_reward=mean_reward,
            last_mean_episode_length=mean_length,
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


def _make_env(config: TrainingConfig, rank: int) -> Callable[[], SATBranchingEnv]:
    """Factory for a single env instance with a unique seed per subprocess."""
    def _init() -> SATBranchingEnv:
        return SATBranchingEnv(
            config.data_dir,
            max_steps=config.max_steps,
            invalid_action_penalty=config.invalid_action_penalty,
            solved_bonus=config.solved_bonus,
            failed_penalty=config.failed_penalty,
            falsified_clause_penalty=config.falsified_clause_penalty,
            seed=config.seed + rank,
        )
    return _init


def train_ppo(
    config: TrainingConfig,
    on_progress: Callable[[TrainingProgress], None] | None = None,
) -> Path:
    n_envs = max(1, config.n_envs)
    if n_envs > 1:
        env = SubprocVecEnv(
            [_make_env(config, i) for i in range(n_envs)],
            start_method="spawn",
        )
    else:
        env = _make_env(config, 0)()

    lr = linear_schedule(config.learning_rate) if config.lr_decay else config.learning_rate

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        learning_rate=lr,
        ent_coef=config.ent_coef,
        policy_kwargs=POLICY_KWARGS,
        max_grad_norm=0.5,
        vf_coef=0.5,
        n_epochs=10,
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
