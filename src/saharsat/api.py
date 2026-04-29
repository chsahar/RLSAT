from __future__ import annotations

import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

from saharsat import SATBranchingEnv
from saharsat.model_io import load_ppo_model
from saharsat.training import TrainingConfig, TrainingProgress, train_ppo

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RewardConfig(BaseModel):
    invalid_action_penalty: float = -1.0
    solved_bonus: float = 100.0
    failed_penalty: float = -100.0


class TrainRequest(BaseModel):
    data_dir: str = "data/uf20-91"
    timesteps: int = Field(default=100_000, ge=1)
    seed: int = 0
    max_steps: int = Field(default=40, ge=1)
    rewards: RewardConfig = Field(default_factory=RewardConfig)
    n_steps: int = Field(default=512, ge=1)
    batch_size: int = Field(default=64, ge=1)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    model_out: str | None = None


class EvalRequest(BaseModel):
    data_dir: str = "data/uf20-91"
    model: str
    episodes: int = Field(default=100, ge=1)
    max_steps: int = Field(default=40, ge=1)
    seed: int = 0
    rewards: RewardConfig = Field(default_factory=RewardConfig)


class JobStatus(BaseModel):
    id: str
    status: Literal["idle", "running", "completed", "failed"]
    config: TrainRequest
    progress: dict | None = None
    model_path: str | None = None
    error: str | None = None


app = FastAPI(title="saharSAT API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, JobStatus] = {}
_jobs_lock = threading.Lock()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/environment")
def environment(data_dir: str = "data/uf20-91") -> dict:
    resolved_data_dir = resolve_project_path(data_dir)
    env = SATBranchingEnv(resolved_data_dir)
    return {
        "data_dir": str(resolved_data_dir),
        "instances": int(len(env.instances)),
        "num_vars": int(env.num_vars),
        "num_clauses": int(env.num_clauses),
        "action_space": int(env.action_space.n),
        "observation_shape": [int(size) for size in env.observation_space.shape],
        "default_max_steps": int(env.max_steps),
    }


@app.get("/api/models")
def models() -> dict:
    model_dir = resolve_project_path("models")
    items = []
    if model_dir.exists():
        for path in sorted(model_dir.glob("*.zip"), reverse=True):
            stat = path.stat()
            items.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size_bytes": int(stat.st_size),
                    "modified_time": float(stat.st_mtime),
                }
            )
    return {"models": items}


@app.post("/api/train", response_model=JobStatus)
def start_training(request: TrainRequest) -> JobStatus:
    with _jobs_lock:
        running = [job for job in _jobs.values() if job.status == "running"]
        if running:
            raise HTTPException(
                status_code=409,
                detail=f"Training job already running: {running[0].id}",
            )

        job_id = str(uuid.uuid4())
        job = JobStatus(id=job_id, status="running", config=request)
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_training_job,
        args=(job_id, request),
        daemon=True,
    )
    thread.start()
    return job


@app.get("/api/train/{job_id}", response_model=JobStatus)
def training_status(job_id: str) -> JobStatus:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found")
        return job


@app.get("/api/train", response_model=list[JobStatus])
def training_jobs() -> list[JobStatus]:
    with _jobs_lock:
        return list(_jobs.values())


@app.post("/api/evaluate")
def evaluate(request: EvalRequest) -> dict:
    env = SATBranchingEnv(
        resolve_project_path(request.data_dir),
        max_steps=request.max_steps,
        invalid_action_penalty=request.rewards.invalid_action_penalty,
        solved_bonus=request.rewards.solved_bonus,
        failed_penalty=request.rewards.failed_penalty,
        seed=request.seed,
    )
    model_path = resolve_project_path(request.model)
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model checkpoint does not exist: {model_path}",
        )

    try:
        model = load_ppo_model(model_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not load model checkpoint {model_path}. "
                "The file may be incomplete or corrupted. Train a new model "
                "or choose another checkpoint."
            ),
        ) from exc

    results = [
        _run_eval_episode(model, env, seed=request.seed + episode)
        for episode in range(request.episodes)
    ]
    solved = sum(result["solved"] for result in results)
    satisfied = np.asarray([result["satisfied_clauses"] for result in results])
    steps = np.asarray([result["steps"] for result in results])

    return {
        "episodes": int(request.episodes),
        "solved": int(solved),
        "solve_rate": float(solved / request.episodes),
        "satisfied_clauses": {
            "mean": float(satisfied.mean()),
            "min": int(satisfied.min()),
            "max": int(satisfied.max()),
        },
        "steps": {
            "mean": float(steps.mean()),
            "min": int(steps.min()),
            "max": int(steps.max()),
        },
        "results": results,
    }


def _run_training_job(job_id: str, request: TrainRequest) -> None:
    def on_progress(progress: TrainingProgress) -> None:
        with _jobs_lock:
            job = _jobs[job_id]
            job.progress = to_jsonable(asdict(progress))

    try:
        model_path = train_ppo(
            TrainingConfig(
                data_dir=str(resolve_project_path(request.data_dir)),
                timesteps=request.timesteps,
                seed=request.seed,
                max_steps=request.max_steps,
                invalid_action_penalty=request.rewards.invalid_action_penalty,
                solved_bonus=request.rewards.solved_bonus,
                failed_penalty=request.rewards.failed_penalty,
                n_steps=request.n_steps,
                batch_size=request.batch_size,
                gamma=request.gamma,
                model_out=(
                    str(resolve_project_path(request.model_out))
                    if request.model_out is not None
                    else None
                ),
            ),
            on_progress=on_progress,
        )
    except Exception as exc:
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "failed"
            job.error = str(exc)
        return

    with _jobs_lock:
        job = _jobs[job_id]
        job.status = "completed"
        job.model_path = str(model_path)


def _run_eval_episode(model: PPO, env: SATBranchingEnv, seed: int) -> dict:
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    final_info = info

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, final_info = env.step(int(action))
        total_reward += float(reward)

    return {
        "solved": bool(final_info["all_satisfied"]),
        "satisfied_clauses": int(final_info["satisfied_clauses"]),
        "assigned_variables": int(final_info["assigned_variables"]),
        "steps": int(env.steps),
        "total_reward": float(total_reward),
        "instance_path": final_info["instance_path"],
    }


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value
