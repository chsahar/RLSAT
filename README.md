# saharSAT

A small reinforcement-learning playground for learning SAT branching policies on
SATLIB `uf20-91` (random 3-SAT, 20 vars / 91 clauses, all satisfiable).

The repo contains:

- a Gymnasium environment with unit propagation (BCP) and pure-literal elimination
- training via `sb3-contrib` `MaskablePPO` (action masking is required)
- a FastAPI service for triggering training, polling progress, listing checkpoints, and running evaluation
- a React + Vite UI that drives the API

---

## 1. Environment

Implemented in [src/saharsat/envs.py](src/saharsat/envs.py).

- **One episode = one random SAT instance** sampled from the data directory on `reset()`.
- **Action space**: `Discrete(num_vars * 2)` — action `2*i` sets `x_{i+1} = False`, `2*i+1` sets `x_{i+1} = True`. With `num_vars=20`, that's `Discrete(40)`.
- **Observation** (`Box(-1, 1, shape=(243,), dtype=float32)` for the default `20 vars / 91 clauses`):
  - `num_vars` assignment values: `-1` unassigned, `0` false, `1` true
  - `num_vars` positive pressure: fraction of unsatisfied clauses containing `+x_i`
  - `num_vars` negative pressure: fraction of unsatisfied clauses containing `-x_i`
  - `num_clauses` clause statuses: `-1` falsified, `0` unresolved, `1` satisfied
  - `num_clauses` unresolved counts: `(unresolved literals)/3`, or `0` if the clause is already resolved
  - `1` progress ratio: `assigned_count / num_vars`
- **BCP**: after every agent step, unit propagation and pure-literal elimination run to fixpoint and assign more variables for free. The agent only makes the genuine branching decisions a DPLL solver would face.
- **Action mask**: `env.action_masks()` returns `True` only for unassigned variables; pass it into `MaskablePPO.predict(...)`.
- **Reward** (per step):
  - `+1 * (newly satisfied clauses)`
  - `falsified_clause_penalty * (newly falsified clauses)` (default `-0.5`)
  - `invalid_action_penalty` on attempting to assign an already-assigned variable (default `-1.0`)
  - terminal bonus: `+solved_bonus` if all clauses satisfied (default `+10`), else `failed_penalty` (default `-10`)
  - Note: `unit_clause_bonus` is a knob exposed by the API/CLI/env but currently has no effect on reward — see *Known limitations* below.

---

## 2. Prerequisites

- Python ≥ 3.10
- Node ≥ 18 (only for the UI)
- `tar`, `curl` (for downloading the dataset)
- A C/C++ build toolchain if your platform doesn't have wheels for `numpy` / `torch`

---

## 3. Install (Python)

Use a fresh virtualenv (or your favorite manager — conda, uv, poetry, all fine).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`pyproject.toml` declares: `fastapi`, `uvicorn[standard]`, `gymnasium`, `numpy`, `stable-baselines3`, `sb3-contrib`. `torch` comes in transitively via `stable-baselines3`.

The package installs as `saharsat`. With `pip install -e .` you don't need `PYTHONPATH=src` anymore, but the scripts in this README keep it for users who skip the editable install.

---

## 4. Dataset

Download SATLIB `uf20-91` (1000 satisfiable random 3-SAT instances):

```bash
mkdir -p data/uf20-91
curl -L https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz \
  -o /tmp/uf20-91.tar.gz
tar -xzf /tmp/uf20-91.tar.gz -C data/uf20-91 --strip-components=0
find data/uf20-91 -name "*.cnf" | wc -l   # should print 1000
```

**Important about layout.** All scripts and the API default to `--data-dir data/uf20-91`. The 1000 `.cnf` files must sit directly inside `data/uf20-91/` (no nested folder). If your checkout has the files in `data/` instead, either:

```bash
mkdir -p data/uf20-91 && mv data/uf20-*.cnf data/uf20-91/
```

or pass `--data-dir data/` to every script.

---

## 5. Sanity-check the environment

```bash
PYTHONPATH=src python scripts/check_env.py --data-dir data/uf20-91
```

Runs Gymnasium's `check_env` and prints the observation shape + a sampled instance path. Expect:

```
Observation shape: (243,)
Sample instance: data/uf20-91/uf20-0001.cnf
Environment check passed.
```

---

## 6. Train PPO

```bash
PYTHONPATH=src python scripts/train_ppo.py \
  --data-dir data/uf20-91 \
  --timesteps 100000
```

Trained checkpoints land in `models/` with the timestep budget and timestamp baked into the filename:

```
models/ppo_uf20_91_100k_YYYYMMDD_HHMMSS.zip
```

All reward and PPO knobs are CLI flags:

```bash
PYTHONPATH=src python scripts/train_ppo.py \
  --data-dir data/uf20-91 \
  --timesteps 200000 \
  --invalid-action-penalty -1 \
  --solved-bonus 50 \
  --failed-penalty -50 \
  --falsified-clause-penalty -0.5 \
  --unit-clause-bonus 2 \
  --n-envs 8 \
  --no-lr-decay \
  --model-out models/my_run.zip
```

Defaults live in `TrainingConfig` ([src/saharsat/training.py](src/saharsat/training.py)): `n_steps=2048`, `batch_size=128`, `gamma=1.0`, `learning_rate=3e-4` with linear decay, `ent_coef=0.02`, `n_epochs=10`, `MlpPolicy` with two `Tanh` MLP heads of `[256, 256, 128]`.

Training uses `SubprocVecEnv` with `start_method="spawn"` and `n_envs = min(cpu_count, 8)` by default.

---

## 7. Evaluate a checkpoint

```bash
PYTHONPATH=src python scripts/evaluate_agent.py \
  --data-dir data/uf20-91 \
  --model models/ppo_uf20_91_100k_YYYYMMDD_HHMMSS.zip \
  --episodes 100 \
  --max-steps 40
```

Prints solve count, solve rate, mean/min/max satisfied clauses, mean/min/max steps, and the worst unsolved instance.

---

## 8. REST API

Start the server (defaults to port **8000**):

```bash
PYTHONPATH=src python scripts/run_api.py
# -> http://127.0.0.1:8000  (uvicorn, --reload)
```

Endpoints (defined in [src/saharsat/api.py](src/saharsat/api.py)):

| Method | Path                  | Purpose                                                              |
|--------|-----------------------|----------------------------------------------------------------------|
| GET    | `/api/health`         | Liveness probe.                                                      |
| GET    | `/api/environment`    | Environment shape (instances, vars, clauses, action/obs dims).       |
| GET    | `/api/models`         | List `models/*.zip`, newest first, with size + mtime.                |
| POST   | `/api/train`          | Start a training job. 409 if another job is running.                 |
| GET    | `/api/train/{job_id}` | Status, progress, and final model path for a job.                    |
| GET    | `/api/train`          | List all known jobs in this process.                                 |
| POST   | `/api/evaluate`       | Synchronously evaluate a checkpoint over N episodes.                 |

Relative paths in requests (e.g. `data/uf20-91`, `models/foo.zip`) are resolved against the project root regardless of where uvicorn was launched.

CORS is restricted to `http://localhost(:PORT)` and `http://127.0.0.1(:PORT)`.

---

## 9. React / Vite UI

```bash
cd frontend
npm install
npm run dev
# -> http://127.0.0.1:5173
```

**Heads up: API port.** [frontend/src/App.jsx](frontend/src/App.jsx) currently defaults `API_BASE` to `http://127.0.0.1:8009`, but the API runs on **`8000`**. Either fix that default in source, or run Vite with the override:

```bash
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

The UI lets you inspect the environment shape, set PPO + reward knobs, start one training job, watch progress + mean episode reward / length, list saved models, and run evaluation on a selected checkpoint.

---

## 10. Project layout

```
src/saharsat/
  __init__.py        # re-exports SATBranchingEnv
  envs.py            # DIMACS loader + Gymnasium env with BCP
  training.py        # TrainingConfig, ProgressCallback, train_ppo()
  model_io.py        # MaskablePPO loader + torch.load workaround
  api.py             # FastAPI app, job runner, eval runner
scripts/
  check_env.py       # gymnasium check_env on the env
  train_ppo.py       # CLI for training
  evaluate_agent.py  # CLI for evaluation
  run_api.py         # uvicorn entrypoint
frontend/            # Vite + React UI
data/uf20-91/        # SATLIB instances (you populate this)
models/              # PPO checkpoints (gitignored)
```

---

## 11. Known limitations / gotchas

- **`unit_clause_bonus` is currently a no-op.** It's plumbed through the env, training config, CLI, and API schema, but `SATBranchingEnv.step()` never reads it. Setting it does nothing today. Leaving in for forward-compat; remove the flag if you don't plan to wire it up.
- **Episodes don't terminate on conflict.** When a clause becomes falsified, the env keeps stepping until all vars are assigned or `max_steps` truncates. A DPLL-style early exit would cut wasted work.
- **`gamma=1.0`** with possible truncation at `max_steps=40` can bias value estimation if episodes truncate often. For 20-var instances this is rare in practice.
- **Mean episode reward** is total reward per finished episode, not reward per step.
- **PPO may run slightly past `--timesteps`** because it completes full rollout chunks.
- **`models/*.zip` is gitignored**; the dataset is not.
