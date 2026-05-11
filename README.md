# RLSAT

Learning a DPLL branching heuristic with PPO on SATLIB `uf20-91`
(random 3-SAT, 20 vars / 91 clauses, all satisfiable).

The repo ships:

- a Gymnasium environment with built-in unit propagation and pure-literal elimination
- training via `sb3-contrib` `MaskablePPO` with action masking
- a FastAPI service for training jobs, progress polling, and evaluation
- a React + Vite UI that drives the API

The paper sources for the accompanying write-up live under [paper/](paper/).

## Approach

A SAT solver makes two kinds of moves: forced (unit propagation, pure-literal
elimination) and free (which variable + polarity to branch on next). The env
hard-codes the forced moves and only exposes branching decisions to the agent,
so PPO is learning a variable-selection heuristic on top of a fixed DPLL kernel,
not a SAT solver from scratch.

- **Action space** — `Discrete(2 * num_vars)` (40 by default): `2*i` sets
  `x_{i+1} = False`, `2*i + 1` sets it to `True`.
- **Observation** — `Box(-1, 1, shape=(243,), float32)` for `20 vars / 91 clauses`:
  per-variable assignment + positive/negative literal pressure, per-clause status
  + unresolved-literal ratio, and a global progress scalar.
- **Action mask** — `env.action_masks()` zeros out already-assigned variables;
  passed into `MaskablePPO.predict(...)`.
- **Reward** — `+1` per newly satisfied clause, `falsified_clause_penalty * Δfalsified`,
  `invalid_action_penalty` on attempts to reassign a variable, terminal
  `solved_bonus` / `failed_penalty`.

## Requirements

- Python ≥ 3.10
- Node ≥ 18 (UI only)
- `curl`, `tar` (dataset download)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Dataset

The dataset is fetched at runtime, not stored in the repo. Run:

```bash
scripts/download_data.sh
```

This downloads SATLIB `uf20-91`, extracts the 1000 `.cnf` files into
`data/uf20-91/`, and verifies the count. All scripts and the API default to
`--data-dir data/uf20-91`.

## CLI

```bash
# sanity-check the env
python scripts/check_env.py --data-dir data/uf20-91

# train
python scripts/train_ppo.py --data-dir data/uf20-91 --timesteps 100000

# evaluate
python scripts/evaluate_agent.py \
  --data-dir data/uf20-91 \
  --model models/ppo_uf20_91_100k_YYYYMMDD_HHMMSS.zip \
  --episodes 100
```

Checkpoints land under `models/` and are gitignored. Run
`python scripts/train_ppo.py --help` for the full flag list (PPO knobs and
reward shape).

## API

```bash
python scripts/run_api.py
# -> http://127.0.0.1:8000
```

Configurable via env vars:

| Variable                | Default     | Purpose             |
|-------------------------|-------------|---------------------|
| `RLSAT_API_HOST`        | `127.0.0.1` | Bind host           |
| `RLSAT_API_PORT`        | `8000`      | Bind port           |
| `RLSAT_API_RELOAD`      | `0`         | `1` for autoreload  |

Endpoints (see [src/rlsat/api.py](src/rlsat/api.py)):

| Method | Path                  | Purpose                                       |
|--------|-----------------------|-----------------------------------------------|
| GET    | `/api/health`         | Liveness probe.                               |
| GET    | `/api/environment`    | Env shape (instances, vars, clauses, dims).   |
| GET    | `/api/models`         | List `models/*.zip`, newest first.            |
| POST   | `/api/train`          | Start a training job (one at a time).         |
| GET    | `/api/train/{job_id}` | Job status + progress + model path.           |
| GET    | `/api/train`          | List jobs known to this process.              |
| POST   | `/api/evaluate`       | Synchronous evaluation over N episodes.       |

Relative paths in requests resolve against the project root. CORS is restricted
to `localhost` / `127.0.0.1`.

## UI

```bash
cd frontend
npm install
npm run dev
# -> http://127.0.0.1:5173
```

Point the UI at a non-default API by setting `VITE_API_BASE`:

```bash
VITE_API_BASE=http://127.0.0.1:9000 npm run dev
```

## Layout

```
src/rlsat/
  envs.py            # DIMACS loader + Gymnasium env with BCP
  training.py        # TrainingConfig, ProgressCallback, train_ppo()
  model_io.py        # MaskablePPO loader + torch.load shim
  api.py             # FastAPI app + job runner
scripts/
  check_env.py
  train_ppo.py
  evaluate_agent.py
  run_api.py
  download_data.sh
frontend/            # Vite + React UI
paper/               # NeurIPS-style write-up
data/                # populated by download_data.sh (gitignored)
models/              # PPO checkpoints (gitignored)
```

## Notes

- Episodes don't terminate the moment a clause becomes falsified; the agent
  keeps stepping until all variables are assigned or `max_steps` truncates.
- PPO may run slightly past `--timesteps` because rollouts complete in full
  chunks.
- "Mean episode reward" is total reward per finished episode, not per step.
