# saharSAT

Minimal PPO starter project for learning SAT assignment policies on SATLIB
`uf20-91` random 3-SAT instances.

The project currently includes:

- a Gymnasium environment for SATLIB `uf20-91`
- Stable-Baselines3 PPO training
- a FastAPI REST API for training, evaluation, and model listing
- a React/Vite UI for configuring rewards and tracking training

## Environment

The environment is deliberately simple:

- one episode samples one satisfiable `uf20-91` CNF instance
- the agent assigns one variable per step
- action space is `Discrete(40)`: `x1=False`, `x1=True`, ..., `x20=True`
- observation is a vector of 111 floats:
  - 20 assignment entries: `-1` unassigned, `0` false, `1` true
  - 91 clause status entries: `-1` falsified, `0` unresolved, `1` satisfied
- reward is shaped by the change in satisfied clauses, with a terminal bonus

Clause status calculation is vectorized with NumPy and cached per step.

## Setup

```bash
conda activate omni
pip install -e .
```

## Dataset

Download and extract SATLIB `uf20-91`:

```bash
mkdir -p data
curl -L https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz -o data/uf20-91.tar.gz
tar -xzf data/uf20-91.tar.gz -C data
find data/uf20-91 -name "*.cnf" | wc -l
```

The final command should print:

```text
1000
```

The dataset directory should be committed if you want the benchmark in the repo.

## Check the environment

```bash
PYTHONPATH=src python scripts/check_env.py --data-dir data/uf20-91
```

## Train PPO

```bash
PYTHONPATH=src python scripts/train_ppo.py --data-dir data/uf20-91 --timesteps 100000
```

The trained model is saved with the timestep count and run timestamp in the name:

```text
models/ppo_uf20_91_100k_YYYYMMDD_HHMMSS.zip
```

Reward parameters can be configured from the CLI:

```bash
PYTHONPATH=src python scripts/train_ppo.py \
  --data-dir data/uf20-91 \
  --timesteps 100000 \
  --invalid-action-penalty -1 \
  --solved-bonus 100 \
  --failed-penalty -100
```

## Evaluate a Model

```bash
PYTHONPATH=src python scripts/evaluate_agent.py \
  --data-dir data/uf20-91 \
  --model models/ppo_uf20_91_100k_YYYYMMDD_HHMMSS.zip \
  --episodes 100 \
  --max-steps 40
```

Evaluation reports solve count, solve rate, satisfied clauses, and steps.

## Run the REST API

```bash
conda activate omni
PYTHONPATH=src python scripts/run_api.py
```

The API runs at:

```text
http://127.0.0.1:8000
```

Useful endpoints:

```text
GET  /api/environment
GET  /api/models
POST /api/train
GET  /api/train/{job_id}
POST /api/evaluate
```

The API resolves relative paths from the project root, so `data/uf20-91` works
from the UI and API even if the server reloads.

## Run the React UI

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

The UI runs at:

```text
http://127.0.0.1:5173
```

The UI lets you:

- inspect the environment shape
- configure PPO training parameters
- configure reward values
- start one training job at a time
- track progress, mean episode reward, and mean episode length
- list saved models
- evaluate a selected model on SATLIB instances

## Notes

- `Mean episode reward` is the average total reward over recent completed
  episodes, not reward per step.
- PPO may run slightly past the requested timestep count because it completes
  full rollout chunks.
- Saved models are ignored by git with `models/*.zip`.
- The dataset is not ignored by git.
