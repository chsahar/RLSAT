from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SATInstance:
    path: Path
    num_vars: int
    clauses: np.ndarray
    variable_indices: np.ndarray
    literal_is_positive: np.ndarray


def load_dimacs_cnf(path: str | Path) -> SATInstance:
    """Load a DIMACS CNF file into an integer clause matrix."""
    cnf_path = Path(path)
    num_vars: int | None = None
    num_clauses: int | None = None
    clauses: list[list[int]] = []
    current_clause: list[int] = []

    with cnf_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c") or line.startswith("%"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) < 4 or parts[1].lower() != "cnf":
                    raise ValueError(f"Invalid DIMACS problem line in {cnf_path}: {line}")
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
                continue

            for token in line.split():
                literal = int(token)
                if literal == 0:
                    if current_clause:
                        clauses.append(current_clause)
                        current_clause = []
                    continue
                current_clause.append(literal)

    if current_clause:
        raise ValueError(f"Unterminated clause in {cnf_path}")
    if num_vars is None or num_clauses is None:
        raise ValueError(f"Missing DIMACS problem line in {cnf_path}")
    if len(clauses) != num_clauses:
        raise ValueError(
            f"{cnf_path} declares {num_clauses} clauses but contains {len(clauses)}"
        )

    clause_lengths = {len(clause) for clause in clauses}
    if clause_lengths != {3}:
        raise ValueError(f"{cnf_path} is expected to contain only 3-CNF clauses")

    clause_array = np.asarray(clauses, dtype=np.int16)
    return SATInstance(
        path=cnf_path,
        num_vars=num_vars,
        clauses=clause_array,
        variable_indices=(np.abs(clause_array) - 1).astype(np.intp),
        literal_is_positive=clause_array > 0,
    )


def load_cnf_directory(data_dir: str | Path) -> list[SATInstance]:
    """Load all .cnf files from a directory."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"CNF directory does not exist: {root}")

    paths = sorted(root.glob("*.cnf"))
    if not paths:
        raise FileNotFoundError(f"No .cnf files found in {root}")

    return [load_dimacs_cnf(path) for path in paths]


class SATBranchingEnv(gym.Env):
    """Simple SAT assignment environment for fixed-size SATLIB uf20-91 formulas."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_dir: str | Path,
        num_vars: int = 20,
        num_clauses: int = 91,
        max_steps: int | None = None,
        invalid_action_penalty: float = -1.0,
        solved_bonus: float = 100.0,
        failed_penalty: float = -100.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.max_steps = max_steps or num_vars * 2
        self.invalid_action_penalty = invalid_action_penalty
        self.solved_bonus = solved_bonus
        self.failed_penalty = failed_penalty

        self.instances = load_cnf_directory(data_dir)
        self._validate_instances()

        self.action_space = spaces.Discrete(self.num_vars * 2)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_vars + self.num_clauses,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.instance: SATInstance | None = None
        self.assignment = np.full(self.num_vars, -1, dtype=np.int8)
        self.clause_statuses = np.zeros(self.num_clauses, dtype=np.int8)
        self.satisfied_count = 0
        self.steps = 0
        self.assigned_count = 0

    def _validate_instances(self) -> None:
        for instance in self.instances:
            if instance.num_vars != self.num_vars:
                raise ValueError(
                    f"{instance.path} has {instance.num_vars} variables, "
                    f"expected {self.num_vars}"
                )
            if instance.clauses.shape != (self.num_clauses, 3):
                raise ValueError(
                    f"{instance.path} has clause shape {instance.clauses.shape}, "
                    f"expected ({self.num_clauses}, 3)"
                )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        instance_index = int(self._rng.integers(len(self.instances)))
        self.instance = self.instances[instance_index]
        self.assignment.fill(-1)
        self.clause_statuses.fill(0)
        self.satisfied_count = 0
        self.steps = 0
        self.assigned_count = 0

        info = {
            "instance_path": str(self.instance.path),
            "satisfied_clauses": self.satisfied_count,
        }
        return self._observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.instance is None:
            raise RuntimeError("Call reset() before step().")

        action = int(action)
        variable_index = action // 2
        value = action % 2
        reward = 0.0
        old_satisfied = self.satisfied_count
        invalid_action = False
        self.steps += 1

        if variable_index < 0 or variable_index >= self.num_vars:
            invalid_action = True
            reward += self.invalid_action_penalty
        elif self.assignment[variable_index] != -1:
            invalid_action = True
            reward += self.invalid_action_penalty
        else:
            self.assignment[variable_index] = value
            self.assigned_count += 1
            self._update_clause_statuses()
            reward += float(self.satisfied_count - old_satisfied)

        terminated = self.assigned_count >= self.num_vars
        truncated = not terminated and self.steps >= self.max_steps

        all_satisfied = self.satisfied_count == self.num_clauses
        if terminated or truncated:
            reward += self.solved_bonus if all_satisfied else self.failed_penalty

        info = {
            "invalid_action": invalid_action,
            "satisfied_clauses": self.satisfied_count,
            "all_satisfied": all_satisfied,
            "assigned_variables": self.assigned_count,
            "instance_path": str(self.instance.path),
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.instance is None:
            print("SATBranchingEnv(not reset)")
            return

        print(
            f"{self.instance.path.name}: "
            f"{self.satisfied_count}/{self.num_clauses} clauses satisfied, "
            f"{self.assigned_count}/{self.num_vars} variables assigned"
        )

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.assignment.astype(np.float32),
                self.clause_statuses.astype(np.float32),
            ]
        )

    def _update_clause_statuses(self) -> None:
        assert self.instance is not None
        assigned_values = self.assignment[self.instance.variable_indices]
        assigned_literals = assigned_values != -1
        true_literals = assigned_literals & np.where(
            self.instance.literal_is_positive,
            assigned_values == 1,
            assigned_values == 0,
        )
        satisfied_clauses = np.any(true_literals, axis=1)
        falsified_clauses = np.all(assigned_literals, axis=1) & ~satisfied_clauses

        self.clause_statuses.fill(0)
        self.clause_statuses[satisfied_clauses] = 1
        self.clause_statuses[falsified_clauses] = -1
        self.satisfied_count = int(np.count_nonzero(satisfied_clauses))

    def _count_satisfied_clauses(self) -> int:
        return self.satisfied_count

    def _all_clauses_satisfied(self) -> bool:
        return self.satisfied_count == self.num_clauses
