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
    """SAT branching environment with unit propagation and rich observations.

    After each agent decision, Boolean Constraint Propagation (BCP) automatically
    assigns any forced unit-clause literals. The agent only makes real branching
    decisions — the same split a DPLL solver makes.

    Observation vector (per variable × 2 + per clause + 1):
      - 20 assignment values:  -1 unassigned, 0 false, 1 true
      - 20 positive pressure:  fraction of unsatisfied clauses containing +x_i
      - 20 negative pressure:  fraction of unsatisfied clauses containing -x_i
      - 91 clause statuses:    -1 falsified, 0 unresolved, 1 satisfied
      - 91 unresolved counts:  (unresolved literals in clause) / 3, 0 if resolved
      - 1  progress ratio:     assigned_count / num_vars

    Total observation size: 20 + 20 + 20 + 91 + 91 + 1 = 243

    Compatible with sb3-contrib MaskablePPO via the action_masks() method.
    """

    metadata = {"render_modes": ["human"]}

    OBS_SIZE_PER_INSTANCE = staticmethod(
        lambda nv, nc: nv + nv + nv + nc + nc + 1
    )

    def __init__(
        self,
        data_dir: str | Path,
        num_vars: int = 20,
        num_clauses: int = 91,
        max_steps: int | None = None,
        invalid_action_penalty: float = -1.0,
        solved_bonus: float = 10.0,
        failed_penalty: float = -10.0,
        falsified_clause_penalty: float = -0.5,
        unit_clause_bonus: float = 2.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.max_steps = max_steps or num_vars * 2
        self.invalid_action_penalty = invalid_action_penalty
        self.solved_bonus = solved_bonus
        self.failed_penalty = failed_penalty
        self.falsified_clause_penalty = falsified_clause_penalty
        self.unit_clause_bonus = unit_clause_bonus

        self.instances = load_cnf_directory(data_dir)
        self._validate_instances()

        self.action_space = spaces.Discrete(self.num_vars * 2)
        obs_dim = self.num_vars * 3 + self.num_clauses * 2 + 1
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.instance: SATInstance | None = None
        self.assignment = np.full(self.num_vars, -1, dtype=np.int8)
        self.clause_statuses = np.zeros(self.num_clauses, dtype=np.int8)
        self.unresolved_counts = np.zeros(self.num_clauses, dtype=np.int8)
        self.satisfied_count = 0
        self.falsified_count = 0
        self.steps = 0
        self.assigned_count = 0
        self.bcp_assignments = 0

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

    def action_masks(self) -> np.ndarray:
        """Boolean mask: only unassigned variables have actions enabled."""
        var_free = self.assignment == -1
        mask = np.zeros(self.num_vars * 2, dtype=bool)
        mask[0::2] = var_free
        mask[1::2] = var_free
        return mask

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
        self.unresolved_counts.fill(3)
        self.satisfied_count = 0
        self.falsified_count = 0
        self.steps = 0
        self.assigned_count = 0
        self.bcp_assignments = 0

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
        old_falsified = self.falsified_count
        invalid_action = False
        self.steps += 1

        if variable_index < 0 or variable_index >= self.num_vars:
            invalid_action = True
            reward += self.invalid_action_penalty
        elif self.assignment[variable_index] != -1:
            invalid_action = True
            reward += self.invalid_action_penalty
        else:
            self._assign_variable(variable_index, value)
            self._run_bcp()

            newly_satisfied = self.satisfied_count - old_satisfied
            newly_falsified = self.falsified_count - old_falsified

            reward += float(newly_satisfied)
            reward += float(newly_falsified) * self.falsified_clause_penalty

        terminated = self.assigned_count >= self.num_vars
        truncated = not terminated and self.steps >= self.max_steps

        all_satisfied = self.satisfied_count == self.num_clauses
        if terminated or truncated:
            reward += self.solved_bonus if all_satisfied else self.failed_penalty

        info = {
            "invalid_action": invalid_action,
            "satisfied_clauses": self.satisfied_count,
            "falsified_clauses": self.falsified_count,
            "all_satisfied": all_satisfied,
            "assigned_variables": self.assigned_count,
            "bcp_assignments": self.bcp_assignments,
            "instance_path": str(self.instance.path),
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.instance is None:
            print("SATBranchingEnv(not reset)")
            return

        print(
            f"{self.instance.path.name}: "
            f"{self.satisfied_count}/{self.num_clauses} sat, "
            f"{self.falsified_count} falsified, "
            f"{self.assigned_count}/{self.num_vars} assigned "
            f"({self.bcp_assignments} via BCP)"
        )

    def _assign_variable(self, var_idx: int, value: int) -> None:
        """Assign a single variable and update clause bookkeeping."""
        assert self.instance is not None
        self.assignment[var_idx] = value
        self.assigned_count += 1

        var_mask = self.instance.variable_indices == var_idx
        clause_indices = np.where(np.any(var_mask, axis=1))[0]

        for ci in clause_indices:
            if self.clause_statuses[ci] != 0:
                continue
            for k in range(3):
                if not var_mask[ci, k]:
                    continue
                is_positive = bool(self.instance.literal_is_positive[ci, k])
                lit_true = (value == 1) == is_positive
                if lit_true:
                    self.clause_statuses[ci] = 1
                    self.satisfied_count += 1
                else:
                    self.unresolved_counts[ci] -= 1
                    if self.unresolved_counts[ci] == 0:
                        self.clause_statuses[ci] = -1
                        self.falsified_count += 1
                break

    def _run_bcp(self) -> None:
        """Unit propagation + pure literal elimination, run to fixpoint.

        Unit propagation: a clause with exactly one unresolved literal forces
        that literal to be set true.

        Pure literal elimination: a variable that only appears in one polarity
        across all unsatisfied clauses can be assigned that polarity for free.
        """
        assert self.instance is not None
        changed = True
        while changed:
            changed = False

            unit_mask = (self.clause_statuses == 0) & (self.unresolved_counts == 1)
            unit_clause_indices = np.where(unit_mask)[0]

            for ci in unit_clause_indices:
                if self.clause_statuses[ci] != 0:
                    continue
                for k in range(3):
                    vi = self.instance.variable_indices[ci, k]
                    if self.assignment[vi] != -1:
                        continue
                    is_positive = bool(self.instance.literal_is_positive[ci, k])
                    forced_value = 1 if is_positive else 0
                    self._assign_variable(vi, forced_value)
                    self.bcp_assignments += 1
                    changed = True
                    break

            if changed:
                continue

            changed = self._run_pure_literal_elimination()

    def _run_pure_literal_elimination(self) -> bool:
        """Assign variables that appear in only one polarity in unsatisfied clauses."""
        assert self.instance is not None
        unsat_mask = self.clause_statuses == 0
        unsat_indices = np.where(unsat_mask)[0]
        if len(unsat_indices) == 0:
            return False

        appears_pos = np.zeros(self.num_vars, dtype=bool)
        appears_neg = np.zeros(self.num_vars, dtype=bool)

        for ci in unsat_indices:
            for k in range(3):
                vi = self.instance.variable_indices[ci, k]
                if self.assignment[vi] != -1:
                    continue
                if self.instance.literal_is_positive[ci, k]:
                    appears_pos[vi] = True
                else:
                    appears_neg[vi] = True

        unassigned = self.assignment == -1
        pure_pos = unassigned & appears_pos & ~appears_neg
        pure_neg = unassigned & appears_neg & ~appears_pos

        changed = False
        for vi in np.where(pure_pos)[0]:
            if self.assignment[vi] != -1:
                continue
            self._assign_variable(int(vi), 1)
            self.bcp_assignments += 1
            changed = True

        for vi in np.where(pure_neg)[0]:
            if self.assignment[vi] != -1:
                continue
            self._assign_variable(int(vi), 0)
            self.bcp_assignments += 1
            changed = True

        return changed

    def _observation(self) -> np.ndarray:
        assert self.instance is not None

        assign_obs = self.assignment.astype(np.float32)

        pos_pressure = np.zeros(self.num_vars, dtype=np.float32)
        neg_pressure = np.zeros(self.num_vars, dtype=np.float32)

        unsat_mask = self.clause_statuses == 0
        n_unsat = max(int(np.count_nonzero(unsat_mask)), 1)

        for ci in np.where(unsat_mask)[0]:
            for k in range(3):
                vi = self.instance.variable_indices[ci, k]
                if self.assignment[vi] != -1:
                    continue
                if self.instance.literal_is_positive[ci, k]:
                    pos_pressure[vi] += 1.0
                else:
                    neg_pressure[vi] += 1.0

        pos_pressure /= n_unsat
        neg_pressure /= n_unsat

        clause_obs = self.clause_statuses.astype(np.float32)

        unresolved_obs = np.where(
            self.clause_statuses == 0,
            self.unresolved_counts.astype(np.float32) / 3.0,
            0.0,
        ).astype(np.float32)

        progress = np.array(
            [self.assigned_count / self.num_vars], dtype=np.float32
        )

        return np.concatenate([
            assign_obs,
            pos_pressure,
            neg_pressure,
            clause_obs,
            unresolved_obs,
            progress,
        ])

    def _update_clause_statuses(self) -> None:
        """Full recompute — only used if needed for debugging."""
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
        self.falsified_count = int(np.count_nonzero(falsified_clauses))

        n_unresolved = np.sum(~assigned_literals, axis=1)
        self.unresolved_counts = n_unresolved.astype(np.int8)

    def _count_satisfied_clauses(self) -> int:
        return self.satisfied_count

    def _all_clauses_satisfied(self) -> bool:
        return self.satisfied_count == self.num_clauses
