"""Adaptive interaction memory matrix W, attention normalization A, and encounter dynamics (Stages 0, 1, 2).

Provides:
1. Integer memory matrix W with strictly zero diagonal (W_{ii} == 0).
2. Row-normalized attention matrix A and received attention vector r with O(N) delta maintenance.
3. Multi-population encounter eligibility matrix C_{ij} = E_{g_i, g_j}(t) with null encounter handling.
4. Candidate selection distribution P_{ij} = epsilon * U_{ij} + (1 - epsilon) * V_{ij} incorporating
   prestige bias a, matching bias c, state similarity s_{ij}, and max-subtraction log-sum-exp stability.
5. Two-step triadic closure search variant (chi > 0), reinforcing only the direct tie.
6. Memory forgetting / decay at rate delta via efficient O(N) hierarchical sampling.
7. Reciprocal memory ensemble preserving W = W^T.

Simulation Stage Mapping:
    - Stage 0: Encounter attempt throughput and memory allocation profiling.
    - Stage 1: Core interaction dynamics and attention accumulation from W = 0.
    - Stage 2: Extension campaigns (multi-population eligibility, reciprocal memory).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np


@dataclass(slots=True)
class EncounterResult:
    """Outcome of an encounter attempt.

    Attributes:
        initiator: Agent index i initiating encounter.
        target: Target agent index j, or -1 if encounter was null.
        is_null: True if encounter was null (no eligible partners), False otherwise.
    """

    initiator: int
    target: int
    is_null: bool


class MemoryManager:
    """Manages dynamic interaction memory W and attention matrix A.

    Attributes:
        N: Population size (int >= 2).
        reciprocal: Whether memory is symmetrically paired (W = W^T).
        epsilon: Exploration mixing weight in (0, 1].
        a: Prestige bias coefficient >= 0.
        c: Matching bias coefficient >= 0.
        chi: Two-step triadic closure weight in [0, 1].
        W: Integer memory matrix of shape (N, N), dtype uint32, strictly zero diagonal.
        row_sums: Out-degree memory counts s_i = sum_k W_{ik}, shape (N,), dtype uint32.
        total_memory: Total memory count S(t) = sum_{ij} W_{ij}.
        A: Row-normalized attention matrix of shape (N, N), dtype float64.
        r: Received attention vector r_j = sum_i A_{ij}, shape (N,), dtype float64.
        C_matrix: Optional global encounter eligibility matrix of shape (N, N).
    """

    def __init__(
        self,
        N: int,
        reciprocal: bool = False,
        epsilon: float = 0.1,
        a: float = 0.5,
        c: float = 1.0,
        chi: float = 0.0,
        initial_W: Optional[np.ndarray] = None,
        C_matrix: Optional[np.ndarray] = None,
        capacity: Optional[Union[int, np.ndarray]] = None,
        row_caps: Optional[Union[int, np.ndarray]] = None,
        opportunity_groups: Optional[np.ndarray] = None,
        between_eligibility: float = 0.05,
        within_eligibility: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if N < 2:
            raise ValueError(f"Population size N must be >= 2, got {N}")
        self.N = N
        self.reciprocal = bool(reciprocal)
        self.epsilon = float(epsilon)
        self.a = float(a)
        self.c = float(c)
        self.chi = float(chi)
        self.rng = rng if rng is not None else np.random.default_rng()

        if initial_W is not None:
            if initial_W.shape != (N, N):
                raise ValueError(f"initial_W shape {initial_W.shape} does not match (N, N) = ({N}, {N})")
            if np.any(np.diag(initial_W) != 0):
                raise ValueError("initial_W must have strictly zero diagonal")
            if np.any(initial_W < 0):
                raise ValueError("initial_W must contain non-negative integer values")
            if self.reciprocal and not np.array_equal(initial_W, initial_W.T):
                raise ValueError("initial_W must be symmetric when reciprocal=True")
            self.W = np.array(initial_W, dtype=np.uint32)
        else:
            self.W = np.zeros((N, N), dtype=np.uint32)

        self.row_sums = np.sum(self.W, axis=1, dtype=np.uint32)
        self.total_memory = int(np.sum(self.row_sums))

        # Row capacity caps (Realism Physics Revision)
        self.row_caps: Optional[np.ndarray] = None
        cap = capacity if capacity is not None else row_caps
        if cap is not None:
            if self.reciprocal:
                raise ValueError(
                    "Reciprocal memory with independent row capacity caps is rejected by the specification; "
                    "only directed capacity is supported."
                )
            self.set_row_caps(cap)

        self.C_matrix: Optional[np.ndarray] = None
        self.opportunity_groups: Optional[np.ndarray] = None
        if C_matrix is not None:
            self.set_eligibility_matrix(C_matrix)
        elif opportunity_groups is not None:
            self.set_opportunity_groups(
                opportunity_groups,
                between_eligibility=between_eligibility,
                within_eligibility=within_eligibility,
            )

        self.A = np.zeros((N, N), dtype=np.float64)
        self.r = np.zeros(N, dtype=np.float64)
        self._recompute_all_attention()

    def _recompute_all_attention(self) -> None:
        """Recompute full attention matrix A and received attention vector r."""
        active = self.row_sums > 0
        self.A.fill(0.0)
        if np.any(active):
            self.A[active] = self.W[active].astype(np.float64) / self.row_sums[active, np.newaxis]
        self.r = np.sum(self.A, axis=0, dtype=np.float64)

    def _update_row_attention(self, i: int) -> None:
        """Update row i of attention matrix A and delta-update vector r in O(N) time."""
        old_row = self.A[i].copy()
        s_i = float(self.row_sums[i])
        if s_i > 0.0:
            new_row = self.W[i].astype(np.float64) / s_i
        else:
            new_row = np.zeros(self.N, dtype=np.float64)
        self.A[i] = new_row
        self.r += (new_row - old_row)

    def set_eligibility_matrix(self, C_matrix: Optional[np.ndarray]) -> None:
        """Set or update the global encounter eligibility matrix C of shape (N, N).

        Args:
            C_matrix: Eligibility matrix of shape (N, N) with non-negative entries
                      and strictly zero diagonal, or None to restore unrestricted eligibility.
        """
        if C_matrix is None:
            self.C_matrix = None
            return

        if C_matrix.shape != (self.N, self.N):
            raise ValueError(
                f"C_matrix shape {C_matrix.shape} does not match (N, N) = ({self.N}, {self.N})"
            )
        if np.any(C_matrix < 0.0):
            raise ValueError("C_matrix entries must be non-negative")

        # Ensure zero diagonal
        C = np.array(C_matrix, dtype=np.float64)
        np.fill_diagonal(C, 0.0)
        self.C_matrix = C

    def set_population_interfaces(
        self,
        population_origins: np.ndarray,
        E: np.ndarray,
    ) -> None:
        """Construct and set full eligibility matrix from population origins and interface matrix E."""
        C = self.create_eligibility_matrix(E, population_origins)
        self.set_eligibility_matrix(C)

    @property
    def capacity(self) -> Optional[int]:
        """Uniform row capacity cap if configured and constant, or None if uncapped/heterogeneous."""
        if self.row_caps is not None and len(np.unique(self.row_caps)) == 1:
            return int(self.row_caps[0])
        return None

    def set_row_caps(self, capacity: Optional[Union[int, float, np.ndarray]]) -> None:
        """Set or update directed memory row capacity caps.

        Args:
            capacity: Scalar integer cap for all rows (e.g. 40), or array of shape (N,)
                      with per-row caps, or None for uncapped memory.
        """
        if capacity is None:
            self.row_caps = None
            return

        if self.reciprocal:
            raise ValueError(
                "Reciprocal memory with independent row capacity caps is rejected by the specification; "
                "only directed capacity is supported."
            )

        if isinstance(capacity, (int, float, np.integer, np.floating)) or np.asarray(capacity).ndim == 0:
            c_val = float(capacity)
            if c_val < 1.0 or not c_val.is_integer():
                raise ValueError(f"Row capacity must be an integer >= 1, got {capacity}")
            c_int = int(c_val)
            if np.any(self.row_sums > c_int):
                raise ValueError(f"Cannot set row capacity to {c_int}: existing row memory exceeds cap")
            self.row_caps = np.full(self.N, c_int, dtype=np.uint32)
        else:
            arr = np.asarray(capacity)
            if arr.shape != (self.N,):
                raise ValueError(f"row_caps shape mismatch: expected ({self.N},), got {arr.shape}")
            if np.any(arr < 1):
                raise ValueError("All row caps must be >= 1")
            if not (np.issubdtype(arr.dtype, np.integer) or np.all(arr == np.floor(arr))):
                raise ValueError("All row caps must be integers >= 1")
            arr_uint = arr.astype(np.uint32)
            if np.any(self.row_sums > arr_uint):
                raise ValueError("Cannot set row capacity: existing row memory exceeds cap")
            self.row_caps = arr_uint

    def set_opportunity_groups(
        self,
        groups: np.ndarray,
        between_eligibility: float = 0.05,
        within_eligibility: float = 1.0,
    ) -> None:
        """Set encounter eligibility matrix from opportunity group assignments.

        Args:
            groups: Array of shape (N,) mapping each agent to their opportunity group.
            between_eligibility: Cross-group encounter eligibility (default 0.05).
            within_eligibility: Within-group encounter eligibility (default 1.0).
        """
        groups_arr = np.asarray(groups, dtype=np.int64)
        if groups_arr.shape != (self.N,):
            raise ValueError(f"opportunity_groups shape mismatch: expected ({self.N},), got {groups_arr.shape}")
        self.opportunity_groups = groups_arr.copy()
        num_g = int(np.max(groups_arr)) + 1
        E = np.full((num_g, num_g), float(between_eligibility), dtype=np.float64)
        np.fill_diagonal(E, float(within_eligibility))
        C = self.create_eligibility_matrix(E, groups_arr)
        self.set_eligibility_matrix(C)

    def set_memory(self, W: Union[np.ndarray, list]) -> None:
        """Update memory matrix W and recompute row sums, total memory, and attention."""
        W_arr = np.asarray(W)
        if W_arr.shape != (self.N, self.N):
            raise ValueError(f"W shape {W_arr.shape} does not match (N, N) = ({self.N}, {self.N})")
        if np.any(np.diag(W_arr) != 0):
            raise ValueError("Self-ties forbidden: diag(W) must be strictly 0")
        if np.any(W_arr < 0):
            raise ValueError("Memory counts W cannot be negative")
        if self.reciprocal and not np.array_equal(W_arr, W_arr.T):
            raise ValueError("W must be symmetric when reciprocal=True")
        if self.row_caps is not None and np.any(np.sum(W_arr, axis=1) > self.row_caps):
            raise ValueError("Row memory counts in W exceed directed row capacity caps")
        self.W = np.array(W_arr, dtype=np.uint32)
        self.row_sums = np.sum(self.W, axis=1, dtype=np.uint32)
        self.total_memory = int(np.sum(self.row_sums))
        self._recompute_all_attention()

    @staticmethod

    def create_eligibility_matrix(
        E: np.ndarray,
        population_origins: np.ndarray,
    ) -> np.ndarray:
        """Construct agent eligibility matrix C from population interface matrix E.

        Computes C_{ij} = E_{g_i, g_j} for i != j, and C_{ii} = 0.0.

        Args:
            E: Population-level encounter eligibility matrix of shape (K, K),
               where E[a, b] is the eligibility factor from population a to b.
            population_origins: Integer array of shape (N,) mapping each agent i
               to their population index in {0, ..., K-1}.

        Returns:
            C: Full agent eligibility matrix of shape (N, N), dtype float64.
        """
        g = np.asarray(population_origins, dtype=np.int64)
        if g.ndim != 1:
            raise ValueError("population_origins must be a 1D array of shape (N,)")
        K = E.shape[0]
        if E.shape != (K, K):
            raise ValueError(f"E must be a square matrix of shape (K, K), got {E.shape}")
        if np.any((g < 0) | (g >= K)):
            raise ValueError(f"All population origin indices must be in [0, {K - 1}]")

        C = E[np.ix_(g, g)].astype(np.float64).copy()
        np.fill_diagonal(C, 0.0)
        return C

    def compute_similarities(self, i: int, y: np.ndarray) -> np.ndarray:
        """Compute state similarity s_{ij} = (1 / D) * sum_alpha y_{i, alpha} * y_{j, alpha}.

        Args:
            i: Agent index.
            y: Expressed belief array of shape (N, D).

        Returns:
            Similarity array of shape (N,), dtype float64, in [-1.0, 1.0].
        """
        D = float(y.shape[1])
        return np.dot(y, y[i]) / D

    def compute_target_distribution(
        self,
        i: int,
        y: np.ndarray,
        C_row: Optional[np.ndarray] = None,
        C_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Compute candidate selection distribution P_{ij} for agent i.

        Args:
            i: Initiating agent index.
            y: Expressed beliefs of shape (N, D).
            C_row: Optional eligibility vector C_{i, :} of shape (N,).
                   If a 2D array of shape (N, N) is passed here, it is automatically
                   treated as C_matrix.
            C_matrix: Optional full eligibility matrix of shape (N, N).
                      If None, falls back to self.C_matrix, or unrestricted eligibility.

        Returns:
            Tuple of (P_i, is_null).
            If is_null is True, P_i is None (null encounter attempt).
            If is_null is False, P_i is a normalized probability vector of shape (N,)
            satisfying sum_j P_{ij} = 1.0 and P_{ii} = 0.0.
        """
        # Resolve full eligibility matrix if provided or duck-typed
        eff_C_matrix = C_matrix if C_matrix is not None else self.C_matrix
        if C_row is not None and C_row.ndim == 2:
            eff_C_matrix = C_row
            C_row = None

        if eff_C_matrix is not None and eff_C_matrix.shape != (self.N, self.N):
            raise ValueError(f"C_matrix shape mismatch: expected ({self.N}, {self.N}), got {eff_C_matrix.shape}")

        # Resolve row i eligibility vector C_i
        if C_row is not None:
            if C_row.shape != (self.N,):
                raise ValueError(f"C_row shape mismatch: expected ({self.N},), got {C_row.shape}")
            C_i = C_row.astype(np.float64).copy()
        elif eff_C_matrix is not None:
            C_i = eff_C_matrix[i].astype(np.float64).copy()
        else:
            # Unrestricted eligibility: all j != i eligible
            C_i = np.ones(self.N, dtype=np.float64)
        C_i[i] = 0.0

        c_i = float(np.sum(C_i))
        if c_i <= 0.0:
            # Null encounter attempt: zero eligible candidates
            return None, True

        # Uniform exploration vector U_i
        U_i = C_i / c_i

        # Belief similarity vector s_{ij}
        s_i = self.compute_similarities(i, y)

        # Candidate active mask: eligible AND remembered
        active_candidates = (C_i > 0.0) & (self.W[i] > 0)

        if not np.any(active_candidates):
            # Fallback: no eligible remembered ties
            V_i = U_i.copy()
        else:
            # Max-subtraction log-sum-exp for numerical stability
            z = self.a * self.r + self.c * s_i
            max_z = np.max(z[active_candidates])
            exp_terms = np.zeros(self.N, dtype=np.float64)
            exp_terms[active_candidates] = np.exp(z[active_candidates] - max_z)
            V_unnorm = C_i * self.W[i].astype(np.float64) * exp_terms
            denom = float(np.sum(V_unnorm))
            if denom > 0.0:
                V_i = V_unnorm / denom
            else:
                V_i = U_i.copy()

        # Two-step triadic closure search if chi > 0
        if self.chi > 0.0:
            # M_ik = C_ik * A_ik / sum_l C_il * A_il
            denom_M = float(np.sum(C_i * self.A[i]))
            if denom_M > 0.0:
                M_i = (C_i * self.A[i]) / denom_M
                active_k = np.where(M_i > 0.0)[0]
                two_step = np.zeros(self.N, dtype=np.float64)
                for k in active_k:
                    # Intermediate hop from k to j: obtain agent k's eligibility C_k
                    if eff_C_matrix is not None:
                        C_k = eff_C_matrix[k].astype(np.float64).copy()
                        C_k[k] = 0.0
                    elif C_row is not None:
                        # Fallback when only single row C_row was provided without matrix:
                        # Homogeneous symmetric assumption with self-exclusion for k
                        C_k = C_i.copy()
                        C_k[i] = 1.0  # k can potentially attend to i
                        C_k[k] = 0.0  # k cannot attend to self
                    else:
                        # Unrestricted eligibility
                        C_k = np.ones(self.N, dtype=np.float64)
                        C_k[k] = 0.0

                    denom_k = float(np.sum(C_k * self.A[k]))
                    if denom_k > 0.0:
                        M_k = (C_k * self.A[k]) / denom_k
                        two_step += M_i[k] * M_k

                two_step *= C_i
                two_step[C_i <= 0.0] = 0.0
                two_step[i] = 0.0  # Strictly exclude self-closure
                denom_H = float(np.sum(two_step))
                if denom_H > 0.0:
                    H_i = two_step / denom_H
                else:
                    H_i = U_i.copy()
            else:
                H_i = U_i.copy()

            pref_i = (1.0 - self.chi) * V_i + self.chi * H_i
        else:
            pref_i = V_i

        P_i = self.epsilon * U_i + (1.0 - self.epsilon) * pref_i
        P_i[C_i <= 0.0] = 0.0  # Multi-population eligibility guard
        P_i[i] = 0.0  # Safeguard: no self-ties
        P_sum = float(np.sum(P_i))
        if P_sum > 0.0:
            P_i /= P_sum
        else:
            P_i = U_i.copy()

        P_i[C_i <= 0.0] = 0.0
        P_i[i] = 0.0
        return P_i, False


    def add_encounter(
        self,
        i: int,
        j: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Increment memory count for directed tie (i, j).

        If row capacity caps are active and row i is full (s_i >= C_i), removes
        one memory unit uniformly at random from existing units in row i, then
        increments (i, j) atomically, preserving total row memory and nonnegativity.
        In reciprocal mode, also applies the update to (j, i).
        Raises ValueError if i == j.
        """
        if i == j:
            raise ValueError(f"Self-encounters are prohibited: i={i}, j={j}")

        eff_rng = rng if rng is not None else self.rng

        if self.row_caps is not None and self.row_sums[i] >= self.row_caps[i]:
            s_i = float(self.row_sums[i])
            if s_i > 0.0:
                p_k = self.W[i].astype(np.float64) / s_i
                k_rem = int(eff_rng.choice(self.N, p=p_k))
                self.W[i, k_rem] -= 1
                self.W[i, j] += 1
                self._update_row_attention(i)
        else:
            self.W[i, j] += 1
            self.row_sums[i] += 1
            self.total_memory += 1
            self._update_row_attention(i)

        if self.reciprocal:
            if self.row_caps is not None and self.row_sums[j] >= self.row_caps[j]:
                s_j = float(self.row_sums[j])
                if s_j > 0.0:
                    p_k = self.W[j].astype(np.float64) / s_j
                    k_rem = int(eff_rng.choice(self.N, p=p_k))
                    self.W[j, k_rem] -= 1
                    self.W[j, i] += 1
                    self._update_row_attention(j)
            else:
                self.W[j, i] += 1
                self.row_sums[j] += 1
                self.total_memory += 1
                self._update_row_attention(j)

    def sample_and_apply_encounter(
        self,
        i: int,
        y: np.ndarray,
        C_row: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        C_matrix: Optional[np.ndarray] = None,
    ) -> EncounterResult:
        """Sample candidate target from distribution P_{ij} and apply memory increment.

        Args:
            i: Initiating agent index.
            y: Expressed beliefs of shape (N, D).
            C_row: Eligibility vector of shape (N,) or matrix of shape (N, N).
            rng: NumPy random Generator.
            C_matrix: Optional full eligibility matrix of shape (N, N).

        Returns:
            EncounterResult with initiator, target, and null status.
        """
        if rng is None:
            rng = np.random.default_rng()

        P_i, is_null = self.compute_target_distribution(
            i=i,
            y=y,
            C_row=C_row,
            C_matrix=C_matrix,
        )
        if is_null or P_i is None:
            return EncounterResult(initiator=i, target=-1, is_null=True)

        target = int(rng.choice(self.N, p=P_i))
        self.add_encounter(i, target, rng=rng)
        return EncounterResult(initiator=i, target=target, is_null=False)

    def sample_and_apply_decay(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, int]:
        """Sample a remembered token with probability W_{ij} / S and decrement it.

        Uses an efficient O(N) hierarchical two-stage sampler.
        In reciprocal mode, decrements both W_{ij} and W_{ji} simultaneously.

        Returns:
            Tuple (i, j) of decremented tie, or (-1, -1) if total memory is 0.
        """
        if self.total_memory == 0:
            return -1, -1

        if rng is None:
            rng = np.random.default_rng()

        if not self.reciprocal:
            # Stage 1: Sample initiator i with prob s_i / S
            p_i = self.row_sums.astype(np.float64) / float(self.total_memory)
            i = int(rng.choice(self.N, p=p_i))

            # Stage 2: Sample target j with prob W_{ij} / s_i
            s_i = float(self.row_sums[i])
            p_j = self.W[i].astype(np.float64) / s_i
            j = int(rng.choice(self.N, p=p_j))

            self.W[i, j] -= 1
            self.row_sums[i] -= 1
            self.total_memory -= 1
            self._update_row_attention(i)
            return i, j
        else:
            # Reciprocal decay: sample pair token from upper triangle (i < j)
            upper_tri = np.triu(self.W, k=1)
            upper_row_sums = np.sum(upper_tri, axis=1)
            total_pairs = int(np.sum(upper_row_sums))
            if total_pairs == 0:
                return -1, -1

            p_i = upper_row_sums.astype(np.float64) / float(total_pairs)
            i = int(rng.choice(self.N, p=p_i))

            p_j = upper_tri[i].astype(np.float64) / float(upper_row_sums[i])
            j = int(rng.choice(self.N, p=p_j))

            self.W[i, j] -= 1
            self.W[j, i] -= 1
            self.row_sums[i] -= 1
            self.row_sums[j] -= 1
            self.total_memory -= 2
            self._update_row_attention(i)
            self._update_row_attention(j)
            return i, j


# ==============================================================================
# Standalone Functional API & E2E Test Adapters
# ==============================================================================

def calculate_out_degrees(W: np.ndarray) -> np.ndarray:
    """Calculate outgoing memory count vector s_i = sum_k W_{ik}."""
    return np.sum(W, axis=1, dtype=np.uint32)


compute_out_degrees = calculate_out_degrees


def record_encounter(W: np.ndarray, i: int, j: int) -> None:
    """Record a single directed encounter increment W_{ij} += 1."""
    if i == j:
        raise ValueError(f"Self-encounters are prohibited: i={i}, j={j}")
    W[i, j] += 1


def get_total_memory_mass(W: np.ndarray) -> int:
    """Return total remembered token mass S = sum_{ij} W_{ij}."""
    return int(np.sum(W))


def calculate_attention_matrix(W: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute attention matrix A and active agent fraction f_active."""
    row_sums = np.sum(W, axis=1, keepdims=True, dtype=np.float64)
    active = row_sums > 0.0
    A = np.zeros(W.shape, dtype=np.float64)
    np.divide(W, row_sums, out=A, where=active)
    f_active = float(np.mean(row_sums.ravel() > 0.0))
    return A, f_active


def calculate_received_attention(A: np.ndarray) -> np.ndarray:
    """Calculate received attention vector r_j = sum_i A_{ij}."""
    return np.sum(A, axis=0, dtype=np.float64)


def calculate_candidate_probabilities(
    i: int,
    W: np.ndarray,
    r: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    a: float = 0.5,
    c: float = 1.0,
    epsilon: float = 0.1,
    C: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate candidate target selection probability vector P_i for agent i."""
    N = W.shape[0]
    mgr = MemoryManager(N=N, epsilon=epsilon, a=a, c=c, initial_W=W, C_matrix=C)
    if r is not None:
        mgr.r = np.asarray(r, dtype=np.float64).copy()
    if y is None:
        y = np.ones((N, 1), dtype=np.float64)
    probs, is_null = mgr.compute_target_distribution(i=i, y=y)
    if is_null or probs is None:
        return np.zeros(N, dtype=np.float64)
    return probs


def calculate_decay_propensity(W: np.ndarray, delta: float = 0.1) -> float:
    """Calculate total memory decay propensity R_forget = delta * S."""
    return float(delta * np.sum(W))


def execute_memory_decay(W: np.ndarray, i: int, j: int) -> None:
    """Decrement memory tie W_{ij} by 1."""
    if W[i, j] <= 0:
        raise ValueError(f"Cannot decay edge ({i}, {j}) with zero memory weight")
    W[i, j] -= 1


def sample_decay_edge(
    W: np.ndarray,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int]:
    """Sample a tie (i, j) for decay with probability proportional to W_{ij}."""
    if rng is None:
        rng = np.random.default_rng(seed)
    N = W.shape[0]
    mgr = MemoryManager(N=N, initial_W=W)
    return mgr.sample_and_apply_decay(rng=rng)


def build_eligibility_matrix(groups: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Construct full agent eligibility matrix C from groups and interface matrix E."""
    return MemoryManager.create_eligibility_matrix(E, groups)


def sample_encounter_target(
    i: int,
    C: np.ndarray,
    W: np.ndarray,
    r: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[int]:
    """Sample target agent j for an encounter by agent i, or None if null."""
    if rng is None:
        rng = np.random.default_rng(seed)
    N = W.shape[0]
    mgr = MemoryManager(N=N, initial_W=W, C_matrix=C)
    res = mgr.sample_and_apply_encounter(i=i, y=y, C_matrix=C, rng=rng)
    return None if res.is_null else res.target


def calculate_closure_weights(A: np.ndarray) -> np.ndarray:
    """Calculate 2-step closure path matrix H* = A @ A with zero diagonal."""
    H_star = np.dot(A, A)
    np.fill_diagonal(H_star, 0.0)
    return H_star


def apply_closure_encounter(W: np.ndarray, i: int, j: int) -> None:
    """Apply direct closure encounter incrementing W_{ij}."""
    record_encounter(W, i, j)


def record_reciprocal_encounter(W: np.ndarray, i: int, j: int) -> None:
    """Record reciprocal encounter incrementing both W_{ij} and W_{ji}."""
    if i == j:
        raise ValueError(f"Self-encounters are prohibited: i={i}, j={j}")
    W[i, j] += 1
    W[j, i] += 1

