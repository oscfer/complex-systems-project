"""Common utilities, mathematical oracles, and test base classes for Modular3 E2E tests."""

import importlib
import sys
import unittest
import numpy as np


def get_module(module_name: str):
    """Safely attempt to import a module, returning None if not found."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


class ModularBaseTestCase(unittest.TestCase):
    """Base test case providing dynamic dependency checking, adaptors, and assertions."""

    def require_module(self, module_name: str):
        """Skip the test if the required implementation module is not available."""
        mod = get_module(module_name)
        if mod is None:
            self.skipTest(f"Required implementation module '{module_name}' is not available.")
        return mod

    def require_attr(self, module_name: str, attr_name: str, aliases: list = None):
        """Skip the test if the required module or attribute is not available."""
        mod = self.require_module(module_name)
        candidates = [attr_name] + (aliases if aliases else [])
        for candidate in candidates:
            if hasattr(mod, candidate):
                return getattr(mod, candidate)
        self.skipTest(f"Module '{module_name}' lacks required attribute '{attr_name}' (tried: {candidates}).")

    def create_state(self, N: int, D: int = 1, continuous: bool = False, **kwargs):
        """Create a State instance compatible with the underlying implementation."""
        State = self.require_attr("src.core.state", "State")
        state_mode = "continuous" if continuous else "binary"
        if hasattr(State, "create_empty"):
            return State.create_empty(N=N, D=D, state_mode=state_mode)
        elif hasattr(State, "create_unbiased_emergence"):
            return State.create_unbiased_emergence(N=N, D=D, state_mode=state_mode)
        else:
            try:
                return State(N=N, D=D, state_mode=state_mode, **kwargs)
            except TypeError:
                return State(N=N, D=D, **kwargs)

    def get_coupling(self, name: str = "B0", D: int = 3):
        """Helper to obtain coupling matrix regardless of keyword naming differences."""
        fn = self.require_attr("src.core.couplings", "get_coupling_matrix")
        try:
            return fn(name, D)
        except TypeError:
            try:
                return fn(name=name, D=D)
            except TypeError:
                return fn(D=D, coupling_type=name)

    def calc_internal_energy(self, x, coupling="reinforcing", L=1.0):
        """Helper to compute internal belief energy supporting both matrix and string forms."""
        fn = self.require_attr(
            "src.core.couplings",
            "calculate_isolated_internal_energy",
            aliases=["calculate_internal_belief_energy"]
        )
        if isinstance(coupling, str):
            B = self.get_coupling(name=coupling, D=len(x))
            try:
                return fn(x, B, L=L)
            except TypeError:
                return fn(x, coupling_type=coupling, L=L)
        else:
            return fn(x, coupling, L=L)



# ==============================================================================
# AUTHORITATIVE MATHEMATICAL ORACLES (Derived from research specifications)
# ==============================================================================

def oracle_binary_glauber_prob(kappa: float) -> float:
    """Exact conditional Glauber probability P(z = +1 | F) = 1 / (1 + exp(-2*kappa))."""
    if kappa < -350.0:
        return 0.0
    if kappa > 350.0:
        return 1.0
    return 1.0 / (1.0 + np.exp(-2.0 * kappa))


def oracle_continuous_inverse_cdf(kappa: float, u: float) -> float:
    """Exact, numerically stable 3-branch continuous heat bath inverse-CDF sample.
    
    f(s | F) = exp(kappa * s) / Z(kappa) on [-1, 1].
    """
    if u <= 0.0:
        return -1.0
    if u >= 1.0:
        return 1.0
    
    if kappa > 1e-5:
        # Positive field branch: factor out exp(kappa)
        val = 1.0 + (1.0 / kappa) * np.log(u + (1.0 - u) * np.exp(-2.0 * kappa))
        return float(np.clip(val, -1.0, 1.0))
    elif kappa < -1e-5:
        # Negative field branch: factor out exp(-kappa)
        val = -1.0 + (1.0 / kappa) * np.log((1.0 - u) + u * np.exp(2.0 * kappa))
        return float(np.clip(val, -1.0, 1.0))
    else:
        # Weak / zero field branch: Taylor expansion around kappa = 0
        # z' ~ (2u - 1) + 2u(1-u)kappa - (4/3)u(1-u)(2u-1)kappa^2
        val = (2.0 * u - 1.0) + 2.0 * u * (1.0 - u) * kappa - (4.0 / 3.0) * u * (1.0 - u) * (2.0 * u - 1.0) * (kappa ** 2)
        return float(np.clip(val, -1.0, 1.0))


def oracle_attention_matrix(W: np.ndarray):
    """Computes row-normalized attention matrix A and active-row fraction f_active.
    
    W: (N, N) non-negative integer memory matrix with zero diagonal.
    """
    N = W.shape[0]
    row_sums = W.sum(axis=1)
    A = np.zeros((N, N), dtype=np.float64)
    active_mask = row_sums > 0
    f_active = float(np.mean(active_mask))
    for i in range(N):
        if row_sums[i] > 0:
            A[i, :] = W[i, :] / float(row_sums[i])
    return A, f_active


def oracle_influence_metrics(W: np.ndarray):
    """Computes exact influence metrics from memory matrix W.
    
    Returns: dict with r, p, C, C_N, effective_leaders, f_active.
    If sum(r) == 0, returns NaN for shares and concentrations.
    """
    N = W.shape[0]
    A, f_active = oracle_attention_matrix(W)
    r = A.sum(axis=0)  # Received attention vector
    r_sum = float(r.sum())
    
    if r_sum == 0.0 or N <= 1:
        return {
            "A": A,
            "r": r,
            "p": np.full(N, np.nan),
            "C": np.nan,
            "C_N": np.nan,
            "effective_leaders": np.nan,
            "f_active": f_active,
        }
    
    p = r / r_sum
    C = float(np.sum(p ** 2))
    C_N = float((N * C - 1.0) / (N - 1.0)) if N > 1 else np.nan
    effective_leaders = float(1.0 / C) if C > 0 else np.nan
    
    return {
        "A": A,
        "r": r,
        "p": p,
        "C": C,
        "C_N": C_N,
        "effective_leaders": effective_leaders,
        "f_active": f_active,
    }


def oracle_spearman_rank_persistence(r1: np.ndarray, r2: np.ndarray) -> float:
    """Computes exact Spearman rank correlation using midpoint fractional ranks for ties.
    
    Returns NaN if either vector has zero variance.
    """
    if len(r1) != len(r2) or len(r1) < 2:
        return np.nan
    
    def _fractional_ranks(x: np.ndarray) -> np.ndarray:
        n = len(x)
        sort_indices = np.argsort(x)
        sorted_x = x[sort_indices]
        ranks = np.zeros(n, dtype=np.float64)
        
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            avg_rank = 1.0 + (i + j) / 2.0
            ranks[sort_indices[i:j + 1]] = avg_rank
            i = j + 1
        return ranks
    
    R1 = _fractional_ranks(r1)
    R2 = _fractional_ranks(r2)
    
    std1 = np.std(R1)
    std2 = np.std(R2)
    if std1 == 0.0 or std2 == 0.0:
        return np.nan
    
    corr = np.corrcoef(R1, R2)[0, 1]
    return float(np.clip(corr, -1.0, 1.0))


def oracle_population_discrepancy(x: np.ndarray, y: np.ndarray) -> float:
    """Computes normalized population discrepancy d_2 in [0, 1].
    
    d_2 = (1 / (4 * N * D)) * sum_{i, alpha} (x_{i, alpha} - y_{i, alpha})^2
    """
    N, D = x.shape
    diff = x.astype(np.float64) - y.astype(np.float64)
    return float(np.sum(diff ** 2) / (4.0 * N * D))


def oracle_frustrated_energy(x: np.ndarray, L: float = 1.0) -> float:
    """Internal energy for D=3 under frustrated triangle matrix B_f.
    
    B_f = 0.5 * [[0, 1, 1], [1, 0, -1], [1, -1, 0]].
    E_B(x) = -L * sum_{alpha < beta} B_{alpha, beta} * x_alpha * x_beta
           = -0.5 * L * (x0*x1 + x0*x2 - x1*x2).
    """
    x0, x1, x2 = x[0], x[1], x[2]
    return float(-0.5 * L * (x0 * x1 + x0 * x2 - x1 * x2))


def oracle_reinforcing_energy(x: np.ndarray, L: float = 1.0) -> float:
    """Internal energy for D=3 under reinforcing triangle matrix B_+.
    
    B_+ = 0.5 * [[0, 1, 1], [1, 0, 1], [1, 1, 0]].
    E_B(x) = -0.5 * L * (x0*x1 + x0*x2 + x1*x2).
    """
    x0, x1, x2 = x[0], x[1], x[2]
    return float(-0.5 * L * (x0 * x1 + x0 * x2 + x1 * x2))


def oracle_participation_payoff(x_i1: float, I: int, rho_i: float,
                                b_act: float = 1.0, g: float = 1.0, c_act: float = 0.5) -> float:
    """Local coordination payoff difference Delta u_i = b_act * q_i + g * rho_i - c_act.
    
    q_i = (1 - I * x_i1) / 2 in [0, 1].
    """
    q_i = (1.0 - I * x_i1) / 2.0
    return float(b_act * q_i + g * rho_i - c_act)


def oracle_analytical_baseline_bound(K: float, T: float, N: int):
    """Computes theoretical no-information equilibrium parameters:
    
    q = (1 - tanh(K/T)) / 2
    MSE_bound = q * (1 - q) / N
    """
    q = float((1.0 - np.tanh(K / T)) / 2.0)
    mse_bound = float(q * (1.0 - q) / N)
    return q, mse_bound


def oracle_origin_stratified_split(origin_ids: list, ratios=(0.6, 0.2, 0.2), seed: int = 42):
    """Origin-stratified deterministic partition of simulation origins.
    
    Guarantees no origin appears in more than one partition.
    """
    unique_origins = sorted(list(set(origin_ids)))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_origins).tolist()
    
    M = len(shuffled)
    n_train = int(np.floor(ratios[0] * M))
    n_val = int(np.floor(ratios[1] * M))
    
    train_origins = set(shuffled[:n_train])
    val_origins = set(shuffled[n_train:n_train + n_val])
    test_origins = set(shuffled[n_train + n_val:])
    
    return train_origins, val_origins, test_origins
