"""Network initial condition and belief state generators (Stages 0, 1, 2).

Implements structural initial condition generators:
1. Unbiased independent sampling (tabula rasa emergence state with W = 0).
2. Decentralized random networks (degree budget m per agent, uniform target selection).
3. Modular block networks (degree budget m, homophilic concentration eta).
4. Concentrated-attention hub networks (degree budget m, directed attention concentration eta).
5. State alignment and discrepancy generators (public bias p_g, private discrepancy q).
6. Continuous endpoint states.

Simulation Stage Mapping:
    - Stage 0: Generates tabula rasa initial states (W = 0) for timing baseline runs (N=256).
    - Stage 1: Generates initial conditions for 450 pilot trajectories.
    - Stage 2: Generates modular blocks, hubs, and aligned/opposing populations for extension campaigns.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from src.core.state import State, StateMode


def generate_emergence_state(
    N: int,
    D: int = 1,
    state_mode: Union[StateMode, str] = StateMode.BINARY,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> State:
    """Generate unbiased emergence initial condition with W = 0 and y(0) = x(0).

    Args:
        N: Population size (int >= 1).
        D: Belief dimensionality (int in {1, 3}).
        state_mode: StateMode.BINARY or StateMode.CONTINUOUS.
        seed: Optional PRNG seed.
        rng: Optional NumPy random Generator.

    Returns:
        State: Valid initialized simulation state.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    state = State.create_unbiased_emergence(N=N, D=D, state_mode=state_mode, rng=rng)
    state.validate()
    return state


def generate_decentralized_network(
    N: int,
    m: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate decentralized random network with exact row sums sum_j W_ij = m and W_ii = 0.

    Each agent draws m targets independently and uniformly from {j != i}.

    Args:
        N: Population size (int >= 2).
        m: Fixed memory budget per agent (int >= 1).
        seed: Optional PRNG seed.
        rng: Optional NumPy random Generator.

    Returns:
        W: Interaction memory matrix of shape (N, N), dtype uint32.
    """
    if N < 2:
        raise ValueError(f"Decentralized network requires N >= 2, got {N}")
    if m < 0:
        raise ValueError(f"Memory budget m must be non-negative, got {m}")

    if rng is None:
        rng = np.random.default_rng(seed)

    W = np.zeros((N, N), dtype=np.uint32)
    if m == 0:
        return W

    if N == 2:
        W[0, 1] = m
        W[1, 0] = m
        return W

    all_nodes = np.arange(N)
    for i in range(N):
        candidates = np.delete(all_nodes, i)
        targets = rng.choice(candidates, size=m, replace=True)
        counts = np.bincount(targets, minlength=N)
        W[i] = counts.astype(np.uint32)

    return W


def generate_modular_network(
    N: int,
    num_blocks: int,
    m: int,
    eta: float = 0.8,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate modular block network with memory budget m and concentration eta in [0, 1].

    Distribution for agent i:
        pi_i = (1 - eta) * u_i + eta * v_i
    where u_i is uniform over all {j != i} and v_i is uniform over {j != i, block(j) == block(i)}.

    Args:
        N: Population size (int >= 2).
        num_blocks: Number of equal or balanced communities (int >= 1).
        m: Fixed memory budget per agent (int >= 0).
        eta: Mixing strength in [0.0, 1.0].
        seed: Optional PRNG seed.
        rng: Optional NumPy random Generator.

    Returns:
        Tuple of:
            W: Interaction memory matrix of shape (N, N), dtype uint32.
            blocks: Community assignment array of shape (N,), dtype int64.
    """
    if N < 2:
        raise ValueError(f"Modular network requires N >= 2, got {N}")
    if num_blocks < 1 or num_blocks > N:
        raise ValueError(f"Invalid num_blocks={num_blocks} for N={N}")
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"Concentration eta must be in [0.0, 1.0], got {eta}")
    if m < 0:
        raise ValueError(f"Memory budget m must be non-negative, got {m}")

    if rng is None:
        rng = np.random.default_rng(seed)

    # Assign balanced blocks
    block_sizes = [N // num_blocks + (1 if b < N % num_blocks else 0) for b in range(num_blocks)]
    blocks = np.repeat(np.arange(num_blocks, dtype=np.int64), block_sizes)

    W = np.zeros((N, N), dtype=np.uint32)
    if m == 0:
        return W, blocks

    all_nodes = np.arange(N)
    for i in range(N):
        b_i = blocks[i]
        same_block = (blocks == b_i) & (all_nodes != i)
        num_same = int(np.sum(same_block))

        if eta >= 1.0 and num_same > 0:
            # Strictly within block
            p = np.zeros(N, dtype=np.float64)
            p[same_block] = 1.0 / float(num_same)
        elif eta <= 0.0 or num_same == 0:
            # Strictly uniform over all other nodes
            p = np.zeros(N, dtype=np.float64)
            p[all_nodes != i] = 1.0 / float(N - 1)
        else:
            u_i = np.zeros(N, dtype=np.float64)
            u_i[all_nodes != i] = 1.0 / float(N - 1)
            v_i = np.zeros(N, dtype=np.float64)
            v_i[same_block] = 1.0 / float(num_same)
            p = (1.0 - eta) * u_i + eta * v_i

        p[i] = 0.0
        p /= np.sum(p)
        targets = rng.choice(all_nodes, size=m, replace=True, p=p)
        W[i] = np.bincount(targets, minlength=N).astype(np.uint32)

    return W, blocks


def generate_hub_network(
    N: int,
    m: int,
    eta: float = 0.2,
    hub: Optional[int] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """Generate concentrated-attention hub network with memory budget m and hub boost eta.

    For node i != h:
        pi_ij = (1 - eta) / (N - 1) + eta * 1[j == h] for j != i
    For hub h:
        pi_hj = 1 / (N - 1) for j != h

    Args:
        N: Population size (int >= 2).
        m: Fixed memory budget per agent (int >= 0).
        eta: Attention concentration factor in [0.0, 1.0].
        hub: Optional designated hub node index in [0, N-1].
        seed: Optional PRNG seed.
        rng: Optional NumPy random Generator.

    Returns:
        Tuple of:
            W: Interaction memory matrix of shape (N, N), dtype uint32.
            hub_idx: Selected hub agent index.
    """
    if N < 2:
        raise ValueError(f"Hub network requires N >= 2, got {N}")
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"Concentration eta must be in [0.0, 1.0], got {eta}")
    if m < 0:
        raise ValueError(f"Memory budget m must be non-negative, got {m}")

    if rng is None:
        rng = np.random.default_rng(seed)

    hub_idx = int(rng.integers(0, N)) if hub is None else int(hub)
    if hub_idx < 0 or hub_idx >= N:
        raise ValueError(f"Hub index {hub_idx} out of range [0, {N-1}]")

    W = np.zeros((N, N), dtype=np.uint32)
    if m == 0:
        return W, hub_idx

    all_nodes = np.arange(N)
    for i in range(N):
        p = np.zeros(N, dtype=np.float64)
        if i == hub_idx or eta == 0.0:
            # Hub or unboosted network: uniform over other nodes
            p[all_nodes != i] = 1.0 / float(N - 1)
        else:
            # Non-hub agent attends to hub with elevated probability
            p[all_nodes != i] = (1.0 - eta) / float(N - 1)
            p[hub_idx] += eta

        p[i] = 0.0
        p /= np.sum(p)
        targets = rng.choice(all_nodes, size=m, replace=True, p=p)
        W[i] = np.bincount(targets, minlength=N).astype(np.uint32)

    return W, hub_idx


def generate_aligned_states(
    N: int,
    D: int = 1,
    p_g: Union[float, np.ndarray, list] = 0.5,
    q: float = 0.0,
    groups: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    state_mode: Union[StateMode, str] = StateMode.BINARY,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate correlated private beliefs x and expressed public stances y with controlled discrepancy q.

    Public stances y have P(y = +1) = p_g.
    Private beliefs x independently flip relative to y with probability q:
        P(x != y) = q.

    Args:
        N: Population size.
        D: Dimensionality of belief space.
        p_g: Alignment probability in [0.0, 1.0], or array-like of group/agent probabilities.
        q: Discrepancy probability in [0.0, 1.0].
        groups: Optional community/population assignment array of shape (N,).
        seed: Optional PRNG seed.
        rng: Optional NumPy random Generator.
        state_mode: StateMode.BINARY or StateMode.CONTINUOUS.

    Returns:
        Tuple of (x, y) arrays of shape (N, D), dtype float64.
    """
    p_g_arr = np.asarray(p_g, dtype=np.float64)
    if np.any(p_g_arr < 0.0) or np.any(p_g_arr > 1.0):
        raise ValueError(f"p_g must be in [0.0, 1.0], got {p_g}")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Discrepancy q must be in [0.0, 1.0], got {q}")

    if rng is None:
        rng = np.random.default_rng(seed)

    mode = StateMode(state_mode.lower()) if isinstance(state_mode, str) else state_mode

    # Reshape or expand p_g to strictly (N, 1) or scalar float to preserve shape (N, D)
    if p_g_arr.ndim == 0 or p_g_arr.size == 1:
        p_g_cmp: Union[float, np.ndarray] = float(p_g_arr.item())
    elif p_g_arr.ndim == 1:
        if len(p_g_arr) == N:
            p_g_cmp = p_g_arr.reshape(N, 1)
        elif 1 < len(p_g_arr) <= N:
            if groups is not None:
                g_arr = np.asarray(groups, dtype=np.int64)
                if len(g_arr) != N or np.any(g_arr < 0) or np.any(g_arr >= len(p_g_arr)):
                    raise ValueError(f"groups shape or values invalid for {len(p_g_arr)} groups")
                p_g_cmp = p_g_arr[g_arr].reshape(N, 1)
            else:
                K = len(p_g_arr)
                block_sizes = [N // K + (1 if b < N % K else 0) for b in range(K)]
                p_g_cmp = np.repeat(p_g_arr, block_sizes).reshape(N, 1)
        else:
            raise ValueError(f"p_g array length {len(p_g_arr)} exceeds population size N={N}")
    else:
        raise ValueError("p_g must be a scalar or 1D array")

    # Generate public stances y
    y_signs = np.where(rng.random((N, D)) < p_g_cmp, 1.0, -1.0)
    if mode == StateMode.CONTINUOUS:
        amplitudes = rng.uniform(0.0, 1.0, size=(N, D))
        y = y_signs * amplitudes
    else:
        y = y_signs.astype(np.float64)

    # Generate private beliefs x with flip probability q
    if q == 0.0:
        x = y.copy()
    else:
        flips = rng.random((N, D)) < q
        x = np.where(flips, -y, y).astype(np.float64)

    return x, y


def generate_endpoint_continuous(
    N: int,
    D: int = 1,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> State:
    """Generate continuous state initialized strictly to discrete endpoints {-1.0, +1.0}."""
    if rng is None:
        rng = np.random.default_rng(seed)
    return State.create_endpoint_continuous(N=N, D=D, rng=rng)


def generate_decentralized_state(
    N: int,
    D: int = 1,
    m: int = 20,
    state_mode: Union[StateMode, str] = StateMode.BINARY,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> State:
    """Generate complete State initialized with decentralized network topology."""
    if rng is None:
        rng = np.random.default_rng(seed)
    state = State.create_unbiased_emergence(N=N, D=D, state_mode=state_mode, rng=rng)
    state.W = generate_decentralized_network(N=N, m=m, rng=rng)
    state.validate()
    return state


def generate_modular_state(
    N: int,
    num_blocks: int = 2,
    D: int = 1,
    m: int = 20,
    eta: float = 0.8,
    state_mode: Union[StateMode, str] = StateMode.BINARY,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    return_blocks: bool = False,
) -> Union[State, Tuple[State, np.ndarray]]:
    """Generate complete State initialized with modular network topology.

    Args:
        N: Population size (int >= 2).
        num_blocks: Number of communities (int >= 1).
        D: Dimensionality of belief space.
        m: Fixed memory budget per agent.
        eta: Homophily concentration factor in [0.0, 1.0].
        state_mode: StateMode.BINARY or StateMode.CONTINUOUS.
        seed: Optional PRNG seed.
        rng: Optional NumPy random Generator.
        return_blocks: If True, returns tuple of (state, blocks). Default False returns
            a valid State instance with .blocks attached as an attribute.

    Returns:
        State (or Tuple[State, np.ndarray] if return_blocks is True).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    state = State.create_unbiased_emergence(N=N, D=D, state_mode=state_mode, rng=rng)
    W, blocks = generate_modular_network(N=N, num_blocks=num_blocks, m=m, eta=eta, rng=rng)
    state.W = W
    state.blocks = blocks
    state.validate()
    if return_blocks:
        return state, blocks
    return state


def generate_hub_state(
    N: int,
    D: int = 1,
    m: int = 20,
    eta: float = 0.2,
    hub: Optional[int] = None,
    state_mode: Union[StateMode, str] = StateMode.BINARY,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    return_hub: bool = False,
) -> Union[State, Tuple[State, int]]:
    """Generate complete State initialized with concentrated-attention hub network topology.

    Args:
        N: Population size (int >= 2).
        D: Belief dimensionality.
        m: Fixed memory budget per agent.
        eta: Attention concentration factor.
        hub: Optional designated hub node index.
        state_mode: StateMode.BINARY or StateMode.CONTINUOUS.
        seed: Optional PRNG seed.
        rng: Optional NumPy random Generator.
        return_hub: If True, returns tuple of (state, hub_idx). Default False returns
            a valid State instance with .hub_idx attached as an attribute.

    Returns:
        State (or Tuple[State, int] if return_hub is True).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    state = State.create_unbiased_emergence(N=N, D=D, state_mode=state_mode, rng=rng)
    W, hub_idx = generate_hub_network(N=N, m=m, eta=eta, hub=hub, rng=rng)
    state.W = W
    state.hub_idx = hub_idx
    state.validate()
    if return_hub:
        return state, hub_idx
    return state


def generate_aligned_state(
    N: int,
    D: int = 1,
    p_g: Union[float, np.ndarray, list] = 0.5,
    q: float = 0.0,
    groups: Optional[np.ndarray] = None,
    state_mode: Union[StateMode, str] = StateMode.BINARY,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> State:
    """Generate complete State initialized with correlated beliefs x, y and empty memory W = 0."""
    if rng is None:
        rng = np.random.default_rng(seed)
    mode = StateMode(state_mode.lower()) if isinstance(state_mode, str) else state_mode
    x, y = generate_aligned_states(N=N, D=D, p_g=p_g, q=q, groups=groups, seed=seed, rng=rng, state_mode=mode)
    W = np.zeros((N, N), dtype=np.uint32)
    n = np.zeros(N, dtype=np.uint8)
    Q = np.zeros((N, D, 2), dtype=np.float64)
    state = State(
        N=N,
        D=D,
        state_mode=mode,
        x=x,
        y=y,
        W=W,
        n=n,
        Q=Q,
        t=0.0,
        blocks=groups.copy() if groups is not None else None,
    )
    state.validate()
    return state


def generate_opposing_populations_state(
    N: int,
    num_populations: int = 2,
    D: int = 1,
    p_align: float = 0.9,
    q: float = 0.0,
    state_mode: Union[StateMode, str] = StateMode.BINARY,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> State:
    """Generate complete State initialized with opposing population orientations.

    Population 0 is oriented with P(y = +1) = p_align.
    Population 1 is opposingly oriented with P(y = +1) = 1.0 - p_align.
    Subsequent populations alternate orientation.

    Returns:
        Valid initialized State instance with community .blocks attached.
    """
    if num_populations < 2:
        raise ValueError(f"Opposing populations requires num_populations >= 2, got {num_populations}")
    if not (0.0 <= p_align <= 1.0):
        raise ValueError(f"Alignment probability p_align must be in [0.0, 1.0], got {p_align}")

    if rng is None:
        rng = np.random.default_rng(seed)

    mode = StateMode(state_mode.lower()) if isinstance(state_mode, str) else state_mode

    p_g = [p_align if k % 2 == 0 else (1.0 - p_align) for k in range(num_populations)]
    block_sizes = [N // num_populations + (1 if b < N % num_populations else 0) for b in range(num_populations)]
    blocks = np.repeat(np.arange(num_populations, dtype=np.int64), block_sizes)

    return generate_aligned_state(
        N=N,
        D=D,
        p_g=p_g,
        q=q,
        groups=blocks,
        state_mode=mode,
        rng=rng,
    )


# Convenience aliases
generate_decentralized = generate_decentralized_state
generate_modular = generate_modular_state
generate_hub = generate_hub_state
generate_emergence = generate_emergence_state
generate_aligned = generate_aligned_state
generate_opposing = generate_opposing_populations_state
generate_opposing_state = generate_opposing_populations_state
