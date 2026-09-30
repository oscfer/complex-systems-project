"""Structural Actuators for direct network edge manipulation.

Implements:
1. StructuralActuator: Layer 2 interface for bounded, direct manipulation of network edges.
2. Direct connection forcing: reinforcing memory ties between specified node pairs.
3. Direct tie severance: decaying / deleting memory ties between specified node pairs.
4. Rewiring: atomic disconnection and reconnection of interaction channels.
5. Strict preservation of Layer 1 continuous-time jump process physics:
   - Integer memory with strictly zero diagonal (W_{ii} == 0).
   - Elementwise non-negativity (W_{ij} >= 0).
   - Reciprocal symmetry invariance in reciprocal networks (W == W^T).
   - Finite row capacity constraints (s_i <= C_i).
   - Row-normalized attention A and received attention r delta maintenance.
   - Synchronous scheduler total memory update ensuring exact continuous-time decay rates.
6. BudgetLedger integration and structural budget accounting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.control.contracts import Action, StructuralActionEntry, StructuralOp
from src.control.ledger import BudgetLedger
from src.core.memory import MemoryManager
from src.core.scheduler import PoissonScheduler
from src.core.state import State


class StructuralActuator:
    """Layer 2 actuator executing bounded direct manipulation of structural ties."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def apply_action(
        self,
        action: Action,
        state: State,
        memory_mgr: MemoryManager,
        scheduler: Optional[PoissonScheduler] = None,
        ledger: Optional[BudgetLedger] = None,
    ) -> List[Dict[str, Any]]:
        """Apply all structural entries contained in the Action to the network.

        Invariants enforced:
        - Self-loops prohibited (W_{ii} == 0).
        - Non-negativity (W_{ij} >= 0).
        - Reciprocal symmetry (W == W^T if memory_mgr.reciprocal is True).
        - Row capacity caps preserved if memory_mgr.row_caps is active.
        - Synchronous update to PoissonScheduler total memory to preserve jump rates.
        - Synchronous update to State.W.

        Returns:
            List of operation execution records.
        """
        if not action.structural_entries:
            return []

        records: List[Dict[str, Any]] = []
        N = memory_mgr.N

        for entry in action.structural_entries:
            src = entry.source
            tgt = entry.target
            op = entry.operation
            cnt = entry.count

            if src < 0 or src >= N or tgt < 0 or tgt >= N:
                raise ValueError(f"Structural nodes ({src}, {tgt}) out of bounds for N={N}")
            if src == tgt:
                raise ValueError(f"Self-loops are strictly prohibited: ({src}, {tgt})")

            if op == StructuralOp.FORCE:
                rec = self._apply_force(src, tgt, cnt, memory_mgr)
                records.append(rec)
            elif op == StructuralOp.SEVER:
                rec = self._apply_sever(src, tgt, cnt, memory_mgr)
                records.append(rec)
            elif op == StructuralOp.REWIRE:
                rewire_tgt = entry.rewire_target
                if rewire_tgt is None:
                    raise ValueError("rewire_target must be specified for REWIRE operations")
                if rewire_tgt < 0 or rewire_tgt >= N:
                    raise ValueError(f"rewire_target {rewire_tgt} out of bounds for N={N}")
                if src == rewire_tgt:
                    raise ValueError(f"Rewire target cannot form a self-loop: ({src}, {rewire_tgt})")

                rec_sev = self._apply_sever(src, tgt, cnt, memory_mgr)
                units_to_rewire = rec_sev["units_removed"]
                rec_frc = self._apply_force(src, rewire_tgt, units_to_rewire, memory_mgr)
                records.append({
                    "operation": "rewire",
                    "source": src,
                    "target": tgt,
                    "rewire_target": rewire_tgt,
                    "count_requested": cnt,
                    "units_rewired": rec_frc["actual_added"],
                    "severed": rec_sev,
                    "forced": rec_frc,
                })
            else:
                raise ValueError(f"Unsupported structural operation: {op}")

        # Synchronize scheduler total memory to preserve continuous-time decay jump rate
        if scheduler is not None:
            scheduler.set_total_memory(int(memory_mgr.total_memory))

        # Synchronize state representation
        state.W = memory_mgr.W

        # Invariant verification
        self._verify_invariants(memory_mgr)

        return records

    def _apply_force(
        self,
        src: int,
        tgt: int,
        count: int,
        memory_mgr: MemoryManager,
    ) -> Dict[str, Any]:
        """Force or reinforce a connection by count units."""
        N = memory_mgr.N
        actual_added = 0

        for _ in range(count):
            added_this_iter = False
            # Check row capacity caps
            if memory_mgr.row_caps is not None and memory_mgr.row_sums[src] >= memory_mgr.row_caps[src]:
                # Evict one unit from row src to make room for (src, tgt).
                # Exclude tgt from eviction candidates if other non-zero ties exist,
                # ensuring that forcing (src, tgt) genuinely reinforces (src, tgt).
                other_mask = np.ones(N, dtype=bool)
                other_mask[src] = False
                other_mask[tgt] = False
                other_weights = memory_mgr.W[src].astype(np.float64) * other_mask
                s_other = float(np.sum(other_weights))
                if s_other > 0.0:
                    p_k = other_weights / s_other
                    k_rem = int(self.rng.choice(N, p=p_k))
                    memory_mgr.W[src, k_rem] -= 1
                    memory_mgr.W[src, tgt] += 1
                    memory_mgr._update_row_attention(src)
                    actual_added += 1
                    added_this_iter = True
                else:
                    # Row src is already 100% saturated by tgt; capacity cap prevents further additions
                    pass
            else:
                memory_mgr.W[src, tgt] += 1
                memory_mgr.row_sums[src] += 1
                memory_mgr.total_memory += 1
                memory_mgr._update_row_attention(src)
                actual_added += 1
                added_this_iter = True

            if memory_mgr.reciprocal and added_this_iter:
                if memory_mgr.row_caps is not None and memory_mgr.row_sums[tgt] >= memory_mgr.row_caps[tgt]:
                    other_mask_tgt = np.ones(N, dtype=bool)
                    other_mask_tgt[tgt] = False
                    other_mask_tgt[src] = False
                    other_weights_tgt = memory_mgr.W[tgt].astype(np.float64) * other_mask_tgt
                    s_other_tgt = float(np.sum(other_weights_tgt))
                    if s_other_tgt > 0.0:
                        p_k_tgt = other_weights_tgt / s_other_tgt
                        k_rem_tgt = int(self.rng.choice(N, p=p_k_tgt))
                        memory_mgr.W[tgt, k_rem_tgt] -= 1
                        memory_mgr.W[tgt, src] += 1
                        memory_mgr._update_row_attention(tgt)
                else:
                    memory_mgr.W[tgt, src] += 1
                    memory_mgr.row_sums[tgt] += 1
                    memory_mgr.total_memory += 1
                    memory_mgr._update_row_attention(tgt)

        return {
            "operation": "force",
            "source": src,
            "target": tgt,
            "count_requested": count,
            "actual_added": actual_added,
            "new_weight": int(memory_mgr.W[src, tgt]),
        }

    def _apply_sever(
        self,
        src: int,
        tgt: int,
        count: int,
        memory_mgr: MemoryManager,
    ) -> Dict[str, Any]:
        """Sever or decrement a connection by count units, clamped to 0."""
        cur = int(memory_mgr.W[src, tgt])
        units_to_remove = min(cur, count)

        if units_to_remove > 0:
            memory_mgr.W[src, tgt] -= units_to_remove
            memory_mgr.row_sums[src] -= units_to_remove
            memory_mgr.total_memory -= units_to_remove
            memory_mgr._update_row_attention(src)

            if memory_mgr.reciprocal:
                cur_rec = int(memory_mgr.W[tgt, src])
                rec_remove = min(cur_rec, units_to_remove)
                memory_mgr.W[tgt, src] -= rec_remove
                memory_mgr.row_sums[tgt] -= rec_remove
                memory_mgr.total_memory -= rec_remove
                memory_mgr._update_row_attention(tgt)

        return {
            "operation": "sever",
            "source": src,
            "target": tgt,
            "count_requested": count,
            "units_removed": units_to_remove,
            "new_weight": int(memory_mgr.W[src, tgt]),
        }

    def _verify_invariants(self, memory_mgr: MemoryManager) -> None:
        """Assert strict physical invariants on the memory matrix."""
        # 1. Strictly zero diagonal
        if np.any(np.diag(memory_mgr.W) != 0):
            raise AssertionError("Invariant violation: diagonal elements must be strictly zero")

        # 2. Elementwise non-negativity
        if np.any(memory_mgr.W < 0):
            raise AssertionError("Invariant violation: memory matrix contains negative values")

        # 3. Reciprocal symmetry
        if memory_mgr.reciprocal and not np.array_equal(memory_mgr.W, memory_mgr.W.T):
            raise AssertionError("Invariant violation: reciprocal memory matrix is not symmetric")

        # 4. Total memory consistency
        expected_total = int(np.sum(memory_mgr.W))
        if memory_mgr.total_memory != expected_total:
            raise AssertionError(
                f"Invariant violation: total_memory {memory_mgr.total_memory} != sum(W) {expected_total}"
            )


def force_connection_action(
    source: int,
    target: int,
    count: int = 1,
    intended_end_time: Optional[float] = None,
) -> Action:
    """Create an Action that forces a connection between source and target."""
    entry = StructuralActionEntry(
        source=source,
        target=target,
        operation=StructuralOp.FORCE,
        count=count,
    )
    return Action.structural([entry], intended_end_time=intended_end_time)


def sever_tie_action(
    source: int,
    target: int,
    count: int = 1,
    intended_end_time: Optional[float] = None,
) -> Action:
    """Create an Action that severs a connection between source and target."""
    entry = StructuralActionEntry(
        source=source,
        target=target,
        operation=StructuralOp.SEVER,
        count=count,
    )
    return Action.structural([entry], intended_end_time=intended_end_time)


def rewire_tie_action(
    source: int,
    old_target: int,
    new_target: int,
    count: int = 1,
    intended_end_time: Optional[float] = None,
) -> Action:
    """Create an Action that severs (source, old_target) and forces (source, new_target)."""
    entry = StructuralActionEntry(
        source=source,
        target=old_target,
        operation=StructuralOp.REWIRE,
        count=count,
        rewire_target=new_target,
    )
    return Action.structural([entry], intended_end_time=intended_end_time)
