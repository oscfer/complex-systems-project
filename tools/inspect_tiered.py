"""Validate and preview the declared Tiered-population constructed origin."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config.tiered_profiles import LIGHT_TIERED
from src.config.scenarios import tiered_origin_provider, load_tiered_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-spec", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()
    profile = LIGHT_TIERED.model_copy(update={"N": args.n})
    spec_path = Path(profile.scenario_spec_path)
    spec = load_tiered_spec(str(spec_path))
    state = tiered_origin_provider(profile, args.seed)
    core = state.blocks <= 3
    row_mass = state.W.sum(axis=1)
    output = {
        "valid": True,
        "schema_version": spec["schema_version"],
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "N": state.N,
        "t": state.t,
        "tier_counts": {str(i): int(np.count_nonzero(state.blocks == i)) for i in range(1, 6)},
        "jurisdiction_counts": {name: state.scenario_metadata["jurisdictions"].count(name)
                                for name in dict.fromkeys(state.scenario_metadata["jurisdictions"])},
        "public_plus_share": float(np.mean(state.y[:, 0] > 0)),
        "private_plus_share": float(np.mean(state.x[:, 0] > 0)),
        "public_mean": float(np.mean(state.y[:, 0])),
        "private_mean": float(np.mean(state.x[:, 0])),
        "attention_core_target_share": spec["core_attention_share"],
        "attention_core_realized_share": float(state.W[:, core].sum() / state.W.sum()) if state.W.sum() else None,
        "total_memory": int(state.W.sum()),
        "row_memory_min": int(row_mass.min()),
        "row_memory_max": int(row_mass.max()),
        "self_edges": int(np.trace(state.W)),
        "encounter_nonzero_fraction": float(np.mean(state.opportunity_matrix > 0)),
        "encounter_within": float(spec["opportunity"]["within_jurisdiction"]),
        "encounter_between": float(spec["opportunity"]["between_jurisdictions"]),
        "provenance": state.scenario_metadata["source"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
