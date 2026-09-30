import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.profiles import RunProfile, build_realism_runner
from src.control.contracts import ControlProblem, TargetType
from src.control.controllers import ObservableSelectorPolicy
from src.control.engine import run_controlled_simulation
from src.visualization.plotting import animate_network_temporal

def main():
    profile = RunProfile(
        topic="campaign", tier="light", N=12, D=1, horizon=10.0,
        establishment_time=4.0, n_origins=1, seed_base=1234
    )
    runner = build_realism_runner(profile, origin_seed=1234)
    traj_est = runner.run(horizon=profile.establishment_time)
    
    problem = ControlProblem(
        problem_id="render_test",
        target_type=TargetType.REVERSAL,
        target_params={"threshold": 0.5},
        N=12,
        D=1,
        k=3,
        exposure_cap=20.0,
        amplitude_cap=4.0,
        release_deadline=5.0,
        holding_duration=2.0,
        horizon=10.0,
        accessible_roster=tuple(range(12))
    )
    
    controller = ObservableSelectorPolicy(
        selector="random", nominal_amplitude=2.0, duration=3.0, quota=0.5, is_feedback=True
    )
    
    res, final_state, ledger = run_controlled_simulation(
        initial_state=traj_est.final_state,
        problem=problem,
        controller=controller,
        sim_rng=np.random.default_rng(42),
        controller_rng=np.random.default_rng(43),
    )
    
    anim_data = animate_network_temporal(
        traj_est.timeseries,
        filename=None,
        max_frames=10
    )
    
    print(f"Successfully rendered animation with {anim_data.get('num_frames', 0)} frames")
    
if __name__ == "__main__":
    main()
