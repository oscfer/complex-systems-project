"""Advanced Control Package.

Exports:
- ModelPredictiveController: Finite-horizon receding-horizon controller with simulated rollouts.
- MPCConfig: Configuration parameters for MPC.
- TrajectoryCandidate: Candidate intervention sequence record.
- StateEstimator: Layer 2 compliant observation-to-state estimator.
- SimulatedRolloutForecaster: Forward network rollout simulator.
- ForecastResult: Outcome record of rollout forecasts.
- StructuralActuator: Layer 2 interface for direct, bounded network edge manipulation.
- force_connection_action, sever_tie_action, rewire_tie_action: Structural action factories.
"""

from src.control.advanced.forecaster import (
    ForecastResult,
    SimulatedRolloutForecaster,
    StateEstimator,
)
from src.control.advanced.mpc import (
    MPCConfig,
    ModelPredictiveController,
    TrajectoryCandidate,
)
from src.control.advanced.structural import (
    StructuralActuator,
    force_connection_action,
    rewire_tie_action,
    sever_tie_action,
)

__all__ = [
    "ForecastResult",
    "MPCConfig",
    "ModelPredictiveController",
    "SimulatedRolloutForecaster",
    "StateEstimator",
    "StructuralActuator",
    "TrajectoryCandidate",
    "force_connection_action",
    "rewire_tie_action",
    "sever_tie_action",
]
