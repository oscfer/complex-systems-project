"""One final notebook entry point; reuse the durable sequential supervisor."""
from src.control.optimizers.response_low_power import final_plan
from src.control.sequential_batch import run_pilot


def run_final(out_dir,*,resume_bundle=None,plan_only=False,wall_time_limit=None):
    return run_pilot(out_dir,resume_bundle=resume_bundle,plan_only=plan_only,
                     wall_time_limit=wall_time_limit,plan=final_plan())
