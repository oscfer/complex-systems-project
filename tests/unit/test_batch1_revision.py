"""Bounded regression checks for batch1 scientific and recovery corrections."""
import time
from dataclasses import asdict
import numpy as np
import pytest
from src.control.contracts import RolloutResult
from src.control.optimizers.search import PolicyCandidate
from src.control.optimizers.campaign import (
    CampaignConfig, CampaignTimeLimit, _cached_forecasts, _actual_queue,
    _case_problem, run_batch1_campaign, reconstruct_t0_observation, _origin_state)
from src.config.tiered_profiles import LIGHT_TIERED


def test_all_access_and_matched_size_comparisons():
    config=CampaignConfig(search_origins=('development_01',))
    p=_case_problem(LIGHT_TIERED,1656358726,'broad_diagnostic',-1,'test')
    assert len(p.accessible_roster)==40
    queue=_actual_queue(config,{},LIGHT_TIERED,'development_01',1656358726)
    assert len(queue)==9
    for size in (4,10):
        sets=[set(nodes) for nodes,*_ in queue if len(nodes)==size]
        assert len(sets)==4 and len({tuple(sorted(s)) for s in sets})>1
    plan=run_batch1_campaign('unused',config,LIGHT_TIERED,{},[],{'development_01':1656358726},plan_only=True)
    assert plan['planned_actual_units']==11
    assert plan['search_forecast_simulations']==36


def test_forecast_cache_survives_interruption(tmp_path):
    candidate=PolicyCandidate('test',(0,),((0,-1.),),1.,1.)
    calls=[]
    def evaluate(c,seeds):
        calls.extend(seeds)
        return [RolloutResult('p',c.candidate_id,'estimated',s,True,True,False,None,6.,1.,1,summaries={'horizon':6.}) for s in seeds]
    wrapped=_cached_forecasts(evaluate,tmp_path,time.monotonic()+60)
    wrapped(candidate,(1,))
    expired=_cached_forecasts(evaluate,tmp_path,time.monotonic()-1)
    assert expired(candidate,(1,))[0].future_id==1
    with pytest.raises(CampaignTimeLimit): expired(candidate,(1,2))
    resumed=_cached_forecasts(evaluate,tmp_path,time.monotonic()+60)
    assert len(resumed(candidate,(1,2)))==2
    assert calls==[1,2]
