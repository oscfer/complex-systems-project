"""Exact kernel NON-equivalence check; not an NDlib simulation/parity claim.

Reference rule was inspected in the checksum-pinned NDlib 6.0.1 wheel.
This enumerates its q=1 two-node rule analytically, then compares the local
positive-temperature public heat-bath kernel in a stripped-down case.
"""
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.core.heat_baths import binary_glauber_prob_plus


def compare():
    states=list(itertools.product((-1,1),repeat=2))
    reference=[[0.]*4 for _ in states]
    heatbath=[[0.]*4 for _ in states]
    for row,state in enumerate(states):
        for listener in range(2):
            neighbour=state[1-listener]
            copied=list(state);copied[listener]=neighbour
            reference[row][states.index(tuple(copied))]+=.5
            p=binary_glauber_prob_plus(neighbour/.3)
            for sign,probability in ((1,p),(-1,1-p)):
                updated=list(state);updated[listener]=sign
                heatbath[row][states.index(tuple(updated))]+=.5*probability
    tv=[sum(abs(a-b) for a,b in zip(r,h))/2 for r,h in zip(reference,heatbath)]
    assert all(abs(sum(row)-1)<1e-12 for row in reference+heatbath)
    assert min(tv)>0
    return {'classification':'non_equivalent_positive_temperature',
            'ndlib_version':'6.0.1','ndlib_wheel_sha256':'3e1fe7426cfb00a1dc0b96c54aadace49744c91259736db9eab56f0e6c171346',
            'ndlib_execution':False,'method':'Exact enumeration of inspected reference rule, not installed package execution',
            'states':states,'qvoter_q1_transition_matrix':reference,'heatbath_T_0_3_transition_matrix':heatbath,
            'row_total_variation':tv,
            'scope':'Two nodes, one symmetric unit edge, public-only updates, J=1, K=b=u=0, frozen graph; conditional on one public update. Not the full production model.'}


if __name__=='__main__':
    path=ROOT/'research/experiments/qvoter_non_equivalence.json'
    result=compare();path.write_text(json.dumps(result,indent=2)+'\n')
    print(result['classification'], 'row TV:',result['row_total_variation'])
    print(path)
