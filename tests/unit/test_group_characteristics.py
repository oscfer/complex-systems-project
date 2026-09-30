import numpy as np
from src.control.optimizers.group_characteristics import describe_group


def test_direction_overlap_missingness_and_separate_private():
    W=np.zeros((5,5));W[2,0]=1;W[2,1]=1;W[3,0]=1;W[0,1]=1
    d=describe_group(W,[1,np.nan,-1,1,1],[0,1],groups=[0,0,1,2,2])
    assert d['external_audience_count']==2
    assert abs(d['audience_overlap_fraction']-1/3)<1e-12
    assert d['within_group_attention_sum']==1
    assert d['recipient_public_observed_count']==1
    assert d['audience_public_blue_fraction']==.5
    assert d['audience_group_count']==2
    assert 'recipient_private_blue_fraction' not in d
