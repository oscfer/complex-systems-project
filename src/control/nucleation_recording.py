"""Passive event diagnostics; no RNG, physics or controller information changes."""
from __future__ import annotations
import numpy as np
from src.control.recording import RolloutRecorder


class NucleationRecorder(RolloutRecorder):
    def __init__(self, config, cores, *, J=1., T=.3):
        super().__init__(config);self.cores={k:list(v) for k,v in cores.items()};self.J=J;self.T=T
        self.flips=[];self.core_samples=[];self.core_transitions={k:[] for k in cores};self._x=None;self._y=None

    def _core(self,state,t):
        for name,nodes in self.cores.items():
            value=bool(np.mean(state.x[nodes,0]<0)>=.75 and np.mean(state.y[nodes,0]<0)>=.75)
            trace=self.core_transitions[name]
            if not trace or trace[-1][1]!=value:trace.append([float(t),value])

    def initialize(self,state,ledger,target_evaluator,problem):
        super().initialize(state,ledger,target_evaluator,problem)
        self._x=state.x.copy();self._y=state.y.copy();self._core(state,0.)

    def _capture_network_sample(self,state,ledger,t,problem):
        if (self.config.record_network_frames and 0 < t < problem.horizon
                and abs(t/2-round(t/2)) < 1e-10
                and len(self.network_frames) < self.config.max_network_frames-1):
            self._record_network_frame(state,ledger,t=t)

    def on_event_state(self,state,current_time,ledger,target_evaluator):
        super().on_event_state(state,current_time,ledger,target_evaluator)
        changed=False
        for layer,old,new in [('x',self._x,state.x),('y',self._y,state.y)]:
            for i,alpha in np.argwhere(old!=new):
                changed=True;i=int(i);alpha=int(alpha)
                row=state.W[i,:].astype(float);total=row.sum();A=row/total if total else np.zeros_like(row)
                social=self.J*float(A@state.y[:,alpha]);consistency=float(state.K[i]*(state.y[i,alpha] if layer=='x' else state.x[i,alpha]))
                bias=float(state.b[i,alpha]) if layer=='x' else 0.
                external=float(state.g[i]*ledger.current_field[i,alpha]) if layer=='y' else 0.
                field=consistency+bias if layer=='x' else social+consistency+external
                self.flips.append(dict(t=float(current_time),node=i,layer=layer,component=alpha,
                    before=float(old[i,alpha]),after=float(new[i,alpha]),field=field,temperature=self.T,
                    social=social if layer=='y' else 0.,consistency=consistency,bias=bias,external=external,
                    blue_neighbor_weights=[[int(j),float(A[j])] for j in np.flatnonzero((A>0)&(state.y[:,alpha]<0))] if layer=='y' else []))
        if changed:self._core(state,current_time)
        self._x[:]=state.x;self._y[:]=state.y

    def _record_scalar_point(self,state,ledger,target_evaluator,problem,t):
        super()._record_scalar_point(state,ledger,target_evaluator,problem,t)
        W=state.W.astype(float);totals=W.sum(axis=1,keepdims=True)
        A=np.divide(W,totals,out=np.zeros_like(W),where=totals>0);received=A.sum(axis=0)
        for name,nodes in self.cores.items():
            outside=[i for i in range(state.N) if i not in nodes]
            self.core_samples.append(dict(t=float(t),core=name,
                private_blue=float(np.mean(state.x[nodes,0]<0)),public_blue=float(np.mean(state.y[nodes,0]<0)),
                outside_private_blue=float(np.mean(state.x[outside,0]<0)) if outside else None,outside_public_blue=float(np.mean(state.y[outside,0]<0)) if outside else None,
                internal_attention=float(A[np.ix_(nodes,nodes)].sum()/len(nodes)),
                outside_attention_to_core=float(A[np.ix_(outside,nodes)].sum()/len(outside)) if outside else None,
                received_mean=float(received[nodes].mean()),received_max=float(received[nodes].max())))

    def to_dict(self):
        data=super().to_dict();data['nucleation']=dict(cores=self.cores,core_transitions=self.core_transitions,core_samples=self.core_samples,flips=self.flips,
            interpretation='Exact changes only. Public field uses neighbors and own private state; private field uses own public state and bias. No claim of counterfactual causation.')
        return data
