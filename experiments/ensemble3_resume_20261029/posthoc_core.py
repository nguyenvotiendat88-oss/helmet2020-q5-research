"""Bounded categorical postprocessing of an equal-weight probability ensemble."""
import math
import numpy as np

TAUS=(0.,.25,.5,1.)

def combine(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=3 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1.,atol=1e-6):
        raise ValueError("Expected seed x proposal x class probabilities")
    return p.mean(0)

def train_prior(counts):
    counts=np.asarray(counts,dtype=np.float64)
    if counts.ndim!=1 or (counts<0).any() or not np.isfinite(counts).all() or counts.sum()<=0:
        raise ValueError("TRAIN counts required")
    return (counts+1.)/(counts.sum()+len(counts))

def adjust(probabilities,temperature=1.,tau=0.,prior=None):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1.,atol=1e-6):
        raise ValueError("Expected normalized class probabilities")
    if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(tau) or tau<0:
        raise ValueError("Invalid T/tau")
    if temperature==1. and tau==0.:return p.copy()
    scores=np.log(np.maximum(p,1e-300))/temperature
    if tau:
        prior=np.asarray(prior,dtype=np.float64)
        if prior.shape!=(p.shape[1],) or (prior<=0).any() or not np.isfinite(prior).all() or not np.isclose(prior.sum(),1.):
            raise ValueError("Positive normalized TRAIN prior required")
        scores=scores-tau*np.log(prior)[None]
    q=np.exp(scores-scores.max(1,keepdims=True))
    return q/q.sum(1,keepdims=True)

def nll(p,ids):
    ids=np.asarray(ids)
    if ids.shape!=(len(p),) or not np.issubdtype(ids.dtype,np.integer) or (ids<0).any() or (ids>=p.shape[1]).any():
        raise ValueError("Aligned class IDs required")
    return float(-np.log(np.maximum(p[np.arange(len(ids)),ids],1e-300)).mean())

def fit_temperature(probabilities,ids):
    def bounded(t):return min(10.,max(.5,math.exp(t)))
    def objective(t):return nll(adjust(probabilities,bounded(t)),ids)
    lo,hi=math.log(.5),math.log(10.);phi=(math.sqrt(5)-1)/2
    x=hi-phi*(hi-lo);y=lo+phi*(hi-lo);fx=objective(x);fy=objective(y)
    for _ in range(64):
        if fx<fy:
            hi,y,fy=y,x,fx;x=hi-phi*(hi-lo);fx=objective(x)
        else:
            lo,x,fx=x,y,fy;y=lo+phi*(hi-lo);fy=objective(y)
    options=[(1.,objective(0.)),(bounded(lo),objective(lo)),(bounded(hi),objective(hi))]
    t,value=min(options,key=lambda x:(x[1],abs(x[0]-1)))
    return {"temperature":t,"fit_class_NLL_before":objective(0.),"fit_class_NLL_after":value,
            "fit_samples":len(ids),"bounds":[.5,10.],"space":"log of arithmetic mean decoded36 probabilities"}

def select_tau(rows):
    base=next(r for r in rows if r["tau"]==0.)
    eligible=[r for r in rows if r["weighted_AP50"]>=base["weighted_AP50"]-.005]
    return min(eligible,key=lambda r:(-r["macro_AP50"],r["tau"]))
