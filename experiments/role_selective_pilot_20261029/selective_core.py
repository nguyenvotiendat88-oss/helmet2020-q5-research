"""Fixed selective role policies; no training, conformal claim, or TEST access."""
import numpy as np

METHODS=("crop_confidence","native_role_confidence","role_marginal",
         "topclass_role_marginal","set_consensus")

def probabilities(logits,states,temperature):
    if not np.isfinite(temperature) or temperature<=0:raise ValueError("Positive T required")
    z=np.asarray(logits,dtype=np.float64)/temperature
    states=np.asarray(states,dtype=int)
    if z.ndim!=2 or z.shape[1]!=2*states.shape[1] or not np.isfinite(z).all():
        raise ValueError("Invalid native logits")
    roles=states.shape[1]
    p=np.exp(-np.logaddexp(0.,-z))
    occupancy,helmet=p[:,:roles],p[:,roles:]
    native=np.stack((1-occupancy,occupancy*helmet,occupancy*(1-helmet)),axis=-1)
    logscore=np.stack([np.log(np.maximum(native[:,np.arange(roles),s],1e-12)).sum(-1)
                       for s in states],axis=1)
    q=np.exp(logscore-logscore.max(1,keepdims=True))
    q/=q.sum(1,keepdims=True)
    return native,q

def policies(native,q,states):
    states=np.asarray(states,dtype=int)
    n,c=q.shape;roles=states.shape[1]
    if c!=len(states) or native.shape!=(n,roles,3):raise ValueError("Shape mismatch")
    if not np.allclose(q.sum(1),1) or not np.isfinite(q).all() or (q<0).any():
        raise ValueError("Invalid distribution")
    top=q.argmax(1);topstates=states[top]
    marginal=np.stack([np.stack([q[:,states[:,r]==s].sum(1) for s in range(3)],axis=1)
                       for r in range(roles)],axis=1)
    order=np.argsort(-q,axis=1,kind="stable")
    ordered=q[np.arange(n)[:,None],order]
    same=states[order]==topstates[:,None,:]
    prefix=np.cumprod(same.astype(int),axis=1).astype(bool)
    consensus=(ordered[:,:,None]*prefix).sum(1)
    topmarg=marginal[np.arange(n)[:,None],np.arange(roles)[None,:],topstates]
    return {
        "crop_confidence":(topstates,np.repeat(q.max(1)[:,None],roles,axis=1)),
        "native_role_confidence":(native.argmax(-1),native.max(-1)),
        "role_marginal":(marginal.argmax(-1),marginal.max(-1)),
        "topclass_role_marginal":(topstates,topmarg),
        "set_consensus":(topstates,np.clip(consensus,0,1))}

def matched_mask(confidence,coverage,tiebreak):
    n,roles=confidence.shape
    if not 0<=coverage<=1:raise ValueError("Coverage out of range")
    count=int(np.floor(n*coverage+1e-9))
    keep=np.zeros((n,roles),dtype=bool)
    for r in range(roles):
        order=np.lexsort((tiebreak,-confidence[:,r]))
        keep[order[:count],r]=True
    return keep

def fit_threshold(confidence,coverage):
    n=confidence.shape[0]
    count=max(1,int(np.floor(n*coverage)))
    # Calibration quantiles use confidence only, never GT labels.
    return np.sort(confidence,axis=0)[n-count]

def ratio(a,b):
    return float(a/b) if b else None

def summarize(pred,keep,truth,states,role_names):
    pred=np.asarray(pred);keep=np.asarray(keep,dtype=bool);truth=np.asarray(truth)
    if pred.shape!=keep.shape or truth.shape!=pred.shape:raise ValueError("Alignment differs")
    n,roles=pred.shape
    errors=(pred!=truth)&keep
    present=truth!=0;nohelmet=truth==2
    false_nohelmet=(pred==2)&(truth==1)&keep
    absent_accusation=(pred==2)&(truth==0)&keep
    partial=np.where(keep,pred,-1)
    consistent=np.any(np.all((partial[:,None,:]<0)|(partial[:,None,:]==states[None,:,:]),axis=-1),axis=1)
    accepted=int(keep.sum())
    result={"crops":n,"role_decisions":n*roles,"answered":accepted,"errors":int(errors.sum()),
        "coverage":ratio(accepted,n*roles),"risk":ratio(int(errors.sum()),accepted),
        "GT_present_answered":int((present&keep).sum()),
        "GT_present_risk":ratio(int((errors&present).sum()),int((present&keep).sum())),
        "GT_present_coverage":ratio(int((present&keep).sum()),int(present.sum())),
        "false_nohelmet_on_helmet":int(false_nohelmet.sum()),
        "false_nohelmet_on_helmet_rate":ratio(int(false_nohelmet.sum()),int((truth==1).sum())),
        "false_nohelmet_on_absent":int(absent_accusation.sum()),
        "false_nohelmet_on_absent_rate":ratio(int(absent_accusation.sum()),int((truth==0).sum())),
        "missed_nohelmet_answered":int((nohelmet&keep&(pred!=2)).sum()),
        "nohelmet_abstained":int((nohelmet&~keep).sum()),
        "nohelmet_recall_including_abstentions":ratio(int((nohelmet&keep&(pred==2)).sum()),int(nohelmet.sum())),
        "inconsistent_partial_crops":int((~consistent).sum()),
        "inconsistent_partial_rate":float((~consistent).mean()),
        "fully_answered_crops":int(keep.all(1).sum()),
        "partial_answered_crops":int((keep.any(1)&~keep.all(1)).sum()),
        "all_abstained_crops":int((~keep.any(1)).sum()),"roles":[]}
    for r,name in enumerate(role_names):
        row={"role":name,"support":n,"answered":int(keep[:,r].sum()),
             "coverage":float(keep[:,r].mean()),"risk":ratio(int(errors[:,r].sum()),int(keep[:,r].sum())),"states":[]}
        for state,label in enumerate(("absent","helmet","nohelmet")):
            gt=truth[:,r]==state
            nr=int(gt.sum());ka=int((gt&keep[:,r]).sum());wrong=int((gt&errors[:,r]).sum())
            row["states"].append({"state":state,"label":label,"GT_support":nr,
                "answered":ka,"coverage":ratio(ka,nr),"errors":wrong,"risk":ratio(wrong,ka)})
        result["roles"].append(row)
    supported=[s["risk"] for r in result["roles"] for s in r["states"] if s["risk"] is not None]
    result["macro_role_state_risk_supported_only"]=float(np.mean(supported)) if supported else None
    result["macro_role_state_supported_cells"]=len(supported)
    return result

def cluster_contributions(pred,keep,truth,clip_index,clip_count):
    errors=((pred!=truth)&keep).sum(1)
    answers=keep.sum(1)
    return np.bincount(clip_index,weights=errors,minlength=clip_count),np.bincount(clip_index,weights=answers,minlength=clip_count)

def paired_cluster_interval(candidate,baseline,weights):
    ec,ac=candidate;eb,ab=baseline
    den_c=weights@ac;den_b=weights@ab
    valid=(den_c>0)&(den_b>0)
    deltas=(weights[valid]@ec)/den_c[valid]-(weights[valid]@eb)/den_b[valid]
    return {"delta_risk_pp":100*(ec.sum()/ac.sum()-eb.sum()/ab.sum()),
            "percentile95_pp":[float(x) for x in np.quantile(100*deltas,[.025,.975])],
            "resamples":len(deltas),"scope":"paired_clip_bootstrap_fixed_masks; no_fit_or_seed_uncertainty"}
