import numpy as np
from selective_core import probabilities,policies,matched_mask,fit_threshold,summarize,cluster_contributions,paired_cluster_interval

def run():
    states=np.array([[1,0],[1,1],[2,0]])
    q=np.array([[.5,.3,.2],[.2,.3,.5]])
    native=np.full((2,2,3),1/3)
    p=policies(native,q,states)
    pred,score=p["set_consensus"]
    np.testing.assert_allclose(score[0],[.8,.5])
    assert (pred[0]==[1,0]).all()
    assert (score[0]>=.7).tolist()==[True,False]
    # Set agreement must coincide with prefix accumulated mass.
    for tau in (.1,.5,.50001,.7,.8,.80001,.99):
        order=np.argsort(-q[0],kind="stable")
        prefix=order[:np.searchsorted(np.cumsum(q[0,order]),tau,side="left")+1]
        agreement=(states[prefix]==states[prefix[0]]).all(0)
        np.testing.assert_array_equal(score[0]>=tau,agreement)
    # Selecting part of top configuration is always extendable to a valid configuration.
    truth=states[[0,2]]
    m=matched_mask(score,.5,np.array([2,1]))
    assert (m.sum(0)==1).all()
    s=summarize(pred,m,truth,states,["D","P1"])
    assert s["inconsistent_partial_crops"]==0
    empty=summarize(pred,np.zeros_like(m),truth,states,["D","P1"])
    assert empty["answered"]==0 and empty["risk"] is None
    assert empty["nohelmet_abstained"]==1
    # Wrong confident labels are errors; unknown is never counted correct.
    summary=summarize(np.array([[2,0],[1,1]]),np.ones((2,2),bool),truth,states,["D","P1"])
    assert summary["errors"]==3 and summary["risk"]==.75
    assert summary["false_nohelmet_on_helmet"]==1
    # Threshold fitting and equal-coverage evaluation use no labels.
    a=np.array([[.1,.4],[.2,.3],[.3,.2],[.4,.1]])
    np.testing.assert_allclose(fit_threshold(a,.5),[.3,.3])
    assert (matched_mask(a,.5,np.arange(4)).sum(0)==2).all()
    assert np.array_equal(matched_mask(a,.5,np.arange(4)),matched_mask(a,.5,np.arange(4)))
    native,q=probabilities(np.array([[1.,2.,3.,4.],[-80.,80.,-80.,80.]]),states,2.)
    assert np.isfinite(q).all() and np.allclose(q.sum(1),1)
    assert np.allclose(native.sum(-1),1)
    e,a=cluster_contributions(pred,m,truth,np.array([0,1]),2)
    result=paired_cluster_interval((e,a),(e,a),np.array([[1,1],[2,0],[0,2]]))
    assert result["delta_risk_pp"]==0 and result["percentile95_pp"]==[0.,0.]
    print("PASS: prefix agreement, exact matched coverage, unknown accounting, error counts, thresholds, numeric stability, paired bootstrap")

if __name__=="__main__":run()
