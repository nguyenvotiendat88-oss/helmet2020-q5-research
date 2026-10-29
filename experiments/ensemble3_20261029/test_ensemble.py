"""CPU regression tests of arithmetic, frozen selection, aligned cache and replay."""
import copy,json,tempfile
from pathlib import Path
import run_ensemble as run
import numpy as np
import torch
torch.set_num_threads(2)
from PIL import Image
from posthoc_core import combine,adjust,fit_temperature,train_prior,select_tau
from queue_ensemble import dependency_state

class Toy(torch.nn.Module):
    kit_arm="G"
    def __init__(self,offset):
        super().__init__()
        self.register_buffer("offset",torch.linspace(-1,1,10)+offset)
    def forward(self,x):
        return self.offset[None].expand(len(x),-1)+x.mean((1,2,3))[:,None]*.01

def main():
    p=np.array([[.8,.2],[.3,.7],[.9,.1],[.4,.6]])
    q=combine([p,p,p])
    np.testing.assert_allclose(q,p,atol=1e-15)
    assert np.array_equal(adjust(p),p)
    np.testing.assert_allclose(combine([p,1-p]),np.full(p.shape,.5))
    prior=train_prior([99,0]);assert (prior>0).all()
    changed=adjust(np.array([[.5,.5]]),1.,1.,prior)
    assert changed[0,1]>changed[0,0]  # subtract log prior promotes rare class
    fit=fit_temperature(p,np.array([0,1,1,0]))
    assert .5<=fit["temperature"]<=10 and fit["fit_class_NLL_after"]<=fit["fit_class_NLL_before"]+1e-12
    # Confidence-only parameter changes preserve categorical class ordering for tau0.
    assert np.array_equal(adjust(p,3).argmax(1),p.argmax(1))
    rows=[{"tau":0.,"macro_AP50":.3,"weighted_AP50":.8},
          {"tau":.25,"macro_AP50":.4,"weighted_AP50":.79},
          {"tau":.5,"macro_AP50":.31,"weighted_AP50":.8},
          {"tau":1.,"macro_AP50":.31,"weighted_AP50":.8}]
    assert select_tau(rows)["tau"]==.5
    assert dependency_state({"status":"completed"},True,False)=="ready"
    assert dependency_state({"status":"running"},False,True)=="waiting"
    try:dependency_state({"status":"failed"},False,False)
    except RuntimeError:pass
    else:raise AssertionError("Failed dependency accepted")
    labels=run.LabelMap(run.ROOT/"manifests/classes.json")
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);run.OUT=root/"out";run.OUT.mkdir();(run.OUT/"G").mkdir()
        run.IMAGES=root;run.DEVICE="cpu"
        Image.new("RGB",(32,32),(31,61,111)).save(root/"image.png")
        hashes={"image.png":run.digest(root/"image.png")}
        frames=[{"image_id":f"test{i:03d}","image":"image.png","clip_id":f"clip{i//20}",
                 "frame_index":i+1,"site_id":"fixture","width":32,"height":32,
                 "objects":[{"bbox":[0,0,32,32],"class_id":0}]} for i in range(120)]
        props=[{**r,"objects":[] if i==0 else [
                  {"bbox":[0,0,32,32],"class_id":0,"score":.8},
                  {"bbox":[0,0,32,32],"class_id":1,"score":.6},
                  {"bbox":[2,2,30,30],"class_id":0,"score":.5}]} for i,r in enumerate(frames)]
        plans=[run.fast_eval._unique_boxes(f,p) for f,p in zip(frames,props)]
        assert len(plans[1].boxes)==2 and plans[1].detector_scores[0]==.8
        offsets=np.r_[0,np.cumsum([len(p.boxes) for p in plans])]
        models=[Toy(0.),Toy(.1),Toy(-.1)]
        binding={"fixture":"not-real-model","n":120}
        calls=0
        def interrupt():
            nonlocal calls
            calls+=1
            if calls==111:raise RuntimeError("intentional mid-cache interruption")
        original_guard=run.guard;run.guard=interrupt
        try:run.cache_probabilities("G",models,binding,frames,plans,offsets,hashes,labels)
        except RuntimeError as exc:
            assert "intentional" in str(exc)
        else:raise AssertionError("Interruption did not happen")
        partial=json.loads((run.OUT/"G/cache_progress.json").read_text())
        assert partial["completed_frames"]==100
        run.guard=original_guard
        cache=run.cache_probabilities("G",models,binding,frames,plans,offsets,hashes,labels)
        assert np.isfinite(cache).all() and np.allclose(cache.sum(1),1)
        resize=run.transforms.Resize((224,224))
        mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
        resized=run.fast_eval._prepare_resized(root,plans[119],.05,resize,hashes,run.km._rgb,run.km.crop_box)
        x=run.fast_eval._batched_tensorize(resized,None,mean,std)
        expected=combine([run.km.decode(m(x),"G",labels) for m in models])
        np.testing.assert_array_equal(cache[offsets[119]:offsets[120]],expected)
        # Same checkpoint single-member output reconstructed from saved probabilities.
        original=run.km.classifier_predictions(models[0],root,frames[:2],props[:2],labels,device="cpu",
                    batch=64,margin=.05,size=224,score_thresh=.001,nms_thresh=.5,max_detections=300,image_hashes=hashes)
        single=run.km.decode(models[0](x),"G",labels)
        replay=[run.postprocess(frames[0],plans[0],np.empty((0,36))),
                run.postprocess(frames[1],plans[1],single)]
        assert replay==original
        # Completed cache accepts unchanged bytes and rejects tampering.
        cached=run.cache_probabilities("G",models,binding,frames,plans,offsets,hashes,labels)
        assert np.array_equal(cached,cache);del cache,cached
        mutable=np.load(run.OUT/"G/proposal_probabilities.npy",mmap_mode="r+")
        mutable[0,0]+=.1;mutable.flush();del mutable
        try:run.cache_probabilities("G",models,binding,frames,plans,offsets,hashes,labels)
        except RuntimeError:pass
        else:raise AssertionError("Tampered cache accepted")
    result={"status":"passed","dataset_training_started":False,"GPU_used":False,"test_accessed":False,
        "checks":["probability_ensemble_exact_definition","T1_tau0_identity",
                  "posthoc_prior_adjustment_direction","Tfit_NLL_nonincreasing","tau_selection_weighted_floor_tie",
                  "dependency_gate","duplicate_boxes_dedup_empty_frames","cache_interrupt_resume_aligned",
                  "replayed_single_model_matches_original_pipeline","cache_tamper_rejected"]}
    path=Path(__file__).parent/"verification.json";path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=="__main__":main()
