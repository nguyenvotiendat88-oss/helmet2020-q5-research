"""Regression: old G path unchanged, original Q5 exact prefix restored."""
import ast,json,sys
from pathlib import Path
import torch
import run_ensemble as e
from reuse_g import verified_original_G

def body(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
def main():
    e.seed_rng(0,2)
    old=Path("/workspace/ensemble3_20261029/run_ensemble.py")
    here=Path(__file__).parent
    for name in ("verify_prefix","cache_probabilities","cal_probs","analyze","postprocess"):
        assert ast.dump(body(old,name),include_attributes=False)==ast.dump(body(here/"run_ensemble.py",name),include_attributes=False)
    e.OUT.mkdir(parents=True,exist_ok=True)
    with e.execution_lock("cuda:0"):
        binding,frames,proposals,plans,offsets,hashes,labels,prior=e.setup()
        g=verified_original_G(binding,e.OUT)
        models=e.load_models("Q5",binding)
        assert all(p.requires_grad for m in models for p in m.parameters())
        hashes_before=[e.digest(Path(binding["checkpoints"][f"Q5_seed{s}"]["path"])) for s in (0,1,2)]
        proof=e.verify_prefix("Q5",models,frames,proposals,plans,hashes,labels)
        assert proof["original_vs_saved_exact"] and proof["replay_vs_saved_exact"]
        assert not any(p.grad is not None for m in models for p in m.parameters())
        assert hashes_before==[e.digest(Path(binding["checkpoints"][f"Q5_seed{s}"]["path"])) for s in (0,1,2)]
        result={"status":"passed","original_Q5_prefix_exact":proof,"G_completed_artifacts_verified":True,
                "inference_and_scoring_functions_unchanged":True,"checkpoint_hashes_unchanged":True,
                "training_updates":0,"no_parameter_gradients":True,"test_accessed":False}
        e.atomic_json(here/"fix_verification.json",result)
        print(json.dumps(result),flush=True)
if __name__=="__main__":main()
