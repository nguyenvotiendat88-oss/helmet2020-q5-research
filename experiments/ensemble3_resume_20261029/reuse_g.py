"""Reuse completed G measurements only after identity/recipe/hash validation."""
import json
from pathlib import Path
from helmetlab.common import digest
from helmetlab.kit_runner import canonical,atomic_json

OLD=Path("/workspace/runs/ensemble3_20261029")
def verified_original_G(new_binding,new_root):
    binding=json.loads((OLD/"study.json").read_text())
    previous={k:v for k,v in binding.items() if k!="implementation"}
    current={k:v for k,v in new_binding.items() if k!="implementation"}
    if previous!=current:raise RuntimeError("G reuse recipe, data, checkpoints or runtime differ")
    for name,h in binding["implementation"].items():
        if digest(Path("/workspace/ensemble3_20261029")/name)!=h:
            raise RuntimeError("Old source changed")
    result=json.loads((OLD/"G/result.json").read_text())
    if result["status"]!="completed" or result["study_sha256"]!=canonical(binding):
        raise RuntimeError("G is not a verified completed result")
    checks={}
    def verify(path,h):
        actual=digest(path)
        if actual!=h:raise RuntimeError("G artifact hash mismatch: "+str(path))
        checks[str(path)]=actual
    for seed in (0,1,2):
        rec=binding["checkpoints"][f"G_seed{seed}"];verify(Path(rec["path"]),rec["sha256"])
    for filename,artifact in (("cache_complete.json","proposal_probabilities.npy"),
                              ("cal_probs_complete.json","cal_probs.npz")):
        rec=json.loads((OLD/"G"/filename).read_text())
        if rec["study_sha256"]!=canonical(binding):raise RuntimeError("G cache identity mismatch")
        verify(OLD/"G"/artifact,rec["sha256"])
    selection=json.loads((OLD/"G/selection.json").read_text())
    if result["selection"]!=selection:raise RuntimeError("G selection drift")
    for name,rec in result["evaluation"].items():
        if rec!=json.loads((OLD/"G"/f"eval_{name}.json").read_text()):
            raise RuntimeError("G metric receipt differs")
        verify(OLD/"G"/f"eval_{name}_predictions.jsonl",rec["predictions_sha256"])
    proof={"source_run":str(OLD),"original_study_sha256":canonical(binding),
           "new_study_sha256":canonical(new_binding),"recipe_identical_except_implementation":True,
           "G_code_path_unchanged":True,"artifact_hashes":checks,
           "original_result_sha256":digest(OLD/"G/result.json"),
           "original_result_retains_original_study_identity":True,"test_accessed":False}
    atomic_json(new_root/"G_reuse_evidence.json",proof)
    atomic_json(new_root/"G/result.json",result)
    return result
