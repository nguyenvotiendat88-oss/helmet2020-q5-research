"""CPU mathematical, implementation-parity and historical-control checks."""
import ast,json,copy
from pathlib import Path
import torch
torch.set_num_threads(2)
import run_pilot as run
import check_valid_state_objective as check
from helmetlab.kit_runner import canonical

class Normalize(ast.NodeTransformer):
    def visit_Subscript(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=="PLAN" and isinstance(node.slice,ast.Constant):
            if node.slice.value=="seed":return ast.copy_location(ast.Constant(0),node)
            if node.slice.value=="epochs_per_arm":return ast.copy_location(ast.Constant(12),node)
        return self.generic_visit(node)

def function(path,name,normalize=False):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    if normalize:node=Normalize().visit(node)
    return ast.dump(node,include_attributes=False)

def main():
    here=Path(__file__).resolve().parent
    old=Path("/workspace/gq5_long12_20261027/run_gq5_12.py")
    assert function(old,"train_arm")==function(here/"pilot_engine.py","train_arm",True)
    assert function(old,"evaluate_milestone")==function(here/"pilot_engine.py","evaluate_milestone",True)
    check.run()
    native_loss=run.native.class_loss
    states=torch.tensor(json.loads((run.e.ROOT/"manifests/classes.json").read_text())["classes"][0]["states"])
    from helmetlab.labels import LabelMap
    labels=LabelMap(run.e.ROOT/"manifests/classes.json")
    states=torch.tensor(labels.states)
    original=json.loads((run.ORIGINAL/"study.json").read_text())
    assert canonical(original)==run.ORIGINAL_SHA
    checkpoints=run.verify_controls(original)
    assert set(checkpoints)=={"G","Q5"}
    for mode in run.VARIANTS:
        run.configure(mode)
        assert run.e.PLAN["epochs_per_arm"]==4 and run.e.PLAN["val_epochs"]==[4]
        assert run.e.PLAN["scheduler_T_max"]==12 and run.e.PLAN["lr"]==1e-4
        assert run.native.class_loss is native_loss and run.e.km.class_loss is run.objective
        for arm in ("G","Q5"):
            model=run.e.km.build_classifier(arm,seed=0,backbone_path=run.e.ROOT/"pretrained/resnet18-f37072fd.pth")
            expected=json.loads((run.ORIGINAL/arm/"initialization.json").read_text())
            assert run.e.tensor_digest(model.state_dict())==expected["initial_model_sha256"]
            torch.manual_seed(123)
            features=torch.randn(4,512,7,7)
            ids=torch.tensor([0,1,2,3])
            logits=model.head(features)
            loss=run.objective(logits,ids,states,arm)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
            del model
    output={"status":"passed","training_started":False,"test_accessed":False,
        "checks":["conditional_objective_mathematics_and_gradcheck",
                  "native_loss_not_monkeypatched","paired_historical_initialization_matches",
                  "historical_control_hashes_verified","training_and_evaluation_AST_same_except_seed_stop_epoch",
                  "four_epoch_cosine12_prefix_frozen","finite_G_Q5_gradients_for_both_objectives"]}
    (here/"pilot_verification.json").write_text(json.dumps(output,indent=2))
    print(json.dumps(output),flush=True)

if __name__=="__main__":main()
