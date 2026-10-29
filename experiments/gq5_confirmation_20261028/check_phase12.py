"""Quick regression checks for the 12-epoch launcher, outside production outputs."""
import json, tempfile
from pathlib import Path
import torch
import engine12 as m

def run():
 assert m.PLAN["epochs_per_arm"]==12 and m.PLAN["scheduler_T_max"]==12
 assert m.PLAN["val_epochs"]==[4,8,12]
 assert m.PLAN["arms"]==["G","Q5"] and m.PLAN["seed"]==0
 assert m.PLAN["amp"] is False and m.PLAN["test_accessed"] is False
 assert m.pending_val_epochs(3)==[]
 assert m.pending_val_epochs(4)==[4]
 assert m.pending_val_epochs(5)==[4]
 assert m.pending_val_epochs(8)==[4,8]
 assert m.pending_val_epochs(12)==[4,8,12]
 records=[{"epoch":12,"macro_AP50":.3,"weighted_AP50":.9},
          {"epoch":4,"macro_AP50":.3,"weighted_AP50":.7},
          {"epoch":8,"macro_AP50":.2,"weighted_AP50":.95}]
 assert m.choose_best(records)["epoch"]==4
 p=torch.nn.Parameter(torch.ones(1))
 o=torch.optim.AdamW([p],lr=1e-4)
 s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=12)
 for _ in range(4):
  o.zero_grad();(p**2).sum().backward();o.step();s.step()
 assert abs(o.param_groups[0]["lr"]-7.5e-5)<1e-12
 o2=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=1e-4)
 s2=torch.optim.lr_scheduler.CosineAnnealingLR(o2,T_max=12)
 o2.load_state_dict(o.state_dict());s2.load_state_dict(s.state_dict())
 assert s2.last_epoch==4 and s2.T_max==12
 assert o2.param_groups[0]["lr"]==o.param_groups[0]["lr"]
 with tempfile.TemporaryDirectory(prefix="gq5_long12_contract_") as name:
  d=Path(name)
  m.atomic_save({"epoch":4,"state_dict":{"x":torch.arange(3)}},d/"last.pt")
  snap=m.snapshot_epoch(d,4); before=m.digest(snap)
  m.atomic_save({"epoch":5,"state_dict":{"x":torch.ones(3)}},d/"last.pt")
  assert m.digest(snap)==before and m.digest(snap)!=m.digest(d/"last.pt")
  b={"test":"fixture_not_a_model_score"};ident=m.canonical(b)
  sub=d/"val_epoch04";sub.mkdir()
  m.atomic_json(sub/"metrics.json",{"fixture":True})
  (sub/"predictions.jsonl").write_text('{"fixture":true}\n')
  receipt={"study_sha256":ident,"epoch":4,"checkpoint_sha256":before,
       "metrics_sha256":m.digest(sub/"metrics.json"),"predictions_sha256":m.digest(sub/"predictions.jsonl")}
  m.atomic_json(sub/"complete.json",receipt)
  assert m.verify_eval(d,4,ident)==receipt
  assert m.verify_eval(d,8,ident) is None
  (sub/"predictions.jsonl").write_text("changed")
  try:m.verify_eval(d,4,ident)
  except RuntimeError:pass
  else:raise AssertionError("Changed evaluation output was not rejected")
 print(json.dumps({"status":"passed","checks":["fixed_12epoch_VAL4_8_12","pending_VAL_resume",
   "macro_selection_tie_earliest","cosine12_resume","snapshot_survives_next_epoch_save",
   "VAL_hash_tamper_rejected"],"training_started":False}),flush=True)
if __name__=="__main__":run()
