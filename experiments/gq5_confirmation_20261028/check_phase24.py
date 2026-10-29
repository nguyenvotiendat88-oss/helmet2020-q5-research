"""Meaningful continuation/scheduler/resume checks; no training-study mutation."""
import copy,json,tempfile
from pathlib import Path
import torch
import engine24 as m

def same(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def step(p,opt,sched):
 opt.zero_grad();((p-.3)**2).sum().backward();opt.step();sched.step()

def main():
 assert m.PLAN["epochs_per_arm"]==24 and m.PLAN["additional_epochs"]==12
 assert m.PLAN["val_epochs"]==[18,24] and m.PLAN["early_stopping"] is False
 assert m.PLAN["primary_comparison"]=="fixed_global_epoch24"
 assert m.pending_val_epochs(12)==[] and m.pending_val_epochs(17)==[]
 assert m.pending_val_epochs(18)==[18] and m.pending_val_epochs(24)==[18,24]
 assert m.choose_best([{"epoch":24,"macro_AP50":.3},{"epoch":18,"macro_AP50":.3}])["epoch"]==18

 p=torch.nn.Parameter(torch.ones(3))
 opt=torch.optim.AdamW([p],lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
 old_sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12)
 for _ in range(12):step(p,opt,old_sched)
 assert opt.param_groups[0]["lr"]==0.0
 parent={"schema":"helmet2020.gq5_long12.checkpoint.v1","study_sha256":m.PARENT_SHA,
  "arm":"G","seed":0,"epoch":12,"history":[{"epoch":n} for n in range(1,13)],
  "scheduler":copy.deepcopy(old_sched.state_dict()),"optimizer":copy.deepcopy(opt.state_dict())}
 m.validate_parent_checkpoint(parent,"G",1)
 wrong=copy.deepcopy(parent);wrong["epoch"]=11
 try:m.validate_parent_checkpoint(wrong,"G",1)
 except RuntimeError:pass
 else:raise AssertionError("Wrong parent epoch accepted")
 before=copy.deepcopy(opt.state_dict()["state"]);rng=torch.get_rng_state().clone()
 sched=m.restart_schedule(opt)
 assert same(opt.state_dict()["state"],before) and torch.equal(rng,torch.get_rng_state())
 assert sched.last_epoch==0 and sched.T_max==12 and sched.base_lrs==[2.5e-5]
 assert opt.param_groups[0]["lr"]==2.5e-5
 for _ in range(6):step(p,opt,sched)
 assert abs(opt.param_groups[0]["lr"]-1.25e-5)<1e-12
 p2=torch.nn.Parameter(p.detach().clone())
 o2=torch.optim.AdamW([p2],lr=2.5e-5,weight_decay=1e-4,foreach=False,fused=False)
 s2=torch.optim.lr_scheduler.CosineAnnealingLR(o2,T_max=12)
 o2.load_state_dict(copy.deepcopy(opt.state_dict()))
 s2.load_state_dict(copy.deepcopy(sched.state_dict()))
 step(p,opt,sched);step(p2,o2,s2)
 assert torch.equal(p,p2) and same(opt.state_dict(),o2.state_dict())
 for _ in range(5):step(p,opt,sched)
 assert sched.last_epoch==12 and opt.param_groups[0]["lr"]==0
 assert all(int(x["step"])==24 for x in opt.state.values())
 with tempfile.TemporaryDirectory() as n:
  d=Path(n)
  m.atomic_save({"epoch":18,"v":torch.arange(2)},d/"last.pt")
  snapshot=m.snapshot_epoch(d,18);h=m.digest(snapshot)
  m.atomic_save({"epoch":19,"v":torch.ones(2)},d/"last.pt")
  assert m.digest(snapshot)==h and m.digest(d/"last.pt")!=h
 print(json.dumps({"status":"passed","checks":["fixed_stop24_and_VAL18_24","parent_epoch_and_identity_guard",
   "optimizer_moments_and_steps_preserved","restart_lr_2.5e-5_not_zero","scheduler_resume_same_next_update",
   "cosine_extension_finishes_after12","milestone_snapshot_preserved"],
   "full_dataset_training":False}))
if __name__=="__main__":main()
