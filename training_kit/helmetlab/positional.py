"""Crop-based RQ2 implementation, explicitly two-stage; shared backbone in flat/position controls."""
from __future__ import annotations
import copy
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from .common import read_jsonl, write_json, digest
from .labels import LabelMap
from .data import verify_lock


def crop_box(image, box, margin=.05):
    x1,y1,x2,y2=box; w,h=x2-x1,y2-y1
    return image.crop((max(0,int(x1-margin*w)),max(0,int(y1-margin*h)),
                       min(image.width,int(np.ceil(x2+margin*w))),min(image.height,int(np.ceil(y2+margin*h)))))


def transform(train=False):
    ops=[transforms.Resize((224,224))]
    if train: ops += [transforms.ColorJitter(.2,.2,.1,.02)]
    return transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])


class CropDataset(Dataset):
    def __init__(self, root, split, stride=5, training=False):
        self.root=Path(root);self.items=[];self.transform=transform(training)
        for r in read_jsonl(self.root/f"{split}.jsonl"):
            if (r["frame_index"]-1)%stride: continue
            for o in r["objects"]:self.items.append((r["image"],o["bbox"],o["class_id"]))
        if not self.items: raise ValueError(f"No crops in {split}")
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        path,box,cid=self.items[i]
        with Image.open(self.root/path) as image: crop=crop_box(image.convert("RGB"),box)
        return self.transform(crop),cid


def build_model(mode, num_classes, pretrained=True):
    if mode not in {"flat", "position", "binary10"}:
        raise ValueError("mode must be flat, position or binary10")
    net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    net.fc=nn.Linear(net.fc.in_features,{"position":15,"binary10":10}.get(mode,num_classes))
    return net


def classification_loss(logits, ids, states, mode):
    if mode == "position":
        return nn.functional.cross_entropy(logits.reshape(-1,3), states[ids].reshape(-1))
    if mode == "binary10":
        # Lin et al., ICIP 2021 Eq. 3: helmet loss only where a rider exists.
        # Divide the per-image sum by five, a declared constant normalization.
        target = states[ids]
        present = (target != 0).to(logits.dtype)
        helmet = (target == 1).to(logits.dtype)
        occupancy_loss = nn.functional.binary_cross_entropy_with_logits(logits[:,:5],present,reduction="none")
        helmet_loss = nn.functional.binary_cross_entropy_with_logits(logits[:,5:],helmet,reduction="none")
        return (occupancy_loss + present*helmet_loss).mean()
    if mode == "flat":
        return nn.functional.cross_entropy(logits,ids)
    raise ValueError("Unknown classifier mode")


def position_probabilities(logits, mode):
    if mode == "position":
        return logits.reshape(-1,5,3).softmax(-1)
    if mode == "binary10":
        occupancy, helmet = logits[:,:5].sigmoid(), logits[:,5:].sigmoid()
        return torch.stack((1-occupancy, occupancy*helmet, occupancy*(1-helmet)),dim=-1)
    raise ValueError("Flat classifier has no independent positional probabilities")


def decode_logits(logits, mode, labels):
    """Matched constrained decoding; also return unconstrained states for diagnostics."""
    if mode == "flat":
        q=logits.softmax(-1).detach().cpu().numpy()
        return q, labels.states[q.argmax(-1)]
    p=position_probabilities(logits,mode)
    if mode == "binary10":
        present=(logits[:,:5]>=0)
        native=torch.where(present,torch.where(logits[:,5:]>=0,1,2),0)
    else:
        native=p.argmax(-1)
    return labels.distribution(p.detach().cpu().numpy()), native.detach().cpu().numpy()


def train_positions(dataset, output, mode="position", epochs=30, batch=64, device="cpu", seed=0, stride=5, pretrained=True,
                    resume=False, stop_after=None):
    from .checkpoints import atomic_save, cpu_tree, rng_state, restore_rng
    from .training import environment, source_hashes
    if mode not in {"position","flat","binary10"}: raise ValueError("mode must be position, flat or binary10")
    if min(epochs,batch,stride) < 1: raise ValueError("epochs, batch and stride must be positive")
    if stop_after is not None and not 1 <= stop_after <= epochs: raise ValueError("stop_after must be within planned epochs")
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    dataset=Path(dataset);output=Path(output)
    previous=None
    if resume:
        if not (output/'last.pt').is_file(): raise FileNotFoundError('Resume needs last.pt saved at an epoch boundary')
        previous=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        if previous.get('training_checkpoint_version')!=1: raise ValueError('Legacy checkpoint lacks optimizer/RNG; choose a new run')
    elif output.exists() and any(output.iterdir()): raise FileExistsError("Choose a new output directory or use --resume")
    lock=verify_lock(dataset);labels=LabelMap(dataset/"classes.json")
    output.mkdir(parents=True,exist_ok=True)
    train=CropDataset(dataset,"train",stride,True);val=CropDataset(dataset,"val",stride)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train,batch_size=batch,shuffle=True,num_workers=0,generator=generator,pin_memory=str(device).startswith('cuda'))
    val_loader=DataLoader(val,batch_size=batch,num_workers=0)
    targets=torch.tensor(labels.states,device=device)
    # Same unweighted CE baseline for both controls; loss balancing is a separate ablation.
    net=build_model(mode,len(labels),pretrained and previous is None).to(device)
    # Different head sizes consume different RNG draws; restart augmentation RNG
    # so a matched seed has the same data order and ColorJitter draws in all arms.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    optim=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,epochs)
    scaler=torch.amp.GradScaler("cuda",enabled=str(device).startswith("cuda"))
    history=[];best=-1.
    settings={"mode":mode,"epochs":epochs,"batch":batch,"seed":seed,"stride":stride,"pretrained":pretrained,
              "class_map_sha256":digest(dataset/"classes.json"),"dataset_lock_sha256":lock,"architecture":"resnet18_crop_two_stage",
              "selection":"validation_macro_f1", "not_single_stage_head":True,
              "decoder":"condition_on_36_valid_combinations", "environment":environment(),"source_sha256":source_hashes(),
              "binary10_reference":"Lin et al., ICIP 2021, Eq. 3; adapted loss with /5 normalization; NOT original training recipe",
              "optimizer":"AdamW", "lr":3e-4, "weight_decay":1e-4, "scheduler":"CosineAnnealingLR",
              "augmentation":"ColorJitter(.2,.2,.1,.02)", "shuffle_seed":seed, 'device':str(device),
              'resume_granularity':'completed_epoch; reruns_interrupted_epoch', 'workers':0}
    start_epoch=0
    if previous is not None:
        old=previous['settings']
        for key in ('mode','epochs','batch','seed','stride','pretrained','class_map_sha256','dataset_lock_sha256','source_sha256','device'):
            if old.get(key)!=settings.get(key): raise ValueError(f'Resume settings changed: {key}; choose a new run')
        if old['environment']['versions']!=settings['environment']['versions']:
            raise ValueError('Package versions changed since checkpoint; restore original environment')
        net.load_state_dict(previous['state_dict']);optim.load_state_dict(previous['optimizer'])
        for state in optim.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):state[key]=value.to(device)
        scheduler.load_state_dict(previous['scheduler']);scaler.load_state_dict(previous['scaler'])
        history=previous['history'];best=previous['best_macro_f1'];start_epoch=previous['epoch']
        restore_rng(previous['rng'],generator)
    if stop_after is not None and stop_after < start_epoch: raise ValueError('stop_after is earlier than saved epoch')
    write_json(output/"run.json",{**settings,'status':'training','resumed_from_epoch':start_epoch})
    for epoch in range(start_epoch,epochs):
        if stop_after is not None and epoch >= stop_after: break
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        net.train();running=0.
        for x,y in train_loader:
            x,y=x.to(device),y.to(device);optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",enabled=str(device).startswith("cuda")):
                logits=net(x)
                loss=classification_loss(logits,y,targets,mode)
            scaler.scale(loss).backward();scaler.step(optim);scaler.update()
            running += loss.item()*len(y)
        scheduler.step();net.eval();truth=[];guess=[];native_states=[]
        with torch.inference_mode():
            for x,y in val_loader:
                logits=net(x.to(device))
                q,native=decode_logits(logits,mode,labels)
                truth.extend(y.tolist());guess.extend(q.argmax(-1).tolist())
                native_states.extend(native.tolist())
        truth=np.array(truth);guess=np.array(guess)
        f1s=[]
        for c in np.unique(truth):
            tp=np.sum((truth==c)&(guess==c));f1s.append(2*tp/(np.sum(truth==c)+np.sum(guess==c)))
        macro=float(np.mean(f1s));accuracy=float(np.mean(truth==guess))
        valid_states=set(map(tuple,labels.states.tolist()))
        native=np.asarray(native_states)
        row={"epoch":epoch+1,"loss":running/len(train),"val_macro_f1":macro,"val_exact_match":accuracy,
             "val_native_state_exact_match":float((native==labels.states[truth]).all(-1).mean()),
             "val_native_invalid_combination_rate":float(np.mean([tuple(s) not in valid_states for s in native.tolist()])),
             "epoch_wall_seconds":time.perf_counter()-start}
        row['peak_cuda_memory_bytes']=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None
        history.append(row);print(row,flush=True)
        if macro>best:
            best=macro
            atomic_save({"state_dict":cpu_tree(net.state_dict()),"mode":mode,"num_classes":len(labels),"settings":settings},output/"best.pt")
        atomic_save({'training_checkpoint_version':1,'state_dict':cpu_tree(net.state_dict()),'mode':mode,'num_classes':len(labels),
            'settings':settings,'optimizer':cpu_tree(optim.state_dict()),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch+1,'history':history,'best_macro_f1':best,'rng':rng_state(generator)},output/'last.pt')
        write_json(output/"history.json",history)
    complete=len(history)==epochs
    write_json(output/'history.json',history)
    write_json(output/'run.json',{**settings,'status':'completed' if complete else 'paused','completed_epochs':len(history)})
    if complete:write_json(output/'complete.json',{'best':str(output/'best.pt'),'epochs':epochs,'benchmark_claim':False})
    return str(output/"best.pt")


class PositionPredictor:
    def __init__(self, weights, labels_path, device="cpu"):
        self.labels=LabelMap(labels_path);self.device=device
        checkpoint=torch.load(weights,map_location="cpu",weights_only=True)
        if checkpoint["settings"]["class_map_sha256"]!=digest(labels_path): raise ValueError("Classifier class map mismatch")
        self.mode=checkpoint["mode"]
        self.net=build_model(self.mode,len(self.labels),False)
        self.net.load_state_dict(checkpoint["state_dict"]);self.net.to(device).eval();self.transform=transform()
    @torch.inference_mode()
    def predict(self, rgb, boxes):
        if not boxes:return np.empty((0,len(self.labels)))
        image=Image.fromarray(rgb)
        crops=torch.stack([self.transform(crop_box(image,b)) for b in boxes]).to(self.device)
        logits=self.net(crops)
        return decode_logits(logits,self.mode,self.labels)[0]
