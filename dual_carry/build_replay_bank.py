"""Cache B of exact published replay observations; no online network over bank."""
import json,hashlib,time
import numpy as np,torch
from environment import HERE,FrozenDualPolicy
from humanoidverse.utils.torch_utils import quat_rotate,quat_rotate_inverse

start=time.perf_counter();source=HERE/'replay_sample.pt';raw=torch.load(source,weights_only=False)
d=raw['data'];S=d['observation-state'].float();P=d['observation-privileged_state'].float()
Q=d['qpos'].float();V=d['qvel'].float();quat=torch.roll(Q[:,3:7],-1,dims=-1)
rootv=quat_rotate_inverse(quat,V[:,:3],w_last=True)
boxz=Q[:,2]+quat_rotate(quat,S[:,64:67],w_last=True)[:,2]
x,z=S[:,67:70],S[:,70:73];y=torch.cross(z,x,dim=-1)
G=torch.as_tensor([.07016202,.97153219,.22227911,.04236142]);G=G/G.norm()
qg=torch.roll(G,-1).repeat(len(S),1);unit=torch.zeros((len(S),3));unit[:,2]=1
geomz=quat_rotate(qg,unit,w_last=True)
obj_geomz=x*geomz[:,0:1]+y*geomz[:,1:2]+z*geomz[:,2:3]
upright=quat_rotate(quat,obj_geomz,w_last=True)[:,2]
F=torch.cat([rootv,boxz[:,None],Q[:,2:3],upright[:,None]],dim=-1)
valid=torch.isfinite(S).all(-1)&torch.isfinite(P).all(-1)&torch.isfinite(F).all(-1)
S,P,F=S[valid],P[valid],F[valid]
policy=FrozenDualPolicy();B=[]
with torch.inference_mode():
    for n in range(0,len(S),1024):
        B.append(policy.model.backward_map({'state':S[n:n+1024].cuda(),'privileged_state':P[n:n+1024].cuda()}).cpu())
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
payload={'state':S,'privileged_state':P,'features':F,'B':torch.cat(B),
    'metadata':{'source':'official physics replay at 166.560M steps','sources':raw['sources'],
        'source_sha256':source_sha,'schema':'state98_privileged497_B256',
        'force_labels_available':False,'feature_columns':['root_vx_local','root_vy_local','root_vz_local','box_z','root_z','box_upright']}}
bank_name=f'replay_bank_{len(S)}_{source_sha[:8]}.pt'
torch.save(payload,HERE/bank_name)
good=(S[:,82:84]>.5).all(-1)&(F[:,5]>.9)&(-S[:,60]>.9)
report={'rows':len(S),'bank_file':bank_name,'ranks':len(raw['sources']),'seconds':time.perf_counter()-start,
    'lowering_height_coverage':[(h,int((good&(abs(F[:,3]-h)<.1)&(S[:,75]<.04)&(S[:,75]>-.25)).sum())) for h in [.18,.25,.35,.45,.55,.65]]}
(HERE/'replay_bank_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
