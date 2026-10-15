"""Cache B for static pose goals derived from measured successful pickup."""
import argparse,json,hashlib
import numpy as np,torch
from environment import *

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('Guide already exists')
ct=json.loads((a.run/'context_10hz.json').read_text());records=json.loads((a.run/'rewards.json').read_text())
traj=np.load(a.run/'trajectory.npz');env=DualCarryEnv(json.loads((a.run/'config.json').read_text()),device='cuda');policy=FrozenDualPolicy()
rows=[];heights=[];times=[]
with torch.inference_mode():
    for c,r in zip(ct,records):
        if r['stage'] not in ('acquire','lift'):continue
        contact=np.array(c['contacts']['hand_contact'])
        if not contact.any(axis=1).all():continue
        idx=round(c['time_s']*50)-1
        if idx<0:continue
        env.data.qpos[:]=traj['qpos'][idx];env.data.qvel[:]=0.;env.current_actions=traj['actions'][idx].copy()
        for observer,target,anchor in zip(env.observers,c['object_targets_m'],env.object_anchors):observer.set_target_object_pos(np.array(target)+anchor)
        env.forward();obs=env.observations()
        # Contact bits come from the original recorded physical force solve,
        # not an offline forward call without the original actuator forces.
        obs['state'][:,82:84]=torch.as_tensor(contact,device='cuda')
        obs['privileged_state'][:,-16:-14]=obs['state'][:,82:84]
        rows.append({k:obs[k].clone() for k in ('state','privileged_state')});heights.append(c['object']['position_m'][2]);times.append(c['time_s'])
    state=torch.stack([x['state'] for x in rows],dim=1)
    priv=torch.stack([x['privileged_state'] for x in rows],dim=1)
    B=policy.model.backward_map({'state':state.reshape(-1,98),'privileged_state':priv.reshape(-1,497)}).reshape(2,len(rows),-1)
    metadata={'source':str(a.run),'source_trajectory_sha256':hashlib.sha256((a.run/'trajectory.npz').read_bytes()).hexdigest(),
              'times_s':times,'heights_m':heights,'method':'measured pickup poses with zero target velocities, cached backward_map; runtime uses reward-weighted height goals; not a reversed physics trajectory'}
    torch.save({'state':state.cpu(),'privileged_state':priv.cpu(),'B':B.cpu(),'height':torch.tensor(heights),'metadata':metadata},a.output)
print(json.dumps({'rows_per_robot':len(rows),'height_min':min(heights),'height_max':max(heights),'path':str(a.output)}))
