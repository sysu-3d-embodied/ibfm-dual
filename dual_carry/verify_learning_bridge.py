"""Component tests, not proof of stable carrying or hardware performance."""
import argparse,copy,json,time,numpy as np,torch
from environment import *
from coordinator import Coordinator

torch.set_grad_enabled(False)
p=argparse.ArgumentParser();p.add_argument('--coordinated-motion',action='store_true');p.add_argument('--goal-steering',action='store_true');args=p.parse_args()
e=DualCarryEnv({'state_bank':'replay_bank_67974_90655a91.pt','canonical_box_symmetry':True,
    'robot_positions_m':[[-.62,0,.78],[.62,0,.78]],'coordinated_motion':args.coordinated_motion,
    'goal_steering':args.goal_steering,'travel_direction':'forward' if args.goal_steering else 'world-y'},device='cuda')
p=FrozenDualPolicy();c=Coordinator(e,p);report={}
indices=torch.linspace(0,len(c.B)-1,64,device=p.device).long()
direct=p.model.backward_map({'state':c.S[indices],'privileged_state':c.bank['privileged_state'][indices]})
cache_error=(direct-c.B[indices]).abs().max().item()
assert cache_error<5e-5,cache_error
report['cached_backward_max_abs_error']=cache_error
assert not any(x.requires_grad for x in p.model.parameters())
base=e.context();base['time_s']=5.
def run(ct):
    c.stage='transport';c.stage_start=2.;c.height=.62;c.last_z=p.latents.clone()
    c.context_box_velocity[:]=0;c.context_root_velocity[:]=0;c.smoothed_targets[:]=0
    out=c(ct)
    return out['latents'].clone(),copy.deepcopy(c.last_reward),c.hand_targets.copy()
z0,r0,h0=run(copy.deepcopy(base))
faster=copy.deepcopy(base);faster['robots'][0]['linear_velocity_mps'][0]+=.3
z1,r1,h1=run(faster)
dv=np.array(r1['velocity_targets'])-r0['velocity_targets']
dz=torch.linalg.vector_norm(z1-z0,dim=-1).cpu().numpy()
assert abs(dv[1,0])>.01 and dz[1]>1e-6,(dv,dz)
report['teammate_speed_change']={'input_mps':.3,'other_robot_target_change_mps':dv[1].tolist(),'other_robot_z_L2_change':float(dz[1])}
loaded=copy.deepcopy(base);loaded['contacts']['hand_force_on_box_N'][0][0][2]=15.
z2,r2,h2=run(loaded)
df=np.array(r2['vertical_force_targets_N'])-r0['vertical_force_targets_N']
assert np.linalg.norm(df[1])>0 and np.linalg.norm(h2-h0)>0
report['teammate_load_change']={'other_robot_load_target_change_N':df[1].tolist(),'max_hand_target_change_m':float(np.abs(h2-h0).max())}
report['frozen_weights']=True;report['passed']=True
if args.coordinated_motion:
    for item in r0['rewards']+r1['rewards']+r2['rewards']:
        assert item['forbidden_probability_mass']==0 and not item['no_feasible_candidates'],item
    report['coordinated_motion_support_checks']=True
name='coordinated_motion_bridge_verification.json' if args.coordinated_motion else 'learning_bridge_verification.json'
if args.goal_steering:name='goal_steering_bridge_verification.json'
(HERE/name).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
