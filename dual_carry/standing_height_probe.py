"""Leader-only height command; follower sees measured leader through ct.

Independent validation entry point. Does not modify the carry mission/controller.
No root locking, object teleportation, weld, or external lifting forces.
"""
import argparse, csv, json, shutil, time
from pathlib import Path
import numpy as np
import torch
import mujoco
from environment import DualCarryEnv, FrozenDualPolicy, HERE, HAND_OFFSETS
from coordinator import Coordinator
from humanoidverse.carry_phased_reward_inference import _masked_object_observation


def hand_points(ct):
    points=[]
    for robot in ct['robots']:
        row=[]
        for h,hand in enumerate(robot['hands']):
            rotation=np.zeros(9)
            mujoco.mju_quat2Mat(rotation,np.array(hand['quaternion_wxyz']))
            row.append(np.array(hand['position_m'])+rotation.reshape(3,3)@HAND_OFFSETS[h])
        points.append(row)
    return np.array(points)


def height_schedule(elapsed):
    """Four 4 s half-cosine ramps, each followed by a 3 s hold; 3 s baseline."""
    if elapsed<3:return 0.,'baseline'
    starts=[0.,.06,.12,.06];ends=[.06,.12,.06,0.]
    segment=min(int((elapsed-3)//7),3)
    u=np.clip(((elapsed-3)-7*segment)/4,0,1)
    delta=starts[segment]+(ends[segment]-starts[segment])*(1-np.cos(np.pi*u))/2
    return float(delta),f'height_{int(ends[segment]*100):02d}cm_'+('ramp' if u<1 else 'hold')


class HeightProbe(Coordinator):
    def __init__(self,env,policy,mode):
        super().__init__(env,policy,speed=.1,travel=5.,dynamic=True,temperature=10.)
        self.mode=mode;self.probe_start=None;self.probe_rows=[];self.probe_z=[]
        self.probe_done=False;self.counterfactual_z=None

    def _change(self,stage,t):
        if stage=='transport':
            if self.probe_start is None:
                self.probe_start=t
                ct=self.env.context();self.initial_hands=hand_points(ct)
                self.base_height=float(ct['object']['position_m'][2])
                self.initial_roots=np.array([r['position_m'] for r in ct['robots']])
                self.initial_box=np.array(ct['object']['position_m'])
                self.filtered_leader_delta=0.;self.height_targets=np.full(2,self.base_height)
                self.events.append({'from':'lift','to':'standing_height_probe','time_s':t})
            self.stable_time=-100.
            return
        super()._change(stage,t)

    def __call__(self,ct):
        active=self.probe_start is not None
        self.current_ct=ct
        old_z=self.last_z.clone()
        if active:
            elapsed=ct['time_s']-self.probe_start
            delta,self.segment=height_schedule(elapsed)
            hp=hand_points(ct)
            measured_delta=float(hp[0,:,2].mean()-self.initial_hands[0,:,2].mean())
            dt=1/self.env.config['coordination_hz']
            self.filtered_leader_delta+=(1-np.exp(-dt/.2))*(measured_delta-self.filtered_leader_delta)
            follower_delta=self.filtered_leader_delta if self.mode=='live' else 0.
            self.frozen_height=self.base_height
            self.height_targets=np.array([self.base_height+delta,
                np.clip(self.base_height+follower_delta,self.base_height-.08,self.base_height+.18)])
            # Base lift controller advances .12*dt; cancel only that ramp.
            self.height=self.height_targets[0]-.12*dt
            self.stable_time=-100.
            self.formation_offsets=self.initial_roots-np.array(ct['object']['position_m'])
            self.formation_offsets[:,2]=0.
            self.probe_done=elapsed>=31.
        update=super().__call__(ct)
        # Activation can happen inside super(); start intervention next tick.
        if not active or self.stage=='failed':return update
        z=update['latents']
        angle=torch.acos(torch.nn.functional.cosine_similarity(z,old_z).clamp(-1,1))
        mix=(.08/angle.clamp_min(.08))[:,None]
        z=self.policy.model.project_z((1-mix)*old_z+mix*z)
        self.last_z_change=(1-torch.nn.functional.cosine_similarity(z,old_z)).cpu().tolist()
        self.last_z=z;update['latents']=z
        # Match the base coordinator's temporal smoothing in the counterfactual.
        cf=self.policy.model.project_z(.55*old_z[1:2]+.45*self.counterfactual_z)
        angle_cf=torch.acos(torch.nn.functional.cosine_similarity(cf,old_z[1:2]).clamp(-1,1))
        mix_cf=(.08/angle_cf.clamp_min(.08))[:,None]
        self.counterfactual_z=self.policy.model.project_z((1-mix_cf)*old_z[1:2]+mix_cf*cf)
        # Keep actor object-goal inputs identical/fixed for both conditions.
        # Thus the changing leader cue enters B only through reward -> z.
        update['object_targets_m']=[self.initial_box.tolist(),self.initial_box.tolist()]
        for i in range(2):self.hand_targets[i,:,2]+=self.height_targets[i]-self.height
        hp=hand_points(ct);roots=np.array([r['position_m'] for r in ct['robots']])
        tilt=float(np.degrees(np.arccos(np.clip(ct['object']['geometry_rotation_world'][2][2],-1,1))))
        force=np.linalg.norm(np.array(ct['contact_window']['hand_mean_force_on_box_N']),axis=-1)
        self.probe_rows.append({'t':ct['time_s'],'elapsed':elapsed,'segment':self.segment,
            'A_height_target':float(self.height_targets[0]),'B_height_target':float(self.height_targets[1]),
            'A_hand_height':float(hp[0,:,2].mean()),'B_hand_height':float(hp[1,:,2].mean()),
            'A_hand_delta':float(hp[0,:,2].mean()-self.initial_hands[0,:,2].mean()),
            'B_hand_delta':float(hp[1,:,2].mean()-self.initial_hands[1,:,2].mean()),
            'box_height':ct['object']['position_m'][2],'box_tilt_deg':tilt,
            'A_root_drift':float(np.linalg.norm(roots[0,:2]-self.initial_roots[0,:2])),
            'B_root_drift':float(np.linalg.norm(roots[1,:2]-self.initial_roots[1,:2])),
            'both_grasp':bool(np.array(ct['contacts']['hand_contact']).any(1).all()),
            'all_four_contacts':bool(np.array(ct['contacts']['hand_contact']).all()),
            'max_hand_mean_force_N':float(force.max()),
            'z_A_step_cosine_change':self.last_z_change[0],'z_B_step_cosine_change':self.last_z_change[1],
            'z_B_same_state_cf_cosine_change':float(1-torch.nn.functional.cosine_similarity(z[1:2],self.counterfactual_z)[0]),
            'B_reward_height_used':float(self.height_targets[1]),'B_reward_height_frozen':self.frozen_height})
        self.probe_z.append(z.cpu().numpy())
        self.last_reward['height_probe']=dict(self.probe_rows[-1])
        return update

    def reward(self,velocity,relative,hand_offsets,root_height,object_velocity,rotation_target,world_rotation=None):
        if self.probe_start is None or not hasattr(self,'segment'):
            return super().reward(velocity,relative,hand_offsets,root_height,object_velocity,rotation_target,world_rotation)
        i=self.reward_robot_index;original_height=self.height
        def evaluate(height):
            self.height=height
            rel=relative.copy();rel+=world_rotation.T@np.array([0.,0.,height-original_height])
            # Standing root, slow vertical object tracking, all in correct frame.
            vel=velocity.copy();world=world_rotation@vel;world[2]=0.;vel=world_rotation.T@world
            box_z=self.current_ct['object']['position_m'][2]
            obj=world_rotation.T@np.array([0.,0.,np.clip(1.5*(height-box_z),-.045,.045)])
            score,parts=super(HeightProbe,self).reward(vel,rel,hand_offsets,root_height,obj,rotation_target,world_rotation)
            # Give standing and height/vertical motion useful priority in lift.
            R=torch.as_tensor(world_rotation,device=self.S.device,dtype=self.S.dtype)
            ow=self.S[:,73:76]@R.T
            desired=torch.as_tensor(world_rotation@obj,device=self.S.device,dtype=self.S.dtype)
            vertical=torch.exp(-((ow-desired)/.10).square().sum(-1))
            still=torch.exp(-((self.F[:,:2]-.0)/.12).square().sum(-1))
            parts.update(probe_object_velocity=vertical,probe_standing=still)
            return (score+.25*parts['height']+.20*vertical+.20*still)/1.65,parts
        score,parts=evaluate(float(self.height_targets[i]))
        if i==1:
            cf_score,_=evaluate(self.frozen_height)
            weight=torch.softmax(self.temperature*cf_score,0)*cf_score.clamp_min(0)
            self.counterfactual_z=self.policy.model.project_z(weight[None]@self.B)
        self.height=original_height
        return score,parts


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--mode',choices=['live','frozen-leader'],default='live');p.add_argument('--seconds',type=float,default=45)
    p.add_argument('--state-bank',default='replay_bank_67974_90655a91.pt')
    a=p.parse_args();out=(HERE/'outputs'/a.tag).resolve()
    if not out.is_relative_to((HERE/'outputs').resolve()) or out.exists():raise ValueError('New output directory required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for name in ['standing_height_probe.py','coordinator.py','environment.py','config.json']:
        shutil.copy2(HERE/name,out/'source'/name)
    (out/'arguments.json').write_text(json.dumps(vars(a),indent=2))
    torch.manual_seed(0);np.random.seed(0)
    env=DualCarryEnv({'box_mass_kg':1.4,'box_size_m':[.60,.30,.30],
        'robot_positions_m':[[-.65,0,.78],[.65,0,.78]],'coordination_hz':10,'physics_hz':200,
        'carry_height_m':.65,'canonical_box_symmetry':True,'state_bank':a.state_bank,
        'grip_x_m':.19,'grip_standoff_m':.04,'filtered_context':True,'mass_adaptive_grasp':False,
        'travel_direction':'forward','layout':'opposed','goal_steering':False,'axial_carry':False},device='cuda')
    policy=FrozenDualPolicy();stand=policy.latents.clone();c=HeightProbe(env,policy,a.mode)
    for _ in range(5):policy.act(_masked_object_observation(env.observations()))
    env.reset();torch.cuda.synchronize()
    frames=[];vels=[];acts=[];timings=[];contexts=[];rewards=[];coord_times=[]
    start=time.perf_counter()
    with torch.inference_mode():
        for step in range(round(a.seconds*50)):
            tick=time.perf_counter();ctick=time.perf_counter();update=env.coordinate(c)
            if update:
                coord_times.append(1000*(time.perf_counter()-ctick))
                policy.set_latents(update['latents']);contexts.append(env.latest_context);rewards.append(c.last_reward)
            obs=env.observations()
            if c.stage=='settle':policy.set_latents(stand);obs=_masked_object_observation(obs)
            action=policy.act(obs)
            if update and c.probe_rows and c.stage!='failed':
                actual_z=policy.latents.clone();cf_z=actual_z.clone();cf_z[1]=c.counterfactual_z[0]
                policy.set_latents(cf_z);cf_action=policy.act(obs);policy.set_latents(actual_z)
                actual_target=np.clip(env.defaults+env.target_scale*np.clip(action[1],-env.action_clip,env.action_clip),env.physical_limits[1,:,0],env.physical_limits[1,:,1])
                cf_target=np.clip(env.defaults+env.target_scale*np.clip(cf_action[1],-env.action_clip,env.action_clip),env.physical_limits[1,:,0],env.physical_limits[1,:,1])
                c.probe_rows[-1]['B_same_obs_joint_target_diff_rad']=float(np.sqrt(np.mean((actual_target-cf_target)**2)))
                c.probe_rows[-1]['B_actual_arm_joint_target_rad']=actual_target[15:].tolist()
                c.probe_rows[-1]['B_cf_arm_joint_target_rad']=cf_target[15:].tolist()
            if c.stage=='failed':break
            env.step(action)
            frames.append(env.data.qpos.copy());vels.append(env.data.qvel.copy());acts.append(action)
            timings.append(1000*(time.perf_counter()-tick))
            if step%50==0:
                row=c.probe_rows[-1] if c.probe_rows else {}
                print(json.dumps({'t':round(env.data.time,2),'stage':'probe' if c.probe_start else c.stage,
                    **{k:round(row[k],4) if isinstance(row[k],float) else row[k] for k in ['segment','A_height_target','B_height_target','A_hand_delta','B_hand_delta','box_tilt_deg'] if k in row}}),flush=True)
            if c.probe_done:break
    np.savez_compressed(out/'trajectory.npz',qpos=frames,qvel=vels,actions=acts,control_ms=timings,z=c.probe_z)
    (out/'config.json').write_text(json.dumps(env.config,indent=2))
    (out/'context_10hz.json').write_text(json.dumps(contexts));(out/'rewards.json').write_text(json.dumps(rewards))
    (out/'probe.json').write_text(json.dumps(c.probe_rows))
    scalar_keys=[k for k,v in (c.probe_rows[0] if c.probe_rows else {}).items() if not isinstance(v,list)]
    with (out/'probe.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=scalar_keys,extrasaction='ignore');w.writeheader();w.writerows(c.probe_rows)
    rows=c.probe_rows
    metrics={}
    if rows:
        values=lambda key:np.array([r[key] for r in rows])
        metrics={'A_actual_height_span_m':float(np.ptp(values('A_hand_delta'))),
            'B_actual_height_span_m':float(np.ptp(values('B_hand_delta'))),
            'A_command_tracking_rmse_m':float(np.sqrt(np.mean((values('A_hand_delta')-(values('A_height_target')-c.base_height))**2))),
            'B_vs_A_displacement_rmse_m':float(np.sqrt(np.mean((values('B_hand_delta')-values('A_hand_delta'))**2))),
            'both_grasp_fraction':float(values('both_grasp').mean()),'max_box_tilt_deg':float(values('box_tilt_deg').max()),
            'max_root_drift_m':float(max(values('A_root_drift').max(),values('B_root_drift').max())),
            'max_hand_mean_force_N':float(values('max_hand_mean_force_N').max()),
            'same_obs_B_action_effect_max_rad':float(values('B_same_obs_joint_target_diff_rad').max()),
            'same_state_B_z_effect_max':float(values('z_B_same_state_cf_cosine_change').max())}
    checks={'finished_schedule':c.probe_done,'no_posture_failure':c.stage!='failed',
        'A_tracks_requested_lift':metrics.get('A_command_tracking_rmse_m',1)<.04 and metrics.get('A_actual_height_span_m',0)>.06,
        'B_tracks_measured_A':metrics.get('B_vs_A_displacement_rmse_m',1)<.04,
        'standing_drift_le_15cm':metrics.get('max_root_drift_m',1)<=.15,
        'box_tilt_le_15deg':metrics.get('max_box_tilt_deg',180)<=15,
        'both_grasp_ge_95percent':metrics.get('both_grasp_fraction',0)>=.95,
        'hand_mean_load_le_60N':metrics.get('max_hand_mean_force_N',1000)<=60,
        'counterfactual_negative_control':a.mode!='frozen-leader' or metrics.get('same_obs_B_action_effect_max_rad',1)<1e-5}
    summary={'experiment':'standing height response','mode':a.mode,'success':all(checks.values()),'checks':checks,
        'metrics':metrics,'events':c.events,'destination_m':[0,0,.65],'simulation_seconds':env.data.time,
        'probe_start_s':c.probe_start,'failure_reason':getattr(c,'failure_reason',None),
        'wall_seconds':time.perf_counter()-start,'coordination_ms_p95':float(np.percentile(coord_times,95)),
        'control_ms_p95':float(np.percentile(timings,95)),
        'notes':['B leader-height cue frozen in control condition; own observations and contact forces remain live.',
                 'Same-observation counterfactual changes only B z, not observation/history or physics.',
                 'Real-time statistics include counterfactual extra policy/reward evaluations and logging.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
