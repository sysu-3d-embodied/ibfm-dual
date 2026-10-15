"""Single fixed protocol: A speed commands, B measured-speed following.

This is a diagnostic entry point, not a retuned production carry controller.
"""
import argparse,copy,csv,json,shutil,time
import numpy as np
import torch
from environment import DualCarryEnv,FrozenDualPolicy,HERE
from coordinator import Coordinator
from humanoidverse.carry_phased_reward_inference import _masked_object_observation


def speed_schedule(elapsed):
    speeds=[.05,.10,.20,.10,0.]
    stage=min(int(max(0,elapsed)//6),len(speeds)-1)
    within=max(0,elapsed)-6*stage
    previous=speeds[stage-1] if stage else 0.
    blend=(1-np.cos(np.pi*np.clip(within/2,0,1)))/2
    return previous+blend*(speeds[stage]-previous),stage,within


class SpeedProbe(Coordinator):
    def __init__(self,env,policy,mode):
        super().__init__(env,policy,speed=.1,travel=20.,dynamic=True,temperature=10.)
        self.mode=mode;self.probe_start=None;self.probe_done=False
        self.probe_target=np.zeros((2,3));self.filtered_actual=np.zeros((2,3))
        self.command=0.;self.segment=0;self.within=0.

    def _change(self,stage,t):
        super()._change(stage,t)
        if stage=='transport' and self.probe_start is None:
            self.probe_start=t;ct=self.env.context()
            roots=np.array([r['position_m'] for r in ct['robots']])
            self.start_roots=roots.copy();self.initial_gap=roots[1,:2]-roots[0,:2]
            self.initial_box=np.array(ct['object']['position_m'])
            self.filtered_actual=np.array([r['linear_velocity_mps'] for r in ct['robots']])
            self.frozen_A_speed=float(self.filtered_actual[0,0])

    def __call__(self,ct):
        self.current_ct=ct
        if self.probe_start is not None:
            elapsed=ct['time_s']-self.probe_start
            self.command,self.segment,self.within=speed_schedule(elapsed)
            raw=np.array([r['linear_velocity_mps'] for r in ct['robots']])
            alpha=1-np.exp(-1/self.env.config['coordination_hz']/.2)
            self.filtered_actual+=alpha*(raw-self.filtered_actual)
            self.probe_done=elapsed>=30.
        return super().__call__(ct)

    def robot_velocity_target(self):
            # This computes body velocity only; LevelSpeedProbe additionally shares the object command.
            # A/B use world +X, despite their opposite body headings.
            i=self.reward_robot_index
            if i==0:desired=np.array([self.command,0.,0.])
            else:
                roots=np.array([r['position_m'] for r in self.current_ct['robots']])
                gap_error=roots[0,:2]+self.initial_gap-roots[1,:2]
                cue=float(self.filtered_actual[0,0]) if self.mode=='live' else self.frozen_A_speed
                desired=np.r_[np.array([cue,0.])+np.clip(.8*gap_error,-.12,.12),0.]
                desired[:2]=np.clip(desired[:2],[-.15,-.12],[.40,.12])
            self.probe_target[i]=desired
            return desired

    def reward(self,velocity,relative,hand_offsets,root_height,object_velocity,rotation_target,world_rotation=None):
        if self.stage=='transport':
            desired=self.robot_velocity_target()
            velocity=world_rotation.T@desired
            object_velocity=velocity.copy()
        return super().reward(velocity,relative,hand_offsets,root_height,object_velocity,rotation_target,world_rotation)

    def snapshot(self):
        """Copy mutable controller state; share only read-only model/bank data."""
        shadow=copy.copy(self)
        shared={'env','policy','bank','S','F','B','B_unit','native','native_turn','placement_guide'}
        for name,value in self.__dict__.items():
            if name not in shared:setattr(shadow,name,copy.deepcopy(value))
        return shadow


def joint_targets(env,action):
    return np.clip(env.defaults+env.target_scale*np.clip(action,-env.action_clip,env.action_clip),
        env.physical_limits[:,:,0],env.physical_limits[:,:,1])


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--mode',choices=['live','frozen-speed'],default='live');p.add_argument('--seconds',type=float,default=45)
    p.add_argument('--level-coordination',action='store_true')
    p.add_argument('--state-bank',default='replay_bank_67974_90655a91.pt')
    a=p.parse_args();out=(HERE/'outputs'/a.tag).resolve()
    if not out.is_relative_to((HERE/'outputs').resolve()) or out.exists():raise ValueError('Require new output directory')
    out.mkdir(parents=True);(out/'source').mkdir()
    for name in ['speed_follow_probe.py','level_speed_coordination.py','coordinator.py','environment.py','goal_steering.py','motion_constraints.py','config.json']:
        shutil.copy2(HERE/name,out/'source'/name)
    (out/'arguments.json').write_text(json.dumps(vars(a),indent=2))
    np.random.seed(0);torch.manual_seed(0)
    env=DualCarryEnv({'box_mass_kg':1.4,'box_size_m':[.60,.30,.30],
        'robot_positions_m':[[-.65,0,.78],[.65,0,.78]],'coordination_hz':10,'physics_hz':200,
        'carry_height_m':.65,'canonical_box_symmetry':True,'state_bank':a.state_bank,
        'grip_x_m':.19,'grip_standoff_m':.04,'filtered_context':True,'mass_adaptive_grasp':False,
        'travel_direction':'forward','layout':'opposed','coordinated_motion':True,'axial_carry':True,
        'goal_steering':True,'box_heading_goal':True,'steering_blend':.5,'transport_z_angle':.15,
        'level_feedback':a.level_coordination},device='cuda')
    policy=FrozenDualPolicy();stand=policy.latents.clone()
    if a.level_coordination:
        from level_speed_coordination import LevelSpeedProbe
        c=LevelSpeedProbe(env,policy,a.mode)
    else:c=SpeedProbe(env,policy,a.mode)
    for _ in range(5):policy.act(_masked_object_observation(env.observations()))
    env.reset();torch.cuda.synchronize()
    frames=[];vels=[];actions=[];timings=[];contexts=[];rewards=[];rows=[];latents=[];coord_ms=[]
    cf_update=None
    def coordinate(ct):
        nonlocal cf_update
        shadow=c.snapshot() if c.probe_start is not None else None
        actual=c(ct);cf_update=None
        if shadow is not None and c.stage=='transport':
            # Same ct, old z, temporal filters, yaw feedback and inference path.
            # Only the A-velocity cue used by B is changed. No physics is stepped.
            shadow.mode='frozen-speed';saved_grips=env.grip_commands.copy()
            cf_update=shadow(ct);env.grip_commands[:]=saved_grips
            if not np.allclose(actual['object_targets_m'],cf_update['object_targets_m'],atol=1e-10):
                raise AssertionError('Counterfactual must not alter actor object-goal inputs')
            if not torch.allclose(actual['latents'][0],cf_update['latents'][0],atol=1e-5,rtol=1e-5):
                raise AssertionError('Follower intervention unexpectedly changed leader latent')
        return actual
    start=time.perf_counter();failure=None
    with torch.inference_mode():
        for step in range(round(a.seconds*50)):
            tick=time.perf_counter();update=env.coordinate(coordinate)
            if update:
                coord_ms.append(1000*(time.perf_counter()-tick));policy.set_latents(update['latents'])
                contexts.append(env.latest_context);rewards.append(c.last_reward)
            obs=env.observations()
            if c.stage=='settle':policy.set_latents(stand);obs=_masked_object_observation(obs)
            action=policy.act(obs)
            if update and c.stage=='transport' and cf_update is not None:
                original=policy.latents.clone();policy.set_latents(cf_update['latents'])
                cf_action=policy.act(obs);policy.set_latents(original)
                targets=joint_targets(env,action);cf_targets=joint_targets(env,cf_action)
                ct=env.latest_context;roots=np.array([r['position_m'] for r in ct['robots']])
                raw=np.array([r['linear_velocity_mps'] for r in ct['robots']]);box=np.array(ct['object']['position_m'])
                gap=roots[1,:2]-roots[0,:2];ge=gap-c.initial_gap
                force=np.linalg.norm(np.array(ct['contact_window']['hand_mean_force_on_box_N']),axis=-1)
                row={'t':ct['time_s'],'elapsed':ct['time_s']-c.probe_start,'segment':c.segment,
                    'within_segment':c.within,'A_command_mps':float(c.command),
                    'A_actual_vx':float(raw[0,0]),'B_actual_vx':float(raw[1,0]),
                    'A_filtered_vx':float(c.filtered_actual[0,0]),'B_filtered_vx':float(c.filtered_actual[1,0]),
                    'B_target_vx':float(c.probe_target[1,0]),'B_target_vy':float(c.probe_target[1,1]),
                    'A_cue_used_by_B':float(c.filtered_actual[0,0]) if a.mode=='live' else c.frozen_A_speed,
                    'box_vx':ct['object']['linear_velocity_mps'][0],
                    'A_dx':float(roots[0,0]-c.start_roots[0,0]),'B_dx':float(roots[1,0]-c.start_roots[1,0]),
                    'box_dx':float(box[0]-c.initial_box[0]),'gap_error_x':float(ge[0]),'gap_error_y':float(ge[1]),
                    'gap_error_norm':float(np.linalg.norm(ge)),'box_height':float(box[2]),
                    'box_tilt_deg':float(np.degrees(np.arccos(np.clip(ct['object']['geometry_rotation_world'][2][2],-1,1)))),
                    'both_grasp':bool(np.array(ct['contacts']['hand_contact']).any(1).all()),
                    'max_hand_mean_force_N':float(force.max()),
                    'B_z_same_state_effect':max(0.,float(1-torch.nn.functional.cosine_similarity(original[1:2],cf_update['latents'][1:2])[0])),
                    'B_joint_target_effect_rad':float(np.sqrt(np.mean((targets[1]-cf_targets[1])**2))),
                    'B_actual_joint_targets_rad':targets[1].tolist(),'B_cf_joint_targets_rad':cf_targets[1].tolist(),
                    'B_z_step_change':float(c.last_z_change[1])}
                if a.level_coordination:
                    row.update(c.level_diagnostics())
                rows.append(row);latents.append(original.cpu().numpy())
                if a.mode=='frozen-speed' and row['B_joint_target_effect_rad']>1e-5:
                    raise AssertionError('Identical cue negative control failed')
                if any(r.get('no_feasible_candidates') for r in c.last_reward.get('rewards',[])):
                    failure='no_feasible_candidates';break
            if c.stage=='failed':failure=getattr(c,'failure_reason','posture_failure');break
            env.step(action,torque_callback=c.level_motor_feedback if a.level_coordination else None)
            frames.append(env.data.qpos.copy());vels.append(env.data.qvel.copy());actions.append(action)
            timings.append(1000*(time.perf_counter()-tick))
            if step%50==0:
                r=rows[-1] if rows else {}
                print(json.dumps({'t':round(env.data.time,2),'stage':c.stage,
                    **{k:round(r[k],4) for k in ['A_command_mps','A_filtered_vx','B_filtered_vx','B_target_vx','gap_error_norm','box_tilt_deg'] if k in r}}),flush=True)
            if c.probe_done:break
    np.savez_compressed(out/'trajectory.npz',qpos=frames,qvel=vels,actions=actions,z=latents,control_ms=timings)
    for name,value in [('config',env.config),('context_10hz',contexts),('rewards',rewards),('probe',rows)]:
        (out/f'{name}.json').write_text(json.dumps(value))
    keys=[k for k,v in (rows[0] if rows else {}).items() if not isinstance(v,list)]
    with (out/'probe.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    metrics={};plateaus=[]
    if rows:
        val=lambda key:np.array([r[key] for r in rows])
        rms=lambda x:float(np.sqrt(np.mean(x*x)))
        metrics={'A_command_speed_rmse':rms(val('A_filtered_vx')-val('A_command_mps')),
            'B_A_filtered_speed_rmse':rms(val('B_filtered_vx')-val('A_filtered_vx')),
            'B_A_raw_speed_rmse':rms(val('B_actual_vx')-val('A_actual_vx')),
            'B_own_target_speed_rmse':rms(val('B_filtered_vx')-val('B_target_vx')),
            'gap_rmse_m':rms(val('gap_error_norm')),'gap_max_m':float(val('gap_error_norm').max()),
            'max_box_tilt_deg':float(val('box_tilt_deg').max()),'both_grasp_fraction':float(val('both_grasp').mean()),
            'max_hand_mean_force_N':float(val('max_hand_mean_force_N').max()),'box_displacement_x_m':float(val('box_dx')[-1]),
            'B_same_obs_joint_effect_max_rad':float(val('B_joint_target_effect_rad').max()),
            'B_same_state_z_effect_max':float(val('B_z_same_state_effect').max())}
        for index,target in enumerate([.05,.10,.20,.10,0.]):
            selected=[r for r in rows if r['segment']==index and 4<=r['within_segment']<6]
            plateaus.append({'segment':index,'target_mps':target,'samples':len(selected),
                **({key:float(np.mean([r[key] for r in selected])) for key in ['A_filtered_vx','B_filtered_vx','B_target_vx','box_vx','gap_error_norm']} if selected else {})})
    checks={'finished_speed_schedule':c.probe_done,'no_failure':failure is None,
        'A_tracks_command_rmse_le_0p05':metrics.get('A_command_speed_rmse',1)<=.05,
        'B_tracks_A_rmse_le_0p05':metrics.get('B_A_filtered_speed_rmse',1)<=.05,
        'formation_max_le_0p15m':metrics.get('gap_max_m',1)<=.15,
        'box_tilt_le_15deg':metrics.get('max_box_tilt_deg',180)<=15,
        'both_grasp_ge_95percent':metrics.get('both_grasp_fraction',0)>=.95,
        'hand_mean_load_le_60N':metrics.get('max_hand_mean_force_N',1000)<=60}
    summary={'experiment':'A variable speed, B following','mode':a.mode,'success':all(checks.values()),
        'checks':checks,'metrics':metrics,'plateaus':plateaus,'failure_reason':failure,'events':c.events,
        'simulation_seconds':env.data.time,'probe_start_s':c.probe_start,'wall_seconds':time.perf_counter()-start,
        'coordination_with_cf_ms_p95':float(np.percentile(coord_ms,95)),
        'control_with_cf_ms_p95':float(np.percentile(timings,95)) if timings else None,
        'notes':['Fixed, predeclared protocol; no tuning after results.',
            'Frozen-speed condition freezes only A speed cue; position formation and physical coupling remain live.',
            'Frozen-speed same-state negative control must have joint-target RMS difference <= 1e-5 rad.',
            'Plateau means use final 2 seconds of each 6 second speed stage; velocities are signed world X.',
            '0.2 s low-pass velocities reported alongside raw velocities; timing includes shadow coordination and extra actor call.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
