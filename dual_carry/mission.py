import argparse,json,time,shutil,gc
import numpy as np,torch
from environment import *
from coordinator import Coordinator
from humanoidverse.carry_phased_reward_inference import _masked_object_observation

p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=40);p.add_argument('--tag',default='mission')
p.add_argument('--native',action='store_true');p.add_argument('--impedance',action='store_true')
p.add_argument('--anchor',type=float,default=0);p.add_argument('--distance',type=float,default=.65)
p.add_argument('--hz',type=int,default=10);p.add_argument('--speed',type=float,default=.12)
p.add_argument('--temperature',type=float,default=10.)
p.add_argument('--canonical-symmetry',action='store_true')
p.add_argument('--state-bank',default='state_bank.pt')
p.add_argument('--force-feedback',action='store_true')
p.add_argument('--physics-hz',type=int,default=200)
p.add_argument('--load-guard',action='store_true')
p.add_argument('--carry-height',type=float,default=.65)
p.add_argument('--wrench-feedback',action='store_true')
p.add_argument('--handle-height',type=float,default=0.)
p.add_argument('--grip-x',type=float,default=.17)
p.add_argument('--grip-standoff',type=float,default=.04)
p.add_argument('--layout',choices=['opposed','parallel'],default='opposed')
p.add_argument('--travel',type=float,default=.60)
p.add_argument('--travel-direction',choices=['world-y','forward'],default='world-y',
               help='forward follows robot 1 initial heading; opposed robot 2 walks backward')
p.add_argument('--z-prior',type=float,default=0.)
p.add_argument('--transport-z-angle',type=float,default=3.141593)
p.add_argument('--hand-retarget',action='store_true')
p.add_argument('--filtered-context',action='store_true')
p.add_argument('--goal-reward',action='store_true')
p.add_argument('--smooth-lower',action='store_true')
p.add_argument('--compliant-release',action='store_true')
p.add_argument('--contact-retreat',action='store_true',help='During lowering, step away from box-leg/pelvis/foot contact')
p.add_argument('--mass-adaptive-grasp',action='store_true')
p.add_argument('--grasp-arm-blend',type=float,default=0.,help='Experimental arm force blend; 0 keeps learned arm control')
p.add_argument('--grasp-arm-limit',type=float,default=16.)
p.add_argument('--grasp-handle-target',action='store_true')
p.add_argument('--box-mass',type=float,default=None,help='Override total box mass in kg, including handles')
p.add_argument('--coordinated-motion',action='store_true')
p.add_argument('--axial-carry',action=argparse.BooleanOptionalAction,default=False)
p.add_argument('--goal-lookahead',type=float,default=.60)
p.add_argument('--goal-steering',action=argparse.BooleanOptionalAction,default=False)
p.add_argument('--turn-source',choices=['official','reward'],default='official')
p.add_argument('--steering-blend',type=float,default=.35)
p.add_argument('--min-ess',type=float,default=16.)
p.add_argument('--actor-goal-margin',type=float,default=.45)
p.add_argument('--lower-anchor',type=float,default=0.)
p.add_argument('--box-heading-goal',action='store_true')
p.add_argument('--placement-servo',action='store_true')
p.add_argument('--placement-xy-feedback',action='store_true')
p.add_argument('--lower-pose-feedback',action='store_true')
p.add_argument('--placement-clearance',type=float,default=0.)
p.add_argument('--box-length',type=float,default=.60)
p.add_argument('--placement-guide',default='')
p.add_argument('--guide-mix',type=float,default=.8)
p.add_argument('--guide-height-lead',type=float,default=0.)
a=p.parse_args();out=(HERE/'outputs'/a.tag).resolve()
if a.goal_steering and not a.axial_carry:p.error('--goal-steering requires --axial-carry')
if not 0<=a.steering_blend<=1 or not 0<=a.lower_anchor<=1:p.error('latent blend weights must be within [0,1]')
if a.min_ess<1 or a.speed<=0 or a.travel<=0:p.error('min-ess must be >= 1; speed and travel must be positive')
if a.box_length<.38:p.error('box-length must be >= .38 to contain the four existing grip bars')
if not 0<a.grip_x<a.box_length/2-.016 or a.grip_standoff<.025:p.error('Grip bars must fit along box length and clear the side wall')
if a.axial_carry and (a.layout!='opposed' or a.travel_direction!='forward' or not a.coordinated_motion):
    p.error('--axial-carry requires opposed layout, forward travel and coordinated motion')
if not out.is_relative_to((HERE/'outputs').resolve()):raise ValueError('Run tag must remain inside outputs')
if (out/'trajectory.npz').exists():raise ValueError('Run already exists; choose a new --tag')
out.mkdir(parents=True,exist_ok=True)
(out/'arguments.json').write_text(json.dumps(vars(a),indent=2))
(out/'source').mkdir(exist_ok=True)
for name in ('environment.py','coordinator.py','mass_grasp.py','contact_retreat.py','motion_constraints.py','axial_motion.py','goal_steering.py','assess.py','mission.py','config.json','g1_revo2_robot.xml','revo2_import.json'):
    shutil.copy2(HERE/name,out/'source'/name)
layout_config={'layout':a.layout,'z_prior':a.z_prior,'transport_z_angle':a.transport_z_angle,'hand_retarget':a.hand_retarget,'filtered_context':a.filtered_context,'goal_reward':a.goal_reward}
layout_config.update(smooth_lower=a.smooth_lower,compliant_release=a.compliant_release)
layout_config['contact_retreat']=a.contact_retreat
layout_config['mass_adaptive_grasp']=a.mass_adaptive_grasp
if not 0<=a.grasp_arm_blend<=1 or not 0<=a.grasp_arm_limit<=16:p.error('Invalid grasp arm blend/limit')
layout_config.update(grasp_arm_blend=a.grasp_arm_blend,grasp_arm_limit=a.grasp_arm_limit)
layout_config['grasp_handle_target']=a.grasp_handle_target
if a.box_mass is not None:
    if not np.isfinite(a.box_mass) or a.box_mass<=.16:p.error('Box mass must exceed 0.16 kg handle mass')
    layout_config['box_mass_kg']=a.box_mass
configured_mass=a.box_mass if a.box_mass is not None else json.loads((HERE/'config.json').read_text())['box_mass_kg']
layout_config['mass_adaptive_requested']=a.mass_adaptive_grasp
# Preserve the validated nominal controller at/below its 1.4 kg reference load.
layout_config['mass_adaptive_grasp']=a.mass_adaptive_grasp and configured_mass>1.4
layout_config['coordinated_motion']=a.coordinated_motion
layout_config['travel_direction']=a.travel_direction
layout_config.update(axial_carry=a.axial_carry,goal_lookahead=a.goal_lookahead)
layout_config.update(goal_steering=a.goal_steering,turn_source=a.turn_source,steering_blend=a.steering_blend,min_ess=a.min_ess)
layout_config['actor_goal_margin']=a.actor_goal_margin
layout_config['lower_anchor']=a.lower_anchor
layout_config['box_heading_goal']=a.box_heading_goal
layout_config['placement_servo']=a.placement_servo
layout_config['placement_xy_feedback']=a.placement_xy_feedback
layout_config['lower_pose_feedback']=a.lower_pose_feedback
layout_config['placement_clearance']=a.placement_clearance
layout_config.update(placement_guide=a.placement_guide,guide_mix=a.guide_mix)
layout_config['guide_height_lead']=a.guide_height_lead
layout_config.update(grip_x_m=a.grip_x,grip_standoff_m=a.grip_standoff)
if a.layout=='parallel':layout_config.update(robot_positions_m=[[-.36,-a.distance,.78],[.36,-a.distance,.78]],robot_yaws_rad=[np.pi/2,np.pi/2],box_size_m=[1.2,.36,.30])
e=DualCarryEnv({'robot_positions_m':[[-a.distance,0,.78],[a.distance,0,.78]],'box_size_m':[a.box_length,.30,.30],
                'coordination_hz':a.hz,'canonical_box_symmetry':a.canonical_symmetry,'state_bank':a.state_bank,'physics_hz':a.physics_hz,'carry_height_m':a.carry_height,'wrench_feedback':a.wrench_feedback,'handle_height_m':a.handle_height,**layout_config},device='cuda')
policy=FrozenDualPolicy();stand=policy.latents.clone()
c=Coordinator(e,policy,speed=a.speed,travel=a.travel,dynamic=not a.native,anchor=a.anchor,temperature=a.temperature)
c.standing_latent=stand.clone()
for _ in range(5):policy.act(_masked_object_observation(e.observations()))
if c.dynamic:
    with torch.inference_mode():
        warm=torch.zeros((1,len(c.B)),device=policy.device)
        for _ in range(3):policy.model.project_z(warm@c.B)
    # Warm reward, quaternion and reduction kernels before timing the control
    # loop. Discard this coordinator and reset the world afterwards.
    for stage in ('acquire','lift','transport','lower'):
        c.stage=stage
        for _ in range(3):c(e.context())
    c=Coordinator(e,policy,speed=a.speed,travel=a.travel,dynamic=True,anchor=a.anchor,temperature=a.temperature)
    c.standing_latent=stand.clone()
torch.cuda.synchronize();e.reset()
# Keep cyclic collection out of this bounded, finite control experiment.
# Refcounted per-tick objects still release normally; restore GC afterwards.
gc.collect();gc_was_enabled=gc.isenabled();gc.disable()
frames=[];velocities=[];actions=[];contexts=[];records=[];times=[];start=time.perf_counter()
for step in range(round(a.seconds*50)):
    tick=time.perf_counter()
    update=e.coordinate(c)
    if update:
        policy.set_latents(update['latents']);contexts.append(e.latest_context);records.append(c.last_reward)
        if a.axial_carry and any(r.get('no_feasible_candidates') for r in c.last_reward.get('rewards',[])):
            c._change('failed',e.data.time)
            print(json.dumps({'failure':'no_feasible_axial_candidates','time_s':e.data.time}),flush=True)
            break
    obs=e.observations()
    if c.stage in ('settle','complete','failed') or (c.stage=='release' and (not np.array(e.contacts()['hand_contact']).any() or (a.compliant_release and e.data.time-c.stage_start>=.7))):
        policy.set_latents(stand);obs=_masked_object_observation(obs)
    action=policy.act(obs)
    e.step(action,torque_callback=c.mass_grasp_feedback if a.mass_adaptive_grasp else c.placement_xy_feedback if a.placement_xy_feedback else c.placement_servo if a.placement_servo else c.release_compliance if a.compliant_release else c.wrench_feedback if a.wrench_feedback else c.grip_load_guard if a.load_guard else c.residual_force_feedback if a.force_feedback else c.impedance if a.impedance else None)
    times.append(1000*(time.perf_counter()-tick))
    frames.append(e.data.qpos.copy());velocities.append(e.data.qvel.copy());actions.append(action)
    if step%50==0:
        print(json.dumps({'t':round(e.data.time,2),'stage':c.stage,'root_z':[round(e.data.xpos[r.base,2],3) for r in e.robots],
           'box':e.data.xpos[e.box_body].round(3).tolist(),'contact':e.contacts()['hand_contact'],
           'z_change':c.last_z_change}),flush=True)
    if c.stage=='failed':break
    if c.stage=='complete' and e.data.time-c.stage_start>2:break
if gc_was_enabled:gc.enable()
np.savez_compressed(out/'trajectory.npz',qpos=np.array(frames),qvel=np.array(velocities),actions=np.array(actions),control_ms=np.array(times))
(out/'config.json').write_text(json.dumps(e.config,indent=2))
(out/'context_10hz.json').write_text(json.dumps(contexts))
(out/'rewards.json').write_text(json.dumps(records))
result={'stage':c.stage,'success':c.stage=='complete','events':c.events,'peak_height_m':c.peak_height,
        'failure_reason':getattr(c,'failure_reason',None),
        'box_final_m':e.data.xpos[e.box_body].tolist(),'destination_m':c.destination.tolist(),
        'wall_seconds':time.perf_counter()-start,'simulation_seconds':e.data.time,
        'mode':('native' if a.native else 'dynamic')+('_impedance' if a.impedance else '')+('_residual_force' if a.force_feedback else '')+('_load_guard' if a.load_guard else ''),'anchor':a.anchor,
        'wrench_feedback':a.wrench_feedback,
        'control_ms':{'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'max':float(np.max(times))}}
result['control_ms']['over_20ms_fraction']=float((np.array(times)>20).mean())
from assess import assess
assessment=assess(result,contexts,records)
result['phase_complete']=result['success'];result['success']=assessment['passed']
(out/'assessment.json').write_text(json.dumps(assessment,indent=2))
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
