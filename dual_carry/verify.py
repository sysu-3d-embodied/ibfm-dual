"""Regression checks for the physics/observation bridge (no policy promises)."""
import json
import numpy as np
import torch
import mujoco
from environment import *
from humanoidverse.carry_goal_sim2sim import _build_model, _backward_observation_from_mujoco, _joint_addresses, _world_body_velocity


def main():
    e=DualCarryEnv(); m,d=e.model,e.data; report={}
    assert (m.nq,m.nv,m.nu,m.neq,m.nmocap)==((123,120,102,0,0) if e.revo2 else (79,76,58,0,0))
    assert set(e.robots[0].actuators).isdisjoint(e.robots[1].actuators)
    assert set(e.robots[0].qpos).isdisjoint(e.robots[1].qpos)
    assert sum(int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_FREE) for j in range(m.njnt))==3
    report['topology']=f'2 free G1 roots, {m.nu} motors, 1 free box, no welds/mocap'
    original_qpos=d.qpos.copy();obs=e.observations()
    assert e.observations() is obs
    assert e.observers[0].history['actions'].data_ptr()!=e.observers[1].history['actions'].data_ptr()
    # Reconstruct the SAME box world position through each actor's local frame.
    reconstructed=[]
    for i,r in enumerate(e.robots):
        local=obs['state'][i,64:67].cpu().numpy()
        reconstructed.append(d.xpos[r.base]+d.xmat[r.base].reshape(3,3)@local)
    assert np.allclose(reconstructed,d.xpos[e.box_body],atol=1e-6)
    report['shared_box_observation_error_m']=float(np.max(np.abs(np.array(reconstructed)-d.xpos[e.box_body])))
    # Privileged observation parity with the original unprefixed single-robot helper.
    single=_build_model(HERE/'g1_revo2_robot.xml' if e.revo2 else e.training.robot.xml_path,{'half_size':np.array(e.config['box_size_m'])/2,
                                                 'mass':e.config['box_mass_kg']},render_size=720)
    sd=mujoco.MjData(single);r=e.robots[0]
    sd.qpos[:7]=d.qpos[r.root_qpos:r.root_qpos+7]
    qa,va=_joint_addresses(single,e.joint_names)
    sd.qpos[qa]=d.qpos[r.qpos]
    sj=single.joint('sim2sim_carry_box_joint').id; sq=int(single.jnt_qposadr[sj])
    sd.qpos[sq:sq+7]=d.qpos[e.box_qpos:e.box_qpos+7]
    # Nonzero velocity also tests head velocity and world/local conventions.
    rng=np.random.default_rng(7)
    d.qvel[r.root_dof:r.root_dof+6]=rng.normal(0,.03,6)
    d.qvel[r.dof]=rng.normal(0,.1,29)
    sd.qvel[:6]=d.qvel[r.root_dof:r.root_dof+6];sd.qvel[va]=d.qvel[r.dof]
    e.forward();mujoco.mj_forward(single,sd)
    ob=e.observations()
    ref=_backward_observation_from_mujoco(model=single,data=sd,source_motion={'body_names':e.body_names},
        actor_observation={k:v[:1] for k,v in ob.items()},include_hand_spatial=True,device=torch.device('cpu'))
    err=(ref['privileged_state']-ob['privileged_state'][:1]).abs().max().item()
    assert err<1e-5,err
    report['official_privileged_max_error']=err
    # Read-only object-state changes affect BOTH observers in different local frames.
    e.reset();e.data.qvel[e.box_dof:e.box_dof+3]=[.1,.2,0];e.forward();ob=e.observations()
    for i,r in enumerate(e.robots):
        assert np.allclose(e.data.xmat[r.base].reshape(3,3)@ob['state'][i,73:76].numpy(),[.1,.2,0],atol=1e-6)
    # Box settled on the floor: gravity balances UPWARD contact force.
    e.reset()
    for _ in range(50):e.step(np.zeros((2,29)))
    force=np.array(e.contacts()['net_contact_force_on_box_N'])
    # Robots have moved in PD-only testing: isolate the floor check with distant roots.
    for i,r in enumerate(e.robots): e.data.qpos[r.root_qpos]=(-5 if i==0 else 5)
    e.data.qpos[e.box_qpos:e.box_qpos+3]=[0,0,.15];e.data.qpos[e.box_qpos+3:e.box_qpos+7]=original_qpos[e.box_qpos+3:e.box_qpos+7];e.data.qvel[:]=0;e.forward()
    for _ in range(100):mujoco.mj_step(m,d)
    e.forward()
    f=e.contacts()['net_contact_force_on_box_N'][2]
    assert abs(f-e.config['box_mass_kg']*9.81)<.2,f
    report['floor_support_N']=f
    # Actual hand-to-box collision fixtures, independently for each robot.
    hits=[]
    for i in range(2):
        e.reset(); r=e.robots[i]; other=e.robots[1-i]
        e.data.qpos[other.root_qpos]=10; e.forward()
        geom=r.hand_geoms[0]
        center=e.data.geom_xpos[geom].copy()
        # Test fixture intentionally places box into a hand, not a grasp controller.
        center[2]+=.05
        e.data.qpos[e.box_qpos:e.box_qpos+3]=center;e.forward()
        info=e.contacts(); assert any(info['hand_contact'][i]),info
        assert np.linalg.norm(info['robot_force_on_box_N'][i])>0
        hits.append(np.linalg.norm(info['robot_force_on_box_N'][i]).item())
        # Contact forces must drive the free object's acceleration with correct sign.
        f=np.array(info['net_contact_force_on_box_N'])
        expected=f/e.config['box_mass_kg']+np.array([0,0,-9.81])
        actual=e.data.qacc[e.box_dof:e.box_dof+3]
        assert np.allclose(expected,actual,rtol=1e-5,atol=1e-5),(expected,actual)
    report['both_robot_collision_forces_N']=hits
    # Scheduler counts and reset/history reproducibility.
    e.reset()
    for _ in range(50): e.coordinate();e.coordinate();e.observations();e.step(np.zeros((2,29)))
    assert (e.coordination_updates,e.policy_steps,e.physics_steps)==(e.config['coordination_hz'],50,e.config['physics_hz'])
    e.reset(); assert np.array_equal(original_qpos,e.data.qpos)
    assert e.latest_context is None and e.observers[0].history['actions'].count_nonzero()==0
    report['scheduler']='5 / 50 / 200 Hz verified'
    # Optional parallel layout observes rigid reference frames on ONE payload.
    # Verify position and velocity at nonzero payload rotation/angular speed.
    p=DualCarryEnv({'layout':'parallel','robot_positions_m':[[-.36,-.65,.78],[.36,-.65,.78]],
        'robot_yaws_rad':[np.pi/2,np.pi/2],'box_size_m':[1.2,.36,.30],'handle_height_m':.10})
    q=p.data.qpos[p.box_qpos+3:p.box_qpos+7].copy();rotated=np.zeros(4)
    mujoco.mju_mulQuat(rotated,np.array([np.cos(.15),0,0,np.sin(.15)]),q)
    p.data.qpos[p.box_qpos+3:p.box_qpos+7]=rotated
    p.data.qvel[p.box_dof:p.box_dof+6]=[.1,.2,.03,.07,-.04,.2];p.forward()
    obs=p.observations();vel,omega=_world_body_velocity(p.model,p.data,p.box_body)
    positions=[];velocities=[]
    for i,r in enumerate(p.robots):
        R=p.data.xmat[r.base].reshape(3,3);offset=p.data.geom_xmat[p.box_geom].reshape(3,3)@p.object_anchors[i]
        center=p.data.xpos[r.base]+R@obs['state'][i,64:67].numpy()-offset
        velocity=R@obs['state'][i,73:76].numpy()-np.cross(omega,offset)
        positions.append(center);velocities.append(velocity)
    assert np.allclose(positions,p.data.xpos[p.box_body],atol=1e-6)
    assert np.allclose(velocities,vel,atol=1e-6)
    assert p.model.neq==0 and p.model.nmocap==0
    report['parallel_reference_frames']='both rigid anchors reconstruct one shared center and velocity'
    report['passed']=True
    (HERE/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__': main()
