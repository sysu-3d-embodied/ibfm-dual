"""Two independent I-BFM observations/controllers, one MuJoCo world and box.

The official repository is read-only. No welds, mocap bodies or root actuators.
Forces in context are contact forces (not inferred rewards or actuator torques).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np
import torch

from humanoidverse.carry_goal_sim2sim import (
    ActorObservation, _actuator_indices, _joint_addresses,
    _match_joint_parameter, _mode15_armature, _world_body_velocity,
)
from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
from humanoidverse.utils.robot_spec import load_robot_training_spec

ROOT = Path(os.environ.get("IBFM_ROOT", str(Path(__file__).resolve().parents[1])))
REPO = ROOT / "repo/ibfm"
HERE = Path(__file__).resolve().parent
HAND_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
HAND_GEOMS = ("left_hand_collision", "right_hand_collision")
HAND_OFFSETS = np.array([[.11, -.01, 0.], [.11, .01, 0.]])


@dataclass
class RobotIndex:
    prefix: str
    root_qpos: int
    root_dof: int
    qpos: np.ndarray
    dof: np.ndarray
    actuators: np.ndarray
    base: int
    torso: int
    bodies: list[int]
    hands: list[int]
    hand_geoms: list[int]
    contact_bodies: list[int]
    hand_subtrees: list[list[int]]


def _vec(value):
    return np.asarray(value, dtype=float).tolist()


class DualCarryEnv:
    def __init__(self, config: dict | None = None, device="cpu"):
        self.config = json.loads((HERE / "config.json").read_text())
        if config:
            self.config.update(config)
        c = self.config
        from mass_grasp import targets
        self.mass_grasp=targets(c['box_mass_kg'],c['box_friction'][0])
        self.grip_force_filtered=np.zeros((2,2))
        self.device = torch.device(device)
        self.revo2 = c.get('robot_model') == 'revo2'
        self.parallel=c.get('layout')=='parallel'
        self.object_anchors=np.array([[-.36,-.18,0],[.36,-.18,0]]) if self.parallel else np.zeros((2,3))
        ph, ah, ch = (int(c[k]) for k in ("physics_hz", "policy_hz", "coordination_hz"))
        if ph < 200 or ah != 50 or ch <= 0 or ph % ah or ah % ch:
            raise ValueError("Physics must be a multiple of 50 Hz and at least 200; coordination must divide 50")
        self.substeps, self.coordination_period = ph // ah, ah // ch
        self.training = load_robot_training_spec(REPO / "configs/robots/g1_29dof.yaml")
        self.joint_names = list(self.training.robot.control_joint_names)
        self.body_names = list(self.training.robot.body_names)
        self.defaults = np.array([self.training.default_joint_angles[n] for n in self.joint_names])
        self.kp = np.array([_match_joint_parameter(n, self.training.stiffness) for n in self.joint_names])
        self.kd = np.array([_match_joint_parameter(n, self.training.damping) for n in self.joint_names])
        self.effort = np.array(self.training.effort_limits)
        self.limits = np.array([self.training.robot.joint_ranges[n] for n in self.joint_names])
        scales = np.array([self.training.action_scale_overrides.get(n, self.training.action_scale) for n in self.joint_names])
        self.target_scale = scales * self.effort / self.kp
        self.action_clip = 5.0
        self.spec, self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.robots = []
        for prefix in ("r1/", "r2/"):
            names = [prefix+n for n in self.joint_names]
            qa, va = _joint_addresses(m, names)
            root = m.joint(prefix+"floating_base_joint").id
            hand_ids = [m.body(prefix+n).id for n in HAND_NAMES]
            all_bodies = [b for b in range(1,m.nbody) if m.body(b).name.startswith(prefix)]
            subtrees=[]
            for h in hand_ids:
                group=[]
                for b in all_bodies:
                    ancestor=b
                    while ancestor>0 and ancestor!=h:ancestor=int(m.body_parentid[ancestor])
                    if ancestor==h:group.append(b)
                subtrees.append(group)
            self.robots.append(RobotIndex(
                prefix, int(m.jnt_qposadr[root]), int(m.jnt_dofadr[root]), qa, va,
                _actuator_indices(m, names), m.body(prefix+"pelvis").id,
                m.body(prefix+"torso_link").id,
                [m.body(prefix+n).id for n in self.body_names],
                hand_ids,
                [m.geom(prefix+n).id for n in (("left_hand_base_link_collision_0","right_hand_base_link_collision_0") if self.revo2 else HAND_GEOMS)],
                all_bodies, subtrees,
            ))
            # Apply the official mode-15 physical parameters using UNPREFIXED names.
            m.dof_armature[va] = [_mode15_armature(n) for n in self.joint_names]
            m.dof_frictionloss[va] = .01
            if not self.revo2:
                for n in HAND_GEOMS:
                    m.geom_friction[m.geom(prefix+n).id] = c["box_friction"]
        self.box_body = m.body("shared_box").id
        self.box_geom = m.geom("shared_box_geom").id
        self.box_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g]==self.box_body and m.geom_contype[g]>0}
        j = m.joint("shared_box_freejoint").id
        self.box_qpos, self.box_dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.physical_limits=np.array([m.jnt_range[[m.joint(r.prefix+n).id for n in self.joint_names]] for r in self.robots])
        self.fingers=[]
        if self.revo2:
            for i,r in enumerate(self.robots):
                for h,side in enumerate(('left','right')):
                    joints=[j for j in range(m.njnt) if m.joint(j).name.startswith(r.prefix+side+'_') and
                            any(token in m.joint(j).name for token in ('thumb_','index_','middle_','ring_','pinky_'))]
                    names=[m.joint(j).name for j in joints]
                    qa,va=_joint_addresses(m,names)
                    closed=np.array([.8 if 'thumb_' in n else (1.1 if 'proximal' in n else 1.3) for n in names])
                    self.fingers.append({'robot':i,'hand':h,'qpos':qa,'dof':va,'act':_actuator_indices(m,names),
                                         'closed':closed,'effort':m.jnt_actfrcrange[joints,1].copy(),'names':names})
        self.observers = [ActorObservation(
            device=self.device, default_dof_pos=self.defaults,
            target_object_pos=np.array(c["object_target_m"])+anchor,
            hand_contact_offsets=HAND_OFFSETS, include_hand_spatial=True,
        ) for anchor in self.object_anchors]
        self._geom_names=[m.geom(g).name for g in range(m.ngeom)]
        self._contact_owners={}
        for i,r in enumerate(self.robots):
            for body in r.contact_bodies:
                hand=next((h for h,group in enumerate(r.hand_subtrees) if body in group),None)
                self._contact_owners[body]=(i,hand)
        self.reset()

    def _build_model(self):
        c = self.config
        spec = mujoco.MjSpec.from_string('''<mujoco model="ibfm_dual_shared_box">
          <option timestep="0.005" gravity="0 0 -9.81"/>
          <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/>
            <headlight ambient=".3 .3 .3" diffuse=".5 .5 .5" specular=".1 .1 .1"/>
            <map znear=".01" zfar="40"/></visual>
          <asset><texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" rgb1=".30 .34 .39" rgb2=".38 .42 .47"/>
            <material name="floor_mat" texture="floor_tex" texrepeat="20 20" reflectance=".05"/></asset>
          <worldbody><geom name="floor" type="plane" size="15 15 .05" material="floor_mat" friction="1 .005 .0001"/>
            <light name="key" pos="-2 -3 5" dir=".2 .3 -1" diffuse=".8 .8 .8"/>
            <light name="fill" pos="2 3 4" dir="-.2 -.3 -1" diffuse=".5 .5 .5" castshadow="false"/>
          </worldbody></mujoco>''')
        for prefix in ("r1/", "r2/"):
            robot_xml = HERE/'g1_revo2_robot.xml' if self.revo2 else self.training.robot.xml_path
            child = mujoco.MjSpec.from_file(str(robot_xml))
            spec.attach(child, prefix=prefix, frame=spec.worldbody.add_frame())
        box = spec.worldbody.add_body(name="shared_box", pos=c["box_position_m"])
        box.add_freejoint(name="shared_box_freejoint")
        geometry_quat = np.array(c["box_geom_quaternion_wxyz"], dtype=float)
        geometry_quat /= np.linalg.norm(geometry_quat)
        geometry_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(geometry_rotation, geometry_quat)
        geometry_rotation = geometry_rotation.reshape(3,3)
        box.add_geom(name="shared_box_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=np.array(c["box_size_m"])/2, mass=c["box_mass_kg"]-(.16 if c.get('box_handles') and not c.get('handle_height_m',0) and not self.parallel else 0),
                     quat=geometry_quat,
                     rgba=[.68, .43, .18, 1], friction=c["box_friction"], condim=3,
                     contype=1, conaffinity=1)
        if c.get('box_handles'):
            # Four rigid vertical grip bars belong to the SAME box body. No weld.
            for sx in (-1,1):
                for sy in (-1,1):
                    point=np.array([sx*.36+sy*.17,-.22,c.get('handle_height_m',0.)]) if self.parallel else np.array([sx*c.get('grip_x_m',.17),sy*(c['box_size_m'][1]/2+c.get('grip_standoff_m',.04)),c.get('handle_height_m',0.)])
                    box.add_geom(name=f'grip_{sx}_{sy}',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=[.016,.055,0],pos=geometry_rotation@point,quat=geometry_quat,
                        rgba=[.12,.14,.16,1],mass=0 if c.get('handle_height_m',0) or self.parallel else .04,friction=[1.5,.01,.001],contype=1,conaffinity=1)
                    if c.get('handle_height_m',0):
                        # Raised handles have their mass lumped in the 1.4 kg
                        # payload's cuboid inertia. All parts are one rigid body.
                        connector_q=np.zeros(4);mujoco.mju_mulQuat(connector_q,geometry_quat,np.array([2**-.5,2**-.5,0,0]))
                        for sz in (-1,1):
                            p=np.array([point[0],-.20,c['handle_height_m']+sz*.05]) if self.parallel else np.array([sx*c.get('grip_x_m',.17),sy*(c['box_size_m'][1]/2+c.get('grip_standoff_m',.04)/2),c['handle_height_m']+sz*.05])
                            box.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=[.007,.02,0],pos=geometry_rotation@p,quat=connector_q,
                                rgba=[.12,.14,.16,1],mass=0,friction=[1.5,.01,.001],contype=1,conaffinity=1)
        # Visual corner markings have no mass or collision influence.
        for sign in (-1, 1):
            box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[.012, c["box_size_m"][1]/2+.001, c["box_size_m"][2]/2+.001],
                         pos=geometry_rotation @ np.array([sign*c["box_size_m"][0]*.32, 0, 0]),
                         quat=geometry_quat, mass=0,
                         contype=0, conaffinity=0, rgba=[.22,.17,.11,1], group=2)
        model = spec.compile()
        model.opt.timestep=1/c['physics_hz']
        for prefix, color in (("r1/", [.24,.55,.86,1]), ("r2/", [.92,.45,.22,1])):
            if not self.revo2:model.mat_rgba[model.material(prefix+"silver").id] = color
            else:
                for gid in range(model.ngeom):
                    body_name=model.body(int(model.geom_bodyid[gid])).name
                    if model.geom_group[gid]==2 and body_name.startswith(prefix) and not any(t in body_name for t in ('hand_base','thumb_','index_','middle_','ring_','pinky_')):
                        model.geom_rgba[gid]=color
        if self.revo2:
            model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            model.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
            model.opt.impratio=10
        model.vis.global_.offwidth = int(c["render_width"])
        model.vis.global_.offheight = int(c["render_height"])
        return spec, model

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for i, r in enumerate(self.robots):
            q = r.root_qpos
            self.data.qpos[q:q+3] = self.config["robot_positions_m"][i]
            yaw = self.config["robot_yaws_rad"][i]
            self.data.qpos[q+3:q+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            self.data.qpos[r.qpos] = self.defaults
        q = self.box_qpos
        self.data.qpos[q:q+3] = self.config["box_position_m"]
        # Preserve the official object's marker/body frame. An upright box does
        # NOT have identity body quaternion in the training data.
        geometry_quat = np.array(self.config["box_geom_quaternion_wxyz"], dtype=float)
        geometry_quat /= np.linalg.norm(geometry_quat)
        self.data.qpos[q+3:q+7] = geometry_quat * [1,-1,-1,-1]
        self.current_actions = np.zeros((2, 29))
        self.grip_commands=np.zeros((2,2));self.grip_closure=np.zeros((2,2))
        self.grip_force_filtered[:]=0.
        self.target_positions = np.tile(self.config["object_target_m"], (2,1)).astype(float)
        for observer,anchor in zip(self.observers,self.object_anchors):
            observer.reset_history()
            observer.set_target_object_pos(np.array(self.config["object_target_m"])+anchor)
        self.policy_steps = self.physics_steps = self.coordination_updates = 0
        self.latest_context = None
        self._observation_cache = None
        self._contact_cache = None
        mujoco.mj_forward(self.model, self.data)
        self._window_impulse = np.zeros((2,2,3))
        self._window_peak = np.zeros((2,2))
        self._window_steps = 0
        return self.context()

    def forward(self):
        """After test-only changes of qpos/qvel; invalidate observation cache."""
        mujoco.mj_forward(self.model, self.data)
        self._observation_cache = None
        self._contact_cache = None

    def _body_state(self, body_id):
        linear, angular = _world_body_velocity(self.model, self.data, body_id)
        return {"position_m": _vec(self.data.xpos[body_id]),
                "quaternion_wxyz": _vec(self.data.xquat[body_id]),
                "linear_velocity_mps": _vec(linear), "angular_velocity_radps": _vec(angular)}

    def contacts(self):
        """Forces on the BOX, in world frame; torque measured about box COM.

        mj_contactForce is the wrench on geom2, in the contact frame. The rows
        of contact.frame are basis vectors: use frame.T, and flip for geom1.
        https://mujoco.readthedocs.io/en/stable/computation/index.html
        """
        if self._contact_cache is not None:return self._contact_cache
        hand_force = np.zeros((2,2,3)); hand_torque = np.zeros((2,2,3));hand_normal=np.zeros((2,2))
        hand_flags = np.zeros((2,2), dtype=bool)
        robot_force = np.zeros((2,3)); robot_torque = np.zeros((2,3))
        net_force = np.zeros(3); net_torque = np.zeros(3)
        details = []
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 not in self.box_geoms and g2 not in self.box_geoms:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            sign = 1 if g2 in self.box_geoms else -1
            frame = contact.frame.reshape(3,3)
            force = sign * (frame.T @ wrench[:3])
            torque_at_contact = sign * (frame.T @ wrench[3:])
            arm = contact.pos - self.data.xipos[self.box_body]
            torque = torque_at_contact + np.cross(arm, force)
            net_force += force; net_torque += torque
            other = g1 if g2 in self.box_geoms else g2
            body = int(self.model.geom_bodyid[other])
            ri,hi = self._contact_owners.get(body,(None,None))
            if ri is not None:
                robot_force[ri] += force; robot_torque[ri] += torque
                if hi is not None:
                    hand_force[ri,hi] += force; hand_torque[ri,hi] += torque
                    hand_normal[ri,hi]+=max(0,float(wrench[0]))
                    hand_flags[ri,hi] |= contact.efc_address >= 0 and abs(wrench[0])>1e-5
            details.append({"other_geom": self._geom_names[other],
                            "robot_index": ri, "hand_index": hi,
                            "position_m": _vec(contact.pos), "distance_m": float(contact.dist),
                            "force_on_box_N": _vec(force), "torque_on_box_about_com_Nm": _vec(torque),
                            "force_on_other_N": _vec(-force),
                            "torque_on_other_at_contact_Nm": _vec(-torque_at_contact)})
        result={"hand_contact": hand_flags.tolist(), "hand_force_on_box_N": _vec(hand_force),
                "hand_normal_contact_load_N":_vec(hand_normal),
                "hand_torque_on_box_about_com_Nm": _vec(hand_torque),
                "robot_force_on_box_N": _vec(robot_force),
                "robot_torque_on_box_about_com_Nm": _vec(robot_torque),
                "net_contact_force_on_box_N": _vec(net_force),
                "net_contact_torque_on_box_about_com_Nm": _vec(net_torque), "pairs": details}
        self._contact_cache=result
        return result

    def context(self):
        contact = self.contacts()
        robots = []
        for r in self.robots:
            robots.append({"id": r.prefix[:-1], **self._body_state(r.base),
                           "joint_position_rad": _vec(self.data.qpos[r.qpos]),
                           "joint_velocity_radps": _vec(self.data.qvel[r.dof]),
                           "hands": [self._body_state(h) for h in r.hands]})
        duration = self._window_steps * self.model.opt.timestep
        return {"schema": "ibfm-dual-context-v1", "time_s": float(self.data.time),
                "frame": "world_z_up", "quaternion_order": "wxyz", "robots": robots,
                "object": {"id": "shared_box", **self._body_state(self.box_body),
                           "geometry_rotation_world": _vec(self.data.geom_xmat[self.box_geom].reshape(3,3)),
                           "mass_kg": float(self.model.body_mass[self.box_body]),
                           "size_m": self.config["box_size_m"]},
                "contacts": contact,
                "finger_closure":self.grip_closure.tolist(),
                "finger_joints":[{'robot_index':h['robot'],'hand_index':h['hand'],
                    'position_rad':_vec(self.data.qpos[h['qpos']]),'velocity_radps':_vec(self.data.qvel[h['dof']]),
                    'actuator_torque_Nm':_vec(self.data.actuator_force[h['act']]),'effort_limit_Nm':_vec(h['effort'])} for h in self.fingers],
                "contact_window": {"duration_s": duration,
                    "hand_impulse_on_box_Ns": _vec(self._window_impulse),
                    "hand_mean_force_on_box_N": _vec(self._window_impulse/max(duration, 1e-12)),
                    "hand_peak_force_N": _vec(self._window_peak)},
                "object_targets_m": self.target_positions.tolist()}

    def coordinate(self, callback: Callable | None = None):
        """Call once before each actor step. Callback receives ONE common snapshot.

        Future callback can return {'object_targets_m': [[...],[...]],
        'latents': tensor(2,256)}. Reward/B inference is deliberately not included.
        """
        if self.policy_steps % self.coordination_period:
            return None
        if getattr(self, "_last_coordinate_step", -1) == self.policy_steps and self.latest_context is not None:
            return None
        self.latest_context = self.context()
        self._last_coordinate_step = self.policy_steps
        self.coordination_updates += 1
        update = callback(self.latest_context) if callback else None
        if update and "object_targets_m" in update:
            targets = np.asarray(update["object_targets_m"], dtype=float)
            if targets.shape != (2,3) or not np.isfinite(targets).all():
                raise ValueError("object_targets_m must have finite shape (2,3)")
            self.target_positions = targets.copy()
            for o,t,anchor in zip(self.observers, targets,self.object_anchors):
                o.set_target_object_pos(t+anchor)
            self._observation_cache = None
        self._window_impulse.fill(0); self._window_peak.fill(0); self._window_steps = 0
        return update

    def _privileged(self, r, state):
        # Adapted from official carry_goal_sim2sim._backward_observation_from_mujoco:
        # same ordered 30 bodies + virtual head, selecting only THIS robot.
        pos = [self.data.xpos[b].copy() for b in r.bodies]
        quat = [np.roll(self.data.xquat[b], -1) for b in r.bodies]
        velocities = [_world_body_velocity(self.model, self.data, b) for b in r.bodies]
        vel = [v[0] for v in velocities]; ang = [v[1] for v in velocities]
        ti = r.bodies.index(r.torso)
        head_offset = self.data.xmat[r.torso].reshape(3,3) @ np.array([0,0,.35])
        pos.append(pos[ti]+head_offset); quat.append(quat[ti])
        vel.append(vel[ti]+np.cross(ang[ti], head_offset)); ang.append(ang[ti])
        tensor = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32, device=self.device)[None]
        parts = compute_humanoid_observations_max(tensor(pos), tensor(quat), tensor(vel), tensor(ang),
                                                local_root_obs=True, root_height_obs=True)
        return torch.cat((*parts.values(), state[:,-34:]), dim=-1)

    @torch.no_grad()
    def observations(self):
        # Reading twice in the same actor tick MUST NOT advance history twice.
        if self._observation_cache is not None:
            return self._observation_cache
        contact = self.contacts()["hand_contact"]
        box_vel, box_angular = _world_body_velocity(self.model, self.data, self.box_body)
        obs = []
        for i,(r,o) in enumerate(zip(self.robots, self.observers)):
            _, angular = _world_body_velocity(self.model, self.data, r.base)
            box_quat=self.data.xquat[self.box_body].copy()
            anchor_world_offset=self.data.geom_xmat[self.box_geom].reshape(3,3)@self.object_anchors[i]
            observed_object_pos=self.data.xpos[self.box_body]+anchor_world_offset
            observed_object_velocity=box_vel+np.cross(box_angular,anchor_world_offset)
            if self.parallel:
                # Rigid grasp-reference frames on ONE physical payload. Each
                # policy keeps the original state98/privileged497 schema.
                geometry_quat=np.array(self.config['box_geom_quaternion_wxyz'],dtype=float);geometry_quat/=np.linalg.norm(geometry_quat)
                q=np.zeros(4);qa=np.zeros(4)
                mujoco.mju_mulQuat(q,box_quat,geometry_quat)
                mujoco.mju_mulQuat(qa,q,np.array([2**-.5,0,0,2**-.5]))
                mujoco.mju_mulQuat(box_quat,qa,geometry_quat*np.array([1,-1,-1,-1]))
            elif self.config.get('canonical_box_symmetry',False):
                # A rectangular unmarked box is invariant to 180 degrees around
                # its geometry Z axis. Choose the equivalent representation
                # whose X axis faces the observer; physical pose/ct is unchanged.
                geometry_quat=np.array(self.config['box_geom_quaternion_wxyz'],dtype=float)
                geometry_quat/=np.linalg.norm(geometry_quat)
                geom_world=np.zeros(4);mujoco.mju_mulQuat(geom_world,box_quat,geometry_quat)
                geom_rot=np.zeros(9);mujoco.mju_quat2Mat(geom_rot,geom_world)
                heading=self.data.xmat[r.base].reshape(3,3)[:,0]
                if geom_rot.reshape(3,3)[:,0]@heading<0:
                    flipped=np.zeros(4);mujoco.mju_mulQuat(flipped,geom_world,np.array([0.,0.,0.,1.]))
                    mujoco.mju_mulQuat(box_quat,flipped,geometry_quat*np.array([1,-1,-1,-1]))
            value = o.build(dof_pos=self.data.qpos[r.qpos], dof_vel=self.data.qvel[r.dof],
                            base_pos=self.data.xpos[r.base], base_quat_wxyz=self.data.xquat[r.base],
                            base_ang_vel_world=angular, object_pos=observed_object_pos,
                            object_quat_wxyz=box_quat, object_lin_vel=observed_object_velocity,
                            object_ang_vel=box_angular, hand_body_pos=self.data.xpos[r.hands],
                            hand_body_quat_wxyz=self.data.xquat[r.hands], hand_contact=np.array(contact[i]),
                            current_action=self.current_actions[i].copy())
            value["privileged_state"] = self._privileged(r, value["state"])
            obs.append(value)
        batch = {k:torch.cat([o[k] for o in obs], dim=0) for k in obs[0]}
        expected = {"state":98, "privileged_state":497, "last_action":29, "history_actor":372}
        for key,dim in expected.items():
            if tuple(batch[key].shape) != (2,dim) or not torch.isfinite(batch[key]).all():
                raise RuntimeError(f"Invalid {key} shape/data: {batch[key].shape}")
        self._observation_cache = batch
        return batch

    def step(self, actions, physics_callback=None, torque_callback=None):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (2,29) or not np.isfinite(actions).all():
            raise ValueError("actions must have finite shape (2,29)")
        self.current_actions = np.clip(actions, -self.action_clip, self.action_clip).copy()
        targets = np.clip(self.defaults+self.target_scale*self.current_actions, self.physical_limits[:,:,0], self.physical_limits[:,:,1])
        for _ in range(self.substeps):
            if physics_callback:
                physics_callback(self)
            for i,r in enumerate(self.robots):
                torque = self.kp*(targets[i]-self.data.qpos[r.qpos])-self.kd*self.data.qvel[r.dof]
                if torque_callback:
                    torque = torque_callback(self, i, torque)
                self.data.ctrl[r.actuators] = np.clip(torque, -self.effort, self.effort)
            if self.revo2:
                # Joint-effort-limited compliant closure; the coordination layer
                # selects grasp/release, and contact load gates further closing.
                measured=np.asarray(self.contacts()['hand_normal_contact_load_N'])
                requested=self.grip_commands.copy()
                if self.config.get('mass_adaptive_grasp'):
                    self.grip_force_filtered+=(1-np.exp(-self.model.opt.timestep/.04))*(measured-self.grip_force_filtered)
                    measured=self.grip_force_filtered
                    hold=self.mass_grasp['finger_hold_N'];relax=self.mass_grasp['finger_relax_N']
                else:hold,relax=30.,50.
                requested=np.where((measured>hold)&(requested>self.grip_closure),self.grip_closure,requested)
                requested=np.where((measured>relax)&(requested>0),np.maximum(0,self.grip_closure-.015),requested)
                self.grip_closure += np.clip(requested-self.grip_closure,-3*self.model.opt.timestep,3*self.model.opt.timestep)
                for hand in self.fingers:
                    target=hand['closed']*self.grip_closure[hand['robot'],hand['hand']]
                    kp=self.mass_grasp['finger_kp'] if self.config.get('mass_adaptive_grasp') else 2.
                    torque=kp*(target-self.data.qpos[hand['qpos']])-.08*np.sqrt(kp/2)*self.data.qvel[hand['dof']]
                    self.data.ctrl[hand['act']]=np.clip(torque,-hand['effort'],hand['effort'])
            mujoco.mj_step(self.model, self.data)
            self._contact_cache=None
            # This wrench is the force applied over the just-integrated interval.
            force = np.asarray(self.contacts()["hand_force_on_box_N"])
            self._window_impulse += force*self.model.opt.timestep
            self._window_peak = np.maximum(self._window_peak, np.linalg.norm(force,axis=-1))
            self._window_steps += 1
            self.physics_steps += 1
        self.policy_steps += 1
        # Refresh once at the observation boundary, not twice per physics tick.
        self.forward()


class FrozenDualPolicy:
    """One copy of frozen weights; two batch rows, independent observations/z."""
    def __init__(self, device="cuda"):
        from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
        import joblib
        self.device = torch.device(device)
        folder = REPO / "tasks/carry_box/checkpoint_ref"
        self.model = load_model_from_checkpoint_dir(folder/"checkpoint", device=self.device.type, strict=False)
        self.model.to(self.device).eval().requires_grad_(False)
        z = torch.as_tensor(joblib.load(REPO/"tasks/carry_box/latents/z_get_up_reward.pkl"),
                            device=self.device, dtype=torch.float32).reshape(-1,256)[0]
        self.latents = z.repeat(2,1).clone()

    def set_latents(self, z):
        value = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        if value.shape != (2,256) or not torch.isfinite(value).all():
            raise ValueError("latents must have finite shape (2,256)")
        self.latents = value.clone()

    @torch.inference_mode()
    def act(self, observation):
        # Official carry controller rescales normalized network output by 5
        # BEFORE clipping and storing it in last_action/history.
        return self.model.act(observation, self.latents, mean=True).cpu().numpy() * 5.0
