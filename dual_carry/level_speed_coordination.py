"""Shared object target and bounded, contact-aware level correction.

The reward/B/z path remains frozen I-BFM. Residuals act only on arm motors.
"""
import itertools
import numpy as np
import mujoco
from environment import HAND_OFFSETS
from coordinator import Coordinator
from speed_follow_probe import SpeedProbe


def level_rotation(rotation):
    yaw=float(np.arctan2(rotation[1,0],rotation[0,0]))
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def allocate_vertical(points,com,contact,total_force,moment,limit=12.):
    """Bounded least squares over real contacts: sum Fz, roll and pitch moments.

    Exhaustive active sets are tiny (at most 3^4) and avoid a solver dependency.
    Outputs nonnegative support, never fictitious pulling through a lost grasp.
    """
    active=np.flatnonzero(np.asarray(contact).reshape(-1));out=np.zeros(4)
    offsets=np.asarray(points).reshape(4,3)-com
    G=np.vstack([np.ones(4),offsets[:,1],-offsets[:,0]])
    target=np.r_[total_force,moment[:2]]
    if not len(active):return out.reshape(2,2),G@out-target
    W=np.diag([1.,3.,3.]);regularizer=.08
    A=np.vstack([W@G[:,active],regularizer*np.eye(len(active))])
    b=np.r_[W@target,np.full(len(active),regularizer*total_force/len(active))]
    best=np.inf;solution=np.zeros(len(active))
    for bounds in itertools.product((-1,0,1),repeat=len(active)):
        bounds=np.array(bounds);free=bounds==0;candidate=np.where(bounds==1,limit,0.).astype(float)
        if free.any():candidate[free]=np.linalg.lstsq(A[:,free],b-A[:,~free]@candidate[~free],rcond=None)[0]
        if (candidate< -1e-8).any() or (candidate>limit+1e-8).any():continue
        error=float(np.square(A@candidate-b).sum())
        if error<best:best=error;solution=candidate
    out[active]=np.clip(solution,0,limit)
    return out.reshape(2,2),G@out-target


class LevelSpeedProbe(SpeedProbe):
    def __init__(self,env,policy,mode):
        super().__init__(env,policy,mode)
        self.level_ready=False;self.level_time=-1.
        self.support_targets=np.zeros((2,2));self.level_hands=np.zeros((2,2,3))
        self.level_offsets=np.zeros((2,2,3));self.grip_rotations=np.zeros((2,2,3,3))
        self.support_filter=np.zeros((2,2));self.support_filter_step=-1
        self.level_motor_peak=0.;self.level_motor_clip_steps=0
        self.common_object_velocity=np.zeros(3);self.reward_object_targets=np.zeros((2,3))

    def _change(self,stage,t):
        super()._change(stage,t)
        if stage=='transport' and not self.level_ready:
            env=self.env;R=env.data.geom_xmat[env.box_geom].reshape(3,3)
            box=env.data.xpos[env.box_body]
            for i,r in enumerate(env.robots):
                for h,b in enumerate(r.hands):
                    hand_R=env.data.xmat[b].reshape(3,3)
                    point=env.data.xpos[b]+hand_R@HAND_OFFSETS[h]
                    self.level_offsets[i,h]=R.T@(point-box)
                    self.grip_rotations[i,h]=R.T@hand_R
            self.level_ready=True

    def prepare_level_targets(self):
        ct=self.current_ct
        if self.level_time==ct['time_s']:return
        self.level_time=ct['time_s'];env=self.env
        box=np.array(ct['object']['position_m']);R=np.array(ct['object']['geometry_rotation_world'])
        Rlevel=level_rotation(R);omega=np.array(ct['object']['angular_velocity_radps'])
        velocity=np.array(ct['object']['linear_velocity_mps'])
        self.common_object_velocity=np.array([self.command,0.,np.clip(1.2*(self.height-box[2])-.2*velocity[2],-.04,.04)])
        self.level_moment=3.*np.cross(R[:,2],np.array([0.,0.,1.]))-.35*np.r_[omega[:2],0.]
        self.level_moment[2]=0.
        self.level_moment*=min(1.,1.2/max(np.linalg.norm(self.level_moment),1e-9))
        points=np.zeros((2,2,3))
        for i,r in enumerate(env.robots):
            for h,b in enumerate(r.hands):
                point=env.data.xpos[b]+env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
                contacts=[pair['position_m'] for pair in ct['contacts']['pairs'] if pair['robot_index']==i and pair['hand_index']==h]
                points[i,h]=np.mean(contacts,axis=0) if contacts else point
                # Preserve the acquired grasp offset, but remove box roll/pitch.
                self.level_hands[i,h]=box+Rlevel@self.level_offsets[i,h]
        floor=sum(pair['force_on_box_N'][2] for pair in ct['contacts']['pairs'] if pair['other_geom']=='floor')
        weight=env.config['box_mass_kg']*9.81
        total=np.clip(weight-floor+5*(self.height-box[2])-2*velocity[2],0,1.5*weight)
        self.support_targets,self.allocation_residual=allocate_vertical(points,env.data.xipos[env.box_body],
            ct['contacts']['hand_contact'],total,self.level_moment)
        self.level_R=Rlevel

    def reward(self,velocity,relative,hand_offsets,root_height,object_velocity,rotation_target,world_rotation=None):
        if self.stage!='transport':
            return Coordinator.reward(self,velocity,relative,hand_offsets,root_height,object_velocity,rotation_target,world_rotation)
        self.prepare_level_targets();i=self.reward_robot_index
        desired=self.robot_velocity_target()
        velocity=world_rotation.T@desired
        object_velocity=world_rotation.T@self.common_object_velocity
        self.reward_object_targets[i]=world_rotation@object_velocity
        box=np.array(self.current_ct['object']['position_m'])
        hand_offsets=np.array([world_rotation.T@(self.level_hands[i,h]-box) for h in range(2)])
        return Coordinator.reward(self,velocity,relative,hand_offsets,root_height,object_velocity,rotation_target,world_rotation)

    def __call__(self,ct):
        result=super().__call__(ct)
        if self.stage=='transport':
            if not np.allclose(self.reward_object_targets[0],self.reward_object_targets[1],atol=1e-9):
                raise AssertionError('Both policies must reward the same world-frame object velocity')
            self.hand_targets=self.level_hands.copy()
            self.vertical_force_targets=self.support_targets.copy()
            self.shared_box_velocity=self.common_object_velocity.copy()
            self.last_reward['level_feedback']=self.level_diagnostics()
            self.last_reward['shared_box_velocity_target_world_mps']=self.common_object_velocity.tolist()
            self.last_reward['vertical_force_targets_N']=self.support_targets.tolist()
        return result

    def level_motor_feedback(self,env,index,torque):
        if self.stage!='transport' or not self.level_ready:return torque
        contacts=env.contacts()
        if self.support_filter_step!=env.physics_steps:
            measured=np.array(contacts['hand_force_on_box_N'])[:,:,2]
            self.support_filter+=(1-np.exp(-env.model.opt.timestep/.04))*(measured-self.support_filter)
            self.support_filter_step=env.physics_steps
        r=env.robots[index];out=torque.copy()
        box=env.data.xpos[env.box_body];R=env.data.geom_xmat[env.box_geom].reshape(3,3)
        Rlevel=level_rotation(R)
        # Coordinate targets refresh at 10 Hz; motor feedback evaluates at 200 Hz.
        ramp=min(1.,max(0.,(env.data.time-self.probe_start)/1.))
        box_velocity=self.context_box_velocity
        for h,b in enumerate(r.hands):
            if not contacts['hand_contact'][index][h]:continue
            point=env.data.xpos[b]+env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
            desired=box+Rlevel@self.level_offsets[index,h]
            jac=np.zeros((3,env.model.nv));jr=np.zeros_like(jac)
            mujoco.mj_jac(env.model,env.data,jac,jr,point,b)
            ez=float(np.clip(desired[2]-point[2],-.05,.05))
            vz=float((jac@env.data.qvel)[2]-box_velocity[2])
            force_z=np.clip(80*ez-8*vz+.25*(self.support_targets[index,h]-self.support_filter[index,h]),-8.,8.)
            actual=env.data.xmat[b].reshape(3,3);wanted=Rlevel@self.grip_rotations[index,h]
            orient=.5*sum(np.cross(actual[:,k],wanted[:,k]) for k in range(3))
            moment=np.clip(.8*orient-.06*(jr@env.data.qvel),-.35,.35)
            arm=np.arange(15+7*h,22+7*h)
            residual=jac[2,r.dof[arm]]*force_z+jr[:,r.dof[arm]].T@moment
            if np.max(np.abs(residual))>2.:self.level_motor_clip_steps+=1
            residual=ramp*np.clip(residual,-2.,2.)
            self.level_motor_peak=max(self.level_motor_peak,float(np.max(np.abs(residual))))
            out[arm]+=residual
        return out

    def level_diagnostics(self):
        return {'common_object_target_world_mps':self.common_object_velocity.tolist(),
            'reward_object_targets_world_mps':self.reward_object_targets.tolist(),
            'level_hand_targets_m':self.level_hands.tolist(),'support_targets_N':self.support_targets.tolist(),
            'level_moment_target_Nm':self.level_moment.tolist(),
            'allocation_residual_N_Nm':self.allocation_residual.tolist(),
            'level_motor_peak_Nm':self.level_motor_peak,'level_motor_clip_steps':self.level_motor_clip_steps}
