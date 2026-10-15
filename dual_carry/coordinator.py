"""Context-conditioned reward -> cached B -> separate z; optional hand impedance.

No reward-program search. Phase changes require measured physical conditions.
The force-labelled state library does not yet exist: force feedback modifies
hand targets online, and the optional impedance layer acts only through motors.
"""
import numpy as np
import mujoco
import torch
from environment import HERE,HAND_OFFSETS
from humanoidverse.utils.torch_utils import quat_conjugate,quat_mul,quat_to_tan_norm


class Coordinator:
    def __init__(self, env, policy, *, speed=.12, travel=.60, dynamic=True, anchor=0., temperature=10.):
        import joblib
        from environment import REPO
        self.env,self.policy=env,policy
        self.speed,self.travel,self.dynamic,self.anchor=speed,travel,dynamic,anchor
        self.temperature=temperature
        self.stage='settle';self.stage_start=0.;self.start_box=np.array(env.config['box_position_m'])
        travel_mode=env.config.get('travel_direction','world-y')
        yaw=float(env.config['robot_yaws_rad'][0]) if travel_mode=='forward' else np.pi/2
        self.destination=self.start_box+travel*np.array([np.cos(yaw),np.sin(yaw),0.])
        self.height=.15;self.grasp_active=False;self.events=[];self.last_reward={}
        self.hand_targets=np.zeros((2,2,3));self.velocity_targets=np.zeros((2,3))
        self.force_target=8.;self.last_z=policy.latents.clone();self.last_z_change=[0.,0.]
        self.mass_adaptive=env.config.get('mass_adaptive_grasp',False)
        if self.mass_adaptive:self.force_target=env.mass_grasp['squeeze_N']
        self.grasp_ready_time=0.;self.last_hand_contact=np.full((2,2),-10.)
        self.grasp_offsets=np.zeros((2,2,3));self.mass_feedback_force=np.zeros((2,2,3))
        self.grasp_rotations=np.zeros((2,2,3,3));self.lost_grasp_time=0.
        self.grasp_since=np.full((2,2),-10.)
        self.vertical_force_targets=np.full((2,2),env.config['box_mass_kg']*9.81/4)
        self.stable_time=0.;self.peak_height=.15
        self.formation_offsets=None
        self.grip_latched=np.zeros((2,2),dtype=bool);self.grip_lost_time=np.zeros((2,2))
        self.allocated_forces=np.zeros((2,2,3));self.filtered_forces=np.zeros((2,2,3))
        self.force_integral=np.zeros((2,29));self.force_filter_step=-1
        self.context_box_velocity=np.zeros(3);self.context_root_velocity=np.zeros((2,3))
        self.smoothed_targets=np.zeros((2,3))
        self.shared_box_velocity=np.zeros(3)
        self.heading_errors=np.zeros(2)
        self.goal_steering=env.config.get('goal_steering',False)
        self.travel_yaw=yaw;self.desired_yaws=np.array(env.config['robot_yaws_rad'],dtype=float)
        self.steering_diagnostics=[{},{}]
        self.arrival_capture=False
        self.preparing_placement=False;self.preparation_stable=0.;self.lower_descent_start=0.
        self.reward_valid=None;self.reward_diagnostics={}
        from contact_retreat import ContactRetreat
        self.retreat=ContactRetreat()
        self.native={name:torch.as_tensor(joblib.load(REPO/f'tasks/carry_box/latents/z_{name}.pkl'),
                      device=policy.device,dtype=torch.float32).reshape(1,-1).repeat(2,1)
                     for name in ('lift_low','hold','lower_low','place','transport_slow')}
        if self.goal_steering:
            self.native_turn={int(k):torch.as_tensor(v,device=policy.device,dtype=torch.float32).reshape(1,-1)
                for k,v in joblib.load(REPO/'tasks/carry_box/latents/z_transport_turn.pkl').items()}
        if dynamic:
            self.bank=torch.load(HERE/env.config.get('state_bank','state_bank.pt'),map_location=policy.device,weights_only=False)
            self.S=self.bank['state'];self.F=self.bank['features'];self.B=self.bank['B']
            self.B_unit=torch.nn.functional.normalize(self.B,dim=-1)
        if env.config.get('placement_guide'):
            self.placement_guide=torch.load(HERE/env.config['placement_guide'],map_location=policy.device,weights_only=False)

    def _change(self, stage, t):
        self.events.append({'from':self.stage,'to':stage,'time_s':t})
        self.stage=stage;self.stage_start=t;self.stable_time=0.
        if stage=='transport':self.height=self.env.config.get('carry_height_m',.65)-.03
        if stage=='lower':
            self.preparing_placement=self.env.config.get('placement_clearance',0)>0
            self.preparation_stable=0.;self.lower_descent_start=t
            self.lower_box_start=float(self.env.data.xpos[self.env.box_body,2])
            self.lower_root_start=np.array([self.env.data.xpos[r.base,2] for r in self.env.robots])
            self.lower_start_joints=np.array([self.env.data.qpos[r.qpos]-self.env.defaults for r in self.env.robots])
            box=self.env.data.xpos[self.env.box_body];R=self.env.data.geom_xmat[self.env.box_geom].reshape(3,3)
            self.lower_box_yaw=float(np.arctan2(R[1,0],R[0,0]))
            self.lower_grip_offsets=np.array([[(self.env.data.xpos[b]+self.env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]-box)@R
                for h,b in enumerate(r.hands)] for r in self.env.robots])
            self.lower_grip_rotations=np.array([[R.T@self.env.data.xmat[b].reshape(3,3) for b in r.hands] for r in self.env.robots])
        if stage=='release':
            self.release_arm_positions=np.array([self.env.data.qpos[r.qpos].copy() for r in self.env.robots])
            self.release_points=np.array([[self.env.data.xpos[b]+self.env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h] for h,b in enumerate(r.hands)] for r in self.env.robots])
            self.release_rotations=np.array([[self.env.data.xmat[b].reshape(3,3).copy() for b in r.hands] for r in self.env.robots])
            self.release_normals=self.release_points-self.env.data.xpos[self.env.box_body]
            self.release_normals[:,:,2]=0
            self.release_normals/=np.linalg.norm(self.release_normals,axis=-1,keepdims=True).clip(1e-6)

    def __call__(self, ct):
        t=ct['time_s'];box=np.array(ct['object']['position_m']);bv=np.array(ct['object']['linear_velocity_mps'])
        dt=1/self.env.config['coordination_hz'];self.peak_height=max(self.peak_height,box[2])
        raw_root_vel=np.array([r['linear_velocity_mps'] for r in ct['robots']])
        alpha=1-np.exp(-dt/.25)
        self.context_box_velocity+=alpha*(bv-self.context_box_velocity)
        self.context_root_velocity+=alpha*(raw_root_vel-self.context_root_velocity)
        if self.env.config.get('filtered_context'):bv=self.context_box_velocity.copy()
        contact=np.array(ct['contacts']['hand_contact'])
        roots=np.array([r['position_m'] for r in ct['robots']]);upright=[self.env.data.xmat[r.base].reshape(3,3)[2,2] for r in self.env.robots]
        both=contact.any(axis=1).all()
        for i,r in enumerate(self.env.robots):
            for h,b in enumerate(r.hands):
                if contact[i,h]:
                    if t-self.last_hand_contact[i,h]>.25:
                        self.grasp_since[i,h]=t
                        point=self.env.data.xpos[b]+self.env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
                        self.grasp_offsets[i,h]=np.array(ct['object']['geometry_rotation_world']).T@(point-box)
                        self.grasp_rotations[i,h]=np.array(ct['object']['geometry_rotation_world']).T@self.env.data.xmat[b].reshape(3,3)
                    self.last_hand_contact[i,h]=t
        self.grasp_ready_time=self.grasp_ready_time+dt if contact.all() else 0.
        geom_up=ct['object']['geometry_rotation_world'][2][2]
        floor_support=sum(pair['force_on_box_N'][2] for pair in ct['contacts']['pairs'] if pair['other_geom']=='floor')
        started=self.retreat.update(ct,self.stage=='lower' and self.env.config.get('contact_retreat',False),dt)
        if started.any():
            self.height=float(box[2])
            self.events.append({'from':self.stage,'to':self.stage,'time_s':t,'reason':'body_contact_retreat',
                                'robots':np.flatnonzero(started).tolist(),'contacts':self.retreat.names})
        if min(roots[:,2])<.28 or min(upright)<.45:
            self.failure_reason='robot_posture_limit'
            if self.stage!='failed':self._change('failed',t)
        if self.mass_adaptive and self.stage in ('lift','transport'):
            self.lost_grasp_time=self.lost_grasp_time+dt if not contact.any() else 0.
            if self.lost_grasp_time>.7 and box[2]<.19:
                self.failure_reason='lost_grasp_box_grounded';self._change('failed',t)
        if self.stage=='settle' and t>=1:self._change('acquire',t)
        grasp_ready=not self.mass_adaptive or self.grasp_ready_time>=.2
        if self.stage=='acquire' and both and box[2]>.22 and grasp_ready:
            self.pickup_joints=np.array([self.env.data.qpos[r.qpos]-self.env.defaults for r in self.env.robots])
            self._change('lift',t);self.height=float(box[2]);self.grasp_active=True
        if self.stage=='lift':
            carry_height=self.env.config.get('carry_height_m',.65)
            lift_speed=self.env.mass_grasp['lift_speed'] if self.mass_adaptive else .12
            if not self.mass_adaptive or contact.all():self.height=min(carry_height,self.height+lift_speed*dt)
            stable=both and abs(box[2]-carry_height)<.08 and np.linalg.norm(bv)<.20
            self.stable_time=self.stable_time+dt if stable else 0
            if self.stable_time>.6:
                self.formation_offsets=roots-box
                self.formation_offsets[:,2]=0
                if self.env.parallel:
                    self.formation_offsets[:,0]=np.clip(self.formation_offsets[:,0],[-.5,.28],[-.28,.5])
                    self.formation_offsets[:,1]=np.clip(self.formation_offsets[:,1],-.75,-.35)
                else:
                    self.formation_offsets[0,0]=np.clip(self.formation_offsets[0,0],-.60,-.40)
                    self.formation_offsets[1,0]=np.clip(self.formation_offsets[1,0],.40,.60)
                    self.formation_offsets[:,1]=np.clip(self.formation_offsets[:,1],-.15,.15)
                if self.goal_steering:
                    angle=float(np.arctan2(ct['object']['geometry_rotation_world'][1][0],ct['object']['geometry_rotation_world'][0][0]))
                    ca,sa=np.cos(angle),np.sin(angle)
                    self.formation_offsets=self.formation_offsets@np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
                self._change('transport',t)
        distance=np.linalg.norm(self.destination[:2]-box[:2])
        if self.stage=='transport':
            if self.goal_steering:
                if distance<.12:self.arrival_capture=True
                elif distance>.20:self.arrival_capture=False
            arrived=distance<.12 and np.linalg.norm(bv)<.12 and both and geom_up>.94
            self.stable_time=self.stable_time+dt if arrived else 0
            if self.stable_time>=.6:
                self._change('lower',t);self.height=float(box[2])
        if self.stage=='lower':
            if self.preparing_placement:
                yaw=self.lower_box_yaw;axis=np.array([np.cos(yaw),np.sin(yaw)])
                offsets=(roots[:,:2]-box[:2])@axis
                clearance=self.env.config['placement_clearance']
                ready=offsets[0]<-clearance+.03 and offsets[1]>clearance-.03 and both
                self.preparation_stable=self.preparation_stable+dt if ready else 0.
                if self.preparation_stable>=.3:
                    self.preparing_placement=False;self.lower_descent_start=t
                elif t-self.stage_start>15.:
                    self.failure_reason='placement_clearance_timeout';self._change('failed',t)
            # Descending reference is monotonic. Never move it UP in response
            # to an erroneous lift, which would reward and lock in that error.
            if not self.retreat.active.any() and not self.preparing_placement and both and geom_up>.9 and (not self.goal_steering or box[2]-self.height<.09):
                self.height=max(.15,self.height-.04*dt)
            stable=box[2]<.18 and abs(bv[2])<.07 and floor_support>.8*self.env.config['box_mass_kg']*9.81 and geom_up>.95
            self.stable_time=self.stable_time+dt if stable else 0
            if self.stable_time>=.2:self._change('release',t);self.grasp_active=False
        if self.stage=='release':
            stable=not contact.any() and box[2]<.18 and np.linalg.norm(bv)<.08
            self.stable_time=self.stable_time+dt if stable else 0
            if self.stable_time>1.:self._change('complete',t)
            if self.env.config.get('compliant_release'):
                self.env.grip_commands[:]=0
                z=self.standing_latent if not contact.any() and self.stable_time>=.3 else self.last_z
                self.last_z_change=(1-torch.nn.functional.cosine_similarity(z,self.last_z)).cpu().tolist()
                self.last_z=z
                self.last_reward={'stage':self.stage,'height_target':self.height,'z_cosine_change':self.last_z_change,'release_open_hands':True}
                return {'latents':z}
            # Stand clear after contact has ended instead of chasing a moved box.
            if not contact.any():
                self.env.grip_commands[:]=0
                self.last_z_change=[0.,0.]
                self.last_reward={'stage':self.stage,'height_target':self.height,'z_cosine_change':[0,0]}
                return {'latents':self.standing_latent if hasattr(self,'standing_latent') else self.last_z}
        if self.stage=='acquire':self.height=.22
        if self.stage=='settle':self.height=.15
        if self.env.revo2:
            for i,r in enumerate(self.env.robots):
                for h,b in enumerate(r.hands):
                    point=self.env.data.xpos[b]+self.env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
                    near=np.linalg.norm(point-box)<.38
                    if self.env.config.get('handle_height_m',0):
                        sy=(1 if h==0 else -1)*(1 if i==0 else -1)
                        local=np.array([(-1 if i==0 else 1)*self.env.config.get('grip_x_m',.17),sy*(self.env.config['box_size_m'][1]/2+self.env.config.get('grip_standoff_m',.04)),self.env.config['handle_height_m']])
                        if self.env.parallel:local=self.env.object_anchors[i]+np.array([(-1 if h==0 else 1)*.17,-.04,self.env.config['handle_height_m']])
                        grip=box+np.array(ct['object']['geometry_rotation_world'])@local
                        distance_to_grip=np.linalg.norm(point-grip)
                        if distance_to_grip<.10 or (contact[i,h] and distance_to_grip<.16):self.grip_latched[i,h]=True
                        self.grip_lost_time[i,h]=self.grip_lost_time[i,h]+dt if not contact[i,h] and distance_to_grip>.16 else 0
                        if self.grip_lost_time[i,h]>.4:self.grip_latched[i,h]=False
                        near=bool(self.grip_latched[i,h])
                    closure=self.env.mass_grasp['closure'] if self.mass_adaptive else .9
                    self.env.grip_commands[i,h]=closure if self.stage in ('acquire','lift','transport','lower') and (near or contact[i,h]) else 0
        target_vel=np.zeros(3)
        if self.stage=='transport':
            target_vel[:2]=np.clip(1.2*(self.destination[:2]-box[:2]),-self.speed,self.speed)
            if self.goal_steering:
                from goal_steering import goal_command
                target_vel[:2],self.travel_yaw,_=goal_command(box[:2],self.destination[:2],self.speed,self.travel_yaw,dt,bv[:2])
                self.desired_yaws=np.array([self.travel_yaw,self.travel_yaw+np.pi])
        if self.stage in ('lower','release'):
            target_vel[:2]=np.clip(1.2*(self.destination[:2]-box[:2])-.5*bv[:2],-.12,.12)
        if self.stage=='lift':target_vel[2]=np.clip(1.5*(self.height-box[2]),-.10,.12)
        if self.stage=='lower':target_vel[2]=np.clip(1.5*(self.height-box[2]),-.10,.05)
        if self.stage=='lower' and self.retreat.active.any():target_vel[2]=0.
        for k,r in enumerate(self.env.robots):
            R=self.env.data.xmat[r.base].reshape(3,3)
            error=self.desired_yaws[k]-np.arctan2(R[1,0],R[0,0])
            self.heading_errors[k]=np.arctan2(np.sin(error),np.cos(error))
        if self.env.config.get('axial_carry') and not self.goal_steering and self.stage=='transport':
            # Reduce shared progress when either body turns away; correct yaw
            # through reward inference, never by writing a root quaternion.
            heading_scale=np.clip((np.deg2rad(45)-max(abs(self.heading_errors)))/np.deg2rad(25),0,1)
            target_vel[:2]*=heading_scale
        # A single world-frame object command, before either robot's formation
        # or teammate corrections. Never use a robot-specific v as this target.
        self.shared_box_velocity=target_vel.copy()
        # Common formation: opposite sides of the same box. Teammate velocity and
        # force imbalance change the two robot targets independently.
        root_offsets=np.array([[-.62,0,0],[.62,0,0]])
        if self.env.parallel:root_offsets=np.array([[-.36,-.60,0],[.36,-.60,0]])
        if self.formation_offsets is not None:root_offsets=self.formation_offsets
        if self.goal_steering and self.stage=='lower' and self.env.config.get('placement_clearance',0)>0:
            d=self.env.config['placement_clearance'];root_offsets=np.array([[-d,0,0],[d,0,0]])
        if self.goal_steering and self.stage in ('transport','lower'):
            # Formation follows the real object, not its initial world axes.
            geometry=np.array(ct['object']['geometry_rotation_world'])
            object_yaw=float(np.arctan2(geometry[1,0],geometry[0,0]))
            c,s=np.cos(object_yaw),np.sin(object_yaw)
            root_offsets=root_offsets@np.array([[c,s,0],[-s,c,0],[0,0,1]])
        measured=np.array(ct['contacts']['hand_force_on_box_N'])
        if self.stage=='lower':root_offsets=root_offsets.copy()+np.c_[self.retreat.offset,np.zeros(2)]
        if self.env.config.get('filtered_context') and ct['contact_window']['duration_s']>0:
            measured=np.array(ct['contact_window']['hand_mean_force_on_box_N'])
        load=max(0,self.env.config['box_mass_kg']*9.81-floor_support)
        robot_load=measured[:,:,2].sum(axis=1)
        for i in range(2):
            correction=.15*(robot_load[1-i]-robot_load[i])
            self.vertical_force_targets[i,:]=np.clip(load/4+correction,0,self.env.config['box_mass_kg']*9.81/2)
        if self.env.config.get('wrench_feedback'):
            self.allocated_forces=self.allocate_wrench(ct,target_vel)
            self.vertical_force_targets=self.allocated_forces[:,:,2].copy()
        geom_rot=np.array(ct['object']['geometry_rotation_world'])
        half=np.array(self.env.config['box_size_m'])/2
        root_vel=self.context_root_velocity if self.env.config.get('filtered_context') else raw_root_vel
        targets=[];rewards=[];new_z=[]
        for i,r in enumerate(self.env.robots):
            rotation=self.env.data.xmat[r.base].reshape(3,3)
            formation_error=box[:2]+root_offsets[i,:2]-roots[i,:2]
            v=target_vel.copy();v[:2]+=np.clip(.8*formation_error+.25*(root_vel[1-i,:2]-root_vel[i,:2]),-.15,.15)
            v[:2]=np.clip(v[:2],-.25,.25)
            self.smoothed_targets[i]+=alpha*(v-self.smoothed_targets[i])
            if self.env.config.get('filtered_context'):v=self.smoothed_targets[i].copy()
            if self.env.config.get('axial_carry') and not self.goal_steering and self.stage in ('transport','lower'):
                desired_yaw=self.env.config['robot_yaws_rad'][i]
                forward=np.array([np.cos(desired_yaw),np.sin(desired_yaw),0.])
                side=np.array([-np.sin(desired_yaw),np.cos(desired_yaw),0.])
                longitudinal=np.clip(np.dot(target_vel,forward)+.3*np.dot(np.r_[formation_error,0.],forward)
                    +.25*np.dot(root_vel[1-i]-root_vel[i],forward),-self.speed,self.speed)
                if self.stage=='transport':
                    longitudinal=(max(0,longitudinal) if i==0 else min(0,longitudinal))*heading_scale
                lateral=np.clip(.3*np.dot(np.r_[formation_error,0.],side),-.025,.025)
                # Explicit body-frame roles, rather than turning world +X into
                # a sideways command when the body has already rotated.
                axial_velocity=np.array([longitudinal,lateral,0.])
                v=rotation@axial_velocity
            if self.stage=='lower' and self.retreat.active[i]:
                # Override inward formation/teammate corrections during escape.
                # Bound total displacement; a limit is reported, never teleported.
                v[:2]=(0. if self.retreat.limited[i] else self.retreat.speed)*self.retreat.direction[i]
                v[2]=0.
            self.velocity_targets[i]=v
            root_height=.48 if self.stage=='acquire' else .76
            if self.stage=='lower':root_height=np.clip(.43+.33*(self.height-.15)/.5,.43,.76)
            if self.goal_steering and self.stage=='lower':
                start_h=getattr(self,'lower_box_start',self.env.config.get('carry_height_m',.65))
                start_root=getattr(self,'lower_root_start',np.array([.76,.76]))[i]
                fraction=np.clip((self.height-.15)/max(start_h-.15,.05),0,1)
                root_height=.43+(start_root-.43)*fraction
                if self.preparing_placement:root_height=start_root
                elif self.env.config.get('lower_pose_feedback'):
                    root_height=np.clip(root_height+.5*(root_height-roots[i,2]),.36,.78)
            self.reward_robot_index=i
            if self.stage=='lower' and self.retreat.active.any():root_height=float(roots[i,2])
            desired_rel_world=np.r_[-root_offsets[i,:2],self.height-root_height]+self.env.object_anchors[i]
            rel_target=rotation.T@desired_rel_world
            hand_offsets=[]
            for h in range(2):
                sy=(1 if h==0 else -1)*(1 if i==0 else -1)
                normal=geom_rot[:,1]*sy
                if self.env.parallel:
                    sy=1 if h==0 else -1
                    normal=geom_rot@np.array([-sy,0.,0.])
                normal_load=-np.dot(measured[i,h],normal)
                # Measured force -> small hand target correction -> reward -> z.
                compliance=np.clip((self.force_target-normal_load)*.0005,-.012,.012)
                offset=np.array([(-1 if i==0 else 1)*self.env.config.get('grip_x_m',.17),sy*(half[1]+self.env.config.get('grip_standoff_m',.04)-.015-compliance),0.])
                if self.env.parallel:offset=np.array([-sy*(.17-compliance),-.04,0.])
                offset[2]=self.env.config.get('handle_height_m',0.)
                if self.stage=='release':
                    if self.env.parallel:offset[0]-=sy*.08
                    else:offset[1]+=sy*.08
                desired_offset=offset if self.stage in ('lift','transport','lower') else geom_rot@offset
                if self.goal_steering and self.stage in ('transport','lower'):
                    desired_offset=geom_rot@offset
                if self.env.config.get('hand_retarget'):
                    # Keep the policy's original virtual point and B schema.
                    # Translate the target by the measured Revo2 grip-center
                    # offset relative to that virtual point, in the wrist frame.
                    grip_delta=np.array([.02,-.025 if h==0 else .025,.005])
                    desired_offset=desired_offset-self.env.data.xmat[r.hands[h]].reshape(3,3)@grip_delta
                if self.stage in ('lift','transport','lower'):
                    desired_offset[2]+=np.clip(.0005*(self.vertical_force_targets[i,h]-measured[i,h,2]),-.008,.008)
                anchor_offset=self.env.object_anchors[i] if self.stage in ('lift','transport','lower') else geom_rot@self.env.object_anchors[i]
                hand_world=box+anchor_offset+desired_offset
                hand_world[2]=self.height+(desired_offset[2] if self.stage in ('lift','transport','lower') else self.env.config.get('handle_height_m',0.))
                self.hand_targets[i,h]=hand_world
                # Keep grasp locations on the object while lowering the OBJECT.
                # Asking the state library for hands below the object selects
                # loss-of-contact states instead of controlled descent.
                hand_offsets.append(rotation.T@(desired_offset if self.stage in ('lift','transport','lower') else hand_world-box-anchor_offset))
            if self.dynamic and self.stage not in ('settle','complete','failed'):
                reward_root_quat=self.env.data.xquat[r.base].copy()
                if self.env.config.get('axial_carry') and not self.goal_steering and self.stage in ('transport','lower'):
                    wanted_yaw=self.env.config['robot_yaws_rad'][i]
                    reward_root_quat=np.array([np.cos(wanted_yaw/2),0.,0.,np.sin(wanted_yaw/2)])
                root_quat=torch.as_tensor(np.roll(reward_root_quat,-1),device=self.policy.device,dtype=torch.float32)[None]
                object_quat=np.array(self.env.config['box_geom_quaternion_wxyz'])*[1,-1,-1,-1]
                object_quat=object_quat/np.linalg.norm(object_quat)
                if self.env.parallel:
                    canonical=np.zeros(4);mujoco.mju_mulQuat(canonical,np.array([2**-.5,0,0,2**-.5]),object_quat);object_quat=canonical
                elif self.env.config.get('canonical_box_symmetry',False) and (np.cos(self.env.config['robot_yaws_rad'][i])<0 if self.env.config.get('axial_carry') else rotation[0,0]<0):
                    canonical=np.zeros(4);mujoco.mju_mulQuat(canonical,np.array([0.,0.,0.,1.]),object_quat)
                    object_quat=canonical
                if self.goal_steering and self.stage in ('transport','lower'):
                    # Follow actual box yaw while rewarding level roll/pitch.
                    box_heading=self.travel_yaw if self.env.config.get('box_heading_goal') else object_yaw
                    qyaw=np.array([np.cos(box_heading/2),0.,0.,np.sin(box_heading/2)])
                    rotated=np.zeros(4);mujoco.mju_mulQuat(rotated,qyaw,object_quat);object_quat=rotated
                object_quat=torch.as_tensor(np.roll(object_quat,-1),device=self.policy.device,dtype=torch.float32)[None]
                rotation_target=quat_to_tan_norm(quat_mul(quat_conjugate(root_quat,w_last=True),object_quat,w_last=True),w_last=True)[0]
                score,parts=self.reward(rotation.T@v,rel_target,np.array(hand_offsets),root_height,
                                        rotation.T@target_vel,rotation_target,world_rotation=rotation)
                yaw=np.arctan2(rotation[1,0],rotation[0,0])
                yaw_error=np.arctan2(np.sin(self.desired_yaws[i]-yaw),np.cos(self.desired_yaws[i]-yaw))
                yaw_target=np.clip(1.5*yaw_error,-.5,.5)
                heading=torch.exp(-((4*self.S[:,63]-yaw_target)/.35).square())
                if self.stage in ('transport','lower'):
                    if self.goal_steering:
                        # Steering is a separate feedback branch. Do not require
                        # the movement bank to already have the final heading.
                        score=score*(.85+.15*heading)
                        parts['yaw_rate']=heading
                        self.reward_diagnostics.update(heading_error_deg=float(np.degrees(yaw_error)),
                            assigned_role='forward' if i==0 else 'backward')
                    elif self.env.config.get('axial_carry'):
                        from axial_motion import axial_candidates
                        axial_valid,angle=axial_candidates(self.S,self.F,rotation_target,1. if i==0 else -1.)
                        before=int(self.reward_valid.sum())
                        self.reward_valid=self.reward_valid&axial_valid
                        posture_heading=torch.exp(-(angle/.25).square())
                        lateral=torch.exp(-((self.F[:,1]-float((rotation.T@v)[1]))/.05).square())
                        score=score*(.1+.9*posture_heading)*(.2+.8*heading)*(.2+.8*lateral)
                        parts.update(body_heading=posture_heading,yaw_rate=heading,lateral_speed=lateral)
                        self.reward_diagnostics.update(axial_candidates_before=before,
                            feasible_candidates=int(self.reward_valid.sum()),heading_error_deg=float(np.degrees(yaw_error)),
                            assigned_role='forward' if i==0 else 'backward')
                    else:score=score*(.6+.4*heading)
                # Official reward-weighted inference, but with B already cached.
                # All library rows participate; no online backward-network pass.
                temperature=max(self.temperature,40.) if self.stage in ('transport','lower') else self.temperature
                continuity=self.B_unit@torch.nn.functional.normalize(self.last_z[i],dim=0)
                prior_weight=self.env.config.get('z_prior',0.) if self.stage in ('transport','lower') else 0.
                valid=self.reward_valid
                exhausted=valid is not None and not bool(valid.any())
                selection_score=score
                if self.env.config.get('axial_carry') and not self.goal_steering and valid is not None and not exhausted:
                    # Multiplicative heading terms change the absolute score
                    # scale. Preserve selective inference instead of silently
                    # flattening softmax towards uniform over sparse support.
                    scale=score[valid].max().clamp_min(1e-8)
                    selection_score=score/scale
                    self.reward_diagnostics['selection_score_scale']=float(scale)
                logits=temperature*selection_score+prior_weight*continuity
                if valid is not None:logits=logits.masked_fill(~valid,-torch.inf)
                # Never silently restore forbidden states when coverage is poor.
                # With zero support retain the previous latent and report failure
                # of reward inference; this is not claimed to be a safe controller.
                prob=torch.zeros_like(score) if exhausted else torch.softmax(logits,dim=0)
                weight=prob*score.clamp_min(0)
                if self.goal_steering and self.stage in ('transport','lower'):
                    from goal_steering import supported_weights
                    weight,distribution=supported_weights(score,valid,temperature=16.,min_ess=self.env.config.get('min_ess',16.))
                    exhausted=distribution['no_feasible_candidates']
                    self.reward_diagnostics.update(distribution)
                inferred=self.last_z[i:i+1] if exhausted else self.policy.model.project_z(weight[None]@self.B)
                if self.goal_steering and self.stage=='transport' and not exhausted:
                    from goal_steering import steering_weight
                    yaw_rate=float(np.array(ct['robots'][i]['angular_velocity_radps'])[2])
                    direction,blend,desired_rate=steering_weight(yaw_error,yaw_rate,self.env.config.get('steering_blend',.35))
                    if self.env.config.get('turn_source','official')=='reward':
                        # Separate turn reward, with true rad/s and no heading gate.
                        turn_score=(self.S[:,82:84].clamp(0,1).prod(-1)*(-self.S[:,60]).clamp(0,1)*self.F[:,5].clamp(0,1)
                            *torch.exp(-((4*self.S[:,63]-.5*direction)/.30).square())
                            *torch.exp(-((self.F[:,3]-self.height)/.18).square()))
                        local_target=torch.as_tensor(rotation.T@v,device=score.device,dtype=score.dtype)
                        turn_motion=torch.exp(-((self.F[:,:2]-local_target[:2])/.20).square().sum(-1))
                        turn_score*=.15+.85*turn_motion
                        turn_valid=(self.S[:,82:84]>.5).all(-1)&(self.F[:,3]>.4)&(self.F[:,5]>.9)&(-self.S[:,60]>.9)
                        turn_weight,turn_diag=supported_weights(turn_score,turn_valid,10.,16.)
                        turning=self.policy.model.project_z(turn_weight[None]@self.B)
                    else:turning=self.native_turn[direction]
                    inferred=self.policy.model.project_z((1-blend)*inferred+blend*turning)
                    self.steering_diagnostics[i]={'blend':blend,'direction':direction,'desired_yaw_rate_radps':desired_rate,
                        'actual_yaw_rate_radps':yaw_rate,'source':self.env.config.get('turn_source','official')}
                if self.anchor:
                    inferred=self.policy.model.project_z((1-self.anchor)*inferred+self.anchor*self._native()[i:i+1])
                if self.goal_steering and self.stage=='lower' and not self.preparing_placement and self.env.config.get('lower_anchor',0)>0:
                    lower_mix=self.env.config['lower_anchor']*min(1.,max(0.,(t-self.lower_descent_start)/1.))
                    inferred=self.policy.model.project_z((1-lower_mix)*inferred+lower_mix*self.native['lower_low'][i:i+1])
                if self.stage=='lower' and hasattr(self,'placement_guide'):
                    guide=self.placement_guide
                    guide_height=min(max(self.height-self.env.config.get('guide_height_lead',0.),float(guide['height'].min())),float(guide['height'].max()))
                    score_guide=torch.exp(-((guide['height']-guide_height)/.045).square())
                    weight_guide=torch.softmax(10*score_guide,dim=0)*score_guide
                    latent_guide=self.policy.model.project_z(weight_guide[None]@guide['B'][i])
                    guide_mix=self.env.config.get('guide_mix',.8)*min(1.,max(0.,t-self.lower_descent_start))
                    if self.retreat.active[i]:guide_mix=min(guide_mix,.15)
                    inferred=self.policy.model.project_z((1-guide_mix)*inferred+guide_mix*latent_guide)
                    self.reward_diagnostics['placement_guide_mix']=guide_mix
                    self.reward_diagnostics['placement_guide_height']=float((weight_guide*guide['height']).sum()/weight_guide.sum().clamp_min(1e-12))
                inferred=self.policy.model.project_z(.55*self.last_z[i:i+1]+.45*inferred)
                if self.stage in ('transport','lower'):
                    cosine=torch.nn.functional.cosine_similarity(inferred,self.last_z[i:i+1]).clamp(-1,1)
                    angle=torch.acos(cosine)
                    limit=self.env.config.get('transport_z_angle',np.pi) if self.stage=='transport' else .08
                    mix=(limit/angle.clamp_min(limit)).item()
                    inferred=self.policy.model.project_z((1-mix)*self.last_z[i:i+1]+mix*inferred)
                new_z.append(inferred)
                mixture=weight/weight.sum().clamp_min(1e-12)
                rewards.append({k:float((value*mixture).sum()) for k,value in parts.items()} | {'max':float(score.max()),
                                'effective_samples':float(1/mixture.square().sum()) if not exhausted else 0.,
                                'selected_box_z':float(mixture@self.F[:,3]),'selected_root_z':float(mixture@self.F[:,4]),
                                'selected_root_velocity':(mixture@self.F[:,:3]).cpu().tolist(),
                                'selected_object_velocity':(mixture@self.S[:,73:76]).cpu().tolist(),
                                'selected_object_velocity_world':((mixture@self.S[:,73:76])@torch.as_tensor(rotation.T,device=score.device,dtype=score.dtype)).cpu().tolist(),
                                'no_feasible_candidates':exhausted,
                                'forbidden_probability_mass':float(prob[~valid].sum()) if valid is not None else 0.,
                                **self.reward_diagnostics})
            else:new_z.append(self._native()[i:i+1])
            target_xy=self.destination[:2] if self.stage in ('transport','lower','release') else box[:2]
            if self.goal_steering and self.stage=='transport':
                delta=self.destination[:2]-box[:2]
                dist=float(np.linalg.norm(delta))
                lookahead=min(self.env.config.get('goal_lookahead',.6),max(dist,self.env.config.get('actor_goal_margin',.45)))
                # The actor's learned goal deadband can stop ~30 cm early. Its
                # rolling subgoal stays ahead until the REAL goal is captured.
                # Completion, reward speed and placement always use the real goal.
                target_xy=self.destination[:2] if self.arrival_capture else box[:2]+delta*lookahead/max(dist,1e-6)
            elif self.env.config.get('axial_carry') and self.stage=='transport':
                delta=self.destination[:2]-box[:2]
                target_xy=box[:2]+delta*min(1.,self.env.config.get('goal_lookahead',.6)*heading_scale/max(np.linalg.norm(delta),1e-6))
            targets.append(np.r_[target_xy,self.height].tolist())
        z=torch.cat(new_z)
        self.last_z_change=(1-torch.nn.functional.cosine_similarity(z,self.last_z)).cpu().tolist()
        self.last_z=z;self.last_reward={'stage':self.stage,'height_target':self.height,'velocity_targets':self.velocity_targets.tolist(),
             'shared_box_velocity_target_world_mps':self.shared_box_velocity.tolist(),
             'heading_errors_deg':np.degrees(self.heading_errors).tolist(),
             'desired_heading_world_deg':np.degrees(self.desired_yaws).tolist(),'steering':[dict(x) for x in self.steering_diagnostics],
             'placement_preparation':self.preparing_placement,
             'contact_retreat':self.retreat.diagnostics(),
             'actor_object_targets_m':targets,
             'vertical_force_targets_N':self.vertical_force_targets.tolist(),
             'allocated_hand_force_N':self.allocated_forces.tolist(),
             'force_target_N_per_hand':self.force_target,'rewards':rewards,'z_cosine_change':self.last_z_change}
        if self.mass_adaptive:self.last_reward['mass_grasp']={**self.env.mass_grasp,'ready_time_s':self.grasp_ready_time,
            'arm_feedback_force_N':self.mass_feedback_force.tolist()}
        return {'latents':z,'object_targets_m':targets}

    def mass_grasp_feedback(self,env,index,torque):
        # Compose with the existing placement/release controller. No forces are
        # applied to the box or floating base, only bounded arm motor torques.
        out=self.placement_xy_feedback(env,index,torque)
        blend=env.config.get('grasp_arm_blend',0.);limit=env.config.get('grasp_arm_limit',16.)
        if not self.mass_adaptive or blend==0:return out
        if self.stage not in ('acquire','lift','transport','lower'):return out
        r=env.robots[index];box=env.data.xpos[env.box_body]
        R=env.data.geom_xmat[env.box_geom].reshape(3,3)
        sensed=np.array(env.contacts()['hand_force_on_box_N'])
        for h,b in enumerate(r.hands):
            self.mass_feedback_force[index,h]=0
            if env.data.time-self.last_hand_contact[index,h]>1.:continue
            point=env.data.xpos[b]+env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
            offset=self.grasp_offsets[index,h].copy()
            if env.config.get('grasp_handle_target'):
                fraction=np.clip((env.data.time-self.grasp_since[index,h])/.7,0,1)
                wanted=np.array([(-1 if index==0 else 1)*env.config.get('grip_x_m',.19),
                    (1 if h==0 else -1)*(1 if index==0 else -1)*(env.config['box_size_m'][1]/2+.025),.035])
                offset=(1-fraction)*offset+fraction*wanted
            target=box+R@offset
            target[2]+=np.clip(self.height-box[2],-.025,.045)
            error=target-point
            if np.linalg.norm(error)>.20:continue
            jac=np.zeros((3,env.model.nv));jr=np.zeros_like(jac)
            mujoco.mj_jac(env.model,env.data,jac,jr,point,b)
            normal=R[:,1]*(1 if h==0 else -1)*(1 if index==0 else -1)
            compression=-float(sensed[index,h]@normal)
            force=600*error-20*(jac@env.data.qvel-self.context_box_velocity)
            force-=np.clip(self.force_target+.3*(self.force_target-compression),0,35)*normal
            force[2]+=self.vertical_force_targets[index,h]
            force=np.clip(force,[-45,-45,-35],[45,45,45])
            self.mass_feedback_force[index,h]=force
            arm=np.arange(15+7*h,22+7*h)
            actual=env.data.xmat[b].reshape(3,3);desired=R@self.grasp_rotations[index,h]
            orient=.5*sum(np.cross(actual[:,k],desired[:,k]) for k in range(3))
            moment=np.clip(3*orient-.2*(jr@env.data.qvel),-2,2)
            servo=env.data.qfrc_bias[r.dof[arm]]+jac[:,r.dof[arm]].T@force+jr[:,r.dof[arm]].T@moment-.15*env.data.qvel[r.dof[arm]]
            out[arm]+=np.clip(blend*(servo-out[arm]),-limit,limit)
        return out

    def _native(self):
        name={'acquire':'lift_low','lift':'hold','transport':'transport_slow','lower':'lower_low','release':'place'}.get(self.stage)
        return self.native[name] if name else self.policy.latents

    def reward(self, velocity,relative,hand_offsets,root_height,object_velocity,rotation_target,world_rotation=None):
        S,F=self.S,self.F;dev=S.device
        self.reward_valid=None;self.reward_diagnostics={}
        tensor=lambda x:torch.as_tensor(x,device=dev,dtype=torch.float32)
        exp=lambda error:torch.exp(-error.square().mean(dim=-1))
        parts={
            'speed':exp((F[:,:3]-tensor(velocity)) / tensor([.22,.22,.18])),
            'formation':exp((S[:,64:67]-tensor(relative)) / tensor([.20,.17,.20])),
            'hands':exp((S[:,90:96]-tensor(hand_offsets.reshape(-1)))/.16),
            'height':torch.exp(-((F[:,3]-self.height)/.16).square()),
            'posture':torch.exp(-((F[:,4]-root_height)/.20).square())*(-S[:,60]).clamp(0,1),
            'box_upright':F[:,5].clamp(0,1),
            'contact':S[:,82:84].mean(-1).clamp(0,1),
            'object_speed':exp((S[:,73:76]-tensor(object_velocity))/tensor([.15,.15,.10])),
            'orientation':exp((S[:,67:73]-rotation_target)/.35),
        }
        weights={'speed':.12,'formation':.12,'hands':.18,'height':.20,'posture':.12,'box_upright':.12,'contact':.14,'object_speed':0.,'orientation':0.}
        if self.stage=='acquire':weights.update(height=.10,contact=.25,hands=.23,speed=.04)
        if self.stage=='transport':
            parts['speed']=torch.exp(-((F[:,:2]-tensor(velocity[:2]))/tensor([.10,.10])).square().sum(-1))
            parts['object_speed']=torch.exp(-((S[:,73:75]-tensor(object_velocity[:2]))/.12).square().sum(-1))
            weights.update(speed=.45,object_speed=.20,height=.25,formation=.10,orientation=.12)
            if self.env.config.get('coordinated_motion'):
                # Full 3-D coordinate conversion keeps world horizontal speed
                # meaningful when the root tilts; local XY slicing does not.
                R=tensor(np.eye(3) if world_rotation is None else world_rotation)
                object_world=S[:,73:76]@R.T
                target_world=tensor(object_velocity)@R.T
                parts['object_speed']=torch.exp(-((object_world-target_world)/tensor([.12,.12,.10])).square().sum(-1))
                weights.update(speed=.25,object_speed=.65)
        if self.stage=='lift':weights.update(orientation=.15)
        if self.stage=='lower':
            parts['height']=torch.exp(-((F[:,3]-self.height)/.09).square())
            parts['posture']=torch.exp(-((F[:,4]-root_height)/.13).square())*(-S[:,60]).clamp(0,1)
            weights.update(speed=.08,formation=.08,hands=.12,height=.25,posture=.12,box_upright=.10,contact=.12,object_speed=.12,orientation=.15)
            if self.env.config.get('lower_pose_feedback') and not self.preparing_placement and hasattr(self,'pickup_joints') and hasattr(self,'lower_start_joints'):
                fraction=np.clip((self.lower_box_start-self.height)/max(self.lower_box_start-.22,.05),0,1)
                index=self.reward_robot_index
                target_joints=(1-fraction)*self.lower_start_joints[index]+fraction*self.pickup_joints[index]
                parts['lower_leg_pose']=exp((S[:,:12]-tensor(target_joints[:12]))/.35)
                weights.update(lower_leg_pose=.8,posture=.3)
            if self.preparing_placement:weights.update(speed=.7,formation=.7,height=.4,object_speed=.5,posture=.2)
            if self.env.config.get('smooth_lower'):
                parts['height']=torch.exp(-((F[:,3]-self.height)/.16).square())
        if self.stage=='release':parts['contact']=1-parts['contact']
        if self.env.config.get('goal_reward') and self.stage in ('lower','release'):
            parts['object_rotation_speed']=torch.exp(-((S[:,76:79]/.30).square().sum(-1)))
            weights['object_rotation_speed']=.15
            if self.stage in ('lower','release'):
                # Ground placement is a terminal near-goal state. Do not
                # reward maintaining the CURRENT nonzero goal error.
                parts['near_goal']=torch.exp(-((S[:,79:82]/.35).square().sum(-1)))
                weights['near_goal']=.20
            parts['box_upright']=torch.exp(-((1-F[:,5].clamp(-1,1))/.05).square())
            weights['box_upright']=.20
            if self.stage=='lower':
                parts['speed']=torch.exp(-((F[:,:2]-tensor(velocity[:2]))/.12).square().sum(-1))
                parts['object_speed']=exp((S[:,73:76]-tensor(object_velocity))/tensor([.10,.10,.10]))
                weights.update(speed=.25,object_speed=.30)
        if self.stage=='lower' and self.retreat.active[self.reward_robot_index]:
            parts['retreat_speed']=torch.exp(-((F[:,:2]-tensor(velocity[:2]))/.07).square().sum(-1))
            weights.update(retreat_speed=1.2,formation=.3,object_speed=.4,hands=.25,contact=.25)
        score=sum(weights[k]*parts[k] for k in weights)/sum(weights.values())
        score*=(-S[:,60]).clamp(0,1)
        if self.stage in ('transport','lower'):
            # Exclude non-carry states instead of letting fast dropped-box
            # states trade off against grasp quality in an additive reward.
            valid=(S[:,82:84]>.5).all(-1)&(F[:,5]>.90)&(-S[:,60]>.90)
            if self.stage=='transport':valid=valid&(F[:,3]>.40)&(F[:,4]>.62)&(S[:,75].abs()<.18)
            else:
                valid=valid&(S[:,75]<.04)&(S[:,75]>-.25)
                if not self.env.config.get('smooth_lower'):valid=valid&((F[:,3]-self.height).abs()<.13)
            if self.env.config.get('coordinated_motion'):
                from motion_constraints import directional_candidates
                R=tensor(np.eye(3) if world_rotation is None else world_rotation)
                object_world=S[:,73:76]@R.T
                target_world=tensor(object_velocity)@R.T
                direction_valid,diagnostics=directional_candidates(object_world,target_world,self.stage)
                if self.stage=='transport':
                    # Higher speed priority must not buy an incompatible box
                    # height or tilted carry pose. Same 20-degree tilt bound as
                    # the physical assessment; height band is +/- 12 cm.
                    valid=valid&((F[:,3]-self.height).abs()<=.12)&(F[:,5]>=np.cos(np.deg2rad(20)))&(-S[:,60]>=np.cos(np.deg2rad(20)))
                base_count=int(valid.sum())
                rejected=int((valid&~direction_valid).sum())
                valid=valid&direction_valid
                if self.goal_steering:
                    # Keep physical carrying constraints, soften exact height,
                    # orientation and lateral-speed targets instead of starving B.
                    valid=(S[:,82:84]>.5).all(-1)&(F[:,5]>.90)&(-S[:,60]>.90)
                    if self.stage=='transport':valid &= (F[:,3]>.40)&(F[:,4]>.58)&(S[:,75].abs()<.25)
                    else:
                        valid &= (S[:,75]<.04)&(S[:,75]>-.25)
                        valid &= (F[:,3]-self.height).abs()<.13
                    valid &= direction_valid
                    base_count=int(valid.sum());rejected=0
                self.reward_valid=valid
                self.reward_diagnostics={**diagnostics,'grasp_candidates':base_count,'feasible_candidates':int(valid.sum()),
                    'direction_rejected_grasp_candidates':rejected}
            elif valid.sum()>=16:score=score*valid
        return score,parts

    def release_compliance(self,env,index,torque):
        if self.stage!='release' or env.data.time-self.stage_start>1.:return torque
        r=env.robots[index];out=torque.copy();fraction=np.clip((env.data.time-self.stage_start-.1)/.5,0,1)
        fraction=fraction*fraction*(3-2*fraction)
        for h,b in enumerate(r.hands):
            arm=np.arange(15+7*h,22+7*h)
            point=env.data.xpos[b]+env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
            J=np.zeros((3,env.model.nv));Jr=np.zeros_like(J)
            mujoco.mj_jac(env.model,env.data,J,Jr,point,b)
            target=self.release_points[index,h]+.10*fraction*self.release_normals[index,h]
            force=np.clip(160*(target-point)-10*(J@env.data.qvel),-25,25)
            R=env.data.xmat[b].reshape(3,3);desired=self.release_rotations[index,h]
            error=.5*sum(np.cross(R[:,k],desired[:,k]) for k in range(3))
            moment=2*error-.15*(Jr@env.data.qvel)
            out[arm]=env.data.qfrc_bias[r.dof[arm]]+J[:,r.dof[arm]].T@force+Jr[:,r.dof[arm]].T@moment-.1*env.data.qvel[r.dof[arm]]
        return out

    def placement_xy_feedback(self,env,index,torque):
        """Bounded arm-motor correction for shared-object horizontal drift.

        Adds no vertical command and never writes object/root forces or states.
        The frozen policy remains responsible for balance and descent.
        """
        if self.stage=='release':return self.release_compliance(env,index,torque)
        if self.stage!='lower':return torque
        box=env.data.xpos[env.box_body]
        error=self.destination[:2]-box[:2]
        desired=np.r_[np.clip(40*error-6*self.context_box_velocity[:2],-8.,8.),0.]
        out=torque.copy();r=env.robots[index]
        for h,b in enumerate(r.hands):
            if not env.contacts()['hand_contact'][index][h]:continue
            point=env.data.xpos[b]+env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
            J=np.zeros((3,env.model.nv));Jr=np.zeros_like(J)
            mujoco.mj_jac(env.model,env.data,J,Jr,point,b)
            arm=np.arange(15+7*h,22+7*h)
            out[arm]+=np.clip(J[:,r.dof[arm]].T@desired,-3.,3.)
        return out

    def placement_servo(self,env,index,torque):
        """Actual arm motors lower held grasp points; legs remain under I-BFM.

        Captured grasp offsets avoid a hand-model-specific reach assumption.
        This optional impedance layer never writes root/object poses or forces.
        """
        if self.stage=='release':return self.release_compliance(env,index,torque)
        if self.stage!='lower' or self.preparing_placement:return torque
        r=env.robots[index];out=torque.copy();box=env.data.xpos[env.box_body]
        R=env.data.geom_xmat[env.box_geom].reshape(3,3)
        yaw=self.lower_box_yaw;c,s=np.cos(yaw),np.sin(yaw)
        target_R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        desired_box=np.r_[box[:2]+np.clip(self.destination[:2]-box[:2],-.05,.05),self.height]
        ramp=min(1.,max(0.,(env.data.time-self.lower_descent_start)/1.))
        for h,b in enumerate(r.hands):
            arm=np.arange(15+7*h,22+7*h)
            point=env.data.xpos[b]+env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
            J=np.zeros((3,env.model.nv));Jr=np.zeros_like(J)
            mujoco.mj_jac(env.model,env.data,J,Jr,point,b)
            desired=desired_box+target_R@self.lower_grip_offsets[index,h]
            normal=target_R[:,1]*np.sign(self.lower_grip_offsets[index,h,1])
            force=np.array([80.,80.,160.])*(desired-point)-np.array([8.,8.,12.])*(J@env.data.qvel)
            force-=4.*normal
            force[2]+=self.vertical_force_targets[index,h]
            force=np.clip(force,[-20,-20,-30],[20,20,30])
            current=env.data.xmat[b].reshape(3,3);wanted=target_R@self.lower_grip_rotations[index,h]
            angle=.5*sum(np.cross(current[:,k],wanted[:,k]) for k in range(3))
            moment=np.clip(3*angle-.2*(Jr@env.data.qvel),-1.5,1.5)
            arm_torque=env.data.qfrc_bias[r.dof[arm]]+J[:,r.dof[arm]].T@force+Jr[:,r.dof[arm]].T@moment
            out[arm]+=ramp*np.clip(arm_torque-torque[arm],-5.,5.)
        return out

    def impedance(self, env, index, torque):
        if not self.grasp_active or self.stage not in ('lift','transport','lower'):
            return torque
        r=env.robots[index];result=torque.copy()
        rotation=env.data.geom_xmat[env.box_geom].reshape(3,3)
        force=np.array(env.contacts()['hand_force_on_box_N'])
        for h,body in enumerate(r.hands):
            point=env.data.xpos[body]+env.data.xmat[body].reshape(3,3)@HAND_OFFSETS[h]
            jac=np.zeros((3,env.model.nv));rotjac=np.zeros_like(jac)
            mujoco.mj_jac(env.model,env.data,jac,rotjac,point,body)
            vel=jac@env.data.qvel
            normal=rotation[:,1]*(1 if h==0 else -1)*(1 if index==0 else -1)
            normal_load=-force[index,h]@normal
            feed=np.clip(self.force_target+.3*(self.force_target-normal_load),0,18)
            wrench=220*(self.hand_targets[index,h]-point)-12*(vel-self.velocity_targets[index])-feed*normal
            wrench[2]+=self.vertical_force_targets[index,h]
            wrench=np.clip(wrench,-35,35)
            arm=np.arange(15+7*h,22+7*h)
            result[arm]=.25*torque[arm]+jac[:,r.dof[arm]].T@wrench
        return result

    def residual_force_feedback(self,env,index,torque):
        """Bounded motor-only correction around the frozen policy's PD torque.

        No root/object forces, inverse pose writing or attachment constraints.
        Correct vertical grasp height and differential load; leave walking to pi.
        """
        if self.stage not in ('lift','transport','lower'):return torque
        r=env.robots[index];result=torque.copy()
        for h,body in enumerate(r.hands):
            point=env.data.xpos[body]+env.data.xmat[body].reshape(3,3)@HAND_OFFSETS[h]
            jac=np.zeros((3,env.model.nv));rotjac=np.zeros_like(jac)
            mujoco.mj_jac(env.model,env.data,jac,rotjac,point,body)
            velocity=jac@env.data.qvel
            error=self.hand_targets[index,h]-point
            wrench=np.array([15.,15.,65.])*error-np.array([2.,2.,5.])*(velocity-self.velocity_targets[index])
            wrench=np.clip(wrench,[-6,-6,-10],[6,6,10])
            arm=np.arange(15+7*h,22+7*h)
            residual=jac[:,r.dof[arm]].T@wrench
            result[arm]+=np.clip(residual,-2.,2.)
        return result

    def grip_load_guard(self,env,index,torque):
        """Relax excessive compression through arm motors; no object forces."""
        if self.stage not in ('lower','release'):return torque
        forces=np.array(env.contacts()['hand_force_on_box_N']);r=env.robots[index];result=torque.copy()
        for h,body in enumerate(r.hands):
            point=env.data.xpos[body]+env.data.xmat[body].reshape(3,3)@HAND_OFFSETS[h]
            normal=point-env.data.xpos[env.box_body];normal[2]=0
            normal/=max(np.linalg.norm(normal),1e-6)
            compression=-forces[index,h]@normal
            if compression<=20:continue
            jac=np.zeros((3,env.model.nv));rotjac=np.zeros_like(jac)
            mujoco.mj_jac(env.model,env.data,jac,rotjac,point,body)
            wrench=min(.7*(compression-20),40)*normal
            arm=np.arange(15+7*h,22+7*h)
            result[arm]+=np.clip(jac[:,r.dof[arm]].T@wrench,-3.,3.)
        return result

    def allocate_wrench(self,ct,velocity):
        """One desired object wrench, shared among four real contact locations."""
        if self.stage not in ('transport','lower'):return np.zeros((2,2,3))
        env=self.env;box=np.array(ct['object']['position_m']);v=np.array(ct['object']['linear_velocity_mps'])
        omega=np.array(ct['object']['angular_velocity_radps']);R=np.array(ct['object']['geometry_rotation_world'])
        floor=sum(p['force_on_box_N'][2] for p in ct['contacts']['pairs'] if p['other_geom']=='floor')
        force=env.config['box_mass_kg']*.7*(velocity-v)
        force[2]+=max(0,env.config['box_mass_kg']*9.81-floor)+5*(self.height-box[2])
        force[2]=np.clip(force[2],0,2*env.config['box_mass_kg']*9.81)
        moment=1.5*np.cross(R[:,2],np.array([0.,0.,1.]))-.15*omega;moment[2]=0
        G=np.zeros((6,12));squeeze=np.zeros(12)
        for i,r in enumerate(env.robots):
            for h,b in enumerate(r.hands):
                k=2*i+h;point=env.data.xpos[b]+env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
                contacts=[p['position_m'] for p in ct['contacts']['pairs'] if p['robot_index']==i and p['hand_index']==h]
                if contacts:point=np.mean(contacts,axis=0)
                arm=point-box;x,y,z=arm
                G[:3,3*k:3*k+3]=np.eye(3);G[3:,3*k:3*k+3]=[[0,-z,y],[z,0,-x],[-y,x,0]]
                normal=arm.copy();normal[2]=0;normal/=max(np.linalg.norm(normal),1e-6)
                squeeze[3*k:3*k+3]=-10*normal
        inverse=np.linalg.pinv(G,rcond=1e-3)
        value=inverse@np.r_[force,moment]+(np.eye(12)-inverse@G)@squeeze
        return np.clip(value.reshape(2,2,3),-25,25)

    def wrench_feedback(self,env,index,torque):
        """Low gain PI force correction through the actual arm motors only."""
        if self.stage not in ('transport','lower','release'):
            self.force_integral[index]=0;return torque
        if self.force_filter_step!=env.physics_steps:
            sensed=np.array(env.contacts()['hand_force_on_box_N'])
            alpha=1-np.exp(-env.model.opt.timestep/.03)
            self.filtered_forces+=alpha*(sensed-self.filtered_forces)
            self.force_filter_step=env.physics_steps
        r=env.robots[index];result=torque.copy();contact=env.contacts()['hand_contact']
        ramp=min(1.,max(0,(env.data.time-self.stage_start)/.5))
        for h,b in enumerate(r.hands):
            arm=np.arange(15+7*h,22+7*h)
            if not contact[index][h]:self.force_integral[index,arm]*=.97;continue
            point=env.data.xpos[b]+env.data.xmat[b].reshape(3,3)@HAND_OFFSETS[h]
            jac=np.zeros((3,env.model.nv));jr=np.zeros_like(jac)
            mujoco.mj_jac(env.model,env.data,jac,jr,point,b)
            error=np.clip(self.allocated_forces[index,h]-self.filtered_forces[index,h],-25,25)
            correction=jac[:,r.dof[arm]].T@error
            self.force_integral[index,arm]=np.clip(self.force_integral[index,arm]+2*env.model.opt.timestep*correction,-2.5,2.5)
            result[arm]+=ramp*np.clip(.15*correction+self.force_integral[index,arm],-3,3)
        return result
