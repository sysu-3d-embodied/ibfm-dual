"""Measured non-hand contact -> bounded, hysteretic outward motion command."""
import numpy as np


class ContactRetreat:
    def __init__(self, speed=.08, max_distance=.16):
        self.speed=speed;self.max_distance=max_distance
        self.until=np.full(2,-1.);self.active=np.zeros(2,dtype=bool)
        self.offset=np.zeros((2,2));self.direction=np.zeros((2,2))
        self.origin=np.zeros((2,2));self.initialized=np.zeros(2,dtype=bool)
        self.force=np.zeros(2);self.names=[[],[]];self.limited=np.zeros(2,dtype=bool)

    def update(self, ct, enabled, dt):
        t=ct['time_s'];roots=np.array([r['position_m'][:2] for r in ct['robots']])
        box=np.array(ct['object']['position_m'][:2]);self.force[:]=0.;self.names=[[],[]]
        for p in ct['contacts']['pairs']:
            i=p['robot_index'];name=p['other_geom'].lower()
            if i not in (0,1) or p['hand_index'] is not None:continue
            if not any(s in name for s in ('hip','knee','ankle','pelvis','thigh','shin','foot','torso','waist')):continue
            f=float(np.linalg.norm(p['force_on_box_N']))
            if f>.5:
                self.force[i]+=f;self.names[i].append(p['other_geom'])
        old=self.active.copy()
        for i in range(2):
            if enabled and self.force[i]>.5:self.until[i]=t+.4
            self.active[i]=enabled and t<self.until[i]
            if self.active[i] and not self.initialized[i]:
                d=roots[i]-box
                self.direction[i]=d/max(float(np.linalg.norm(d)),1e-6)
                self.origin[i]=roots[i];self.initialized[i]=True
            self.limited[i]=self.initialized[i] and float((roots[i]-self.origin[i])@self.direction[i])>=self.max_distance
            if self.active[i] and not self.limited[i]:
                self.offset[i]+=self.speed*dt*self.direction[i]
                length=float(np.linalg.norm(self.offset[i]))
                if length>self.max_distance:self.offset[i]*=self.max_distance/length
        return self.active & ~old

    def diagnostics(self):
        return {'active':self.active.tolist(),'force_N':self.force.tolist(),
                'geometries':self.names,'offset_world_m':self.offset.tolist(),
                'distance_limit_reached':self.limited.tolist()}
