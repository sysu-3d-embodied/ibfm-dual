"""Forward-kinematics inspection only; not a physics rollout or success test."""
import json,numpy as np
from environment import DualCarryEnv
e=DualCarryEnv()
for h in e.fingers:e.data.qpos[h['qpos']]=.9*h['closed']
e.forward();report=[]
for r in e.robots[:1]:
    for side,body in zip(('left','right'),r.hands):
        points={}
        for finger in ('thumb','index','middle','ring','pinky'):
            b=e.model.body(r.prefix+side+'_'+finger+'_touch_link').id
            points[finger]=(e.data.xmat[body].reshape(3,3).T@(e.data.xpos[b]-e.data.xpos[body])).tolist()
        report.append({'hand':side,'fingertip_touch_body_origins_in_wrist':points,
            'thumb_index_midpoint':((np.array(points['thumb'])+points['index'])/2).tolist()})
print(json.dumps(report),flush=True)
