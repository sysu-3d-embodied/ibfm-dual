"""Load-aware force targets; motor limits remain those of the robot model."""
import numpy as np


def targets(mass, friction=1., gravity=9.81):
    if not np.isfinite(mass) or mass<=0:raise ValueError('Positive finite box mass required')
    load=mass*gravity/4
    squeeze=float(np.clip(1.7*load/max(friction,.3)+2.,8.,35.))
    return {'per_hand_weight_N':load,'squeeze_N':squeeze,
            'finger_hold_N':float(np.clip(2*squeeze+12,30,65)),
            'finger_relax_N':float(np.clip(3*squeeze+15,50,100)),
            'finger_kp':float(np.clip(2*np.sqrt(mass/1.4),2,4.5)),
            'closure':float(np.clip(.9+.025*(mass-1.4),.9,1.)),
            'lift_speed':float(np.clip(.12*np.sqrt(1.4/mass),.045,.12))}
