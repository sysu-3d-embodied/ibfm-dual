"""Goal-relative steering and supported reward mixtures; no trajectory tracking."""
import numpy as np
import torch


def wrap(angle):
    return np.arctan2(np.sin(angle),np.cos(angle))


def goal_command(box_xy,goal_xy,max_speed,previous_yaw,dt,velocity_xy=None):
    error=np.asarray(goal_xy)-np.asarray(box_xy)
    distance=float(np.linalg.norm(error))
    # Near arrival preserve the last reference: atan2 becomes ill-conditioned.
    raw=float(np.arctan2(error[1],error[0])) if distance>.12 else previous_yaw
    yaw=previous_yaw+float(np.clip(wrap(raw-previous_yaw),-.45*dt,.45*dt))
    speed=min(max_speed,.65*distance,np.sqrt(2*.20*distance))
    velocity=speed*error/max(distance,1e-8)
    # No heading-based speed multiplier and no original-path cross-track term.
    return velocity,wrap(yaw),distance


def supported_weights(score,valid,temperature=16.,min_ess=16.):
    """Flatten a peaky supported mixture; never add physically invalid rows."""
    positive=valid&torch.isfinite(score)&(score>0)
    if not bool(positive.any()):return torch.zeros_like(score),{'no_feasible_candidates':True,'effective_samples':0.,'temperature_used':0.}
    score=score.clamp_min(0);beta=float(temperature)
    count=int(positive.sum());target=min(float(min_ess),count)
    for _ in range(9):
        probability=torch.softmax((beta*score).masked_fill(~positive,-torch.inf),dim=0)
        weight=probability*score
        mixture=weight/weight.sum().clamp_min(1e-20)
        ess=float(1/mixture.square().sum())
        if ess>=target or beta==0:break
        beta=beta*.5 if beta>.25 else 0.
    return weight,{'no_feasible_candidates':False,'effective_samples':ess,
                   'temperature_used':beta,'feasible_candidates':count,'ess_target_met':ess>=target}


def steering_weight(error,yaw_rate,max_weight=.35):
    desired=float(np.clip(1.2*error,-.5,.5))
    residual=desired-.25*yaw_rate
    alpha=min(max_weight,abs(residual)/.5*max_weight)
    if abs(error)<np.deg2rad(2) and abs(yaw_rate)<.05:alpha=0.
    return (1 if residual>=0 else -1),alpha,desired
