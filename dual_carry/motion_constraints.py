"""World-frame, phase-dependent support constraints for cached-B inference.

These constraints concern library samples, not guaranteed future dynamics.
Keep stationary samples for gait support; reject only clear reverse motion.
"""
import torch


def directional_candidates(object_velocity_world, target_world, stage):
    valid=torch.ones(len(object_velocity_world),device=object_velocity_world.device,dtype=torch.bool)
    speed=torch.linalg.vector_norm(target_world[:2])
    active=stage in ('transport','lower') and float(speed)>.025
    tolerance=.015 if stage=='transport' else .025
    if active:
        direction=target_world[:2]/speed
        forward=object_velocity_world[:,:2]@direction
        valid &= forward>=-tolerance
    # Lowering cannot select fast upward object states. Lift/settle/release do
    # not inherit the transport horizontal rule, and near-zero targets have no
    # direction (avoid normalizing a zero vector).
    if stage=='lower':valid &= object_velocity_world[:,2]<=.04
    return valid,{'direction_filter_active':active,'reverse_tolerance_mps':tolerance,
                  'shared_target_world_mps':target_world.detach().cpu().tolist()}
