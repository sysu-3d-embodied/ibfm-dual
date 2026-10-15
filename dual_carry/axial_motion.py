"""Reward-space constraints for opposed forward/backward carrying.

Sample orientation is object-relative, not a global root yaw observation.
Infer the root heading compatible with the desired object orientation.
This constrains inference support, not the actual MuJoCo root pose.
"""
import torch


def heading_error_from_relative_rotation(state, desired_relative_rotation):
    x,z=state[:,67:70],state[:,70:73]
    relative=torch.stack([x,torch.cross(z,x,dim=-1),z],dim=-1)
    x,z=desired_relative_rotation[:3],desired_relative_rotation[3:]
    desired=torch.stack([x,torch.cross(z,x,dim=-1),z],dim=-1)
    difference=desired@relative.transpose(-1,-2)
    return torch.atan2(difference[:,1,0],difference[:,0,0])


def axial_candidates(state,features,desired_relative_rotation,sign,heading_limit_deg=25.):
    error=heading_error_from_relative_rotation(state,desired_relative_rotation)
    valid=(error.abs()<=heading_limit_deg*torch.pi/180)&(features[:,1].abs()<=.08)
    # Allow stationary gait support, but not clear motion against assigned role.
    valid &= sign*features[:,0]>=-.015
    return valid,error
