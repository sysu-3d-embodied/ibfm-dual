"""Conservative validation of recorded shared-box dynamics, independent of FSM."""
import json,argparse
from pathlib import Path
import numpy as np

def assess(summary,contexts,rewards):
    if not contexts:return {'passed':False,'reason':'No recorded contexts'}
    stages=np.array([r.get('stage','unknown') for r in rewards])
    box=np.array([c['object']['position_m'] for c in contexts])
    up=np.array([c['object']['geometry_rotation_world'][2][2] for c in contexts])
    contacts=np.array([c['contacts']['hand_contact'] for c in contexts])
    transport=stages=='transport';lower=stages=='lower'
    goal=np.array(summary['destination_m']);error=float(np.linalg.norm(np.array(summary['box_final_m'])[:2]-goal[:2]))
    tilt=np.degrees(np.arccos(np.clip(up,-1,1)))
    support=np.array([sum(x['force_on_box_N'][2] for x in c['contacts']['pairs'] if x['other_geom']=='floor') for c in contexts])
    bilateral=contacts.any(-1).all(-1)
    meanforce=np.array([c['contact_window']['hand_mean_force_on_box_N'] for c in contexts])
    times=np.array([c['time_s'] for c in contexts]);dt=float(np.median(np.diff(times))) if len(times)>1 else 0.
    velocity=np.array([c['object']['linear_velocity_mps'] for c in contexts])
    distance=np.linalg.norm(box[:,:2]-goal[:2],axis=-1)
    # Task progress is separate from FSM completion and bilateral contact.
    stalled=transport&(distance>.12)&(np.linalg.norm(velocity[:,:2],axis=-1)<.02)
    current=longest=0.
    for value in stalled:
        current=current+dt if value else 0.;longest=max(longest,current)
    nonhand=np.array([sum(max(0.,pair['force_on_box_N'][2]) for pair in c['contacts']['pairs']
        if pair['other_geom']!='floor' and pair.get('hand_index') is None) for c in contexts])
    target=np.array([r.get('shared_box_velocity_target_world_mps',[np.nan]*3) for r in rewards])
    target_available=transport&np.isfinite(target).all(-1)
    metrics={'goal_error_m':error,'transport_contexts':int(transport.sum()),
        'transport_max_tilt_deg':float(tilt[transport].max()) if transport.any() else None,
        'transport_both_robot_contact_fraction':float(bilateral[transport].mean()) if transport.any() else 0.,
        'transport_unplanned_floor_contacts':int(((support>1.)&transport).sum()),
        'peak_hand_mean_force_N':float(np.linalg.norm(meanforce,axis=-1).max()),
        'lower_contexts':int(lower.sum()),'validation_sample_hz':1/dt if dt else None,
        'transport_stalled_fraction':float(stalled[transport].mean()) if transport.any() else None,
        'transport_longest_stall_s':longest,
        'transport_mean_horizontal_speed_mps':float(np.linalg.norm(velocity[transport,:2],axis=-1).mean()) if transport.any() else None,
        'transport_velocity_rmse_mps':float(np.sqrt(np.mean(np.sum((velocity[target_available,:2]-target[target_available,:2])**2,axis=-1)))) if target_available.any() else None,
        'transport_peak_nonhand_upward_force_N':float(nonhand[transport].max()) if transport.any() else None,
        'no_feasible_candidate_updates':sum(bool(x.get('no_feasible_candidates')) for r in rewards for x in r.get('rewards',[]))}
    checks={'phase_complete':summary['stage']=='complete','goal_error_le_12cm':error<=.12,
        'real_transport_observed':bool(transport.sum()>=10),'lowering_observed':bool(lower.sum()>=5),
        'no_transport_floor_contact':metrics['transport_unplanned_floor_contacts']==0,
        'both_robots_grasp_at_least_90pct':metrics['transport_both_robot_contact_fraction']>=.90,
        'transport_tilt_le_20deg':metrics['transport_max_tilt_deg'] is not None and metrics['transport_max_tilt_deg']<=20,
        'hand_mean_load_le_60N':metrics['peak_hand_mean_force_N']<=60,
        'no_transport_stall_over_3s':longest<=3.,
        'reward_inference_has_support':metrics['no_feasible_candidate_updates']==0}
    axial=any(x.get('assigned_role') for r in rewards for x in r.get('rewards',[]))
    if axial:
        errors=np.array([r.get('heading_errors_deg',[180,180]) for r in rewards])
        aligned=(np.abs(errors)<=25).all(-1)
        metrics['transport_heading_within_25deg_fraction']=float(aligned[transport].mean()) if transport.any() else 0.
        checks['opposed_heading_at_least_90pct']=metrics['transport_heading_within_25deg_fraction']>=.90
        local=[]
        for ct in contexts:
            row=[]
            for robot in ct['robots']:
                w,x,y,z=robot['quaternion_wxyz'];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                vx,vy=robot['linear_velocity_mps'][:2]
                row.append([vx*np.cos(yaw)+vy*np.sin(yaw),-vx*np.sin(yaw)+vy*np.cos(yaw)])
            local.append(row)
        local=np.array(local);moving=transport[:,None]&(np.linalg.norm(local,axis=-1)>.03)
        signed=local[:,:,0]*np.array([1,-1])
        wrong=(signed<-.015)&moving
        metrics['moving_wrong_role_fraction']=float(wrong.sum()/moving.sum()) if moving.any() else None
        metrics['transport_mean_abs_body_lateral_speed_mps']=float(np.abs(local[transport,:,1]).mean()) if transport.any() else None
        checks['forward_backward_role_at_least_90pct']=bool(moving.any() and wrong.sum()/moving.sum()<=.10)
    return {'passed':all(checks.values()),'checks':checks,'metrics':metrics}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    result=assess(json.loads((a.folder/'summary.json').read_text()),json.loads((a.folder/'context_10hz.json').read_text()),json.loads((a.folder/'rewards.json').read_text()))
    (a.folder/'assessment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
