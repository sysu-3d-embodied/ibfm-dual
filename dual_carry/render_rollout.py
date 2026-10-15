"""Render saved physics trajectories without altering or regenerating actions."""
import argparse,json
import imageio.v2 as imageio,mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from environment import *
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--stride',type=int,default=2)
p.add_argument('--fixed-camera',action='store_true');a=p.parse_args()
config=json.loads((a.folder/'config.json').read_text()) if (a.folder/'config.json').exists() else {'box_size_m':[.60,.30,.30]}
e=DualCarryEnv(config);frames=np.load(a.folder/'trajectory.npz')['qpos']
summary=json.loads((a.folder/'summary.json').read_text()) if (a.folder/'summary.json').exists() else {}
events=summary.get('events',[]);renderer=mujoco.Renderer(e.model,height=720,width=1280)
contexts=json.loads((a.folder/'context_10hz.json').read_text()) if (a.folder/'context_10hz.json').exists() else []
context_times=np.array([c['time_s'] for c in contexts])
camera=mujoco.MjvCamera();camera.distance=3.8;camera.azimuth=105;camera.elevation=-18
goal=np.array(summary.get('destination_m',[0,0,.15]),dtype=float)
start_position=np.array(config.get('box_position_m',[0,0,.15]),dtype=float)
travel=float(np.linalg.norm(goal[:2]-start_position[:2]))
if a.fixed_camera:
    camera.lookat[:]=.5*(start_position+goal);camera.lookat[2]=.55
    camera.distance=max(3.8,1.15*travel+2.0)
    camera.azimuth=float(np.degrees(np.arctan2(goal[1]-start_position[1],goal[0]-start_position[0]))-90)
    camera.elevation=-25
opt=mujoco.MjvOption();opt.geomgroup[3]=0
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
writer=imageio.get_writer(str(a.folder/'preview.mp4'),fps=50/a.stride,codec='libx264',quality=8,macro_block_size=1)
for step in range(0,len(frames),a.stride):
    e.data.qpos[:]=frames[step];e.forward();t=(step+1)/50
    if not a.fixed_camera:
        camera.lookat[:]=.5*(e.data.xpos[e.robots[0].base]+e.data.xpos[e.robots[1].base]);camera.lookat[2]=.65
    renderer.update_scene(e.data,camera=camera,scene_option=opt)
    goal=np.array(summary.get('destination_m',[0,0,.15]),dtype=float)
    scene=renderer.scene
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([.12,.003,.003]),np.array([goal[0],goal[1],.005]),np.eye(3).reshape(-1),np.array([.12,.85,.25,.55],dtype=np.float32))
    scene.ngeom+=1
    if a.fixed_camera:
        # Visual markers only; the saved physics and collision model are untouched.
        markers=[(start_position,.12,[.15,.55,1.,.8])]
        if travel>0:
            for distance in np.arange(1.,travel,1.):
                markers.append((start_position+(goal-start_position)*distance/travel,.035,[.95,.85,.25,.8]))
        for position,radius,color in markers:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([radius,.003,.003]),np.array([position[0],position[1],.005]),np.eye(3).reshape(-1),np.array(color,dtype=np.float32))
            scene.ngeom+=1
    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
    stage='settle'
    for event in events:
        if event['time_s']<=t:stage=event['to']
    draw.rectangle((0,0,1280,76),fill=(18,23,30))
    draw.text((18,10),f"I-BFM dual carry | t={t:.2f}s | phase: {stage} | {summary.get('mode','probe')}",font=font,fill='white')
    draw.text((18,43),f"One shared physical box | {'Nominal checklist PASS; inspect lowering quality' if summary.get('success') else 'DEVELOPMENT TEST - NOT SUCCESS'}",font=font,fill=(255,215,125))
    if a.fixed_camera:
        draw.rectangle((0,76,1280,110),fill=(18,23,30))
        draw.text((18,80),f"Fixed camera | Blue: START | Green: GOAL | Yellow: 1 m spacing | Task distance: {travel:.2f} m",font=font,fill='white')
    if contexts:
        ct=contexts[max(0,min(len(contexts)-1,int(np.searchsorted(context_times,t,side='right')-1)))]
        forces=np.linalg.norm(ct['contact_window']['hand_mean_force_on_box_N'],axis=-1).max(axis=-1)
        contact=np.array(ct['contacts']['hand_contact']).any(axis=-1)
        error=np.linalg.norm(e.data.xpos[e.box_body,:2]-goal[:2])
        if config.get('axial_carry'):
            yaw_errors=[]
            desired=config['robot_yaws_rad']
            if config.get('goal_steering'):
                delta=goal[:2]-np.array(ct['object']['position_m'][:2])
                travel_yaw=float(np.arctan2(delta[1],delta[0]))
                desired=[travel_yaw,travel_yaw+np.pi]
            for i,robot in enumerate(ct['robots']):
                w,x,y,z=robot['quaternion_wxyz'];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                delta=desired[i]-yaw
                yaw_errors.append(np.degrees(np.arctan2(np.sin(delta),np.cos(delta))))
            draw.rectangle((0,110,1280,142),fill=(18,23,30))
            draw.text((18,113),f"Desired: blue forward / orange backward | Heading error R1/R2: {yaw_errors[0]:+.1f} / {yaw_errors[1]:+.1f} deg",font=font,fill='white')
        draw.rectangle((0,676,1280,720),fill=(18,23,30))
        draw.text((18,687),f"Box z={e.data.xpos[e.box_body,2]:.2f} m | Goal error={error:.2f} m | Contact R1/R2={int(contact[0])}/{int(contact[1])} | Hand load R1/R2={forces[0]:.1f}/{forces[1]:.1f} N",font=font,fill='white')
    writer.append_data(np.asarray(frame))
    if step%(50*2)==0:frame.save(a.folder/f'still_{step:04d}.png')
writer.close();renderer.close();print(a.folder/'preview.mp4')
