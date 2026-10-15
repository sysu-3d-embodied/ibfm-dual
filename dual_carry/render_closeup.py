"""1080p following-camera view of saved physics; no simulation or policy rerun."""
import argparse,json
import imageio.v2 as imageio,mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from environment import DualCarryEnv,Path

p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
p.add_argument('--preview-only',action='store_true');p.add_argument('--stride',type=int,default=5)
a=p.parse_args();w,h=1920,1080
config=json.loads((a.folder/'config.json').read_text())
config.update(render_width=w,render_height=h)
env=DualCarryEnv(config)
trajectory=np.load(a.folder/'trajectory.npz');frames=trajectory['qpos']
summary=json.loads((a.folder/'summary.json').read_text())
goal=np.array(summary['destination_m']);events=summary['events']
renderer=mujoco.Renderer(env.model,height=h,width=w)
camera=mujoco.MjvCamera();camera.distance=2.8;camera.azimuth=120;camera.elevation=-15
option=mujoco.MjvOption();option.geomgroup[3]=0
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
out=a.folder/'preview_closeup.mp4'
steps=sorted(set(min(len(frames)-1,int(t*50)) for t in [0,10,40,130,len(frames)/50-.1])) if a.preview_only else range(0,len(frames),a.stride)
writer=None if a.preview_only else imageio.get_writer(str(out),fps=50/a.stride,codec='libx264',quality=8,macro_block_size=1)
previews=[]
for step in steps:
    env.data.qpos[:]=frames[step];env.forward();t=(step+1)/50
    camera.lookat[:]=.5*(env.data.xpos[env.robots[0].base]+env.data.xpos[env.robots[1].base])
    camera.lookat[2]=.72
    renderer.update_scene(env.data,camera=camera,scene_option=option)
    scene=renderer.scene
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([.12,.003,.003]),np.array([goal[0],goal[1],.005]),np.eye(3).reshape(-1),np.array([.12,.85,.25,.55],dtype=np.float32))
    scene.ngeom+=1
    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
    stage='settle'
    for event in events:
        if event['time_s']<=t:stage=event['to']
    draw.rectangle((0,0,w,88),fill=(18,23,30))
    draw.text((24,10),f"I-BFM dual carry | CLOSE-UP FOLLOWING CAMERA | t={t:.2f} s | {stage}",font=font,fill='white')
    result='PASS (nominal checklist)' if summary['success'] else 'NOT PASSED (original assessment)'
    draw.text((24,48),f"Same recorded physics | {result} | Grip positions: +/-{config.get('grip_x_m',.17):.2f} m",font=font,fill=(255,215,125))
    box=env.data.xpos[env.box_body];error=np.linalg.norm(box[:2]-goal[:2])
    draw.rectangle((0,h-54,w,h),fill=(18,23,30))
    draw.text((24,h-43),f"Box height: {box[2]:.3f} m | Horizontal goal error: {error:.3f} m | Green circle: goal | Blue forward / orange backward",font=font,fill='white')
    if writer:writer.append_data(np.asarray(frame))
    if a.preview_only:
        frame.save(a.folder/f'closeup_{step:05d}.jpg',quality=95)
        previews.append(frame.resize((960,540)))
    elif step%500==0:print(f'rendered {step}/{len(frames)}',flush=True)
if writer:writer.close()
if previews:
    grid=Image.new('RGB',(1920,540*((len(previews)+1)//2)))
    for i,frame in enumerate(previews):grid.paste(frame,((i%2)*960,(i//2)*540))
    grid.save(a.folder/'closeup_contact_sheet.jpg',quality=93)
renderer.close();trajectory.close()
print(a.folder/'closeup_contact_sheet.jpg' if a.preview_only else out,flush=True)
