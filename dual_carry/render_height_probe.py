"""Render recorded physical trajectories and plot leader/follower diagnostics."""
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--compare',type=Path)
p.add_argument('--plot-only',action='store_true');a=p.parse_args()
rows=json.loads((a.folder/'probe.json').read_text());summary=json.loads((a.folder/'summary.json').read_text())
if not rows:raise RuntimeError('No active probe samples; inspect summary')
v=lambda key:np.array([r[key] for r in rows]);t=v('elapsed');base=v('A_height_target')[0]
fig,ax=plt.subplots(3,2,figsize=(15,11),sharex=True)
ax[0,0].plot(t,v('A_height_target')-base,'k--',label='A requested lift')
ax[0,0].plot(t,v('A_hand_delta'),label='A actual hand lift')
ax[0,0].plot(t,v('B_hand_delta'),label='B actual hand lift')
ax[0,0].set_ylabel('Vertical displacement (m)')
ax[0,1].plot(t,v('A_hand_delta'),label='Measured A cue')
ax[0,1].plot(t,v('B_height_target')-base,label='B reward height offset')
ax[0,1].set_ylabel('ct -> B target (m)')
ax[1,0].plot(t,v('box_tilt_deg'),label='Box tilt')
ax[1,0].axhline(15,color='red',ls='--',label='15 deg criterion')
ax[1,0].set_ylabel('Tilt (degrees)')
ax[1,1].plot(t,v('max_hand_mean_force_N'),label='Largest hand mean load')
ax[1,1].axhline(60,color='red',ls='--',label='60 N criterion')
ax[1,1].set_ylabel('Force norm (N)')
ax[2,0].plot(t,v('A_root_drift'),label='A root horizontal drift')
ax[2,0].plot(t,v('B_root_drift'),label='B root horizontal drift')
ax[2,0].axhline(.15,color='red',ls='--',label='15 cm criterion')
ax[2,0].set_ylabel('Standing drift (m)')
ax[2,1].plot(t,v('B_same_obs_joint_target_diff_rad'),label='B joint target effect: live vs frozen cue')
ax[2,1].set_ylabel('Same-observation RMS difference (rad)')
if a.compare:
    control=json.loads((a.compare/'probe.json').read_text())
    if control:
        ax[0,0].plot([r['elapsed'] for r in control],[r['B_hand_delta'] for r in control],ls=':',label='B in frozen-cue rollout')
        ax[1,0].plot([r['elapsed'] for r in control],[r['box_tilt_deg'] for r in control],ls=':',label='Frozen-cue rollout tilt')
for axis in ax.flat:axis.legend(fontsize=8);axis.grid(alpha=.25)
for axis in ax[2]:axis.set_xlabel('Time since standing probe began (s)')
fig.suptitle('1.4 kg shared box: A commanded, B follows measured A via reward -> z\n'+
    ('ALL CRITERIA PASSED' if summary['success'] else 'NOT ALL CRITERIA PASSED — see measurements'))
fig.tight_layout();fig.savefig(a.folder/'height_response.png',dpi=160);plt.close(fig)
if a.plot_only:raise SystemExit()

import imageio.v2 as imageio,mujoco
from PIL import Image,ImageDraw,ImageFont
from environment import DualCarryEnv
config=json.loads((a.folder/'config.json').read_text());config.update(render_width=1920,render_height=1080)
env=DualCarryEnv(config);frames=np.load(a.folder/'trajectory.npz')['qpos']
renderer=mujoco.Renderer(env.model,height=1080,width=1920)
cam=mujoco.MjvCamera();cam.distance=2.8;cam.azimuth=120;cam.elevation=-15
option=mujoco.MjvOption();option.geomgroup[3]=0
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
rt=v('t');keyframes=[]
writer=imageio.get_writer(str(a.folder/'height_response_1080p.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1)
for step in range(0,len(frames),2):
    env.data.qpos[:]=frames[step];env.forward();now=(step+1)/50
    cam.lookat[:]=.5*(env.data.xpos[env.robots[0].base]+env.data.xpos[env.robots[1].base]);cam.lookat[2]=.72
    renderer.update_scene(env.data,camera=cam,scene_option=option)
    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
    draw.rectangle((0,0,1920,110),fill=(18,23,30));draw.rectangle((0,978,1920,1080),fill=(18,23,30))
    draw.text((24,10),f"Standing leader/follower test | A: blue | B: orange | 1.4 kg | t={now:.2f}s",font=font,fill='white')
    if now>=rt[0]:
        r=rows[max(0,np.searchsorted(rt,now,side='right')-1)]
        draw.text((24,49),f"{r['segment']} | A reward height: {r['A_height_target']:.3f}m | B reward height from measured A: {r['B_height_target']:.3f}m",font=font,fill=(255,220,135))
        draw.text((24,990),f"Hand lift: A {r['A_hand_delta']:+.3f}m / B {r['B_hand_delta']:+.3f}m | Box tilt {r['box_tilt_deg']:.1f} deg",font=font,fill='white')
        draw.text((24,1030),f"Standing drift: A {r['A_root_drift']:.3f}m / B {r['B_root_drift']:.3f}m | Measured physics; no fixed roots",font=font,fill='white')
    else:draw.text((24,49),'Initial grasp and lift before the standing test',font=font,fill='white')
    writer.append_data(np.asarray(frame))
    if step%400==0:keyframes.append(frame.resize((960,540)));print(f'rendered {step}/{len(frames)}',flush=True)
writer.close();renderer.close()
grid=Image.new('RGB',(1920,540*((len(keyframes)+1)//2)))
for i,frame in enumerate(keyframes):grid.paste(frame,((i%2)*960,(i//2)*540))
grid.save(a.folder/'height_response_contact_sheet.jpg',quality=92)
print(a.folder/'height_response_1080p.mp4',flush=True)
