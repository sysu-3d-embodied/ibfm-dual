"""Plots and video from saved speed-following physics; no rerun or tuning."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('live',type=Path);p.add_argument('control',type=Path)
p.add_argument('--plot-only',action='store_true');a=p.parse_args()
rows=json.loads((a.live/'probe.json').read_text());control=json.loads((a.control/'probe.json').read_text())
summary=json.loads((a.live/'summary.json').read_text())
if not rows:raise RuntimeError('No speed-test samples')
v=lambda key:np.array([r[key] for r in rows]);t=v('elapsed')
fig,ax=plt.subplots(3,2,figsize=(15,11),sharex=True)
ax[0,0].plot(t,v('A_command_mps'),'k--',label='A commanded speed')
ax[0,0].plot(t,v('A_filtered_vx'),label='A actual (0.2s LPF)')
ax[0,0].plot(t,v('B_filtered_vx'),label='B actual (0.2s LPF)')
ax[0,0].set_ylabel('World X speed (m/s)')
ax[0,1].plot(t,v('A_cue_used_by_B'),label='Measured A speed cue')
ax[0,1].plot(t,v('B_target_vx'),label='B reward target with formation correction')
ax[0,1].plot(t,v('B_filtered_vx'),label='B actual (0.2s LPF)')
ax[0,1].set_ylabel('B speed (m/s)')
ax[1,0].plot(t,v('B_filtered_vx')-v('A_filtered_vx'),label='B - A speed, live cue')
ax[1,0].set_ylabel('Speed mismatch (m/s)')
ax[1,1].plot(t,v('gap_error_norm'),label='Formation error, live cue')
ax[1,1].set_ylabel('Formation error (m)')
ax[2,0].plot(t,v('box_tilt_deg'),label='Box tilt, live cue')
ax[2,0].set_ylabel('Box tilt (deg)')
ax[2,1].plot(t,v('B_joint_target_effect_rad'),label='B same-observation target difference')
ax[2,1].set_ylabel('Joint target RMS difference (rad)')
if control:
    ct=np.array([r['elapsed'] for r in control])
    ax[1,0].plot(ct,[r['B_filtered_vx']-r['A_filtered_vx'] for r in control],ls=':',label='B - A speed, frozen cue')
    ax[1,1].plot(ct,[r['gap_error_norm'] for r in control],ls=':',label='Formation error, frozen cue')
    ax[2,0].plot(ct,[r['box_tilt_deg'] for r in control],ls=':',label='Box tilt, frozen cue')
for axes in ax.flat:axes.grid(alpha=.25);axes.legend(fontsize=8)
for axes in ax[2]:axes.set_xlabel('Time since transport started (s)')
level_enabled=json.loads((a.live/'config.json').read_text()).get('level_feedback',False)
fig.suptitle(('1.4 kg speed following: shared object target + level feedback' if level_enabled else '1.4 kg speed-following validation: A commanded, B follows measured speed')+'\n'+
    ('ALL CRITERIA PASSED' if summary['success'] else 'NOT ALL CRITERIA PASSED — unoptimized test'))
fig.tight_layout();fig.savefig(a.live/'speed_follow_response.png',dpi=160);plt.close(fig)
if a.plot_only:raise SystemExit()
import imageio.v2 as imageio,mujoco
from PIL import Image,ImageDraw,ImageFont
from environment import DualCarryEnv
config=json.loads((a.live/'config.json').read_text());config.update(render_width=1920,render_height=1080)
env=DualCarryEnv(config);frames=np.load(a.live/'trajectory.npz')['qpos']
renderer=mujoco.Renderer(env.model,height=1080,width=1920)
camera=mujoco.MjvCamera();camera.distance=2.8;camera.azimuth=120;camera.elevation=-15
option=mujoco.MjvOption();option.geomgroup[3]=0
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
rt=v('t');keyframes=[]
writer=imageio.get_writer(str(a.live/'speed_follow_1080p.mp4'),fps=25,codec='libx264',
    macro_block_size=1,ffmpeg_params=['-crf','27','-preset','fast'])
for step in range(0,len(frames),2):
    env.data.qpos[:]=frames[step];env.forward();now=(step+1)/50
    camera.lookat[:]=.5*(env.data.xpos[env.robots[0].base]+env.data.xpos[env.robots[1].base]);camera.lookat[2]=.72
    renderer.update_scene(env.data,camera=camera,scene_option=option)
    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
    draw.rectangle((0,0,1920,112),fill=(18,23,30));draw.rectangle((0,978,1920,1080),fill=(18,23,30))
    label='Shared object goal + level feedback' if level_enabled else 'Unoptimized'
    draw.text((24,10),f'Speed-following | A: blue, B: orange | 1.4 kg | t={now:.2f}s | {label}',font=font,fill='white')
    if now>=rt[0]:
        r=rows[max(0,np.searchsorted(rt,now,side='right')-1)]
        draw.text((24,50),f"A command: {r['A_command_mps']:.3f} m/s | Actual A: {r['A_filtered_vx']:.3f} / B: {r['B_filtered_vx']:.3f} m/s (0.2s LPF)",font=font,fill=(255,220,135))
        draw.text((24,990),f"B reward speed target: {r['B_target_vx']:.3f} m/s | Formation error: {r['gap_error_norm']:.3f} m",font=font,fill='white')
        draw.text((24,1030),f"Box tilt: {r['box_tilt_deg']:.1f} deg | Box travel: {r['box_dx']:.2f} m | Signed world +X speed",font=font,fill='white')
    else:draw.text((24,50),'Initial grasp and lift',font=font,fill='white')
    writer.append_data(np.asarray(frame))
    if step%400==0:keyframes.append(frame.resize((960,540)));print(f'rendered {step}/{len(frames)}',flush=True)
writer.close();renderer.close()
grid=Image.new('RGB',(1920,540*((len(keyframes)+1)//2)))
for i,frame in enumerate(keyframes):grid.paste(frame,((i%2)*960,(i//2)*540))
grid.save(a.live/'speed_follow_contact_sheet.jpg',quality=90)
print(a.live/'speed_follow_1080p.mp4',flush=True)
