"""Import the user-provided G1/Revo2 assets without executing archive scripts."""
from pathlib import Path
import hashlib,json,zipfile,copy
import xml.etree.ElementTree as ET
from environment import HERE,REPO

src=HERE/'g1_revo2_assets.zip';out=(HERE/'model_assets').resolve();out.mkdir(exist_ok=True)
with zipfile.ZipFile(src) as z:
    for info in z.infolist():
        target=(out/info.filename).resolve()
        if not target.is_relative_to(out):raise ValueError(f'Unsafe archive path: {info.filename}')
        if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink is unsupported')
        if info.is_dir():target.mkdir(parents=True,exist_ok=True)
        else:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
preview=out/'portable_preview/scene.xml';source=ET.parse(preview).getroot()
root=ET.Element('mujoco',model='user_g1_revo2_robot_only')
ET.SubElement(root,'compiler',angle='radian',autolimits='true',balanceinertia='true')
asset=copy.deepcopy(source.find('asset'));root.append(asset)
for mesh in asset.findall('mesh'):
    mesh.set('file',str((preview.parent/mesh.get('file')).resolve()))
world=ET.SubElement(root,'worldbody');pelvis=copy.deepcopy(source.find('./worldbody/body[@name="pelvis"]'));world.append(pelvis)
removed=[]
for body in pelvis.iter('body'):
    for node in list(body):
        if node.tag in ('camera','light') or (node.tag=='geom' and 'rubber_hand' in node.get('mesh','')):
            removed.append(node.get('name',node.get('mesh',node.tag)));body.remove(node)
    for gi,geom in enumerate(body.findall('geom')):
        if geom.get('name') is None:geom.set('name',body.get('name')+'_geom_'+str(gi))
        is_visual=geom.get('contype')=='0' and geom.get('conaffinity')=='0'
        geom.set('group','2' if is_visual else '3')
        if not is_visual:
            geom.set('contype','1')
            is_hand=any(token in body.get('name','') for token in ('hand_base','thumb_','index_','middle_','ring_','pinky_'))
            geom.set('conaffinity','6' if is_hand else '1')
act=ET.SubElement(root,'actuator');joint_names=[];finger_names=[]
for joint in pelvis.iter('joint'):
    if joint.get('type')=='free':continue
    name=joint.get('name');joint_names.append(name)
    ET.SubElement(act,'motor',name=name+'_motor',joint=name,gear='1')
    if any(x in name for x in ('thumb_','index_','middle_','ring_','pinky_')):finger_names.append(name)
destination=HERE/'g1_revo2_robot.xml';ET.indent(root,space='  ');ET.ElementTree(root).write(destination,encoding='unicode')
report={'archive_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'source':str(preview),
        'robot_xml':str(destination),'joint_count':len(joint_names),'finger_joints':finger_names,
        'removed_obsolete_rubber_geoms_and_target_cameras':removed,
        'adaptation':'Preserved supplied G1+Revo2 bodies, masses, limits and meshes. Removed scene table/bottle/cap and duplicate rubber hands. Added torque motors; finger commands will use coupled flexion targets.',
        'constraints':False}
(HERE/'revo2_import.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
