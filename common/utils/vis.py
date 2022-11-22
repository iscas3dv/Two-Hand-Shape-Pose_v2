# This part reuses the code of InterHand2.6M.
# We add function vis_mesh() for visualizing predicted mesh.
# Thanks to Gyeongsik Moon for excellent work.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/InterHand2.6M/blob/main/LICENSE
#

import os
import os.path as osp
import cv2
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from config import cfg
from PIL import Image, ImageDraw
import pyrender
import trimesh


def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        
    return rgb_dict

def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None):
    
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
    
    if save_path is None:
        _img.save(osp.join(cfg.vis_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))


def vis_3d_keypoints(kps_3d, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)
    
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')

    #plt.show()
    #cv2.waitKey(0)
    
    fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)

def vis_mesh(img, verts, faces, focal, princpt, mano_valid, filename):
    prev_depth = None
    for hand_id, (v, f) in enumerate(zip(np.split(verts,2,axis=0),np.split(faces,2,axis=0))):
        if not mano_valid[hand_id]:continue
        v = v / 1000 # milimeter to meter
        mesh = trimesh.Trimesh(v, f)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        scene.add(camera)
        
        # renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
        
        # light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        # render
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgb = rgb[:,:,:3].astype(np.float32)
        depth = depth[:,:,None]
        valid_mask = (depth > 0)

        if prev_depth is None:
            render_mask = valid_mask
            img = rgb * render_mask + img * (1 - render_mask)
            prev_depth = depth
        else:
            render_mask = valid_mask * np.logical_or(depth < prev_depth, prev_depth==0)
            img = rgb * render_mask + img * (1 - render_mask)
            prev_depth = depth * render_mask + prev_depth * (1 - render_mask)
    cv2.imwrite(osp.join(cfg.vis_dir, filename), img)


