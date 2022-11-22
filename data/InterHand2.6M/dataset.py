# This part reuses the code of InterHand2.6M.
# Thanks to Gyeongsik Moon for excellent work.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/InterHand2.6M/blob/main/LICENSE
#

import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from config import cfg
from utils.preprocessing import load_img, generate_mano_skeleton, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints, vis_mesh
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import scipy.io as sio
import smplx
import os
from tqdm import tqdm
from eval import EvalUtil

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode, use_sh:bool=False, use_ih:bool=True):
        self.mode = mode # train, test, val
        self.dataset_path = '/data/baowen20/dataset/InterHand2.6M/InterHand2.6M_5fps_batch1'
        self.img_path = os.path.join(self.dataset_path,'images')
        self.annot_path = os.path.join(self.dataset_path,'annotations')
        self.transform = transform
        self.joint_num = 21 # single hand
        self.shape_dim = 10
        self.pose_dim = 48
        self.root_joint_idx = {'right': 4, 'left': 4+self.joint_num}
        self.wrist_joint_idx = {'right': 0, 'left': self.joint_num}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.verts_type = {'right': np.arange(0,778), 'left': np.arange(778,778*2)}
        self.shape_type = {'right': np.arange(0,self.shape_dim), 'left': np.arange(self.shape_dim,self.shape_dim*2)}
        self.pose_type = {'right': np.arange(0,self.pose_dim), 'left': np.arange(self.pose_dim,self.pose_dim*2)}
        self.trans_type = {'right': np.arange(0,3), 'left': np.arange(3,6)}
        # self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num*2)
        self.skeleton = generate_mano_skeleton()
        self.use_sh = use_sh
        self.use_ih = use_ih
        
        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []

        mano_layer = {'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True),
                        'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False)}
        

        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
            print('Fix shapedirs bug of MANO')
            mano_layer['left'].shapedirs[:,0,:] *= -1
        
        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.mode))
        db = COCO(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)
        
        MANO_joints_filename = osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_MANO_joints.json')
        if os.path.exists(MANO_joints_filename):
            with open(MANO_joints_filename) as f:
                joints = json.load(f)
        else:
            joints = {c:dict() for c in mano_params.keys()}

        update_mano_joints = False
        
        for aid in tqdm(db.anns.keys(), ncols=100, mininterval=10):
            ann = db.anns[aid]
            hand_type = ann['hand_type']
            if (hand_type == 'right' or hand_type == 'left') and not use_sh: continue
            if (hand_type != 'right' and hand_type != 'left') and not use_ih: continue
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
 
            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']

            if str(frame_idx) not in mano_params[str(capture_id)]:continue
            img_path = osp.join(self.img_path, self.mode, img['file_name'])
            
            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            
            shape_param = np.zeros([self.shape_dim*2], dtype=np.float32)
            pose_param = np.zeros([self.pose_dim * 2], dtype=np.float32)
            trans_param = np.zeros([3 * 2], dtype=np.float32)
            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']] * joint_valid[self.wrist_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']] * joint_valid[self.wrist_joint_idx['left']]
            mano_valid = np.ones([2],dtype=np.float32)
            if str(frame_idx) in joints[str(capture_id)]:
                joint_world = np.array(joints[str(capture_id)][str(frame_idx)],np.float32)
                for str2id, ht in enumerate(('right','left')):
                    mano_param = mano_params[str(capture_id)][str(frame_idx)][ht]
                    if mano_param is None:
                        mano_valid[str2id] = 0
                        joint_valid[self.joint_type[ht]] = 0
                        continue
                    shape_param[self.shape_type[ht]] = mano_param['shape']
                    pose_param[self.pose_type[ht]] = mano_param['pose']
                    trans_param[self.trans_type[ht]] = mano_param['trans']
            else:
                update_mano_joints = True
                joint_world = np.zeros([self.joint_num*2,3], dtype=np.float32)
                for str2id, ht in enumerate(('right','left')):
                    mano_param = mano_params[str(capture_id)][str(frame_idx)][ht]
                    if mano_param is None:
                        mano_valid[str2id] = 0
                        joint_valid[self.joint_type[ht]] = 0
                        continue
                    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
                    root_pose = mano_pose[0].view(1,3)
                    hand_pose = mano_pose[1:,:].view(1,-1)
                    shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
                    trans = torch.FloatTensor(mano_param['trans']).view(1,3)
                    output = mano_layer[ht](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                    J = output.joints[0].numpy()
                    fingertips = output.vertices[0].numpy()[[745, 317, 444, 556, 673]]
                    J = np.vstack([J,fingertips]) * 1000
                    joint_world[self.joint_type[ht]] = J
                    shape_param[self.shape_type[ht]] = mano_param['shape']
                    pose_param[self.pose_type[ht]] = mano_param['pose']
                    trans_param[self.trans_type[ht]] = mano_param['trans']
                joints[str(capture_id)][str(frame_idx)] = joint_world.tolist()
            
            
            joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]

            
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
            
            img_width, img_height = img['width'], img['height']
            bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
            bbox = process_bbox(bbox, (img_height, img_width))

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bbox, 
                    'joint': joint, 'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 
                    'shape_param': shape_param,'pose_param':pose_param,'trans_param':trans_param,'mano_valid':mano_valid,
                    'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx,
                    'camrot':camrot, 'campos':campos}
            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)
        
        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))
        if update_mano_joints:
            print("update %s"%os.path.basename(MANO_joints_filename))
            with open(MANO_joints_filename,'w') as file_object:
                file_object.write(json.dumps(joints))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid, shape_param = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid'], data['shape_param']
        mano_valid = data['mano_valid']
        joint_cam = joint['cam_coord'].copy(); joint_valid = joint['valid'].copy();
        hand_type = self.handtype_str2array(hand_type)
        
        # image load
        img = load_img(img_path)
        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans, shape_param = augmentation(img, bbox, joint_cam, joint_valid, hand_type, self.mode, self.joint_type, shape_param, self.shape_type)
        root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]],dtype=np.float32).reshape(1) if hand_type[0]*hand_type[1] == 1 else np.zeros((1),dtype=np.float32)
        img = self.transform(img.astype(np.float32))/255.
        
        inputs = {'img': img}
        targets = {'joint_coord': joint_coord, 'hand_type': hand_type, 'shape_param':shape_param}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid, 'mano_valid':mano_valid,
                     'inv_trans': inv_trans, 'capture': int(data['capture']), 'cam': int(data['cam']), 'frame': int(data['frame'])}
        return inputs, targets, meta_info

    def evaluate(self, preds):
        mano_layer = {'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True),
                        'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False)}
        

        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
            print('Fix shapedirs bug of MANO')
            mano_layer['left'].shapedirs[:,0,:] *= -1
        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, trans, preds_verts = preds['joints3d'], preds['trans'], preds['verts3d']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        
        mpjpe_sh = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih = [[] for _ in range(self.joint_num*2)]

        mpvpe_sh = []
        mpvpe_ih = []
        util = EvalUtil(num_kp=self.joint_num*2)
        for n in range(sample_num):
            data = gts[n]
            bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
            shape_param, pose_param, trans_param, mano_valid = data['shape_param'], data['pose_param'], data['trans_param'], data['mano_valid']
            camrot, campos = data['camrot'], data['campos']
            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']
            joint_valid = joint['valid']

            pred_joint_coord_cam = preds_joint_coord[n].copy()
            pred_verts_coord_cam = preds_verts[n].copy()
            # root joint alignment
            for str2id,h in enumerate(('right', 'left')):
                gt_normalization_length = np.linalg.norm(gt_joint_coord[self.root_joint_idx[h]]-gt_joint_coord[self.wrist_joint_idx[h]])
                pred_normalization_length = np.linalg.norm(pred_joint_coord_cam[self.root_joint_idx[h]]-pred_joint_coord_cam[self.wrist_joint_idx[h]])
                if mano_valid[str2id]:
                    mano_pose = torch.FloatTensor(pose_param[self.pose_type[h]]).view(-1,3)
                    root_pose = mano_pose[0].view(1,3)
                    hand_pose = mano_pose[1:,:].view(1,-1)
                    shape = torch.FloatTensor(shape_param[self.shape_type[h]]).view(1,-1)
                    trans = torch.FloatTensor(trans_param[self.trans_type[h]]).view(1,3)
                    output = mano_layer[h](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                    gt_verts_coord_cam = output.vertices[0].numpy() * 1000
                    gt_verts_coord_cam = world2cam(gt_verts_coord_cam.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
                    pred_verts_coord_cam[self.verts_type[h]] = \
                                (pred_verts_coord_cam[self.verts_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h],None,:])\
                                * (gt_normalization_length/pred_normalization_length) + gt_joint_coord[self.root_joint_idx[h],None,:]
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpvpe_sh.append(np.sqrt(np.sum((pred_verts_coord_cam[self.verts_type[h]] - gt_verts_coord_cam)**2,axis=-1)))
                    else:
                        mpvpe_ih.append(np.sqrt(np.sum((pred_verts_coord_cam[self.verts_type[h]] - gt_verts_coord_cam)**2,axis=-1)))
                pred_joint_coord_cam[self.joint_type[h]] = \
                            (pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h],None,:])\
                            * (gt_normalization_length/pred_normalization_length) + gt_joint_coord[self.root_joint_idx[h],None,:]
                
            

            # mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                    else:
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
            
            
            
            util.feed(gt_joint_coord,joint_valid,pred_joint_coord_cam)

            vis = False
            if vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                pred_joint_coord_img = (pred_joint_coord_cam[:,:2]/pred_joint_coord_cam[:,2:]) * focal[None,:] + princpt[None,:]
                vis_kps = pred_joint_coord_img.copy()
                vis_valid = joint_valid.copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
                vis_keypoints(_img, vis_kps, vis_valid, self.skeleton, filename)

            vis = False
            if vis:
                filename = 'out_' + str(n) + '_3d.jpg'
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton, filename)

            vis = False
            if vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = 'out_' + str(n) + '_mesh.jpg'
                vis_mesh(cvimg, pred_verts_coord_cam, np.concatenate([mano_layer['right'].faces,mano_layer['left'].faces]),
                            data['cam_param']['focal'],data['cam_param']['princpt'],mano_valid,filename)
                
 
        tot_err = []
        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            if not self.use_sh:
                tot_err_j=np.mean(np.stack(mpjpe_ih[j]))
            else:
                tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        print(eval_summary)
        print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
        

        eval_summary = 'MPVPE: \n'
        if not self.use_sh: 
            tot_err_v = np.mean(mpvpe_ih)
        else:
            tot_err_v = np.mean(np.concatenate(mpvpe_ih,mpvpe_sh))
        print('MPVPE for all hand sequences: %.2f' % (np.mean(tot_err_v)))
        print()
        
        mean, median, auc, pck_curve_all, threshs = util.get_measures(0, 50, 200)
        # print('Average mean EPE: %.3f mm' %mean)
        # print('Average median EPE: %.3f mm' %median)
        print('Area under curve between 0mm - 50mm: %.3f' % auc)
        util.plot_pck(threshs,pck_curve_all,auc)



