import torch
import torch.nn as nn
from config import cfg
import numpy as np
T_JOINT_NUM = 16
JOINT_NUM = 21
SHAPE_DIM = 10
ROOT_JOINT_IDX = {'right': 4, 'left': 4+JOINT_NUM}
WRIST_JOINT_IDX = {'right': 0, 'left': JOINT_NUM}
T_ROOT_JOINT_IDX = {'right': 4, 'left': 4+T_JOINT_NUM}
T_WRIST_JOINT_IDX = {'right': 0, 'left': T_JOINT_NUM}
T_JOINT_TYPE = {'right': np.arange(0,T_JOINT_NUM), 'left': np.arange(T_JOINT_NUM,T_JOINT_NUM*2)}
JOINT_TYPE = {'right': np.arange(0,JOINT_NUM), 'left': np.arange(JOINT_NUM,JOINT_NUM*2)}
SHAPE_TYPE = {'right': np.arange(0,SHAPE_DIM), 'left': np.arange(SHAPE_DIM,SHAPE_DIM*2)}
# JOINT_LEVEL_IDX={'right':np.array([[0,0,0,0,0],[13,1,4,10,7],[14,2,5,11,8],[15,3,6,12,9],[16,17,18,19,20]],dtype=int),
#                 'left':np.array([[0,0,0,0,0],[13,1,4,10,7],[14,2,5,11,8],[15,3,6,12,9],[16,17,18,19,20]],dtype=int)+JOINT_NUM,}
JOINT_LEVEL_IDX={'right':np.array([[0,0,0,0,0],[13,1,4,10,7],[14,2,5,11,8],[15,3,6,12,9]],dtype=int),
                'left':np.array([[0,0,0,0,0],[13,1,4,10,7],[14,2,5,11,8],[15,3,6,12,9]],dtype=int)+JOINT_NUM,}
T_JOINT_LEVEL_IDX={'right':np.array([[0,0,0,0,0],[13,1,4,10,7],[14,2,5,11,8],[15,3,6,12,9]],dtype=int),
                'left':np.array([[0,0,0,0,0],[13,1,4,10,7],[14,2,5,11,8],[15,3,6,12,9]],dtype=int)+T_JOINT_NUM,}

def calculate_loss(outputs, targets, meta_info):
    joint_coord, hand_type, shape_param = targets['joint_coord'], targets['hand_type'].bool(), targets['shape_param']
    with torch.no_grad():
        gt_normalization_length = {ht:torch.norm(joint_coord[:,ROOT_JOINT_IDX[ht]]-joint_coord[:,WRIST_JOINT_IDX[ht]],dim=-1)\
                                        for ht in ('right','left')}

        gt_joint_right_coor = (joint_coord-joint_coord[:,ROOT_JOINT_IDX['right'],None,:])/gt_normalization_length['right'][:,None,None]
        gt_joint_left_coor = (joint_coord-joint_coord[:,ROOT_JOINT_IDX['left'],None,:])/gt_normalization_length['left'][:,None,None]
        
        gt_offset_right_coor = gt_joint_right_coor[:,JOINT_TYPE['left']]-gt_joint_right_coor[:,JOINT_TYPE['right']]
        gt_offset_left_coor = gt_joint_left_coor[:,JOINT_TYPE['left']]-gt_joint_left_coor[:,JOINT_TYPE['right']]

        gt_right_bone_length = torch.norm(gt_joint_right_coor[:,JOINT_LEVEL_IDX['right'][1:,:]] - gt_joint_right_coor[:,JOINT_LEVEL_IDX['right'][:-1,:]],dim=-1)
        gt_left_bone_length = torch.norm(gt_joint_left_coor[:,JOINT_LEVEL_IDX['left'][1:,:]] - gt_joint_left_coor[:,JOINT_LEVEL_IDX['left'][:-1,:]],dim=-1)

    predict_joint_coord, predict_shape_param, predict_pose_param, predict_trans = outputs["joints3d"], outputs["shape"], outputs["pose"], outputs["trans"]
    predict_T_coord = outputs["T_pose"]
    predict_normalization_length = {ht:torch.norm(predict_T_coord[:,:,T_ROOT_JOINT_IDX[ht]]-predict_T_coord[:,:,T_WRIST_JOINT_IDX[ht]],dim=-1).clamp_min(1e-8)\
                                        for ht in ('right','left')}
    predict_normalized_rTpose = predict_T_coord[:,:,T_JOINT_TYPE["right"]]/predict_normalization_length["right"][:,:,None,None]
    predict_normalized_lTpose = predict_T_coord[:,:,T_JOINT_TYPE["left"]]/predict_normalization_length["left"][:,:,None,None]
    predict_normalized_rjoint = (predict_joint_coord[:,:,JOINT_TYPE["right"]]-predict_joint_coord[:,:,ROOT_JOINT_IDX["right"],None])/predict_normalization_length["right"][:,:,None,None]
    predict_normalized_ljoint = (predict_joint_coord[:,:,JOINT_TYPE["left"]]-predict_joint_coord[:,:,ROOT_JOINT_IDX["left"],None])/predict_normalization_length["left"][:,:,None,None]
    
    predict_joint_right_coor = torch.cat([predict_normalized_rjoint,
                                            (predict_normalized_ljoint+predict_trans[:,None,None,1:])*torch.exp(predict_trans[:,None,:1,None])],dim=2)
    predict_joint_left_coor = torch.cat([predict_normalized_rjoint*torch.exp(-predict_trans[:,None,:1,None])-predict_trans[:,None,None,1:],
                                            predict_normalized_ljoint],dim=2)

    predict_offset_right_coor = predict_joint_right_coor[:,:,JOINT_TYPE['left']]-predict_joint_right_coor[:,:,JOINT_TYPE['right']]
    predict_offset_left_coor = predict_joint_left_coor[:,:,JOINT_TYPE['left']]-predict_joint_left_coor[:,:,JOINT_TYPE['right']]

    predict_right_bone_length = torch.norm(predict_normalized_rTpose[:,:,T_JOINT_LEVEL_IDX['right'][1:,:]] - predict_normalized_rTpose[:,:,T_JOINT_LEVEL_IDX['right'][:-1,:]],dim=-1)
    predict_left_bone_length = torch.norm(predict_normalized_lTpose[:,:,T_JOINT_LEVEL_IDX['right'][1:,:]] - predict_normalized_lTpose[:,:,T_JOINT_LEVEL_IDX['right'][:-1,:]],dim=-1)

    #joint_valid(B,42,1)
    joint_valid = meta_info["joint_valid"].unsqueeze(-1).bool()
    #mano_valid(B,3)
    mano_valid = meta_info["mano_valid"].bool()


    offset_valid = joint_valid[:,JOINT_TYPE['right']]*joint_valid[:,JOINT_TYPE['left']]
    batchsize = joint_coord.shape[0]
    right_joint_offset_loss = torch.where(offset_valid.unsqueeze(1),
                                            predict_offset_right_coor - gt_offset_right_coor.unsqueeze(1),
                                            torch.zeros_like(predict_offset_right_coor))
    left_joint_offset_loss = torch.where(offset_valid.unsqueeze(1),
                                            predict_offset_left_coor - gt_offset_left_coor.unsqueeze(1),
                                            torch.zeros_like(predict_offset_left_coor))

    shape_consistence_loss = torch.where((hand_type[:,0]*hand_type[:,1])[:,None,None],
                                            predict_shape_param[:,:,SHAPE_TYPE['right']]-predict_shape_param[:,:,SHAPE_TYPE['left']],
                                            torch.zeros_like(predict_shape_param[:,:,SHAPE_TYPE['right']]))

    right_joint_loss = torch.where(joint_valid[:,None,JOINT_TYPE['right']],
                                    predict_normalized_rjoint-gt_joint_right_coor[:,None,JOINT_TYPE['right']],
                                    torch.zeros_like(predict_normalized_rjoint))
    left_joint_loss = torch.where(joint_valid[:,None,JOINT_TYPE['left']],
                                    predict_normalized_ljoint-gt_joint_left_coor[:,None,JOINT_TYPE['left']],
                                    torch.zeros_like(predict_normalized_ljoint))

    right_bone_valid = joint_valid[:,JOINT_LEVEL_IDX['right'][1:,:]]*joint_valid[:,JOINT_LEVEL_IDX['right'][:-1,:]]
    right_bone_loss = torch.where(right_bone_valid.unsqueeze(1).squeeze(-1),
                                    predict_right_bone_length-gt_right_bone_length.unsqueeze(1),
                                    torch.zeros_like(predict_right_bone_length))
    left_bone_valid = joint_valid[:,JOINT_LEVEL_IDX['left'][1:,:]]*joint_valid[:,JOINT_LEVEL_IDX['left'][:-1,:]]
    left_bone_loss = torch.where(left_bone_valid.unsqueeze(1).squeeze(-1),
                                    predict_left_bone_length-gt_left_bone_length.unsqueeze(1),
                                    torch.zeros_like(predict_left_bone_length))

    right_shape_loss = torch.where(hand_type[:,None,:1] * mano_valid[:,None,:1],
                                    predict_shape_param[:,:,SHAPE_TYPE['right']]-shape_param[:,None,SHAPE_TYPE['right']],
                                    torch.zeros_like(predict_shape_param[:,:,SHAPE_TYPE['right']]))
    left_shape_loss = torch.where(hand_type[:,None,1:] * mano_valid[:,None,1:],
                                    predict_shape_param[:,:,SHAPE_TYPE['left']]-shape_param[:,None,SHAPE_TYPE['left']],
                                    torch.zeros_like(predict_shape_param[:,:,SHAPE_TYPE['left']]))
    regularizer_loss = torch.sum(predict_shape_param**2) + 0.1 * torch.sum(predict_pose_param[...,3:]**2)



    return {
        "offset":(torch.square(right_joint_offset_loss).sum()+torch.square(left_joint_offset_loss).sum())/batchsize * 1,
        "consis":torch.square(shape_consistence_loss).sum()/batchsize * 1e-2,
        "joint":(torch.abs(right_joint_loss).sum()+torch.abs(left_joint_loss).sum())/batchsize * 1e1,
        "bone":(torch.square(right_bone_loss).sum()+torch.square(left_bone_loss).sum())/batchsize *1e2,
        "shape":(torch.square(right_shape_loss).sum()+torch.square(left_shape_loss).sum())/batchsize * 1e-1,
        "regularizer":regularizer_loss/batchsize * 1e-1,
    }
