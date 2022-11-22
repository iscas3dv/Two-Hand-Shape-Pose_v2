import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    args = parse_args()
    test_folder = './test_folder'
    model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
    model=Model(resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,cascaded_num=3)
    device_run=torch.device('cuda:%d'%(args.gpu))
    ckpt=torch.load("{}".format(model_path))
    model.load_state_dict({k.split('.',1)[1]:v for k,v in ckpt['network'].items()})
    model.to(device_run)
    model.eval()
    print('load success')
    INPUT_SIZE=256
    right_face=model.mesh_reg.mano_layer['r'].faces
    left_face=model.mesh_reg.mano_layer['l'].faces
    for img_name in os.listdir(test_folder):
        img=cv2.imread(os.path.join(test_folder,img_name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:continue
        ratio=INPUT_SIZE/max(*img.shape[:2])
        M=np.array([[ratio,0,0],[0,ratio,0]],dtype=np.float32)
        img=cv2.warpAffine(img,M,(INPUT_SIZE,INPUT_SIZE),flags=cv2.INTER_LINEAR,borderValue=[0,0,0])
        img=img[:,:,::-1].astype(np.float32)/255
        input_tensor=torch.tensor(img.copy().transpose(2,0,1),device=device_run,dtype=torch.float32).unsqueeze(0)
        out = model({'img':input_tensor}, None, None, 'test')

        joint_coord_out = out['joints3d']
        trans = out['trans']
        verts3d = out['verts3d']
        right_mano_para={
            'joints3d':joint_coord_out[:,:21,:]-joint_coord_out[:,4,None,:],
            'verts3d':verts3d[:,:verts3d.shape[1]//2,:]-joint_coord_out[:,4,None,:],
        }
        left_mano_para = {
            'joints3d':joint_coord_out[:,21:,:]-joint_coord_out[:,4+21,None,:],
            'verts3d':verts3d[:,verts3d.shape[1]//2:,:]-joint_coord_out[:,4+21,None,:],
        }


        predict_right_length=(right_mano_para['joints3d'][:,4]-right_mano_para['joints3d'][:,0]).norm(dim=1)
        predict_left_length=(left_mano_para['joints3d'][:,4]-left_mano_para['joints3d'][:,0]).norm(dim=1)
        predict_right_verts=right_mano_para['verts3d']/predict_right_length[:,None,None]
        predict_left_verts=left_mano_para['verts3d']/predict_left_length[:,None,None]

        predict_left_verts_trans=(predict_left_verts+trans[:,1:].view(-1,1,3))*torch.exp(trans[:,0,None,None])

        output_file_name=img_name.split('.')[0]
        
        with open(os.path.join(test_folder,output_file_name+'_right.obj'),'w') as file_object:
            for v in predict_right_verts[0]:
                print("v %f %f %f"%(v[0],v[1],v[2]),file=file_object)
            for f in right_face+1:
                print("f %d %d %d"%(f[0],f[1],f[2]),file=file_object)

        with open(os.path.join(test_folder,output_file_name+'_left.obj'),'w') as file_object:
            for v in predict_left_verts_trans[0]:
                print("v %f %f %f"%(v[0],v[1],v[2]),file=file_object)
            for f in left_face+1:
                print("f %d %d %d"%(f[0],f[1],f[2]),file=file_object)

        with open(os.path.join(test_folder,output_file_name+'_interacting.obj'),'w') as file_object:
            for v in predict_right_verts[0]:
                print("v %f %f %f"%(v[0],v[1],v[2]),file=file_object)
            for v in predict_left_verts_trans[0]:
                print("v %f %f %f"%(v[0],v[1],v[2]),file=file_object)
            for f in right_face+1:
                print("f %d %d %d"%(f[0],f[1],f[2]),file=file_object)
            for f in left_face+1+predict_right_verts.shape[1]:
                print("f %d %d %d"%(f[0],f[1],f[2]),file=file_object)
            

if __name__=='__main__':
    with torch.no_grad():
        main()

