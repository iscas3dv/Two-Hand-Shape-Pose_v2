# This part reuses the code of InterHand2.6M.
# Thanks to Gyeongsik Moon for excellent work.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/InterHand2.6M/blob/main/LICENSE
#

import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn
from utils.transforms import flip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--test_set', type=str, dest='test_set')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    
    if cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
    else:
        args.test_set = 'test'

    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set)
    tester._make_model()
    
    preds = {'joints3d': [], 'trans': [], 'verts3d': []}
    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
            
            # forward
            out = tester.model(inputs, targets, meta_info, 'test')

            joint_coord_out = out['joints3d'].cpu().numpy()
            trans = out['trans'].cpu().numpy()
            verts3d = out['verts3d'].cpu().numpy()

            preds['joints3d'].append(joint_coord_out)
            preds['trans'].append(trans)
            preds['verts3d'].append(verts3d)
            
    # evaluate
    preds = {k: np.concatenate(v) for k,v in preds.items()}
    tester._evaluate(preds)   

if __name__ == "__main__":
    main()
