"""
This part reuses code from https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/meshregnet.py
Thanks to Yana Hasson for the excellent work.
"""
import torch
from torch import nn

from nets import manobranch
from nets import resnet_anydim as resnet
from nets import absolutebranch
import smplx
from config import cfg

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, outplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MeshRegNet(nn.Module):
    def __init__(
        self,
        fc_dropout=0,
        resnet_version=50,
        mano_neurons=[512, 512],
        mano_comps=45,
        mano_use_pca=True,
        input_size=3,
        addition_channels=42*64,
        cascaded_num=1,
    ):
        super(MeshRegNet, self).__init__()
        if int(resnet_version) == 18:
            img_feature_size = 512
            base_net = resnet.resnet18(input_size=input_size,pretrained=False,addition_channel=addition_channels,return_inter=True)
        elif int(resnet_version) == 50:
            img_feature_size = 2048
            base_net = resnet.resnet50(input_size=input_size,pretrained=False,addition_channel=addition_channels,return_inter=True)
        elif int(resnet_version) == 101:
            img_feature_size = 2048
            base_net = resnet.resnet101(input_size=input_size,pretrained=False,addition_channel=addition_channels,return_inter=True)

        mano_base_neurons = [img_feature_size] + mano_neurons
        self.base_net = base_net
        self.mano_layer = nn.ModuleDict(
            {
                'r':smplx.create(cfg.smplx_path, 'mano', use_pca=mano_use_pca, is_rhand=True),
                'l':smplx.create(cfg.smplx_path, 'mano', use_pca=mano_use_pca, is_rhand=False)
            }
        )
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs( self.mano_layer['l'].shapedirs[:,0,:] -  self.mano_layer['r'].shapedirs[:,0,:])) < 1:
            print('Fix shapedirs bug of MANO')
            self.mano_layer['l'].shapedirs[:,0,:] *= -1

        # for quick T pose
        for ht in ('r','l'):
            beta2J = (self.mano_layer[ht].J_regressor[:,:,None,None] * self.mano_layer[ht].shapedirs.unsqueeze(0)).sum(dim=1)
            self.mano_layer[ht].register_buffer('beta2J', beta2J)
            template_J = (self.mano_layer[ht].J_regressor[:,:,None] * self.mano_layer[ht].v_template.unsqueeze(0)).sum(dim=1)
            self.mano_layer[ht].register_buffer('template_J', template_J)

        # Predict left hand
        self.left_mano_branch = manobranch.ManoBranch(
            ncomps=mano_comps,
            base_neurons=mano_base_neurons,
            dropout=fc_dropout,
            mano_pose_coeff=1,
            use_shape=True,
            use_pca=mano_use_pca,
        )
        

        # Predict right hand
        self.right_mano_branch = manobranch.ManoBranch(
            ncomps=mano_comps,
            base_neurons=mano_base_neurons,
            dropout=fc_dropout,
            mano_pose_coeff=1,
            use_shape=True,
            use_pca=mano_use_pca,
        )

        self.trans_branch = absolutebranch.AbsoluteBranch(
            base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=4
        )

        self.MANO_cascaded=cascaded_num>0
        self.cascaded_num=cascaded_num

        if self.MANO_cascaded:
            self.cascaded_left_mano_branch_list=nn.ModuleList([])
            self.cascaded_right_mano_branch_list=nn.ModuleList([])

            mano_base_neurons_channels = [img_feature_size+(10+48)*2+4]+mano_neurons
                
            for i in range(cascaded_num):
                self.cascaded_left_mano_branch_list.append(
                    manobranch.ManoBranch(
                        ncomps=mano_comps,
                        base_neurons=mano_base_neurons_channels,
                        dropout=fc_dropout,
                        mano_pose_coeff=1,
                        use_shape=True,
                        use_pca=mano_use_pca,
                    ))
                
                self.cascaded_right_mano_branch_list.append(
                    manobranch.ManoBranch(
                        ncomps=mano_comps,
                        base_neurons=mano_base_neurons_channels,
                        dropout=fc_dropout,
                        mano_pose_coeff=1,
                        use_shape=True,
                        use_pca=mano_use_pca,
                    ))

            feature_channel=[256,512,1024,2048]
            self.cascaded_left_feature_extractor_list=nn.ModuleList([])
            self.cascaded_right_feature_extractor_list=nn.ModuleList([])
            for i in range(cascaded_num):
                C=feature_channel[-(i+1)]
                input_channel=C
                stride=2**i
                downsample_right = nn.Conv2d(input_channel, feature_channel[-1], kernel_size=1, stride=stride, bias=False)
                downsample_left = nn.Conv2d(input_channel, feature_channel[-1], kernel_size=1, stride=stride, bias=False)
                self.cascaded_right_feature_extractor_list.append(Bottleneck(input_channel,C//2,feature_channel[-1],stride=stride,downsample=downsample_right))
                self.cascaded_left_feature_extractor_list.append(Bottleneck(input_channel,C//2,feature_channel[-1],stride=stride,downsample=downsample_left))

    def generate_batch_heatmap(self,batch_points,image_size,sigma):
        x_value=torch.arange(image_size,device=batch_points.device,dtype=torch.float32)
        x_value=batch_points[:,:,0].unsqueeze(2)-x_value[None,None,:]
        y_value=torch.arange(image_size,device=batch_points.device,dtype=torch.float32)
        y_value=batch_points[:,:,1].unsqueeze(2)-y_value[None,None,:]
        heatmap=torch.exp(-(x_value.unsqueeze(2)**2+y_value.unsqueeze(3)**2)/sigma)
        #heatmap=torch.exp(-(x_value.unsqueeze(2)**2+y_value.unsqueeze(3)**2)/sigma**2/2)
        return heatmap

    def register_heatmap(self,xyc:torch.Tensor,J:torch.Tensor,origin_size:int,output_size:int):
        device_run=xyc.device
        batch_size=xyc.shape[0]
        M=torch.cat([J[:,:,:2,None],torch.eye(2,device=device_run)[None,None,:,:].repeat(batch_size,J.shape[1],1,1)],dim=-1)
        wM=xyc[:,:,2,None,None]*M
        wB=xyc[:,:,2,None,None]*xyc[:,:,:2,None]
        wM=wM.reshape(batch_size,-1,3)
        wB=wB.reshape(batch_size,-1,1)
        MTM=torch.bmm(wM.transpose(2,1),wM)
        MTB=torch.bmm(wM.transpose(2,1),wB)
        sT=torch.bmm(torch.inverse(MTM),MTB)[:,:,0].detach()
        ratio=output_size/origin_size
        projected_xy=(J[:,:,:2]*sT[:,None,0,None]+sT[:,None,1:])*ratio
        #sigma = cfg.sigma * 2 * max(ratio, 0.25)
        heatmap=self.generate_batch_heatmap(projected_xy,output_size, 3)
        output_heatmap=1-torch.prod(1-heatmap,dim=1)
        return output_heatmap

    def pose_shape2verts_joints(self,pose,shape,side):
        output = self.mano_layer[side](global_orient=pose[:,:3], hand_pose=pose[:,3:], betas=shape)
        J = output.joints
        fingertips = output.vertices[:,[745, 317, 444, 556, 673]]
        joints = torch.cat([J,fingertips], dim=1) * 1000
        verts = output.vertices * 1000
        T_pose = self.mano_layer[side].template_J.unsqueeze(0) + \
                    (self.mano_layer[side].beta2J[None,:,:,:] * shape[:,None,None,:]).sum(dim = -1)
        T_pose = T_pose * 1000
        results = {"verts3d": verts, "joints3d": joints, "shape": shape, "pose": pose, "T_pose": T_pose}
        return results

    def forward(self, input, heatmap,xyc,attention):
        #xyc:(Batch,42,3) left,right, x,y,confidence 
        features, intermediates = self.base_net(input,heatmap)

        right_attention_map=attention[:,0,None,:,:]
        left_attention_map=attention[:,1,None,:,:]
        right_features=intermediates['res_layer4']*nn.functional.interpolate(right_attention_map,size=[8,8])
        left_features=intermediates['res_layer4']*nn.functional.interpolate(left_attention_map,size=[8,8])
        right_features=right_features.mean([2,3])
        left_features=left_features.mean([2,3])
        
        left_mano_para_list = {"verts3d": [], "joints3d": [], "shape": [], "pose": [], "T_pose": []}
        right_mano_para_list = {"verts3d": [], "joints3d": [], "shape": [], "pose": [], "T_pose": []}

        pose,shape=self.left_mano_branch(left_features)
        left_mano_para=self.pose_shape2verts_joints(pose,shape,'l')
        pose,shape=self.right_mano_branch(right_features)
        right_mano_para=self.pose_shape2verts_joints(pose,shape,'r')
        for k,v in left_mano_para.items():
            left_mano_para_list[k].append(v)
        for k,v in right_mano_para.items():
            right_mano_para_list[k].append(v)
        
        trans = self.trans_branch(features)


        if self.MANO_cascaded:
            for i in range(self.cascaded_num):
                lower_left_mano_para=left_mano_para
                lower_right_mano_para=right_mano_para

                feature_name='res_layer%d'%(4-i)
                left_heatmap=self.register_heatmap(xyc[:,21:,:],lower_left_mano_para['joints3d'],\
                                                    64,intermediates[feature_name].shape[2])
                right_heatmap=self.register_heatmap(xyc[:,:21,:],lower_right_mano_para['joints3d'],\
                                                    64,intermediates[feature_name].shape[2])
                # import cv2
                # cv2.imwrite("%s_left.png"%feature_name,(left_heatmap[0,:,:,None]*255).cpu().numpy().astype('uint8'))
                # cv2.imwrite("%s_right.png"%feature_name,(right_heatmap[0,:,:,None]*255).cpu().numpy().astype('uint8'))
                right_features=intermediates[feature_name]*right_heatmap[:,None,:,:]
                left_features=intermediates[feature_name]*left_heatmap[:,None,:,:]
                right_features=self.cascaded_right_feature_extractor_list[i](right_features)
                left_features=self.cascaded_left_feature_extractor_list[i](left_features)
                right_features=right_features.mean([2,3])
                left_features=left_features.mean([2,3])

                left_pose,left_shape=self.cascaded_left_mano_branch_list[i](torch.cat([lower_right_mano_para["shape"],lower_right_mano_para["pose"],lower_left_mano_para["shape"],lower_left_mano_para["pose"],left_features,trans,],dim=1))
                right_pose,right_shape=self.cascaded_right_mano_branch_list[i](torch.cat([lower_left_mano_para["shape"],lower_left_mano_para["pose"],lower_right_mano_para["shape"],lower_right_mano_para["pose"],right_features,trans,],dim=1))
                
                left_mano_para=self.pose_shape2verts_joints(left_pose,left_shape,'l')
                right_mano_para=self.pose_shape2verts_joints(right_pose,right_shape,'r')
                for k,v in left_mano_para.items():
                    left_mano_para_list[k].append(v)
                for k,v in right_mano_para.items():
                    right_mano_para_list[k].append(v)
        for k in left_mano_para_list.keys():
            left_mano_para_list[k]=torch.stack(left_mano_para_list[k],dim=1)
        for k in right_mano_para_list.keys():
            right_mano_para_list[k]=torch.stack(right_mano_para_list[k],dim=1)
                
        return right_mano_para_list,left_mano_para_list,trans
