import torch
import torch.nn as nn
from nets.meshreg import MeshRegNet
from nets.module import BackboneNet, PoseNet
from config import cfg
from nets.loss import calculate_loss
import torchvision.transforms as transforms

class InferenceModel(nn.Module):
    def __init__(self, backbone_net, pose_net):
        super(InferenceModel, self).__init__()

        # modules
        self.backbone_net = backbone_net
        self.pose_net = pose_net
          
    def forward(self, inputs):
        input_img = inputs
        img_feat = self.backbone_net(input_img)
        joint_heatmap_out, rel_root_depth_out, hand_type = self.pose_net(img_feat)
        return joint_heatmap_out,rel_root_depth_out, hand_type

class Model(nn.Module):
    def __init__(self,resnet_version,mano_use_pca,mano_neurons,cascaded_num):
        super(Model, self).__init__()
        self.joint_num = 21
        self.depth_downsample_factor=4
        self.spatial_downsample_factor=2
        self.mesh_reg=MeshRegNet(input_size=3, resnet_version=resnet_version, mano_use_pca=mano_use_pca, mano_neurons=mano_neurons,\
                                    addition_channels=42*64//self.depth_downsample_factor, cascaded_num=cascaded_num)
        backbone_net = BackboneNet()
        pose_net = PoseNet(self.joint_num)
        self.heatmap_predictor = InferenceModel(backbone_net, pose_net)

        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        
    def forward(self, inputs, targets, meta_info, mode):
        heatmap,_,_=self.heatmap_predictor(inputs['img'])
        downsampled_heatmap=heatmap[:,:,::self.depth_downsample_factor,::self.spatial_downsample_factor,::self.spatial_downsample_factor]
        B,K,D,H,W=downsampled_heatmap.shape
        
        val_z, idx_z = torch.max(heatmap,2)
        val_zy, idx_zy = torch.max(val_z,2)
        val_zyx, joint_x = torch.max(val_zy,2)
        joint_x = joint_x[:,:,None]
        joint_y = torch.gather(idx_zy, 2, joint_x)
        xyc=torch.cat((joint_x, joint_y, val_zyx[:,:,None]),2).float()

        attention_map=heatmap.reshape(heatmap.shape[0],-1,heatmap.shape[3],heatmap.shape[4])
        right_attention_map,_=attention_map[:,:(self.joint_num*64),:,:].max(dim=1,keepdim=True)
        left_attention_map,_=attention_map[:,(self.joint_num*64):,:,:].max(dim=1,keepdim=True)
        attention=torch.cat([right_attention_map,left_attention_map],dim=1)
        input_img = self.normalize_img(inputs['img']).detach()
        right_mano_para_list,left_mano_para_list,trans=self.mesh_reg(input_img, downsampled_heatmap.reshape(B,K*D,H,W),xyc,attention)
        if mode == 'train':
            outputs = {
                        "joints3d":torch.cat([right_mano_para_list["joints3d"],left_mano_para_list["joints3d"]],dim=2),
                        "shape":torch.cat([right_mano_para_list["shape"],left_mano_para_list["shape"]],dim=2),
                        "pose":torch.cat([right_mano_para_list["pose"],left_mano_para_list["pose"]],dim=2),
                        "T_pose":torch.cat([right_mano_para_list["T_pose"],left_mano_para_list["T_pose"]],dim=2),
                        "trans":trans
                        }
            loss=calculate_loss(outputs,targets,meta_info)
            return loss
        elif mode == 'test':
            outputs = {
                "joints3d":torch.cat([right_mano_para_list["joints3d"][:,-1,:,:],left_mano_para_list["joints3d"][:,-1,:,:]],dim=1),
                "verts3d":torch.cat([right_mano_para_list["verts3d"][:,-1,:,:],left_mano_para_list["verts3d"][:,-1,:,:]],dim=1),
                "trans":trans,
            }
            return outputs

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.01)
        if not m.bias is None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        if not m.bias is None:
            nn.init.constant_(m.bias, 0)

def get_model(mode):
    model = Model(resnet_version=50,
                    mano_neurons=[512, 512, 512, 512],
                    mano_use_pca=False,
                    cascaded_num=3)
    
    if mode == 'train':
        model.apply(init_weights)
        ckpt=torch.load(cfg.InterNet_model_dir)
        param_dict={k.split('.',1)[1]:v for k,v in ckpt['network'].items()}
        model.heatmap_predictor.load_state_dict(param_dict)
        model.mesh_reg.base_net.init_weights("resnet50")

    return model
