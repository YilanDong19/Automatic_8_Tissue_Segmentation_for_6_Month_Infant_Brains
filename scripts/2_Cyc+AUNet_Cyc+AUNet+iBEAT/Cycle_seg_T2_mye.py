import os
import numpy as np
import glob
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import DataLoader, CacheDataset
from monai.losses import DiceLoss, LocalNormalizedCrossCorrelationLoss, DiceCELoss
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    LoadImaged,
    ConvertToMultiChannelBasedOnBratsClassesd,
    HistogramNormalized,
    RandAffined,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    Spacingd,
    EnsureChannelFirstd,
    ScaleIntensityd,
)
from monai.utils import set_determinism
from monai.data import NibabelReader
import torch
import matplotlib.pyplot as plt
import functools
from torch.nn import init
import torch.nn as nn
import copy
import os
import shutil
import itertools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from argparse import ArgumentParser
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance, DiceMetric
import csv

print_config()



class ConvertToMultiChannel_dhcp(MapTransform):

    # 0 background
    # 1 CSF
    # 2 GM
    # 3 WM
    # 4 bone
    # 5 Ventricle
    # 6 Cerebellum
    # 7 Basal Ganglia
    # 8 Brainstem
    # 9 Hippocampus / Amygdala

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(d[key] == 0) # Background
            result.append(d[key] == 1) # CSF
            result.append(d[key] == 2) # GM
            result.append(d[key] == 3) # WM
            # result.append(d[key] == 4) # bone
            result.append(d[key] == 5) # Ventricle
            result.append(d[key] == 6) # Cerebellum
            result.append(d[key] == 7) # Basal Ganglia
            result.append(d[key] == 8) # Brainstem
            result.append(d[key] == 9) # Hippocampus / Amygdala
            d[key] = torch.stack(result, axis=0).float()
        return d



class ConvertToMultiChannel_6month(MapTransform):

    # 0 Background
    # 1 CSF
    # 2 Gray matter
    # 3 White matter
    # 4 Ventricle
    # 5 Cerebellum
    # 6 Basal Ganglia
    # 7 Brainstem
    # 8 Hippocampus / Amygdala
    # 9 Myelination


    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(d[key] == 0) # Background
            result.append(d[key] == 1) # CSF
            result.append(d[key] == 2) # GM
            result.append(torch.logical_or(d[key] == 3, d[key] == 9)) # WM
            result.append(d[key] == 4) # Ventricle
            result.append(d[key] == 5) # Cerebellum
            result.append(d[key] == 6) # Basal Ganglia
            result.append(d[key] == 7) # Brainstem
            result.append(d[key] == 8) # Hippocampus / Amygdala
            # result.append(d[key] == 9) # Myelination
            d[key] = torch.stack(result, axis=0).float()
        return d
        

class ConvertToMultiChannel_com(MapTransform):

    # 0 Background
    # 1 CSF
    # 2 Gray matter
    # 3 White matter
    # 4 Ventricle
    # 5 Cerebellum
    # 6 Basal Ganglia
    # 7 Brainstem
    # 8 Hippocampus / Amygdala
    # 9 Myelination


    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(d[key] == 0) # Background
            result.append(d[key] == 1) # CSF
            result.append(d[key] == 2) # GM
            result.append(d[key] == 3) # WM
            result.append(d[key] == 4) # Ventricle
            result.append(d[key] == 5) # Cerebellum
            result.append(d[key] == 6) # Basal Ganglia
            result.append(d[key] == 7) # Brainstem
            result.append(d[key] == 8) # Hippocampus / Amygdala
            # result.append(d[key] == 9) # Myelination
            d[key] = torch.stack(result, axis=0).float()
        return d


class ConvertToMultiChannel_mye(MapTransform):

    # 0 Background
    # 1 CSF
    # 2 Gray matter
    # 3 White matter
    # 4 Ventricle
    # 5 Cerebellum
    # 6 Basal Ganglia
    # 7 Brainstem
    # 8 Hippocampus / Amygdala
    # 9 Myelination


    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(d[key] == 0) # Background
            result.append(d[key] == 1) # CSF
            result.append(d[key] == 2) # GM
            result.append(d[key] == 3) # WM
            result.append(d[key] == 4) # Ventricle
            result.append(d[key] == 5) # Cerebellum
            result.append(d[key] == 6) # Basal Ganglia
            result.append(d[key] == 7) # Brainstem
            result.append(d[key] == 8) # Hippocampus / Amygdala
            result.append(d[key] == 9) # Myelination
            d[key] = torch.stack(result, axis=0).float()
        return d







class ConvertToMultiChannel_ibeat(MapTransform):

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(d[key] == 0) # Background
            result.append(d[key] == 1) # CSF
            result.append(d[key] == 2) # GM
            result.append(d[key] == 3) # WM
            d[key] = torch.stack(result, axis=0).float()
        return d

################################### setting ###################################
################################### setting ###################################
################################### setting ###################################

root_dir = '/nfs/home/ydong/code/pippi/long_cy/save_model'
train_path_dhcp = '/nfs/home/ydong/dataset/DHCP_long/va_train/'
train_path_6_month = '/nfs/home/ydong/dataset/6_month_long/va_train/'
val_path_6month = '/nfs/home/ydong/dataset/6_month_long/va_before_val_5/'
test_path_6_month = '/nfs/home/ydong/dataset/6_month_long/va_test_10_subjects/'

val_result_path = root_dir + '/val_result'
test_result_path = root_dir + '/test_result'
logger_path = root_dir + '/log'

if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(val_result_path):
    os.makedirs(val_result_path)
if not os.path.exists(test_result_path):
    os.makedirs(test_result_path)
if not os.path.exists(logger_path):
    os.makedirs(logger_path)

################################### DHCP train ###################################
################################### DHCP train ###################################
################################### DHCP train ###################################
set_determinism(seed=0)
# train_images_T1 = sorted(glob.glob(os.path.join(train_path, 'sub-*_ses-*_desc-T1.nii.gz')))
train_images_T2 = sorted(glob.glob(os.path.join(train_path_dhcp, 'sub-*_ses-*_desc-T2.nii.gz')))
train_images_label = sorted(glob.glob(os.path.join(train_path_dhcp, 'sub-*_ses-*_desc-8label.nii.gz')))

train_dicts_dhcp = [
    {"image_T2": image_T2, "image_label": image_label, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:-10]}
    for image_T2, image_label in zip(train_images_T2, train_images_label)
]
train_transform_dhcp = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label"]),
        ConvertToMultiChannel_dhcp(keys="image_label"),
        EnsureChannelFirstd(keys=["image_T2"]),
        # ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),
        RandAffined(keys=["image_T2","image_label"], 
              mode=("bilinear",'nearest'), 
              prob=0.5,
              translate_range=(10, 10, 10),
              rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
              scale_range=(0.1, 0.1, 0.1),
              padding_mode="zeros",
          ),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),
        # HistogramNormalized(keys=["image_T2"],min=-1,max=1),

    ]
)


################################### 6month val ###################################
################################### 6month val ###################################
################################### 6month val ###################################
# train_images_T1 = sorted(glob.glob(os.path.join(val_path_6month, 'sub-*_ses-*_desc-T1.nii.gz')))
val_images_T2 = sorted(glob.glob(os.path.join(val_path_6month, '*_T2.nii.gz')))
val_images_label = sorted(glob.glob(os.path.join(val_path_6month, '*_fixed_basalganglia_prediction.nii.gz')))

val_dicts_6month = [
    {"image_T2": image_T2, "image_label": image_label, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:12]}
    for image_T2, image_label in zip(val_images_T2, val_images_label)
]
val_transform_6month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        EnsureChannelFirstd(keys=["image_T2"]),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),

    ]
)



################################### 6_month train ###################################
################################### 6_month train ###################################
################################### 6_month train ###################################

# train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T1.nii.gz')))
train_images_T2 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2.nii.gz')))

train_dicts_6_month = [
    {"image_T2": image_T2, "affine": nib.load(image_T2[0]).affine, "image_name": os.path.basename(image_T2[0])[0:12]}
    for image_T2 in zip(train_images_T2)
]

train_transform_6_month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2"]),
        EnsureChannelFirstd(keys=["image_T2"]),
        RandAffined(keys=["image_T2"], 
              mode=("bilinear"), 
              prob=0.5,
              translate_range=(10, 10, 10),
              rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
              scale_range=(0.1, 0.1, 0.1),
              padding_mode="zeros",
          ),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),
        # HistogramNormalized(keys=["image_T2"],min=-1,max=1),

    ]
)



################################### 6_month test ###################################
################################### 6_month test ###################################
################################### 6_month test ###################################

# train_images_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T1.nii.gz')))
test_images_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2.nii.gz')))
test_images_label_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_VK.nii.gz')))
test_images_label_ibeat = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2_iBEAT_outputs.nii.gz')))
# train_images_label_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*T2*-tissue.nii.gz')))

test_dicts_6_month = [
    {"image_T2": image_T2, "image_label": image_label, "image_name": os.path.basename(image_T2)[0:12], "affine": nib.load(image_T2).affine, "ibeat": image_ibeat}
    for image_T2,image_label,image_ibeat in zip(test_images_T2,test_images_label_T1,test_images_label_ibeat)
]

test_transform_6_month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label","ibeat"]),
        EnsureChannelFirstd(keys=["image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        ConvertToMultiChannel_ibeat(keys="ibeat"),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),

    ]
)

#############################################################################
#############################################################################
#############################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('Network initialized with weights sampled from N(0,0.02).')
    net.apply(init_func)


def init_network(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm3d, bias = False):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.LeakyReLU(0.2,True))

def conv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm3d, bias = False):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))

'''
def dconv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0, output_padding=0,
                                 norm_layer = nn.BatchNorm3d, bias = False):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))
'''
def dconv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0, output_padding=0,
                                 norm_layer = nn.BatchNorm3d, bias = False):
    return nn.Sequential(
        # nn.ConvTranspose3d(in_dim, out_dim, kernel_size, stride,
        #                    padding, output_padding, bias = bias),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad3d(1),
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad3d(1),
                     conv_norm_relu(dim, dim, kernel_size=3, 
                     norm_layer= norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad3d(1),
                      nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


#############################################################################
#############################################################################
#############################################################################

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, 
                                innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            # upconv = nn.ConvTranspose3d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            upsample = nn.Upsample(scale_factor=2)
            upconv = nn.Conv3d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [nn.ReLU(True), upsample, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # upconv = nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upsample = nn.Upsample(scale_factor=2)
            upconv = nn.Conv3d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upsample, upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            # upconv = nn.ConvTranspose3d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upsample = nn.Upsample(scale_factor=2)
            upconv = nn.Conv3d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upsample, upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, input):
        return self.unet_model(input)



class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=True, num_blocks=6):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        res_model = [nn.ReflectionPad3d(3),
                    conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]

        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]

        res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 1, 0, 1, norm_layer=norm_layer, bias=use_bias),
                      dconv_norm_relu(ngf * 2, ngf * 1, 3, 1, 0, 1, norm_layer=norm_layer, bias=use_bias),
                      nn.ReflectionPad3d(3),
                      nn.Conv3d(ngf, output_nc, 7),
                      nn.Tanh()]
        self.res_model = nn.Sequential(*res_model)

    def forward(self, x):
        return self.res_model(x)




def define_Gen(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, gpu_ids=[0]):
    gen_net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=9)
    elif netG == 'resnet_6blocks':
        gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=6)
    elif netG == 'unet_128':
        gen_net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        gen_net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_32':
        gen_net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_network(gen_net, gpu_ids)
   
#############################################################################
#############################################################################
#############################################################################

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        dis_model = [nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2,
                                               norm_layer= norm_layer, padding=1, bias=use_bias)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1,
                                               norm_layer= norm_layer, padding=1, bias=use_bias)]
        dis_model += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_bias=False):
        super(PixelDiscriminator, self).__init__()
        dis_model = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)



def define_Dis(input_nc, ndf, netD, n_layers_D=3, norm='batch', gpu_ids=[0]):
    dis_net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
        use_bias = norm_layer == nn.InstanceNorm3d

    if netD == 'n_layers':
        dis_net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_bias=use_bias)
    elif netD == 'pixel':
        dis_net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_bias=use_bias)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    return init_network(dis_net, gpu_ids)

   
#############################################################################
#############################################################################
#############################################################################

# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


def save_best_model(model, epoch, filename="long_cy_apply_6month.pt", best_acc=0, dir_add=''):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)





# To save the checkpoint 
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items

class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)

def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i=i+1
    print('-----------------------------------------------')


                                               
def record_metric(outputs_labels, path):                                                                                                                    

    
    names = ['CSF','GM','WM','Ventricle','Cerebellum','Basal Ganglia','Brainstem','Hippocampus/Amygdala']   
    post_pred = AsDiscrete(to_onehot=outputs_labels[0][1].shape[1])
    subject_number = len(outputs_labels)
    dice_metric = DiceMetric(include_background=False)
    
    average_dice = 0
    average_hausdoff = 0
    average_surface = 0
    
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['Brain tissue','Dice','Hausdoff95','Average surface']
        for x in range(subject_number):
            csv_write.writerow(csv_head)
            y_pred, y_label = outputs_labels[x]
            y_pred = torch.from_numpy(y_pred)
            y_label = torch.from_numpy(y_label)
            y_pred=torch.argmax(y_pred,dim=1)
            y_pred = post_pred(y_pred).unsqueeze(0)
    
            output_dice = dice_metric(y_pred,y_label)
            output_hausdorff = compute_hausdorff_distance(y_pred,y_label,include_background=False,percentile=95)
            output_surface = compute_average_surface_distance(y_pred,y_label,include_background=False,symmetric=True)
    
            output_dice = output_dice[0].numpy()
            mean_dice = np.mean(output_dice)
            output_hausdorff = output_hausdorff[0].numpy()
            mean_hausdoff = np.mean(output_hausdorff)
            output_surface = output_surface[0].numpy()
            mean_surface = np.mean(output_surface)
            
            for i in range(len(names)):
                data = (names[i],output_dice[i],output_hausdorff[i],output_surface[i])
                csv_write.writerow(data)
            data = ('Average',mean_dice,mean_hausdoff,mean_surface)
            csv_write.writerow(data)
            data=([])
            csv_write.writerow(data)
            
        
            average_dice = average_dice + mean_dice
            average_hausdoff = average_hausdoff + mean_hausdoff
            average_surface = average_surface + mean_surface
        
            if x == 0:
                detail_output_dice = np.copy(output_dice)
                detail_output_hausdoff = np.copy(output_hausdorff)
                detail_output_surface = np.copy(output_surface)
            else:
                detail_output_dice = detail_output_dice + output_dice
                detail_output_hausdoff = detail_output_hausdoff + output_hausdorff            
                detail_output_surface = detail_output_surface + output_surface
    
        average_dice = average_dice / subject_number
        average_hausdoff = average_hausdoff / subject_number
        average_surface = average_surface / subject_number
    
        detail_output_dice = detail_output_dice / subject_number
        detail_output_hausdoff = detail_output_hausdoff / subject_number
        detail_output_surface = detail_output_surface / subject_number
    
        
    
        data = ([])
        csv_write.writerow(data)
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)
        for i in range(len(names)):
            data = (names[i],detail_output_dice[i],detail_output_hausdoff[i],detail_output_surface[i])
            csv_write.writerow(data)
        
        data = ('Average',average_dice,average_hausdoff,average_surface)
        csv_write.writerow(data)
    
     

#############################################################################
#############################################################################
#############################################################################

class cycleGAN(object):
    def __init__(self,args):

        # Define the network 
        #####################################################
        self.Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=1, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=1, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Seg=AttentionUnet(
                               spatial_dims=3,
                               in_channels=1,
                               out_channels=9,
                               channels=(32, 64, 128, 256, 512),
                               strides=(2, 2, 2, 2),
                               ).to(device)
        

        print_networks([self.Gab,self.Gba,self.Da,self.Db,self.Seg], ['Gab','Gba','Da','Db','Seg'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        # self.LNCC = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=args.LNCC_kernel_size)
        self.dice_loss = DiceLoss(to_onehot_y=False,include_background=True,softmax=True)
        self.dicece_loss = DiceCELoss(to_onehot_y=False,include_background=True,softmax=True)

        # Optimizers
        #####################################################
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(),self.Gba.parameters()), lr=args.lr_g, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr_d, betas=(0.5, 0.999))
        self.Seg_optimizer = torch.optim.Adam(self.Seg.parameters(), lr=args.lr_seg, betas=(0.5, 0.999))
        

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.Seg_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.Seg_optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
        
        
        
        #test_ds_6_month = CacheDataset(data=test_dicts_6_month_test, transform=test_transform_6_month)
        #self.test_loader_6_month = DataLoader(test_ds_6_month, batch_size=args.batch_size, num_workers=0,pin_memory=True)
        test_ds_6_month = CacheDataset(data=test_dicts_6_month, transform=test_transform_6_month)
        self.test_loader_6_month = DataLoader(test_ds_6_month, batch_size=args.batch_size, num_workers=0,pin_memory=True)
        
        

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.logger=SummaryWriter(log_dir=logger_path)
            self.start_epoch = ckpt['epoch']
            self.total_number = ckpt['total_number']
            self.best_val_accuracy = ckpt['best_val_accuracy']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.Seg.load_state_dict(ckpt['Seg'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
            self.Seg_optimizer.load_state_dict(ckpt['Seg_optimizer'])
            print('The start epoch is: ', self.start_epoch)
            
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 1
            self.logger=SummaryWriter(log_dir=logger_path)
            self.total_number=0
            self.best_val_accuracy = -1
            print('The start epoch is: ', self.start_epoch)



    def train(self,args):


        #a_fake_sample = Sample_from_Pool()
        #b_fake_sample = Sample_from_Pool()
        
        # Data
        #####################################################     
        train_ds_dhcp = CacheDataset(data=train_dicts_dhcp, transform=train_transform_dhcp)
        self.train_loader_dhcp = DataLoader(train_ds_dhcp, batch_size=args.batch_size,shuffle=True, num_workers=0,pin_memory=True)
        
        train_ds_6month = CacheDataset(data=train_dicts_6_month, transform=train_transform_6_month)
        self.train_loader_6_month = DataLoader(train_ds_6month, batch_size=args.batch_size,shuffle=True, num_workers=0,pin_memory=True)
        
        val_ds_6month = CacheDataset(data=val_dicts_6month, transform=val_transform_6month)
        self.val_loader_6month = DataLoader(val_ds_6month, batch_size=args.batch_size, num_workers=0,pin_memory=True)
        
        
        
        
        Gen_a_loss_list = []
        Gen_b_loss_list = []
        Dis_a_r_loss_list=[]
        Dis_a_f_loss_list=[]
        Dis_b_r_loss_list=[]
        Dis_b_f_loss_list=[]

        
        Gen_loss_list=[]
        Dis_loss_list=[]
        test_seg_loss = {}


        
        

        for epoch in range(self.start_epoch, args.epochs+1):
            
            self.Seg.train()
            lr_gen = self.g_optimizer.param_groups[0]['lr']
            lr_dis = self.d_optimizer.param_groups[0]['lr']
            print('learning rate of Gen = %.7f' % lr_gen)
            print('learning rate of Dis = %.7f' % lr_dis)
            
            log_gen = 0
            log_dis = 0
            log_a_gen = 0
            log_b_gen = 0
            log_a_dis_real = 0
            log_a_dis_fake = 0
            log_b_dis_real = 0
            log_b_dis_fake = 0
            log_seg = 0


           
            # for idx, batch_data in enumerate(train_loader):
            for idx, (dhcp_real, month_6_real) in enumerate(zip(self.train_loader_dhcp, self.train_loader_6_month)):
                self.total_number = self.total_number + 1
                
                dhcp_real_T2 = dhcp_real["image_T2"]
                a_real = dhcp_real_T2
                dhcp_label = dhcp_real["image_label"]

                
                month_6_real_T2 = month_6_real["image_T2"]               
                b_real = month_6_real_T2



                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)
                self.g_optimizer.zero_grad()

                a_real = Variable(a_real)
                b_real = Variable(b_real)
                a_real, b_real, dhcp_label = cuda([a_real, b_real, dhcp_label])

                # Forward pass through generators
                ##################################################
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)
                
                # Segmentation losses
                ###################################################
                Seg_out = self.Seg(b_fake)
                seg_loss = self.dicece_loss(Seg_out, dhcp_label)


                
                
                ########## my edition ###########
                if idx==2 :
                    a_fake_plot = a_fake.detach().cpu().numpy()
                    a_real_plot = a_real.detach().cpu().numpy()
                    b_fake_plot = b_fake.detach().cpu().numpy()
                    b_real_plot = b_real.detach().cpu().numpy()
                    a_recon_plot = a_recon.detach().cpu().numpy()
                    b_recon_plot = b_recon.detach().cpu().numpy()
                    
                    
                    Seg_out_plot = Seg_out.detach().cpu().numpy()
                    dhcp_label_plot = dhcp_label.detach().cpu().numpy()
                    Seg_out_argmax = np.argmax(Seg_out_plot, axis=1)
                    
                    
                    dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,0,:,:,:]==1]=0
                    dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,1,:,:,:]==1]=1
                    dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,2,:,:,:]==1]=2
                    dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,3,:,:,:]==1]=3
                    dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,4,:,:,:]==1]=4
                    dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,5,:,:,:]==1]=5
                    dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,6,:,:,:]==1]=6
                    dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,7,:,:,:]==1]=7
                    dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,8,:,:,:]==1]=8
                    # dhcp_label_plot[:,0,:,:,:][dhcp_label_plot[:,9,:,:,:]==1]=9
                    
                    
                    
                    fig2 = plt.figure("2", (30, 36))
                    ax2 = fig2.subplots(6,5)
                    
                    
                    ax2[0,0].title.set_text('DHCP_real_T2')
                    ax2[0,0].imshow(np.rot90(a_real_plot[0, 0, :, :,round(a_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[0,1].title.set_text('6_MONTH_fake_T2')
                    ax2[0,1].imshow(np.rot90(b_fake_plot[0, 0, :, :, round(b_fake_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[0,2].title.set_text('DHCP_recon_T2')
                    ax2[0,2].imshow(np.rot90(a_recon_plot[0 ,0, :, :, round(a_recon_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[0,3].title.set_text('DHCP_label')
                    ax2[0,3].imshow(np.rot90(dhcp_label_plot[0 ,0, :, :, round(dhcp_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)   
                    ax2[0,4].title.set_text('Prediction')
                    ax2[0,4].imshow(np.rot90(Seg_out_argmax[0, :, :, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)                                     
                    ax2[1,0].title.set_text('6_month_real_T2')
                    ax2[1,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[1,1].title.set_text('DHCP_fake_T2')
                    ax2[1,1].imshow(np.rot90(a_fake_plot[0 , 0, :,:, round(a_fake_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[1,2].title.set_text('6_MONTH_recon_T2')
                    ax2[1,2].imshow(np.rot90(b_recon_plot[0 , 0, :,:, round(b_recon_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)

                    

                    ax2[2,0].title.set_text('DHCP_real_T2')
                    ax2[2,0].imshow(np.rot90(a_real_plot[0, 0, :, round(a_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[2,1].title.set_text('6_MONTH_fake_T2')
                    ax2[2,1].imshow(np.rot90(b_fake_plot[0, 0, :, round(b_fake_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[2,2].title.set_text('DHCP_recon_T2')
                    ax2[2,2].imshow(np.rot90(a_recon_plot[0, 0, :, round(a_recon_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)  
                    ax2[2,3].title.set_text('DHCP_label')
                    ax2[2,3].imshow(np.rot90(dhcp_label_plot[0, 0, :, round(dhcp_label_plot.shape[3]/2),:]), cmap="jet",vmin=0,vmax=9)   
                    ax2[2,4].title.set_text('Prediction')
                    ax2[2,4].imshow(np.rot90(Seg_out_argmax[0,:, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)                                 
                    ax2[3,0].title.set_text('6_month_real_T2')
                    ax2[3,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[3,1].title.set_text('DHCP_fake_T2')
                    ax2[3,1].imshow(np.rot90(a_fake_plot[0, 0, :, round(a_fake_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[3,2].title.set_text('6_MONTH_recon_T2')
                    ax2[3,2].imshow(np.rot90(b_recon_plot[0, 0, :, round(b_recon_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)

                    
                    
                    ax2[4,0].title.set_text('DHCP_real_T2')
                    ax2[4,0].imshow(np.rot90(a_real_plot[0, 0, round(a_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[4,1].title.set_text('6_MONTH_fake_T2')
                    ax2[4,1].imshow(np.rot90(b_fake_plot[0, 0, round(b_fake_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[4,2].title.set_text('DHCP_recon_T2')
                    ax2[4,2].imshow(np.rot90(a_recon_plot[0, 0, round(a_recon_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2) 
                    ax2[4,3].title.set_text('DHCP_label')
                    ax2[4,3].imshow(np.rot90(dhcp_label_plot[0, 0, round(dhcp_label_plot.shape[2]/2), :,:]), cmap="jet",vmin=0,vmax=9)   
                    ax2[4,4].title.set_text('Prediction')
                    ax2[4,4].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)                                       
                    ax2[5,0].title.set_text('6_month_real_T2')
                    ax2[5,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[5,1].title.set_text('DHCP_fake_T2')
                    ax2[5,1].imshow(np.rot90(a_fake_plot[0, 0, round(a_fake_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax2[5,2].title.set_text('6_MONTH_recon_T2')
                    ax2[5,2].imshow(np.rot90(b_recon_plot[0, 0, round(b_recon_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    
                    fig2.savefig(root_dir + '/'+ str(epoch) +'.png')
                    


               

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef
                
                # LNCC losses
                ###################################################
                # LNCC_gen_loss_a1 = self.LNCC(b_fake,a_real)
                # LNCC_gen_loss_a2 = self.LNCC(b_fake,a_recon)     
                # LNCC_gen_loss_b1 = self.LNCC(a_fake,b_real) 
                # LNCC_gen_loss_b2 = self.LNCC(a_fake,b_recon)      
                # LNCC_gen_loss_total = (LNCC_gen_loss_a1 + LNCC_gen_loss_a2 + LNCC_gen_loss_b1 + LNCC_gen_loss_b2) * 1

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                real_label = cuda(Variable(torch.ones(a_fake_dis.size())))

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # Total generators losses
                ###################################################
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss + seg_loss

                # Update generators
                ###################################################
                gen_loss.backward()
                self.g_optimizer.step()


                # Discriminator Computations
                #################################################

                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                #################################################
                a_fake = Variable(a_fake)
                b_fake = Variable(b_fake)
                a_fake, b_fake = cuda([a_fake, b_fake])

                # Forward pass through discriminators
                ################################################# 
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label = cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = cuda(Variable(torch.zeros(a_fake_dis.size())))

                # Discriminator losses
                ##################################################
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                # Total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

                # Update discriminators
                ##################################################
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()
                
                
                
                # Segmentation losses
                ###################################################               
                set_grad([self.Seg], True)
                self.Seg_optimizer.zero_grad()
                Seg_out = self.Seg(b_fake)
                seg_loss = self.dicece_loss(Seg_out, dhcp_label)
                background_loss = self.dice_loss(Seg_out[:,0,:,:,:], dhcp_label[:,0,:,:,:])
                csf_loss = self.dice_loss(Seg_out[:,1,:,:,:], dhcp_label[:,1,:,:,:])
                WM_loss = self.dice_loss(Seg_out[:,2,:,:,:], dhcp_label[:,2,:,:,:])
                GM_loss = self.dice_loss(Seg_out[:,3,:,:,:], dhcp_label[:,3,:,:,:])
                csf_dice= 1-csf_loss
                WM_dice= 1-WM_loss
                GM_dice= 1-GM_loss
                background_dice= 1-background_loss
                seg_dice= 1-seg_loss
                
                seg_loss.backward()
                self.Seg_optimizer.step()
                
                
                
                print("Epoch: (%3d) (%5d/%5d) | Gen_a Loss:%.2e | Gen_b loss:%.2e | Dis_a_r loss:%.2e | Dis_a_f loss:%.2e | Dis_b_r loss:%.2e | Dis_b_f loss:%.2e" % 
                                            (epoch, idx + 1, min(len(self.train_loader_dhcp), len(self.train_loader_6_month)),
                                                            a_gen_loss,b_gen_loss,a_dis_real_loss,a_dis_fake_loss,b_dis_real_loss,b_dis_fake_loss))
                                                            
                with open(root_dir+'/val_result/'+'/logs.txt',"a") as file:
                    file.write('Epoch: ')
                    file.write(str(epoch))
                    file.write(', ')
                    file.write('( ')
                    file.write(str(idx))
                    file.write(' / ')
                    file.write(str(len(self.train_loader_6_month)))
                    file.write(' )\n ')
                
                self.logger.add_scalars('Gen_Dis', {'Gen':gen_loss.item(),
                                                    'Dis':(a_dis_loss+b_dis_loss).item()},
                                                     self.total_number)
                self.logger.add_scalar('Gen_loss', gen_loss.item(), self.total_number)  
                self.logger.add_scalar('Dis_loss', (a_dis_loss+b_dis_loss).item(), self.total_number)   
                self.logger.add_scalars('Gen_dhcp_6month', {'Fake_dhcp':a_gen_loss.item(),
                                                            'Fake_6month':b_gen_loss.item()},
                                                             self.total_number)   
                self.logger.add_scalars('Dis_dhcp', {'Real_dhcp':a_dis_real_loss.item(),
                                                     'Fake_dhcp':a_dis_fake_loss.item()},
                                                      self.total_number)       
                self.logger.add_scalars('Dis_6month', {'Real_6monyh':b_dis_real_loss.item(),
                                                       'Fake_6month':b_dis_fake_loss.item()},
                                                       self.total_number)
                self.logger.add_scalar('Seg_loss', seg_loss.item(), self.total_number) 
                
                Gen_loss_list.append(gen_loss.item())
                Dis_loss_list.append((a_dis_loss+b_dis_loss).item())                
                Gen_a_loss_list.append(a_gen_loss.item())
                Gen_b_loss_list.append(b_gen_loss.item())
                Dis_a_r_loss_list.append(a_dis_real_loss.item())
                Dis_a_f_loss_list.append(a_dis_fake_loss.item())
                Dis_b_r_loss_list.append(b_dis_real_loss.item())
                Dis_b_f_loss_list.append(b_dis_fake_loss.item())
                
                
                log_gen = log_gen + gen_loss.item()
                log_dis = log_dis + (a_dis_loss+b_dis_loss).item()
                log_a_gen = log_a_gen + a_gen_loss.item()
                log_b_gen = log_b_gen + b_gen_loss.item()
                log_a_dis_real = log_a_dis_real + a_dis_real_loss.item()
                log_a_dis_fake = log_a_dis_fake + a_dis_fake_loss.item()
                log_b_dis_real = log_b_dis_real + b_dis_real_loss.item()
                log_b_dis_fake = log_b_dis_fake + b_dis_fake_loss.item()
                log_seg = log_seg + seg_loss.item()

            log_gen = log_gen / len(self.train_loader_dhcp)
            log_dis = log_dis / len(self.train_loader_dhcp)
            log_a_gen = log_a_gen / len(self.train_loader_dhcp)
            log_b_gen = log_b_gen / len(self.train_loader_dhcp)
            log_a_dis_real = log_a_dis_real / len(self.train_loader_dhcp)
            log_a_dis_fake = log_a_dis_fake / len(self.train_loader_dhcp)
            log_b_dis_real = log_b_dis_real / len(self.train_loader_dhcp)
            log_b_dis_fake = log_b_dis_fake / len(self.train_loader_dhcp)
            log_seg = log_seg / len(self.train_loader_dhcp)
            
            self.logger.add_scalars('Gen_Dis_epoch', {'Gen':log_gen,
                                                    'Dis':log_dis},
                                                     epoch)
            self.logger.add_scalar('Gen_loss_epoch', log_gen, epoch)  
            self.logger.add_scalar('Dis_loss_epoch', log_dis, epoch)   
            self.logger.add_scalars('Gen_dhcp_6month_epoch', {'Fake_dhcp':log_a_gen,
                                                            'Fake_6month':log_b_gen},
                                                             epoch)   
            self.logger.add_scalars('Dis_dhcp_epoch', {'Real_dhcp':log_a_dis_real,
                                                     'Fake_dhcp':log_a_dis_fake},
                                                      epoch)       
            self.logger.add_scalars('Dis_6month_epoch', {'Real_6monyh':log_b_dis_real,
                                                       'Fake_6month':log_b_dis_fake},
                                                       epoch) 
            self.logger.add_scalar('Seg_loss_epoch', log_seg, epoch)
                            
            if epoch % 10 ==0:
            
                average_val_6month_dice = 0
                self.Seg.eval()
                for idx, month6_real in enumerate(self.val_loader_6month):
                    
                    month6_real_T2 = month6_real["image_T2"]
                    month6_label = month6_real["image_label"]
                    month6_name = month6_real["image_name"][0]
                    b_real = month6_real_T2
                    b_real = Variable(b_real)          
                    b_real, month6_label = cuda([b_real, month6_label]) 
                    #b_fake = self.Gba(a_real)
                    
                    Seg_out = self.Seg(b_real)
                    seg_loss = self.dicece_loss(Seg_out, month6_label)
                    seg_dice= 1-seg_loss
                    average_val_6month_dice = average_val_6month_dice + seg_dice.item()
                    
                    Seg_out_plot = Seg_out.detach().cpu().numpy()
                    month6_label_plot = month6_label.detach().cpu().numpy()
                    Seg_out_argmax = np.argmax(Seg_out_plot, axis=1)      
                    b_real_plot = b_real.detach().cpu().numpy()

                    
                    month6_label_plot[:,0,:,:,:][month6_label_plot[:,0,:,:,:]==1]=0
                    month6_label_plot[:,0,:,:,:][month6_label_plot[:,1,:,:,:]==1]=1
                    month6_label_plot[:,0,:,:,:][month6_label_plot[:,2,:,:,:]==1]=2
                    month6_label_plot[:,0,:,:,:][month6_label_plot[:,3,:,:,:]==1]=3
                    month6_label_plot[:,0,:,:,:][month6_label_plot[:,4,:,:,:]==1]=4
                    month6_label_plot[:,0,:,:,:][month6_label_plot[:,5,:,:,:]==1]=5
                    month6_label_plot[:,0,:,:,:][month6_label_plot[:,6,:,:,:]==1]=6
                    month6_label_plot[:,0,:,:,:][month6_label_plot[:,7,:,:,:]==1]=7
                    month6_label_plot[:,0,:,:,:][month6_label_plot[:,8,:,:,:]==1]=8
                    
                    
                                
                    fig1 = plt.figure("1", (18, 18))
                    ax1 = fig1.subplots(3,3)
                    
                    ax1[0,0].title.set_text('6month_T2')
                    ax1[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax1[0,1].title.set_text('6month_label')
                    ax1[0,1].imshow(np.rot90(month6_label_plot[0 ,0, :, :, round(month6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
                    ax1[0,2].title.set_text('Prediction')
                    ax1[0,2].imshow(np.rot90(Seg_out_argmax[0 , :,:, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

                    ax1[1,0].title.set_text('6month_T2')
                    ax1[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax1[1,1].title.set_text('6month_label')
                    ax1[1,1].imshow(np.rot90(month6_label_plot[0, 0, :, round(month6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9) 
                    ax1[1,2].title.set_text('Prediction')
                    ax1[1,2].imshow(np.rot90(Seg_out_argmax[0 , :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

                    ax1[2,0].title.set_text('6month_T2')
                    ax1[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax1[2,1].title.set_text('6month_label')
                    ax1[2,1].imshow(np.rot90(month6_label_plot[0, 0, round(month6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
                    ax1[2,2].title.set_text('Prediction')
                    ax1[2,2].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

                    suptitle_name = month6_name + ' dice score: ' + str(seg_dice.item())
                    fig1.suptitle(suptitle_name)
                    fig1.savefig(val_result_path+'/'+ str(epoch) + '_' + month6_name +'.png')
            
            
                average_val_6month_dice = average_val_6month_dice / (idx+1)
                with open(val_result_path+'/'+'results.txt',"a") as file:
                    file.write('\n')
                    file.write(str(epoch))
                    file.write(' val dhcp dice: ')
                    file.write(str(average_val_6month_dice))
                    file.write('\n')
                
                
                
                if average_val_6month_dice>=self.best_val_accuracy:
                    self.best_val_accuracy = average_val_6month_dice+0
                    # save_best_model(self.Seg,epoch,best_acc=self.best_val_accuracy,dir_add = root_dir)
                    save_checkpoint({'epoch': epoch + 1,
                                     'total_number': self.total_number,
                                     'best_val_accuracy': self.best_val_accuracy,
                                     'Da': self.Da.state_dict(),
                                     'Db': self.Db.state_dict(),
                                     'Gab': self.Gab.state_dict(),
                                     'Gba': self.Gba.state_dict(),
                                     'Seg': self.Seg.state_dict()},
                                     '%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
                    
                    
                    with open(val_result_path+'/'+'results.txt',"a") as file:
                        file.write("save model")
                        file.write('\n')
                
                
                average_test_loss = 0
                for idx, month_6_real in enumerate(self.test_loader_6_month):

                    month_6_real_T2 = month_6_real["image_T2"]
                    month_6_label = month_6_real["image_label"]
                    month_6_name = month_6_real["image_name"][0]
                    month_6_affine = month_6_real["affine"]

                    b_real = month_6_real_T2

                
                    b_real = Variable(b_real)
                    b_real,month_6_label = cuda([b_real,month_6_label])

                
                    # Segmentation losses
                    ###################################################
                    

                    Seg_out = self.Seg(b_real)
                    seg_loss = self.dicece_loss(Seg_out, month_6_label)
                    seg_dice = 1-seg_loss

                    Seg_out_plot = Seg_out.detach().cpu().numpy()
                    month_6_label_plot = month_6_label.detach().cpu().numpy()
                    Seg_out_argmax = np.argmax(Seg_out_plot, axis=1)      
                    b_real_plot = b_real.detach().cpu().numpy()

                    
                   

                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
                    
                

                   
                    fig = plt.figure("0", (18, 18))
                    ax = fig.subplots(3,3)
                    
                    ax[0,0].title.set_text('6_month_real_T2')
                    ax[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax[0,1].title.set_text('VK_label')
                    ax[0,1].imshow(np.rot90(month_6_label_plot[0 , 0, :,:, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
                    ax[0,2].title.set_text('Prediction')
                    ax[0,2].imshow(np.rot90(Seg_out_argmax[0 , :,:, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)
                    
                    ax[1,0].title.set_text('6_month_real_T2')
                    ax[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax[1,1].title.set_text('VK_label')
                    ax[1,1].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:]), cmap="jet",vmin=0,vmax=9)
                    ax[1,2].title.set_text('Prediction')
                    ax[1,2].imshow(np.rot90(Seg_out_argmax[0 , :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)
                    
                    ax[2,0].title.set_text('6_month_real_T2')
                    ax[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
                    ax[2,1].title.set_text('VK_label')
                    ax[2,1].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:]), cmap="jet",vmin=0,vmax=9)
                    ax[2,2].title.set_text('Prediction')
                    ax[2,2].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)
                    
                    suptitle_name = str(epoch) +' : ' + month_6_name + ' dice score: ' + str(seg_dice.item())
                    fig.suptitle(suptitle_name)
                    fig.savefig(val_result_path+'/'+ str(epoch) + '_' + month_6_name +'.png')                    
                    
                    average_test_loss = average_test_loss + seg_dice.item()                     
                average_test_loss = average_test_loss / (idx+1) 

                with open(val_result_path+'/'+'results.txt',"a") as file:
                    file.write('\n')
                    file.write(str(epoch))
                    file.write(' test 6month dice ')
                    file.write(str(average_test_loss))
                    file.write('\n')
                
                
            # Override the latest checkpoint
            #######################################################
            save_checkpoint({'epoch': epoch + 1,
                             'total_number': self.total_number,
                             'best_val_accuracy': self.best_val_accuracy,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'Seg': self.Seg.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict(),
                                   'Seg_optimizer': self.Seg_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
            # self.Seg_lr_scheduler.step()
    

        
        x_axis = list(range(1, len(Gen_loss_list)+1))
        plt.figure(figsize=(15,10))
        plt.plot(x_axis, Gen_loss_list, c='b',label='Gen loss')
        plt.plot(x_axis, Dis_loss_list, c='r',label='Dis loss')
        plt.title("Gen and Dis loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(root_dir+'/Gen_Dis_loss.png')
        
        plt.figure(figsize=(15,10))
        plt.plot(x_axis, Gen_a_loss_list, c='b',label='Gen_a loss')
        plt.title("Gen_a loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(root_dir+'/Gen_a_loss.png')
        
        plt.figure(figsize=(15,10))
        plt.plot(x_axis, Gen_b_loss_list, c='b',label='Gen_b loss')
        plt.title("Gen_b loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(root_dir+'/Gen_b_loss.png')
        
        plt.figure(figsize=(15,10))
        plt.plot(x_axis, Dis_a_r_loss_list, c='b',label='a_real loss')
        plt.plot(x_axis, Dis_a_f_loss_list, c='r',label='a_fake loss')
        plt.title("a real and fake loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(root_dir+'/a_real_fake.png')
        
        plt.figure(figsize=(15,10))
        plt.plot(x_axis, Dis_b_r_loss_list, c='b',label='b_real loss')
        plt.plot(x_axis, Dis_b_f_loss_list, c='r',label='b_fake loss')
        plt.title("b real and fake loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(root_dir+'/b_real_fake.png')
        
        # print(test_seg_loss)
        
        self.logger.close()

    def test(self,args):
        print("####################### Doing the whole brain segmentation for 6month infants #######################")
        print("Loading the data")
        
        
        dice_loss = DiceLoss(softmax=True)
        
        all_seg_path = root_dir + '/do_seg_on_6month'       
        if not os.path.exists(all_seg_path):
            os.makedirs(all_seg_path)
        
        print("Start the segmentation for 6month test")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        average_6month = 0
        csv_record_data = []

        
        for idx, month_6_real in enumerate(self.test_loader_6_month):

            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            
            b_real = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
            b_real, month_6_label = cuda([b_real, month_6_label])
            
                        
            Seg_out = self.Seg(b_real)
            csv_record_data.append([Seg_out.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
            seg_loss = dice_loss(Seg_out, month_6_label)
            seg_dice= 1-seg_loss
            average_6month = average_6month + seg_dice.item()
            
            month_6_label_plot = month_6_label.detach().cpu().numpy()
            
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
            
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)   
            img_save = nib.Nifti1Image(np.float64(Seg_out_argmax[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_prediction' + '.nii.gz')

            b_real_plot = b_real.detach().cpu().numpy()
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                         
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_T2' + '.nii.gz') 
            
            img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_VK' + '.nii.gz')       
            
            fig3 = plt.figure("3", (18, 18))
            ax3 = fig3.subplots(3,3)
                    
            ax3[0,0].title.set_text('6_month_real_T2')
            ax3[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax3[0,1].title.set_text('VK_labels')
            ax3[0,1].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax3[0,2].title.set_text('Prediction')
            ax3[0,2].imshow(np.rot90(Seg_out_argmax[0 , :,:, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax3[1,0].title.set_text('6_month_real_T2')
            ax3[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax3[1,1].title.set_text('VK_labels')
            ax3[1,1].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax3[1,2].title.set_text('Prediction')
            ax3[1,2].imshow(np.rot90(Seg_out_argmax[0 , :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax3[2,0].title.set_text('6_month_real_T2')
            ax3[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax3[2,1].title.set_text('VK_labels')
            ax3[2,1].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax3[2,2].title.set_text('Prediction')
            ax3[2,2].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name + ' dice score: ' + str(seg_dice.item())
            fig3.suptitle(suptitle_name)
            fig3.savefig(all_seg_path+'/'+ month_6_name +'.png')
        
        average_6month = average_6month / (idx+1)
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The average dice score of 6month subjects are: ')
            file.write(str(average_6month))
            file.write(' \n ')  
        
        record_metric(csv_record_data, all_seg_path+'/2_Cyc+Att.csv')  


    def add_ibeat(self,args):
        print("####################### Add iBeat segmentation outputs to CycleGAN+Seg 8-tissue segmentation outputs #######################")
        print("Loading the data")
        
        dice_loss = DiceLoss()
        add_ibeat_path = root_dir + '/add_ibeat_6month_test'       
        if not os.path.exists(add_ibeat_path):
            os.makedirs(add_ibeat_path)
        
        print("Start the segmentation for 6month test")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(add_ibeat_path+'/add_ibeat_test_6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        
        
        for idx, month_6_real in enumerate(self.test_loader_6_month):

            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_ibeat = month_6_real["ibeat"]
            
            b_real = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
            month_6_ibeat = Variable(month_6_ibeat)
            b_real, month_6_label, month_6_ibeat = cuda([b_real, month_6_label, month_6_ibeat])
            
                        
            Seg_out = self.Seg(b_real)
            #combined_outputs = month_6_label + 0
            #combined_outputs[:,0,:,:,:][month_6_ibeat[:,0,:,:,:]==1]=1
            #combined_outputs[:,1,:,:,:] = month_6_ibeat[:,1,:,:,:]
            #combined_outputs[:,2,:,:,:] = month_6_ibeat[:,2,:,:,:]
            
            
            #seg_loss = self.dicece_loss(Seg_out, month_6_label)
            #seg_dice= 1-seg_loss
            #average_6month = average_6month + seg_dice.item()
            
            month_6_label_plot = month_6_label.detach().cpu().numpy()
            month_6_ibeat_plot = month_6_ibeat.detach().cpu().numpy()
            
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
            
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,0,:,:,:]==1]=0
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,1,:,:,:]==1]=1
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,2,:,:,:]==1]=2
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,3,:,:,:]==1]=3
            month_6_ibeat_plot = month_6_ibeat_plot[:,0,:,:,:]
            
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)  
            
            combination = np.copy(month_6_ibeat_plot)
            combination[Seg_out_argmax==1]=1
            combination[month_6_ibeat_plot==1]=1
            combination[month_6_ibeat_plot==2]=2
            combination[month_6_ibeat_plot==3]=3

            combination[Seg_out_argmax==4]=4
            combination[Seg_out_argmax==5]=5
            combination[Seg_out_argmax==6]=6
            combination[Seg_out_argmax==7]=7
            combination[Seg_out_argmax==8]=8
            
            img_save = nib.Nifti1Image(np.float64(combination[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month_6_name + '_combined' + '.nii.gz')

            
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                         
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month_6_name + '_T2' + '.nii.gz') 
            
            img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month_6_name + '_VK' + '.nii.gz')
            
            
        ################################### 6_month test ###################################
        ################################### 6_month test ###################################
        ################################### 6_month test ###################################
        
        # train_images_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T1.nii.gz')))
        test_images_T2 = sorted(glob.glob(os.path.join(add_ibeat_path, '*_T2.nii.gz')))
        test_images_label_T1 = sorted(glob.glob(os.path.join(add_ibeat_path, '*_VK.nii.gz')))
        test_images_label_ibeat = sorted(glob.glob(os.path.join(add_ibeat_path, '*_combined.nii.gz')))
        # train_images_label_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*T2*-tissue.nii.gz')))

        test_dicts_6_month_com = [
                     {"image_T2": image_T2, "image_label": image_label, "image_name": os.path.basename(image_T2)[0:12], "affine": nib.load(image_T2).affine, "com": image_ibeat}
                     for image_T2,image_label,image_ibeat in zip(test_images_T2,test_images_label_T1,test_images_label_ibeat)
        ]

        test_transform_6_month_com = Compose(
        [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label","com"]),
        EnsureChannelFirstd(keys=["image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        ConvertToMultiChannel_com(keys="com"),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1)])

        test_ds_6_month_com = CacheDataset(data=test_dicts_6_month_com, transform=test_transform_6_month_com)
        self.test_loader_6_month_com = DataLoader(test_ds_6_month_com, batch_size=args.batch_size, num_workers=0,pin_memory=True)

        average_6month = 0
        csv_record_data = []
        for idx, month_6_real in enumerate(self.test_loader_6_month_com):

            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_com = month_6_real["com"]
            
            b_real = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
            month_6_com = Variable(month_6_com)
            b_real, month_6_label, month_6_com = cuda([b_real, month_6_label, month_6_com])
            
            csv_record_data.append([month_6_com.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
               
            seg_loss = dice_loss(month_6_com, month_6_label)
            seg_dice= 1-seg_loss
            average_6month = average_6month + seg_dice.item()
            
            month_6_label_plot = month_6_label.detach().cpu().numpy()
            month_6_com_plot = month_6_com.detach().cpu().numpy()
            b_real_plot = b_real.detach().cpu().numpy()
            
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
            
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,0,:,:,:]==1]=0
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,1,:,:,:]==1]=1
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,2,:,:,:]==1]=2
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,3,:,:,:]==1]=3
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,4,:,:,:]==1]=4
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,5,:,:,:]==1]=5
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,6,:,:,:]==1]=6
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,7,:,:,:]==1]=7
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,8,:,:,:]==1]=8
            month_6_com_plot = month_6_com_plot[:,0,:,:,:]

         
            fig4 = plt.figure("4", (18, 18))
            ax4 = fig4.subplots(3,3)
                    
            ax4[0,0].title.set_text('6_month_real_T2')
            ax4[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax4[0,1].title.set_text('VK_labels')
            ax4[0,1].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax4[0,2].title.set_text('Prediction_ibeat_combined')
            ax4[0,2].imshow(np.rot90(month_6_com_plot[0 , :,:, round(month_6_com_plot.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax4[1,0].title.set_text('6_month_real_T2')
            ax4[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax4[1,1].title.set_text('VK_labels')
            ax4[1,1].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax4[1,2].title.set_text('Prediction_ibeat_combined')
            ax4[1,2].imshow(np.rot90(month_6_com_plot[0 , :, round(month_6_com_plot.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax4[2,0].title.set_text('6_month_real_T2')
            ax4[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax4[2,1].title.set_text('VK_labels')
            ax4[2,1].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax4[2,2].title.set_text('Prediction_ibeat_combined')
            ax4[2,2].imshow(np.rot90(month_6_com_plot[0, round(month_6_com_plot.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name + ' dice score: ' + str(seg_dice.item())
            fig4.suptitle(suptitle_name)
            fig4.savefig(add_ibeat_path+'/'+ month_6_name +'.png')
        
        average_6month = average_6month / (idx+1)
        with open(add_ibeat_path+'/add_ibeat_test_6month_results.txt',"a") as file:
            file.write('The average dice score of 6month subjects are: ')
            file.write(str(average_6month))
            file.write(' \n ')  
        
        record_metric(csv_record_data, add_ibeat_path+'/2_Cyc+Att+iBeat.csv')   

    def add_ibeat_T1w_only(self,args):
        
        ibeat_T1_T2_only_path = '/nfs/home/ydong/dataset/6_month_long/T1_T2_ibeat_test_set/'

        test_images_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2.nii.gz')))
        test_images_label_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_VK.nii.gz')))
        test_images_label_ibeat = sorted(glob.glob(os.path.join(ibeat_T1_T2_only_path, '*_T1_modality_only_ibeat_outputs.nii.gz')))


        test_dicts_6_month_com = [
            {"image_T2": image_T2, "image_label": image_label, "image_name": os.path.basename(image_T2)[0:12], "affine": nib.load(image_T2).affine, "ibeat": image_ibeat}
            for image_T2,image_label,image_ibeat in zip(test_images_T2,test_images_label_T1,test_images_label_ibeat)
        ]

        test_transform_6_month_com = Compose(
            [
                LoadImaged(reader=NibabelReader, keys=["image_T2","image_label","ibeat"]),
                EnsureChannelFirstd(keys=["image_T2"]),
                ConvertToMultiChannel_6month(keys="image_label"),
                ConvertToMultiChannel_ibeat(keys="ibeat"),
                ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),

            ]
        )
    
        test_ds_6_month_com = CacheDataset(data=test_dicts_6_month_com, transform=test_transform_6_month_com)
        self.test_loader_6_month_com = DataLoader(test_ds_6_month_com, batch_size=args.batch_size, num_workers=0,pin_memory=True)
        


        print("####################### Add iBeat segmentation outputs to CycleGAN+Seg 8-tissue segmentation outputs #######################")
        print("Loading the data")
        
        dice_loss = DiceLoss()
        add_ibeat_path = root_dir + '/add_ibeat_T1w_only_6month_test'       
        if not os.path.exists(add_ibeat_path):
            os.makedirs(add_ibeat_path)
        
        print("Start the segmentation for 6month test")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(add_ibeat_path+'/add_ibeat_T1w_only_test_6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        
        
        for idx, month_6_real in enumerate(self.test_loader_6_month_com):

            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_ibeat = month_6_real["ibeat"]
            
            b_real = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
            month_6_ibeat = Variable(month_6_ibeat)
            b_real, month_6_label, month_6_ibeat = cuda([b_real, month_6_label, month_6_ibeat])
            
                        
            Seg_out = self.Seg(b_real)
            #combined_outputs = month_6_label + 0
            #combined_outputs[:,0,:,:,:][month_6_ibeat[:,0,:,:,:]==1]=1
            #combined_outputs[:,1,:,:,:] = month_6_ibeat[:,1,:,:,:]
            #combined_outputs[:,2,:,:,:] = month_6_ibeat[:,2,:,:,:]
            
            
            #seg_loss = self.dicece_loss(Seg_out, month_6_label)
            #seg_dice= 1-seg_loss
            #average_6month = average_6month + seg_dice.item()
            
            month_6_label_plot = month_6_label.detach().cpu().numpy()
            month_6_ibeat_plot = month_6_ibeat.detach().cpu().numpy()
            
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4

            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
            
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,0,:,:,:]==1]=0
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,1,:,:,:]==1]=1
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,2,:,:,:]==1]=2
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,3,:,:,:]==1]=3
            month_6_ibeat_plot = month_6_ibeat_plot[:,0,:,:,:]
            
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)  
            
            combination = np.copy(month_6_ibeat_plot)
            combination[Seg_out_argmax==1]=1
            combination[month_6_ibeat_plot==1]=1
            combination[month_6_ibeat_plot==2]=2
            combination[month_6_ibeat_plot==3]=3

            combination[Seg_out_argmax==4]=4
            combination[Seg_out_argmax==5]=5
            combination[Seg_out_argmax==6]=6
            combination[Seg_out_argmax==7]=7
            combination[Seg_out_argmax==8]=8
            
            img_save = nib.Nifti1Image(np.float64(combination[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month_6_name + '_combined' + '.nii.gz')

            
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                         
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month_6_name + '_T2' + '.nii.gz') 
            
            img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month_6_name + '_VK' + '.nii.gz')
            


        ################################### 6_month test ###################################
        ################################### 6_month test ###################################
        ################################### 6_month test ###################################
        
        # train_images_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T1.nii.gz')))
        test_images_T2 = sorted(glob.glob(os.path.join(add_ibeat_path, '*_T2.nii.gz')))
        test_images_label_T1 = sorted(glob.glob(os.path.join(add_ibeat_path, '*_VK.nii.gz')))
        test_images_label_ibeat = sorted(glob.glob(os.path.join(add_ibeat_path, '*_combined.nii.gz')))
        # train_images_label_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*T2*-tissue.nii.gz')))

        test_dicts_6_month_com = [
                     {"image_T2": image_T2, "image_label": image_label, "image_name": os.path.basename(image_T2)[0:12], "affine": nib.load(image_T2).affine, "com": image_ibeat}
                     for image_T2,image_label,image_ibeat in zip(test_images_T2,test_images_label_T1,test_images_label_ibeat)
        ]

        test_transform_6_month_com = Compose(
        [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label","com"]),
        EnsureChannelFirstd(keys=["image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        ConvertToMultiChannel_com(keys="com"),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1)])

        test_ds_6_month_com = CacheDataset(data=test_dicts_6_month_com, transform=test_transform_6_month_com)
        self.test_loader_6_month_com = DataLoader(test_ds_6_month_com, batch_size=args.batch_size, num_workers=0,pin_memory=True)

        average_6month = 0
        csv_record_data = []
        for idx, month_6_real in enumerate(self.test_loader_6_month_com):

            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_com = month_6_real["com"]
            
            b_real = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
            month_6_com = Variable(month_6_com)
            b_real, month_6_label, month_6_com = cuda([b_real, month_6_label, month_6_com])
            
            csv_record_data.append([month_6_com.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
               
            seg_loss = dice_loss(month_6_com, month_6_label)
            seg_dice= 1-seg_loss
            average_6month = average_6month + seg_dice.item()
            
            month_6_label_plot = month_6_label.detach().cpu().numpy()
            month_6_com_plot = month_6_com.detach().cpu().numpy()
            b_real_plot = b_real.detach().cpu().numpy()
            
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
            
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,0,:,:,:]==1]=0
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,1,:,:,:]==1]=1
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,2,:,:,:]==1]=2
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,3,:,:,:]==1]=3
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,4,:,:,:]==1]=4
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,5,:,:,:]==1]=5
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,6,:,:,:]==1]=6
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,7,:,:,:]==1]=7
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,8,:,:,:]==1]=8
            month_6_com_plot = month_6_com_plot[:,0,:,:,:]

         
            fig4 = plt.figure("4", (18, 18))
            ax4 = fig4.subplots(3,3)
                    
            ax4[0,0].title.set_text('6_month_real_T2')
            ax4[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax4[0,1].title.set_text('VK_labels')
            ax4[0,1].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax4[0,2].title.set_text('Prediction_ibeat_combined')
            ax4[0,2].imshow(np.rot90(month_6_com_plot[0 , :,:, round(month_6_com_plot.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax4[1,0].title.set_text('6_month_real_T2')
            ax4[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax4[1,1].title.set_text('VK_labels')
            ax4[1,1].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax4[1,2].title.set_text('Prediction_ibeat_combined')
            ax4[1,2].imshow(np.rot90(month_6_com_plot[0 , :, round(month_6_com_plot.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax4[2,0].title.set_text('6_month_real_T2')
            ax4[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax4[2,1].title.set_text('VK_labels')
            ax4[2,1].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax4[2,2].title.set_text('Prediction_ibeat_combined')
            ax4[2,2].imshow(np.rot90(month_6_com_plot[0, round(month_6_com_plot.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name + ' dice score: ' + str(seg_dice.item())
            fig4.suptitle(suptitle_name)
            fig4.savefig(add_ibeat_path+'/'+ month_6_name +'.png')
        
        average_6month = average_6month / (idx+1)
        with open(add_ibeat_path+'/add_ibeat_T1w_only_test_6month_results.txt',"a") as file:
            file.write('The average dice score of 6month subjects are: ')
            file.write(str(average_6month))
            file.write(' \n ')  
        
        record_metric(csv_record_data, add_ibeat_path+'/4_Cyc+Att+iBeat_T1.csv')                      
        
        



    def add_ibeat_T2w_only(self,args):



        ibeat_T1_T2_only_path = '/nfs/home/ydong/dataset/6_month_long/T1_T2_ibeat_test_set/'

        test_images_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2.nii.gz')))
        test_images_label_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_VK.nii.gz')))
        test_images_label_ibeat = sorted(glob.glob(os.path.join(ibeat_T1_T2_only_path, '*_T2_modality_only_ibeat_outputs.nii.gz')))


        test_dicts_6_month_com = [
            {"image_T2": image_T2, "image_label": image_label, "image_name": os.path.basename(image_T2)[0:12], "affine": nib.load(image_T2).affine, "ibeat": image_ibeat}
            for image_T2,image_label,image_ibeat in zip(test_images_T2,test_images_label_T1,test_images_label_ibeat)
        ]

        test_transform_6_month_com = Compose(
            [
                LoadImaged(reader=NibabelReader, keys=["image_T2","image_label","ibeat"]),
                EnsureChannelFirstd(keys=["image_T2"]),
                ConvertToMultiChannel_6month(keys="image_label"),
                ConvertToMultiChannel_ibeat(keys="ibeat"),
                ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),

            ]
        )
    
        test_ds_6_month_com = CacheDataset(data=test_dicts_6_month_com, transform=test_transform_6_month_com)
        self.test_loader_6_month_com = DataLoader(test_ds_6_month_com, batch_size=args.batch_size, num_workers=0,pin_memory=True)
        


        print("####################### Add iBeat segmentation outputs to CycleGAN+Seg 8-tissue segmentation outputs #######################")
        print("Loading the data")
        
        dice_loss = DiceLoss()
        add_ibeat_path = root_dir + '/add_ibeat_T2w_only_6month_test'       
        if not os.path.exists(add_ibeat_path):
            os.makedirs(add_ibeat_path)
        
        print("Start the segmentation for 6month test")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(add_ibeat_path+'/add_ibeat_T2w_only_test_6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        
        
        for idx, month_6_real in enumerate(self.test_loader_6_month_com):

            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_ibeat = month_6_real["ibeat"]
            
            b_real = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
            month_6_ibeat = Variable(month_6_ibeat)
            b_real, month_6_label, month_6_ibeat = cuda([b_real, month_6_label, month_6_ibeat])
            
                        
            Seg_out = self.Seg(b_real)
            #combined_outputs = month_6_label + 0
            #combined_outputs[:,0,:,:,:][month_6_ibeat[:,0,:,:,:]==1]=1
            #combined_outputs[:,1,:,:,:] = month_6_ibeat[:,1,:,:,:]
            #combined_outputs[:,2,:,:,:] = month_6_ibeat[:,2,:,:,:]
            
            
            #seg_loss = self.dicece_loss(Seg_out, month_6_label)
            #seg_dice= 1-seg_loss
            #average_6month = average_6month + seg_dice.item()
            
            month_6_label_plot = month_6_label.detach().cpu().numpy()
            month_6_ibeat_plot = month_6_ibeat.detach().cpu().numpy()
            
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
            
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,0,:,:,:]==1]=0
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,1,:,:,:]==1]=1
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,2,:,:,:]==1]=2
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,3,:,:,:]==1]=3
            month_6_ibeat_plot = month_6_ibeat_plot[:,0,:,:,:]
            
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)  
            
            combination = np.copy(month_6_ibeat_plot)
            combination[Seg_out_argmax==1]=1
            combination[month_6_ibeat_plot==1]=1
            combination[month_6_ibeat_plot==2]=2
            combination[month_6_ibeat_plot==3]=3

            combination[Seg_out_argmax==4]=4
            combination[Seg_out_argmax==5]=5
            combination[Seg_out_argmax==6]=6
            combination[Seg_out_argmax==7]=7
            combination[Seg_out_argmax==8]=8
            
            img_save = nib.Nifti1Image(np.float64(combination[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month_6_name + '_combined' + '.nii.gz')

            
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                         
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month_6_name + '_T2' + '.nii.gz') 
            
            img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month_6_name + '_VK' + '.nii.gz')
            


        ################################### 6_month test ###################################
        ################################### 6_month test ###################################
        ################################### 6_month test ###################################
        
        # train_images_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T1.nii.gz')))
        test_images_T2 = sorted(glob.glob(os.path.join(add_ibeat_path, '*_T2.nii.gz')))
        test_images_label_T1 = sorted(glob.glob(os.path.join(add_ibeat_path, '*_VK.nii.gz')))
        test_images_label_ibeat = sorted(glob.glob(os.path.join(add_ibeat_path, '*_combined.nii.gz')))
        # train_images_label_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*T2*-tissue.nii.gz')))

        test_dicts_6_month_com = [
                     {"image_T2": image_T2, "image_label": image_label, "image_name": os.path.basename(image_T2)[0:12], "affine": nib.load(image_T2).affine, "com": image_ibeat}
                     for image_T2,image_label,image_ibeat in zip(test_images_T2,test_images_label_T1,test_images_label_ibeat)
        ]

        test_transform_6_month_com = Compose(
        [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label","com"]),
        EnsureChannelFirstd(keys=["image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        ConvertToMultiChannel_com(keys="com"),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1)])

        test_ds_6_month_com = CacheDataset(data=test_dicts_6_month_com, transform=test_transform_6_month_com)
        self.test_loader_6_month_com = DataLoader(test_ds_6_month_com, batch_size=args.batch_size, num_workers=0,pin_memory=True)

        average_6month = 0
        csv_record_data = []
        for idx, month_6_real in enumerate(self.test_loader_6_month_com):

            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_com = month_6_real["com"]
            
            b_real = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
            month_6_com = Variable(month_6_com)
            b_real, month_6_label, month_6_com = cuda([b_real, month_6_label, month_6_com])
            
            csv_record_data.append([month_6_com.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
               
            seg_loss = dice_loss(month_6_com, month_6_label)
            seg_dice= 1-seg_loss
            average_6month = average_6month + seg_dice.item()
            
            month_6_label_plot = month_6_label.detach().cpu().numpy()
            month_6_com_plot = month_6_com.detach().cpu().numpy()
            b_real_plot = b_real.detach().cpu().numpy()
            
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
            
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,0,:,:,:]==1]=0
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,1,:,:,:]==1]=1
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,2,:,:,:]==1]=2
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,3,:,:,:]==1]=3
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,4,:,:,:]==1]=4
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,5,:,:,:]==1]=5
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,6,:,:,:]==1]=6
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,7,:,:,:]==1]=7
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,8,:,:,:]==1]=8
            month_6_com_plot = month_6_com_plot[:,0,:,:,:]

         
            fig4 = plt.figure("4", (18, 18))
            ax4 = fig4.subplots(3,3)
                    
            ax4[0,0].title.set_text('6_month_real_T2')
            ax4[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax4[0,1].title.set_text('VK_labels')
            ax4[0,1].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax4[0,2].title.set_text('Prediction_ibeat_combined')
            ax4[0,2].imshow(np.rot90(month_6_com_plot[0 , :,:, round(month_6_com_plot.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax4[1,0].title.set_text('6_month_real_T2')
            ax4[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax4[1,1].title.set_text('VK_labels')
            ax4[1,1].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax4[1,2].title.set_text('Prediction_ibeat_combined')
            ax4[1,2].imshow(np.rot90(month_6_com_plot[0 , :, round(month_6_com_plot.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax4[2,0].title.set_text('6_month_real_T2')
            ax4[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax4[2,1].title.set_text('VK_labels')
            ax4[2,1].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax4[2,2].title.set_text('Prediction_ibeat_combined')
            ax4[2,2].imshow(np.rot90(month_6_com_plot[0, round(month_6_com_plot.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name + ' dice score: ' + str(seg_dice.item())
            fig4.suptitle(suptitle_name)
            fig4.savefig(add_ibeat_path+'/'+ month_6_name +'.png')
        
        average_6month = average_6month / (idx+1)
        with open(add_ibeat_path+'/add_ibeat_T2w_only_test_6month_results.txt',"a") as file:
            file.write('The average dice score of 6month subjects are: ')
            file.write(str(average_6month))
            file.write(' \n ')  
        
        record_metric(csv_record_data, add_ibeat_path+'/4_Cyc+Att+iBeat_T2.csv')                      
        
        


    def create_combined_trainset(self,args):
        print("####################### Add iBeat segmentation outputs to CycleGAN+Seg 8-tissue segmentation outputs #######################")
        print("Loading the data")
        
        
        
        train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T1.nii.gz')))
        train_images_T2 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2.nii.gz')))
        train_images_ibeat = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2_iBEAT_outputs.nii.gz')))
        

        train_dicts_6_month = [
            {"image_T1": image_T1,"image_T2": image_T2,"ibeat": image_ibeat, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:12]}
            for image_T1, image_T2, image_ibeat in zip(train_images_T1, train_images_T2,train_images_ibeat)
        ]


        train_transform_6_month_com = Compose(
            [
                LoadImaged(reader=NibabelReader, keys=["image_T1","image_T2","ibeat"]),
                EnsureChannelFirstd(keys=["image_T1","image_T2"]),
                ConvertToMultiChannel_ibeat(keys="ibeat"),
                ScaleIntensityd(keys=["image_T1","image_T2"],minv=-1,maxv=1),

            ]
        )
        
        
        
        train_ds_6month_com = CacheDataset(data=train_dicts_6_month, transform=train_transform_6_month_com)
        self.train_loader_6_month_com = DataLoader(train_ds_6month_com, batch_size=args.batch_size,shuffle=False, num_workers=0,pin_memory=True)
        
        
        
        dice_loss = DiceLoss()
        create_combined_train_path = root_dir + '/create_combined_train_set'       
        if not os.path.exists(create_combined_train_path):
            os.makedirs(create_combined_train_path)
        
        print("Start the segmentation for 6month train")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(create_combined_train_path+'/add_ibeat_train_6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        

        
        for idx, month_6_real in enumerate(self.train_loader_6_month_com):

            month_6_real_T1 = month_6_real["image_T1"]
            month_6_real_T2 = month_6_real["image_T2"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_ibeat = month_6_real["ibeat"]
            
            b_real = Variable(month_6_real_T2)
            month_6_ibeat = Variable(month_6_ibeat)
            b_real, month_6_ibeat = cuda([b_real, month_6_ibeat])
            
                        
            Seg_out = self.Seg(b_real)
            

            month_6_ibeat_plot = month_6_ibeat.detach().cpu().numpy()            
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,0,:,:,:]==1]=0
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,1,:,:,:]==1]=1
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,2,:,:,:]==1]=2
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,3,:,:,:]==1]=3
            month_6_ibeat_plot = month_6_ibeat_plot[:,0,:,:,:]
            
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)  
            
            combination = np.copy(month_6_ibeat_plot)
            combination[Seg_out_argmax==1]=1
            combination[month_6_ibeat_plot==1]=1
            combination[month_6_ibeat_plot==2]=2
            combination[month_6_ibeat_plot==3]=3

            combination[Seg_out_argmax==4]=4
            combination[Seg_out_argmax==5]=5
            combination[Seg_out_argmax==6]=6
            combination[Seg_out_argmax==7]=7
            combination[Seg_out_argmax==8]=8
            
            img_save = nib.Nifti1Image(np.float64(combination[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, create_combined_train_path +'/' + month_6_name + '_combined' + '.nii.gz')

            
            month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                
            img_save = nib.Nifti1Image(month_6_real_T1[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, create_combined_train_path +'/' + month_6_name + '_T1' + '.nii.gz') 
                     
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, create_combined_train_path +'/' + month_6_name + '_T2' + '.nii.gz') 
            
         
            fig5 = plt.figure("5", (12, 18))
            ax5 = fig5.subplots(3,2)
                    
            ax5[0,0].title.set_text('6_month_real_T2')
            ax5[0,0].imshow(np.rot90(month_6_real_T2[0 , 0, :,:, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax5[0,1].title.set_text('Prediction_ibeat_combined')
            ax5[0,1].imshow(np.rot90(combination[0 , :,:, round(combination.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax5[1,0].title.set_text('6_month_real_T2')
            ax5[1,0].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax5[1,1].title.set_text('Prediction_ibeat_combined')
            ax5[1,1].imshow(np.rot90(combination[0 , :, round(combination.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax5[2,0].title.set_text('6_month_real_T2')
            ax5[2,0].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax5[2,1].title.set_text('Prediction_ibeat_combined')
            ax5[2,1].imshow(np.rot90(combination[0, round(combination.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name
            fig5.suptitle(suptitle_name)
            fig5.savefig(create_combined_train_path+'/'+ month_6_name +'.png')
   
    def myelination(self,args):  
         
        print("####################### utilize myelination #######################")
        print("Loading the data")
        
        
        dice_loss = DiceLoss(softmax=True)
        
        all_seg_path = root_dir + '/mye_do_seg_on_6month'       
        if not os.path.exists(all_seg_path):
            os.makedirs(all_seg_path)
        
        print("Start the segmentation for 6month test")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        average_6month = 0
        csv_record_data = []


        test_images_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2.nii.gz')))
        test_images_label_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_VK.nii.gz')))
        test_images_label_mye = sorted(glob.glob(os.path.join(test_path_6_month, '*_fixed_basalganglia_prediction.nii.gz')))

        test_dicts_6_month = [
            {"image_T2": image_T2, "image_label": image_label,"image_label_mye": image_label_mye, "image_name": os.path.basename(image_T2)[0:12], "affine": nib.load(image_T2).affine}
            for image_T2,image_label,image_label_mye in zip(test_images_T2,test_images_label_T1,test_images_label_mye)
        ]

        test_transform_6_month = Compose(
            [
                LoadImaged(reader=NibabelReader, keys=["image_T2","image_label","image_label_mye"]),
                EnsureChannelFirstd(keys=["image_T2"]),
                ConvertToMultiChannel_6month(keys="image_label"),
                ConvertToMultiChannel_mye(keys="image_label_mye"),
                # ConvertToMultiChannel_ibeat(keys="ibeat"),
                ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),

            ]
        )     
        
        test_ds_6_month = CacheDataset(data=test_dicts_6_month, transform=test_transform_6_month)
        self.test_loader_6_month = DataLoader(test_ds_6_month, batch_size=args.batch_size, num_workers=0,pin_memory=True)  
              
        for idx, month_6_real in enumerate(self.test_loader_6_month):

            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_label_mye = month_6_real["image_label_mye"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            
            b_real = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
            month_6_label_mye = Variable(month_6_label_mye)
            b_real, month_6_label, month_6_label_mye = cuda([b_real, month_6_label, month_6_label_mye])
            
                        
            Seg_out = self.Seg(b_real)
            seg_out_argmax = torch.argmax(Seg_out,dim=1)
            seg_out_argmax = seg_out_argmax[0,:,:,:]
            index_ = torch.where(seg_out_argmax==3)
            mask_mye = month_6_label_mye[0,9,:,:,:]
            index_mye = torch.where(mask_mye==1)
            month6_wm = b_real[0,0,:,:,:][index_]
            mean_month6 = torch.mean(month6_wm)
            b_real_after = b_real + 0
            b_real_after[0,0,:,:,:][index_mye] = mean_month6
            
            Seg_out = self.Seg(b_real_after)
            
            csv_record_data.append([Seg_out.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
            seg_loss = dice_loss(Seg_out, month_6_label)
            seg_dice= 1-seg_loss
            average_6month = average_6month + seg_dice.item()
            
            month_6_label_plot = month_6_label.detach().cpu().numpy()
            
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
            
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)   
            img_save = nib.Nifti1Image(np.float64(Seg_out_argmax[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_prediction' + '.nii.gz')

            b_real_plot = b_real.detach().cpu().numpy()
            b_real_after_plot = b_real_after.detach().cpu().numpy()
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                         
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_T2' + '.nii.gz') 
            
            img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_VK' + '.nii.gz')       
            
            fig6 = plt.figure("6", (24, 18))
            ax6 = fig6.subplots(3,4)
                    
            ax6[0,0].title.set_text('6_month_real_T2')
            ax6[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax6[0,1].title.set_text('6_month_real_T2_corrected')
            ax6[0,1].imshow(np.rot90(b_real_after_plot[0 , 0, :,:, round(b_real_after_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax6[0,2].title.set_text('VK_labels')
            ax6[0,2].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax6[0,3].title.set_text('Prediction')
            ax6[0,3].imshow(np.rot90(Seg_out_argmax[0 , :,:, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax6[1,0].title.set_text('6_month_real_T2')
            ax6[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax6[1,1].title.set_text('6_month_real_T2_corrected')
            ax6[1,1].imshow(np.rot90(b_real_after_plot[0, 0, :, round(b_real_after_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax6[1,2].title.set_text('VK_labels')
            ax6[1,2].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax6[1,3].title.set_text('Prediction')
            ax6[1,3].imshow(np.rot90(Seg_out_argmax[0 , :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax6[2,0].title.set_text('6_month_real_T2')
            ax6[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax6[2,1].title.set_text('6_month_real_T2_corrected')
            ax6[2,1].imshow(np.rot90(b_real_after_plot[0, 0, round(b_real_after_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax6[2,2].title.set_text('VK_labels')
            ax6[2,2].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax6[2,3].title.set_text('Prediction')
            ax6[2,3].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name + ' dice score: ' + str(seg_dice.item())
            fig6.suptitle(suptitle_name)
            fig6.savefig(all_seg_path+'/'+ month_6_name +'.png')
        
        average_6month = average_6month / (idx+1)
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The average dice score of 6month subjects are: ')
            file.write(str(average_6month))
            file.write(' \n ')  
        
        record_metric(csv_record_data, all_seg_path+'/mye_6month_results.csv')  


    def mye_add_ibeat(self,args):  
         
        print("####################### utilize myelination #######################")
        print("Loading the data")
        
        
        dice_loss = DiceLoss(softmax=True)
        
        all_seg_path = root_dir + '/mye_6month_add_ibeat'       
        if not os.path.exists(all_seg_path):
            os.makedirs(all_seg_path)
        
        print("Start the segmentation for 6month test")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        average_6month = 0
        csv_record_data = []


        test_images_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2.nii.gz')))
        test_images_label_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_VK.nii.gz')))
        test_images_label_ibeat = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2_iBEAT_outputs.nii.gz')))
        test_images_label_mye = sorted(glob.glob(os.path.join(test_path_6_month, '*_fixed_basalganglia_prediction.nii.gz')))

        test_dicts_6_month = [
            {"image_T2": image_T2, "image_label": image_label,"image_label_mye": image_label_mye, "image_name": os.path.basename(image_T2)[0:12], "affine": nib.load(image_T2).affine, "ibeat": image_ibeat}
            for image_T2,image_label,image_ibeat,image_label_mye  in zip(test_images_T2,test_images_label_T1,test_images_label_ibeat, test_images_label_mye)
        ]

        test_transform_6_month = Compose(
            [
                LoadImaged(reader=NibabelReader, keys=["image_T2","image_label","image_label_mye","ibeat"]),
                EnsureChannelFirstd(keys=["image_T2"]),
                ConvertToMultiChannel_6month(keys="image_label"),
                ConvertToMultiChannel_mye(keys="image_label_mye"),
                ConvertToMultiChannel_ibeat(keys="ibeat"),
                ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),

            ]
        )     
        
        test_ds_6_month = CacheDataset(data=test_dicts_6_month, transform=test_transform_6_month)
        self.test_loader_6_month = DataLoader(test_ds_6_month, batch_size=args.batch_size, num_workers=0,pin_memory=True)  
        
        post_pred = AsDiscrete(to_onehot=9)
              
        for idx, month_6_real in enumerate(self.test_loader_6_month):

            month_6_real_T2 = month_6_real["image_T2"]

            month_6_label = month_6_real["image_label"]
            month_6_label_mye = month_6_real["image_label_mye"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_ibeat = month_6_real["ibeat"]
            
            b_real = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
            month_6_label_mye = Variable(month_6_label_mye)
            month_6_ibeat = Variable(month_6_ibeat)
            b_real, month_6_label, month_6_label_mye, month_6_ibeat = cuda([b_real, month_6_label, month_6_label_mye, month_6_ibeat])
            
            month_6_ibeat_plot = month_6_ibeat + 0           
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,0,:,:,:]==1]=0
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,1,:,:,:]==1]=1
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,2,:,:,:]==1]=2
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,3,:,:,:]==1]=3
            month_6_ibeat_plot = month_6_ibeat_plot[:,0,:,:,:]
            
                        
            Seg_out = self.Seg(b_real)
            seg_out_argmax = torch.argmax(Seg_out,dim=1)
            seg_out_argmax = seg_out_argmax[0,:,:,:]
            index_ = torch.where(seg_out_argmax==3)
            mask_mye = month_6_label_mye[0,9,:,:,:]
            index_mye = torch.where(mask_mye==1)
            month6_wm = b_real[0,0,:,:,:][index_]
            mean_month6 = torch.mean(month6_wm)
            b_real_after = b_real + 0
            b_real_after[0,0,:,:,:][index_mye] = mean_month6
            
            Seg_out = self.Seg(b_real_after)
            

            Seg_out_argmax = torch.argmax(Seg_out, dim=1)  
            
            combination = month_6_ibeat_plot + 0
            combination[Seg_out_argmax==1]=1
            combination[month_6_ibeat_plot==1]=1
            combination[month_6_ibeat_plot==2]=2
            combination[month_6_ibeat_plot==3]=3

            combination[Seg_out_argmax==4]=4
            combination[Seg_out_argmax==5]=5
            combination[Seg_out_argmax==6]=6
            combination[Seg_out_argmax==7]=7
            combination[Seg_out_argmax==8]=8
            
            month_6_com = post_pred(combination).unsqueeze(0)
            csv_record_data.append([month_6_com.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
            seg_loss = dice_loss(month_6_com, month_6_label)
            seg_dice= 1-seg_loss
            average_6month = average_6month + seg_dice.item()
            
            month_6_label_plot = month_6_label.detach().cpu().numpy()
            month_6_com_plot = month_6_com.detach().cpu().numpy()
            b_real_plot = b_real.detach().cpu().numpy()
            
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
            month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
            
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,0,:,:,:]==1]=0
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,1,:,:,:]==1]=1
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,2,:,:,:]==1]=2
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,3,:,:,:]==1]=3
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,4,:,:,:]==1]=4
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,5,:,:,:]==1]=5
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,6,:,:,:]==1]=6
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,7,:,:,:]==1]=7
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,8,:,:,:]==1]=8
            month_6_com_plot = month_6_com_plot[:,0,:,:,:]
            
            
            img_save = nib.Nifti1Image(np.float64(month_6_com_plot[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_prediction' + '.nii.gz')
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                         
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_T2' + '.nii.gz') 
            
            img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_VK' + '.nii.gz')    
         
            fig7 = plt.figure("7", (18, 18))
            ax7 = fig7.subplots(3,3)
                    
            ax7[0,0].title.set_text('6_month_real_T2')
            ax7[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax7[0,1].title.set_text('VK_labels')
            ax7[0,1].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax7[0,2].title.set_text('Prediction_ibeat_combined')
            ax7[0,2].imshow(np.rot90(month_6_com_plot[0 , :,:, round(month_6_com_plot.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax7[1,0].title.set_text('6_month_real_T2')
            ax7[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax7[1,1].title.set_text('VK_labels')
            ax7[1,1].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax7[1,2].title.set_text('Prediction_ibeat_combined')
            ax7[1,2].imshow(np.rot90(month_6_com_plot[0 , :, round(month_6_com_plot.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax7[2,0].title.set_text('6_month_real_T2')
            ax7[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax7[2,1].title.set_text('VK_labels')
            ax7[2,1].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax7[2,2].title.set_text('Prediction_ibeat_combined')
            ax7[2,2].imshow(np.rot90(month_6_com_plot[0, round(month_6_com_plot.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name + ' dice score: ' + str(seg_dice.item())
            fig7.suptitle(suptitle_name)
            fig7.savefig(all_seg_path+'/'+ month_6_name +'.png')
        
        average_6month = average_6month / (idx+1)
        with open(all_seg_path+'/mye_6month_add_ibeat.txt',"a") as file:
            file.write('The average dice score of 6month subjects are: ')
            file.write(str(average_6month))
            file.write(' \n ')  
        
        record_metric(csv_record_data, all_seg_path+'/mye_6month_add_ibeat.csv')  


    def create_cover_mye_iBeat_combined_trainset(self,args):  
         
        print("####################### utilize myelination #######################")
        print("Loading the data")
        
                
        train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T1.nii.gz')))
        train_images_T2 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2.nii.gz')))
        train_images_ibeat = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2_iBEAT_outputs.nii.gz')))
        test_images_label_mye = sorted(glob.glob(os.path.join(train_path_6_month, '*_fixed_basalganglia_prediction.nii.gz')))
        

        train_dicts_6_month = [
            {"image_T1": image_T1,"image_T2": image_T2,"ibeat": image_ibeat,"image_label_mye": image_label_mye, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:12]}
            for image_T1, image_T2, image_ibeat, image_label_mye in zip(train_images_T1, train_images_T2,train_images_ibeat, test_images_label_mye)
        ]


        train_transform_6_month_com = Compose(
            [
                LoadImaged(reader=NibabelReader, keys=["image_T1","image_T2","ibeat","image_label_mye"]),
                EnsureChannelFirstd(keys=["image_T1","image_T2"]),
                ConvertToMultiChannel_ibeat(keys="ibeat"),
                ConvertToMultiChannel_mye(keys="image_label_mye"),
                ScaleIntensityd(keys=["image_T1","image_T2"],minv=-1,maxv=1),

            ]
        )
        
        
        
        train_ds_6month_com = CacheDataset(data=train_dicts_6_month, transform=train_transform_6_month_com)
        self.train_loader_6_month_com = DataLoader(train_ds_6month_com, batch_size=args.batch_size,shuffle=False, num_workers=0,pin_memory=True)
        
        
        
        
        
        dice_loss = DiceLoss(softmax=True)
        
        all_seg_path = root_dir + '/create_cover_mye_ibeat_combined_train_set'       
        if not os.path.exists(all_seg_path):
            os.makedirs(all_seg_path)
        
        print("Start the segmentation for 6month train")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        average_6month = 0
        # csv_record_data = []

        
        post_pred = AsDiscrete(to_onehot=9)
              
        for idx, month_6_real in enumerate(self.train_loader_6_month_com):

            month_6_real_T2 = month_6_real["image_T2"]

            # month_6_label = month_6_real["image_label"]
            month_6_label_mye = month_6_real["image_label_mye"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_ibeat = month_6_real["ibeat"]
            
            b_real = Variable(month_6_real_T2)
            # month_6_label = Variable(month_6_label)
            month_6_label_mye = Variable(month_6_label_mye)
            month_6_ibeat = Variable(month_6_ibeat)
            b_real, month_6_label_mye, month_6_ibeat = cuda([b_real, month_6_label_mye, month_6_ibeat])
            
            month_6_ibeat_plot = month_6_ibeat + 0           
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,0,:,:,:]==1]=0
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,1,:,:,:]==1]=1
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,2,:,:,:]==1]=2
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,3,:,:,:]==1]=3
            month_6_ibeat_plot = month_6_ibeat_plot[:,0,:,:,:]
            
                        
            Seg_out = self.Seg(b_real)
            seg_out_argmax = torch.argmax(Seg_out,dim=1)
            seg_out_argmax = seg_out_argmax[0,:,:,:]
            index_ = torch.where(seg_out_argmax==3)
            mask_mye = month_6_label_mye[0,9,:,:,:]
            index_mye = torch.where(mask_mye==1)
            month6_wm = b_real[0,0,:,:,:][index_]
            mean_month6 = torch.mean(month6_wm)
            b_real_after = b_real + 0
            b_real_after[0,0,:,:,:][index_mye] = mean_month6
            
            Seg_out = self.Seg(b_real_after)
            

            Seg_out_argmax = torch.argmax(Seg_out, dim=1)  
            
            combination = month_6_ibeat_plot + 0
            combination[Seg_out_argmax==1]=1
            combination[month_6_ibeat_plot==1]=1
            combination[month_6_ibeat_plot==2]=2
            combination[month_6_ibeat_plot==3]=3

            combination[Seg_out_argmax==4]=4
            combination[Seg_out_argmax==5]=5
            combination[Seg_out_argmax==6]=6
            combination[Seg_out_argmax==7]=7
            combination[Seg_out_argmax==8]=8
            
            month_6_com = post_pred(combination).unsqueeze(0)
            # csv_record_data.append([month_6_com.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
            #seg_loss = dice_loss(month_6_com, month_6_label)
            #seg_dice= 1-seg_loss
            #average_6month = average_6month + seg_dice.item()
            
            # month_6_label_plot = month_6_label.detach().cpu().numpy()
            month_6_com_plot = month_6_com.detach().cpu().numpy()
            b_real_plot = b_real.detach().cpu().numpy()
            

            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,0,:,:,:]==1]=0
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,1,:,:,:]==1]=1
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,2,:,:,:]==1]=2
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,3,:,:,:]==1]=3
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,4,:,:,:]==1]=4
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,5,:,:,:]==1]=5
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,6,:,:,:]==1]=6
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,7,:,:,:]==1]=7
            month_6_com_plot[:,0,:,:,:][month_6_com_plot[:,8,:,:,:]==1]=8
            month_6_com_plot = month_6_com_plot[:,0,:,:,:]

            
            img_save = nib.Nifti1Image(np.float64(month_6_com_plot[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_combined' + '.nii.gz')
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                           
         
            fig11 = plt.figure("11", (12, 18))
            ax11 = fig11.subplots(3,2)
                    
            ax11[0,0].title.set_text('6_month_real_T2')
            ax11[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax11[0,1].title.set_text('Prediction_ibeat_combined')
            ax11[0,1].imshow(np.rot90(month_6_com_plot[0 , :,:, round(month_6_com_plot.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax11[1,0].title.set_text('6_month_real_T2')
            ax11[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax11[1,1].title.set_text('Prediction_ibeat_combined')
            ax11[1,1].imshow(np.rot90(month_6_com_plot[0 , :, round(month_6_com_plot.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax11[2,0].title.set_text('6_month_real_T2')
            ax11[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax11[2,1].title.set_text('Prediction_ibeat_combined')
            ax11[2,1].imshow(np.rot90(month_6_com_plot[0, round(month_6_com_plot.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name
            fig11.suptitle(suptitle_name)
            fig11.savefig(all_seg_path+'/'+ month_6_name +'.png')
        

        
        # record_metric(csv_record_data, all_seg_path+'/mye_6month_add_ibeat.csv')  






    def create_not_combined_trainset(self,args):
        print("####################### Add iBeat segmentation outputs to CycleGAN+Seg 8-tissue segmentation outputs #######################")
        print("Loading the data")
        
        
        
        train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T1.nii.gz')))
        train_images_T2 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2.nii.gz')))
        

        train_dicts_6_month = [
            {"image_T1": image_T1,"image_T2": image_T2, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:12]}
            for image_T1, image_T2 in zip(train_images_T1, train_images_T2)
        ]


        train_transform_6_month_com = Compose(
            [
                LoadImaged(reader=NibabelReader, keys=["image_T1","image_T2"]),
                EnsureChannelFirstd(keys=["image_T1","image_T2"]),
                # ConvertToMultiChannel_ibeat(keys="ibeat"),
                ScaleIntensityd(keys=["image_T1","image_T2"],minv=-1,maxv=1),

            ]
        )
        
        
        
        train_ds_6month_com = CacheDataset(data=train_dicts_6_month, transform=train_transform_6_month_com)
        self.train_loader_6_month_com = DataLoader(train_ds_6month_com, batch_size=args.batch_size,shuffle=False, num_workers=0,pin_memory=True)
        
        
        
        dice_loss = DiceLoss()
        create_combined_train_path = root_dir + '/create_not_combined_train_set'       
        if not os.path.exists(create_combined_train_path):
            os.makedirs(create_combined_train_path)
        
        print("Start the segmentation for 6month train")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))

        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(create_combined_train_path+'/seg_not_combinedtrain_6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        

        
        for idx, month_6_real in enumerate(self.train_loader_6_month_com):

            month_6_real_T1 = month_6_real["image_T1"]
            month_6_real_T2 = month_6_real["image_T2"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            
            b_real = Variable(month_6_real_T2)
            b_real = cuda(b_real)
            
                        
            Seg_out = self.Seg(b_real)
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)  

            
            img_save = nib.Nifti1Image(np.float64(Seg_out_argmax[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, create_combined_train_path +'/' + month_6_name + '_not_combined' + '.nii.gz')

            
            month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()

            
         
            fig8 = plt.figure("8", (12, 18))
            ax8 = fig8.subplots(3,2)
                    
            ax8[0,0].title.set_text('6_month_real_T2')
            ax8[0,0].imshow(np.rot90(month_6_real_T2[0 , 0, :,:, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax8[0,1].title.set_text('Prediction_ibeat_not_combined')
            ax8[0,1].imshow(np.rot90(Seg_out_argmax[0 , :,:, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax8[1,0].title.set_text('6_month_real_T2')
            ax8[1,0].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax8[1,1].title.set_text('Prediction_ibeat_not_combined')
            ax8[1,1].imshow(np.rot90(Seg_out_argmax[0 , :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax8[2,0].title.set_text('6_month_real_T2')
            ax8[2,0].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax8[2,1].title.set_text('Prediction_ibeat_not_combined')
            ax8[2,1].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name
            fig8.suptitle(suptitle_name)
            fig8.savefig(create_combined_train_path+'/'+ month_6_name +'.png')

    def create_mye_cyc_trainset(self,args):
        print("####################### Add iBeat segmentation outputs to CycleGAN+Seg 8-tissue segmentation outputs #######################")
        print("Loading the data")
        
        
        
        train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T1.nii.gz')))
        train_images_T2 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2.nii.gz')))
        train_images_mye = sorted(glob.glob(os.path.join(train_path_6_month, '*_fixed_basalganglia_prediction.nii.gz')))
        

        train_dicts_6_month = [
            {"image_T1": image_T1,"image_T2": image_T2,"image_label_mye": image_label, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:12]}
            for image_T1, image_T2, image_label in zip(train_images_T1, train_images_T2,train_images_mye)
        ]


        train_transform_6_month_com = Compose(
            [
                LoadImaged(reader=NibabelReader, keys=["image_T1","image_T2","image_label_mye"]),
                EnsureChannelFirstd(keys=["image_T1","image_T2"]),
                ConvertToMultiChannel_mye(keys="image_label_mye"),
                ScaleIntensityd(keys=["image_T1","image_T2"],minv=-1,maxv=1),

            ]
        )
        
        
        
        train_ds_6month_com = CacheDataset(data=train_dicts_6_month, transform=train_transform_6_month_com)
        self.train_loader_6_month_com = DataLoader(train_ds_6month_com, batch_size=args.batch_size,shuffle=False, num_workers=0,pin_memory=True)
        
        
        
        dice_loss = DiceLoss()
        create_combined_train_path = root_dir + '/create_mye_cyc_train_set'       
        if not os.path.exists(create_combined_train_path):
            os.makedirs(create_combined_train_path)
        
        print("Start the segmentation for 6month train")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))

        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(create_combined_train_path+'/mye_cyc_train_6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        

        
        for idx, month_6_real in enumerate(self.train_loader_6_month_com):

            month_6_real_T1 = month_6_real["image_T1"]
            month_6_real_T2 = month_6_real["image_T2"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_mye = month_6_real["image_label_mye"]
            
            b_real = Variable(month_6_real_T2)
            month_6_mye = Variable(month_6_mye)
            b_real, month_6_mye = cuda([b_real, month_6_mye])
            
                        
            Seg_out = self.Seg(b_real)
            

            month_6_mye_plot = month_6_mye.detach().cpu().numpy()            
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,0,:,:,:]==1]=0
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,1,:,:,:]==1]=1
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,2,:,:,:]==1]=2
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,3,:,:,:]==1]=3
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,4,:,:,:]==1]=4
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,5,:,:,:]==1]=5
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,6,:,:,:]==1]=6
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,7,:,:,:]==1]=7
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,8,:,:,:]==1]=8
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,9,:,:,:]==1]=9
            month_6_mye_plot = month_6_mye_plot[:,0,:,:,:]
            
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)  
            
            combination = np.copy(Seg_out_argmax)
            combination[month_6_mye_plot==9]=9
            
            
            img_save = nib.Nifti1Image(np.float64(combination[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, create_combined_train_path +'/' + month_6_name + '_mye_cyc' + '.nii.gz')

            
            month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
            
            fig9 = plt.figure("9", (12, 18))
            ax9 = fig9.subplots(3,2)
                    
            ax9[0,0].title.set_text('6_month_real_T2')
            ax9[0,0].imshow(np.rot90(month_6_real_T2[0 , 0, :,:, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax9[0,1].title.set_text('Prediction_mye_cyc')
            ax9[0,1].imshow(np.rot90(combination[0 , :,:, round(combination.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax9[1,0].title.set_text('6_month_real_T2')
            ax9[1,0].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax9[1,1].title.set_text('Prediction_mye_cyc')
            ax9[1,1].imshow(np.rot90(combination[0 , :, round(combination.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax9[2,0].title.set_text('6_month_real_T2')
            ax9[2,0].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax9[2,1].title.set_text('Prediction_mye_cyc')
            ax9[2,1].imshow(np.rot90(combination[0, round(combination.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name
            fig9.suptitle(suptitle_name)
            fig9.savefig(create_combined_train_path+'/'+ month_6_name +'.png')

    def create_mye_com_trainset(self,args):
        print("####################### Add iBeat segmentation outputs to CycleGAN+Seg 8-tissue segmentation outputs #######################")
        print("Loading the data")
        
        
        
        train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T1.nii.gz')))
        train_images_T2 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2.nii.gz')))
        train_images_mye = sorted(glob.glob(os.path.join(train_path_6_month, '*_fixed_basalganglia_prediction.nii.gz')))
        train_images_label_ibeat = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2_iBEAT_outputs.nii.gz')))
        

        train_dicts_6_month = [
            {"image_T1": image_T1,"image_T2": image_T2,"image_label_mye": image_label_mye,"ibeat": ibeat, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:12]}
            for image_T1, image_T2, image_label_mye, ibeat in zip(train_images_T1, train_images_T2,train_images_mye, train_images_label_ibeat)
        ]


        train_transform_6_month_com = Compose(
            [
                LoadImaged(reader=NibabelReader, keys=["image_T1","image_T2","image_label_mye", "ibeat"]),
                EnsureChannelFirstd(keys=["image_T1","image_T2"]),
                ConvertToMultiChannel_mye(keys="image_label_mye"),
                ConvertToMultiChannel_ibeat(keys="ibeat"),
                ScaleIntensityd(keys=["image_T1","image_T2"],minv=-1,maxv=1),

            ]
        )
        
        
        
        train_ds_6month_com = CacheDataset(data=train_dicts_6_month, transform=train_transform_6_month_com)
        self.train_loader_6_month_com = DataLoader(train_ds_6month_com, batch_size=args.batch_size,shuffle=False, num_workers=0,pin_memory=True)
        
        
        
        dice_loss = DiceLoss()
        create_combined_train_path = root_dir + '/create_mye_combined_train_set'       
        if not os.path.exists(create_combined_train_path):
            os.makedirs(create_combined_train_path)
        
        print("Start the segmentation for 6month train")
        
        long_cy_apply_6month = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (args.checkpoint_dir))
        self.Seg.load_state_dict(long_cy_apply_6month['Seg'])
        self.Seg.eval()
        
        with open(create_combined_train_path+'/mye_combined_train_6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        

        
        for idx, month_6_real in enumerate(self.train_loader_6_month_com):

            month_6_real_T1 = month_6_real["image_T1"]
            month_6_real_T2 = month_6_real["image_T2"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]
            month_6_mye = month_6_real["image_label_mye"]
            month_6_ibeat = month_6_real["ibeat"]
            
            b_real = Variable(month_6_real_T2)
            month_6_mye = Variable(month_6_mye)
            month_6_ibeat = Variable(month_6_ibeat)
            b_real, month_6_mye, month_6_ibeat = cuda([b_real, month_6_mye, month_6_ibeat])
            
                        
            Seg_out = self.Seg(b_real)
            
            month_6_ibeat_plot = month_6_ibeat + 0   
            month_6_ibeat_plot = month_6_ibeat_plot.detach().cpu().numpy()        
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,0,:,:,:]==1]=0
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,1,:,:,:]==1]=1
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,2,:,:,:]==1]=2
            month_6_ibeat_plot[:,0,:,:,:][month_6_ibeat_plot[:,3,:,:,:]==1]=3
            month_6_ibeat_plot = month_6_ibeat_plot[:,0,:,:,:]
            
                        
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)  
            
            combination = np.copy(month_6_ibeat_plot)
            combination[Seg_out_argmax==1]=1
            combination[month_6_ibeat_plot==1]=1
            combination[month_6_ibeat_plot==2]=2
            combination[month_6_ibeat_plot==3]=3

            combination[Seg_out_argmax==4]=4
            combination[Seg_out_argmax==5]=5
            combination[Seg_out_argmax==6]=6
            combination[Seg_out_argmax==7]=7
            combination[Seg_out_argmax==8]=8
            

            month_6_mye_plot = month_6_mye.detach().cpu().numpy()            
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,0,:,:,:]==1]=0
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,1,:,:,:]==1]=1
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,2,:,:,:]==1]=2
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,3,:,:,:]==1]=3
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,4,:,:,:]==1]=4
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,5,:,:,:]==1]=5
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,6,:,:,:]==1]=6
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,7,:,:,:]==1]=7
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,8,:,:,:]==1]=8
            month_6_mye_plot[:,0,:,:,:][month_6_mye_plot[:,9,:,:,:]==1]=9
            month_6_mye_plot = month_6_mye_plot[:,0,:,:,:]
            

            combination[month_6_mye_plot==9]=9
            
            
            img_save = nib.Nifti1Image(np.float64(combination[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, create_combined_train_path +'/' + month_6_name + '_mye_combined' + '.nii.gz')

            
            month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()
            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
            
            fig10 = plt.figure("10", (12, 18))
            ax10 = fig10.subplots(3,2)
                    
            ax10[0,0].title.set_text('6_month_real_T2')
            ax10[0,0].imshow(np.rot90(month_6_real_T2[0 , 0, :,:, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax10[0,1].title.set_text('Prediction_mye_combined')
            ax10[0,1].imshow(np.rot90(combination[0 , :,:, round(combination.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax10[1,0].title.set_text('6_month_real_T2')
            ax10[1,0].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax10[1,1].title.set_text('Prediction_mye_combined')
            ax10[1,1].imshow(np.rot90(combination[0 , :, round(combination.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax10[2,0].title.set_text('6_month_real_T2')
            ax10[2,0].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax10[2,1].title.set_text('Prediction_mye_combined')
            ax10[2,1].imshow(np.rot90(combination[0, round(combination.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month_6_name
            fig10.suptitle(suptitle_name)
            fig10.savefig(create_combined_train_path+'/'+ month_6_name +'.png')






                   
        
#############################################################################
#############################################################################
#############################################################################

def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--decay_epoch', type=int, default=150)
    parser.add_argument('--LNCC_kernel_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_g', type=float, default=.00008)
    parser.add_argument('--lr_d', type=float, default=.00008)
    parser.add_argument('--lr_seg', type=float, default=.0004)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=0.5)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--add_ibeat', type=bool, default=False)
    parser.add_argument('--add_ibeat_T1', type=bool, default=True)
    parser.add_argument('--add_ibeat_T2', type=bool, default=True)
    parser.add_argument('--mye', type=bool, default=False)
    parser.add_argument('--mye_add_ibeat', type=bool, default=False)
    parser.add_argument('--create_com_train', type=bool, default=False)
    parser.add_argument('--create_not_com_train', type=bool, default=False)
    parser.add_argument('--create_mye_cyc_com_train', type=bool, default=False)
    parser.add_argument('--create_mye_ibeat_com_train', type=bool, default=False)
    parser.add_argument('--create_cover_mye_ibeat_com_train', type=bool, default=False)
    # parser.add_argument('--dataset_dir', type=str, default='./datasets/horse2zebra')
    parser.add_argument('--checkpoint_dir', type=str, default=root_dir)
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--dropout', default=True)
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='unet_128')
    parser.add_argument('--dis_net', type=str, default='n_layers')
    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()
    return args


def main():
  args = get_args()


  str_ids = args.gpu_ids.split(',')
  args.gpu_ids = []
  for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
      args.gpu_ids.append(id)
  print(args.gpu_ids)
  print(args.dropout)
  
  if args.training:
    print("Training")
    model = cycleGAN(args)
    model.train(args)
  if args.testing:
    print("Testing")
    model = cycleGAN(args)
    model.test(args)
  if args.add_ibeat:
    print("adding iBeat segmentation outputs")
    model = cycleGAN(args)
    model.add_ibeat(args)
  if args.add_ibeat_T1:
    print("adding iBeat segmentation outputs")
    model = cycleGAN(args)
    model.add_ibeat_T1w_only(args)
  if args.add_ibeat_T2:
    print("adding iBeat segmentation outputs")
    model = cycleGAN(args)
    model.add_ibeat_T2w_only(args)
  if args.create_com_train:
    print("create the combined outputs for 6-month train")
    model = cycleGAN(args)
    model.create_combined_trainset(args)
  if args.mye:
    print("replace mye with WM mean")
    model = cycleGAN(args)
    model.myelination(args)
  if args.mye_add_ibeat:
    print("add ibeat to mye outputs")
    model = cycleGAN(args)
    model.mye_add_ibeat(args)
  if args.create_not_com_train:
    print("create the not combined outputs for 6-month train")
    model = cycleGAN(args)
    model.create_not_combined_trainset(args)
  if args.create_mye_cyc_com_train:
    print("create the mye+cyc outputs for 6-month train")
    model = cycleGAN(args)
    model.create_mye_cyc_trainset(args)
  if args.create_mye_ibeat_com_train:
    print("create the mye+combined outputs for 6-month train")
    model = cycleGAN(args)
    model.create_mye_com_trainset(args)
  if args.create_cover_mye_ibeat_com_train:
    print("create the mye+combined outputs for 6-month train")
    model = cycleGAN(args)
    model.create_cover_mye_iBeat_combined_trainset(args)



if __name__ == '__main__':
    main()






