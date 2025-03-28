import os
import numpy as np
import glob
import shutil
import monai
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import DataLoader, CacheDataset
from monai.losses import LocalNormalizedCrossCorrelationLoss, BendingEnergyLoss, DiceCELoss, DiceLoss
import torch.nn.functional as F
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
    ScaleIntensity,
)
from monai.utils import set_determinism
# import itk
from monai.data import NibabelReader
import torch
import matplotlib.pyplot as plt
import functools
from torch.nn import init
import torch.nn as nn
import copy
import itertools
from torch.autograd import Variable
import torchvision.datasets as dsets
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
path = '/nfs/home/ydong/code/pippi/long_cy/save_model'
root_dir = '/nfs/home/ydong/code/pippi/long_cy_register/la6_re_long/save_model'
train_path_dhcp = '/nfs/home/ydong/dataset/DHCP_long/va_train_long/'
train_path_dhcp_test = '/nfs/home/ydong/dataset/DHCP_long/va_test_long/'
train_path_6month = '/nfs/home/ydong/dataset/6_month_long/va_train_long/'
train_path_6month_test = '/nfs/home/ydong/dataset/6_month_long/va_test_long/'
val_path_6month = '/nfs/home/ydong/dataset/6_month_long/va_before_val_5/'   # these are the 5 outputs without vk correction
test_path_6_month = '/nfs/home/ydong/dataset/6_month_long/va_test_10_subjects/'  # these are the 3 outputs with vk correction


val_result_path = root_dir + '/val_result'
logger_path = root_dir + '/log'


if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(val_result_path):
    os.makedirs(val_result_path)
if not os.path.exists(logger_path):
    os.makedirs(logger_path)

################################### DHCP ###################################
################################### DHCP ###################################
################################### DHCP ###################################
set_determinism(seed=0)
# train_images_T1 = sorted(glob.glob(os.path.join(train_path, 'sub-*_ses-*_desc-T1.nii.gz')))
train_images_T2 = sorted(glob.glob(os.path.join(train_path_dhcp, '*sub-*_ses-*_desc-T2.nii.gz')))
train_images_label = sorted(glob.glob(os.path.join(train_path_dhcp, '*sub-*_ses-*_desc-8label.nii.gz')))

train_dicts_dhcp = [
    {"image_T2": image_T2, "image_label": image_label,  "image_name": os.path.basename(image_T2)[0:-10],"affine": nib.load(image_T2).affine}
    for image_T2, image_label in zip(train_images_T2, train_images_label)
]
train_transform_dhcp = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label"]),
        ConvertToMultiChannel_dhcp(keys="image_label"),
        EnsureChannelFirstd(keys=["image_T2"]),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),

    ]
)


##########################################################################################################################################################################################################
# train_images_T1 = sorted(glob.glob(os.path.join(train_path, 'sub-*_ses-*_desc-T1.nii.gz')))
train_images_T2 = sorted(glob.glob(os.path.join(train_path_dhcp_test, '*sub-*_ses-*_desc-T2.nii.gz')))
train_images_label = sorted(glob.glob(os.path.join(train_path_dhcp_test, '*sub-*_ses-*_desc-8label.nii.gz')))

train_dicts_dhcp_test = [
    {"image_T2": image_T2, "image_label": image_label, "image_name": os.path.basename(image_T2)[0:-10],"affine": nib.load(image_T2).affine}
    for image_T2, image_label in zip(train_images_T2, train_images_label)
]
train_transform_dhcp_test = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label"]),
        ConvertToMultiChannel_dhcp(keys="image_label"),
        EnsureChannelFirstd(keys=["image_T2"]),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),
    ]
)



################################### 6_month ###################################
################################### 6_month ###################################
################################### 6_month ###################################

# train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T1.nii.gz')))
train_images_T2 = sorted(glob.glob(os.path.join(train_path_6month, '*_T2.nii.gz')))
train_images_label = sorted(glob.glob(os.path.join(train_path_6month, '*_fixed_basalganglia_prediction.nii.gz')))

train_dicts_6_month = [
    {"image_T2": image_T2, "image_label": image_label, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:-10]}
    for image_T2, image_label in zip(train_images_T2, train_images_label)
]

train_transform_6_month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label"]),
        EnsureChannelFirstd(keys=["image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),


    ]
)
#########################################################################################################################################################################################################

# train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T1.nii.gz')))
train_images_T2 = sorted(glob.glob(os.path.join(train_path_6month_test, '*_T2.nii.gz')))
train_images_label = sorted(glob.glob(os.path.join(train_path_6month_test, '*_fixed_basalganglia_prediction.nii.gz')))

train_dicts_6_month_test = [
    {"image_T2": image_T2, "image_label": image_label, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:-10]}
    for image_T2, image_label in zip(train_images_T2, train_images_label)
]

train_transform_6_month_test = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label"]),
        EnsureChannelFirstd(keys=["image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),


    ]
)


################################### 6_month val ###################################
################################### 6_month val ###################################
################################### 6_month val ###################################

# train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month_test, '*_T1.nii.gz')))
val_images_T2 = sorted(glob.glob(os.path.join(val_path_6month, '*_T2.nii.gz')))
val_images_label = sorted(glob.glob(os.path.join(val_path_6month, '*_fixed_basalganglia_prediction.nii.gz')))
# train_images_label_T2 = sorted(glob.glob(os.path.join(train_path_6_month_test, '*T2*-tissue.nii.gz')))

val_dicts_6month = [
    {"image_T2": image_T2, "image_label": image_label, "image_name": os.path.basename(image_T2)[0:-10], "affine": nib.load(image_T2).affine}
    for image_T2,image_label in zip(val_images_T2,val_images_label)
]

val_transform_6month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label"]),
        EnsureChannelFirstd(keys=["image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        # RandFlipd(keys=["image_T2","image_label"], prob=1, spatial_axis=1),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),
        # HistogramNormalized(keys=["image_T2"],min=-1,max=1),

    ]
)




################################### 6_month test ###################################
################################### 6_month test ###################################
################################### 6_month test ###################################

# train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month_test, '*_T1.nii.gz')))
test_images_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2.nii.gz')))
test_images_label_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_VK.nii.gz')))
test_images_label_ibeat = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2_iBEAT_outputs.nii.gz')))
# train_images_label_T2 = sorted(glob.glob(os.path.join(train_path_6_month_test, '*T2*-tissue.nii.gz')))

test_dicts_6month = [
    {"image_T2": image_T2, "image_label": image_label, "image_name": os.path.basename(image_T2)[0:-10], "affine": nib.load(image_T2).affine, "ibeat": image_ibeat}
    for image_T2,image_label,image_ibeat in zip(test_images_T2,test_images_label_T1,test_images_label_ibeat)
]

test_transform_6month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T2","image_label","ibeat"]),
        EnsureChannelFirstd(keys=["image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        ConvertToMultiChannel_ibeat(keys="ibeat"),
        # RandFlipd(keys=["image_T2","image_label"], prob=1, spatial_axis=1),
        ScaleIntensityd(keys=["image_T2"],minv=-1,maxv=1),
        # HistogramNormalized(keys=["image_T2"],min=-1,max=1),

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

# cycleGAN  -utils

import copy
import os
import shutil

import numpy as np
import torch

# To make directories 
def mkdir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]

def save_best_model(model, epoch, filename="long_cy_reg_apply_6month.pt", best_acc=0, dir_add=''):
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



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
#                                Voxelmorph                                    #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #


def diff_and_trim(array, axis):
    # Compute discrete spatial derivatives

    _, H, W, D = array.shape
    return np.diff(array, axis=axis)[:, : (H - 1), : (W - 1), : (D - 1)]

def jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.

    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.

    Returns a numpy array of shape (H-1,W-1,D-1).
    """

    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] = dx[0] + 1
    dy[1] = dy[1] + 1
    dz[2] = dz[2] + 1

    # Compute determinant at each spatial location
    det = (
        dx[0] * (dy[1] * dz[2] - dz[1] * dy[2])
        - dy[0] * (dx[1] * dz[2] - dz[1] * dx[2])
        + dz[0] * (dx[1] * dy[2] - dy[1] * dx[2])
    )

    return det



def plot_2D_deformation(vector_field, grid_spacing=5, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot an x-y grid warped by this deformation.

    vector_field should be a tensor of shape (2,H,W)
    """
    _, H, W = vector_field.shape
    grid_img = np.zeros((H, W))
    grid_img[np.arange(0, H, grid_spacing), :] = 1
    grid_img[:, np.arange(0, W, grid_spacing)] = 1
    grid_img = torch.tensor(grid_img, dtype=vector_field.dtype).unsqueeze(0)  # adds channel dimension, now (C,H,W)
    warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")
    grid_img_warped = warp(grid_img.unsqueeze(0), vector_field.unsqueeze(0))[0]

    return grid_img_warped[0]

#####################################################################################
# REGISTRATION NETWORK # 
#####################################################################################


def initialize_weights_(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)) or \
                    isinstance(module, nn.Linear):
                # nn.init.xavier_normal_(module.weight)
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def get_smoothing_kernel(sigma, spatial_rank):
    """
    Function that creates a smoothing kernel
    written by Wenqi
    https://github.com/NifTK/NiftyNet/blob/dev/niftynet/network/interventional_dense_net.py
    :param sigma: standard deviation of the gaussian
    :param spatial_rank: spatial rank of the tensor
    :return: the kernel
    """
    if sigma <= 0:
        raise NotImplementedError
    tail = int(sigma * 2)
    if spatial_rank == 2:
        x, y = np.mgrid[-tail:tail + 1, -tail:tail + 1]
        g = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    elif spatial_rank == 3:
        x, y, z = np.mgrid[-tail:tail + 1, -tail:tail + 1, -tail:tail + 1]
        g = np.exp(-0.5 * (x * x + y * y + z * z) / (sigma * sigma))
    else:
        raise NotImplementedError
    return g / g.sum()


# Smoothes the velocity field
def smooth_field(dense_field, sigma, spatial_rank):
    """
    Function that applies smoothing to the field
    :param dense_field: the field
    :param sigma: the standard deviation of the smoothing kernel
    :param spatial_rank: the spatial rank of the tensor
    :return: the smoothed field
    """
    kernel = get_smoothing_kernel(sigma, spatial_rank)
    kernel = cuda(torch.from_numpy(kernel).type(torch.FloatTensor))
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    ksize = kernel.shape[-1]

    smoothed = [
        F.conv3d(coord.unsqueeze(1), kernel, padding=ksize // 2)
        for coord in torch.unbind(dense_field, dim=1)]

    return torch.cat(smoothed, dim=1)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, is_half=False):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)
        self.is_half = is_half

    def forward(self, vec):
        vec_half = None
        # vec = vec * self.scale
        for i in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
            if i == self.nsteps // 2:
                vec_half = vec
        if self.is_half:
            return vec, vec_half
        else:
            return vec


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', padding='zeros'):
        super().__init__()

        self.mode = mode
        self.padding = padding

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = cuda(grid.type(torch.FloatTensor))

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations

        # return print(flow.shape)

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                                  (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode=self.padding)


class RegNetwork(nn.Module):

    def __init__(self, in_channels=4, out_channels=3, inshape=(128, 128, 128), smooth_sigma=2, name='RegNetwork'):
        super(RegNetwork, self).__init__()

        self.name = name
        print('Network chosen is {}'.format(name))

        # other params
        self.ddf_channels = out_channels
        self.steps = 4
        self.smooth_sigma = smooth_sigma
        self.integrate = VecInt(inshape, self.steps,
                                is_half=True) if self.steps > 0 else None
        self.transformer = SpatialTransformer(inshape)

        # activations
        self.tanh = nn.Tanh()

        # encoder
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

        # decoder
        self.uconv1 = nn.ConvTranspose3d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.uconv2 = nn.ConvTranspose3d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.uconv3 = nn.ConvTranspose3d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)

        # velocity field
        self.vconv1 = nn.Conv3d(
            in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.vconv2 = nn.Conv3d(
            in_channels=16, out_channels=self.ddf_channels, kernel_size=3, stride=1, padding=1)
        self.smoothing = smooth_field

        initialize_weights_(self)

    def forward(self, x):
        # encoder
        # Nb x 16 x 128 x 128
        enc1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        # Nb x 32 x  64 x  64
        enc2 = F.leaky_relu(self.conv2(enc1), negative_slope=0.2)
        # Nb x 32 x  32 x  32
        enc3 = F.leaky_relu(self.conv3(enc2), negative_slope=0.2)
        # Nb x 64 x  16 x  16
        enc4 = F.leaky_relu(self.conv4(enc3), negative_slope=0.2)

        # decoder
        # Nb x 64 x  32 x  32
        dec1 = torch.cat(
            [F.leaky_relu(self.uconv1(enc4), negative_slope=0.2), enc3], dim=1)
        # Nb x 64 x  64 x  64
        dec2 = torch.cat(
            [F.leaky_relu(self.uconv2(dec1), negative_slope=0.2), enc2], dim=1)
        # Nb x 48 x 128 x 128
        dec3 = torch.cat(
            [F.leaky_relu(self.uconv3(dec2), negative_slope=0.2), enc1], dim=1)

        # velocity field
        vel = F.leaky_relu(self.vconv1(dec3), negative_slope=0.2)
        vel = self.tanh(self.vconv2(vel))  # z (Nb x 2 (3) x 128 x 128 )
        vel = self.smoothing(vel, self.smooth_sigma, 3)

        flow_pos = vel
        flow_neg = -vel

        # Do scaling and squaring forwards and backwards
        # Here is a bit different from the official code, 
        # official integrate: 
        # this place 
        disp_field_pos, disp_half_pos = self.integrate(flow_pos) 
        disp_field_neg, disp_half_neg = self.integrate(flow_neg)

        # Moving          Fixed             Moving         Fixed
        return (disp_half_pos, disp_half_neg), (disp_field_pos, disp_field_neg), (flow_pos, flow_neg)



def record_metric(outputs_labels, path):

    
    names = ['CSF','GM','WM','Ventricle','Cerebellum','Basal Ganglia','Brainstem','Hippocampus/Amygdala']   
    post_pred = AsDiscrete(to_onehot=outputs_labels[0][1].shape[1])
    subject_number = len(outputs_labels)
    dice_metric = DiceMetric(include_background=False)
    
    average_dice = 0
    average_hausdoff = 0
    average_surface = 0
    
    for x in range(subject_number):
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
    
        
    
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['Brain tissue','Dice','Hausdoff95','Average surface']
        csv_write.writerow(csv_head)
        for i in range(len(names)):
            data = (names[i],detail_output_dice[i],detail_output_hausdoff[i],detail_output_surface[i])
            csv_write.writerow(data)
        
        data = ('Average',average_dice,average_hausdoff,average_surface)
        csv_write.writerow(data)



#############################################################################
#############################################################################
#############################################################################

class cycle_register(object):
    def __init__(self,args):

        # Define the network 
        #####################################################
        # self.Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
        #                                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        # self.Da = define_Dis(input_nc=1, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        # self.Db = define_Dis(input_nc=1, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.Seg=AttentionUnet(
                               spatial_dims=3,
                               in_channels=1,
                               out_channels=9,
                               channels=(32, 64, 128, 256, 512),
                               strides=(2, 2, 2, 2),
                               ).to(device)
        
        
        self.model = RegNetwork(in_channels=2,
                                out_channels=3,
                                inshape=args.im_size,
                                smooth_sigma=args.smooth_sigma).to(device)

        print_networks([self.model,self.Seg], ['Voxelmorph','Seg'])
        print_networks([self.Gba], ['Gba'])

        # Define Loss criterias
        self.LNCC = LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=args.LNCC_kernel_size)
        self.regularization = BendingEnergyLoss()
        self.diceceloss = DiceCELoss(to_onehot_y=False,include_background=True,softmax=True)
        self.min_max01 = ScaleIntensity(minv=0, maxv=1)
        self.min_max11 = ScaleIntensity(minv=-1, maxv=1)


        # Optimizers
        #####################################################
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr_r, weight_decay=args.weight_decay_r)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
        
        self.Seg_optimizer = torch.optim.Adam(self.Seg.parameters(), lr=args.lr_seg, betas=(0.5, 0.999))
        self.Seg_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.Seg_optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
        
           
        test_ds_6_month = CacheDataset(data=test_dicts_6month, transform=test_transform_6month)
        self.test_loader_6_month = DataLoader(test_ds_6_month, batch_size=args.batch_size, num_workers=0,pin_memory=True)
        
        

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            register = load_checkpoint('%s/register_seg.ckpt' % (args.checkpoint_dir))          
            self.logger=SummaryWriter(log_dir=logger_path)
            self.start_epoch = register['epoch']
            self.total_number = register['total_number']
            self.best_val_accuracy = register['best_val_accuracy']
            self.model.load_state_dict(register['model'])
            self.optimizer.load_state_dict(register['optimizer'])
            self.Seg.load_state_dict(register['Seg'])
            self.Seg_optimizer.load_state_dict(register['Seg_optimizer'])
            
            ckpt = load_checkpoint('%s/latest.ckpt' % (path))
            # self.Da.load_state_dict(ckpt['Da'])
            # self.Db.load_state_dict(ckpt['Db'])
            # self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            print('The start epoch is: ', self.start_epoch)


        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 1
            self.logger=SummaryWriter(log_dir=logger_path)
            self.total_number=0
            self.best_val_accuracy = -1
            
            ckpt_g = load_checkpoint('%s/latest.ckpt' % (path))
            ckpt_seg = load_checkpoint('%s/long_cy_apply_6month.ckpt' % (path))
            self.Seg.load_state_dict(ckpt_seg['Seg'])
            # self.Da.load_state_dict(ckpt['Da'])
            # self.Db.load_state_dict(ckpt['Db'])
            # self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt_g['Gba'])
            print('The start epoch is: ', self.start_epoch)
            



    def train(self,args):

        set_grad([self.Gba], False)   
        
        # Data
        #####################################################     
        train_ds_dhcp = CacheDataset(data=train_dicts_dhcp, transform=train_transform_dhcp)
        self.train_loader_dhcp = DataLoader(train_ds_dhcp, batch_size=args.batch_size,shuffle=False, num_workers=0,pin_memory=True)
        
        train_ds_dhcp_test = CacheDataset(data=train_dicts_dhcp_test, transform=train_transform_dhcp_test)
        self.train_loader_dhcp_test = DataLoader(train_ds_dhcp_test, batch_size=args.batch_size,shuffle=False, num_workers=0,pin_memory=True)
        
        
        train_ds_6month = CacheDataset(data=train_dicts_6_month, transform=train_transform_6_month)
        self.train_loader_6month = DataLoader(train_ds_6month, batch_size=args.batch_size,shuffle=False, num_workers=0,pin_memory=True)
        
        train_ds_6month_test = CacheDataset(data=train_dicts_6_month_test, transform=train_transform_6_month_test)

        self.train_loader_6month_test = DataLoader(train_ds_6month_test, batch_size=args.batch_size,shuffle=False, num_workers=0,pin_memory=True)
        
        
        
        
        val_ds_6_month = CacheDataset(data=val_dicts_6month, transform=val_transform_6month)
        self.val_loader_6month = DataLoader(val_ds_6_month, batch_size=args.batch_size, num_workers=0,pin_memory=True)
        
        
           

        for epoch in range(self.start_epoch, args.epochs+1):

            lr_model = self.optimizer.param_groups[0]['lr']
            print('learning rate of model = %.7f' % lr_model)
            
            log_LNCC = 0
            log_reg = 0
            epoch_loss = 0
            average_LNCC= 0
            average_regularization = 0
            average_label_loss = 0
            average_seg_loss = 0
           
            # for idx, batch_data in enumerate(train_loader):
            for idx, (dhcp_real, month_6_real) in enumerate(zip(self.train_loader_dhcp, self.train_loader_6month)):

                set_grad([self.model], True)
                set_grad([self.Seg], False)
                self.model.train()
                self.optimizer.zero_grad()
                self.total_number = self.total_number + 1
                
                a_real = dhcp_real["image_T2"]
                a_real_name = dhcp_real["image_name"][0]
                moving_label = dhcp_real["image_label"]
                mask = torch.zeros_like(a_real)
                mask[0,0,:,:,:] = 1-moving_label[0,0,:,:,:]

                
                b_real = month_6_real["image_T2"] 
                b_real_name = month_6_real["image_name"][0]
                fixed_label = month_6_real["image_label"]               



                # Generator Computations
                ##################################################
                a_real = Variable(a_real)
                b_real = Variable(b_real)
                a_real, b_real, moving_label, fixed_label, mask = cuda([a_real, b_real, moving_label, fixed_label, mask])

                # Forward pass through generators
                ##################################################
                b_fake = self.Gba(a_real)
                moving_image = self.min_max01(b_fake)
                moving_image = moving_image * mask
                fixed_image = self.min_max01(b_real)
                
                
                (disp_half_pos, disp_half_neg), (disp_field_pos, disp_field_neg), (flow_pos, flow_neg) = self.model(torch.cat((moving_image, fixed_image), dim=1))
                moving_warped = SpatialTransformer(size=args.im_size)(moving_image, disp_field_pos)
                moving_label_warped = SpatialTransformer(size=args.im_size, mode='bilinear')(moving_label, disp_field_pos)             
    
                LNCC_loss = self.LNCC(moving_warped, fixed_image)
                regularization_loss = self.regularization(disp_field_pos)
                label_loss = self.diceceloss(moving_label_warped, fixed_label)
                loss = 1 * LNCC_loss +  args.lamuda * regularization_loss
                loss.backward(retain_graph=True)
                self.optimizer.step()
                
                
                # Segmentation losses
                ################################################### 
                set_grad([self.model], False)
                set_grad([self.Seg], True)
                self.Seg.train()
                self.Seg_optimizer.zero_grad()

                moving_image = Variable(torch.Tensor(moving_image.cpu().data.numpy()))
                moving_label = Variable(torch.Tensor(moving_label.cpu().data.numpy()))
                disp_field_pos = Variable(torch.Tensor(disp_field_pos.cpu().data.numpy()))

                moving_image, moving_label, disp_field_pos = cuda([moving_image, moving_label, disp_field_pos])                  
                moving_warped_copy = SpatialTransformer(size=args.im_size)(moving_image, disp_field_pos)
                moving_label_warped_copy = SpatialTransformer(size=args.im_size, mode='bilinear')(moving_label, disp_field_pos)
                moving_warped_copy = self.min_max11(moving_warped_copy)

                Seg_out = self.Seg(moving_warped_copy)
                seg_loss = self.diceceloss(Seg_out, moving_label_warped_copy)       
                seg_loss.backward()
                self.Seg_optimizer.step()
                
                epoch_loss += loss.item()
                average_LNCC += LNCC_loss.item()
                average_regularization += regularization_loss.item()
                average_label_loss += label_loss.item()
                average_seg_loss = average_seg_loss + seg_loss.item()             
                
                
                

       
                print("Epoch: (%3d) (%5d/%5d) | LNCC Loss:%.2e | regularization loss:%.2e | label dice loss:%.2e | Seg dice loss:%.2e" % 
                                            (epoch, idx + 1, min(len(self.train_loader_dhcp), len(self.train_loader_6month)),LNCC_loss.item(),regularization_loss.item(),label_loss.item(),seg_loss.item()))
        
        


                with open(root_dir+'/val_result/'+'/logs.txt',"a") as file:
                    file.write('Epoch: ')
                    file.write(str(epoch))
                    file.write(', ')
                    file.write('( ')
                    file.write(str(idx+1))
                    file.write(' / ')
                    file.write(str(len(self.train_loader_dhcp)))
                    file.write(' )\n ')
                    
                    
                    
                
                
                self.logger.add_scalar('LNCC_loss', LNCC_loss.item(), self.total_number)  
                self.logger.add_scalar('Regularization_loss', regularization_loss.item(), self.total_number)   
                self.logger.add_scalar('label_dice_loss', label_loss.item(), self.total_number) 
                self.logger.add_scalar('Seg_dice_loss', seg_loss.item(), self.total_number) 
                
                if idx == 1:
                    moving_image = moving_image.detach().cpu().numpy()
                    moving_label_plot = moving_label.detach().cpu().numpy()
                    fixed_image = fixed_image.detach().cpu().numpy()
                    fixed_label_plot = fixed_label.detach().cpu().numpy()
                    moving_warped = moving_warped.detach().cpu().numpy()
                    moving_label_warped_plot = moving_label_warped.detach().cpu().numpy()
                    moving_label_warped_plot = np.argmax(moving_label_warped_plot, axis=1)
                    moving_label_warped_plot = moving_label_warped_plot[0,:,:,:]
                    Seg_out_plot = Seg_out.detach().cpu().numpy()
                    Seg_out_argmax = np.argmax(Seg_out_plot, axis=1)   
                    Seg_out_argmax = Seg_out_argmax[0,:,:,:]
                    # moving_label_warped_plot[moving_label_warped_plot>=0.5]=1
                    # moving_label_warped_plot[moving_label_warped_plot<0.5]=0
                    
                    jacobian = jacobian_determinant(disp_field_pos.cpu().detach()[0,:,:,:,:])
                    
                    moving_image = moving_image[0,0,:,:,:]
                    fixed_image = fixed_image[0,0,:,:,:]
                    moving_warped = moving_warped[0,0,:,:,:]

                    
                    moving_label_plot[:,0,:,:,:][moving_label_plot[:,0,:,:,:]==1]=0
                    moving_label_plot[:,0,:,:,:][moving_label_plot[:,1,:,:,:]==1]=1
                    moving_label_plot[:,0,:,:,:][moving_label_plot[:,2,:,:,:]==1]=2
                    moving_label_plot[:,0,:,:,:][moving_label_plot[:,3,:,:,:]==1]=3
                    moving_label_plot[:,0,:,:,:][moving_label_plot[:,4,:,:,:]==1]=4
                    moving_label_plot[:,0,:,:,:][moving_label_plot[:,5,:,:,:]==1]=5
                    moving_label_plot[:,0,:,:,:][moving_label_plot[:,6,:,:,:]==1]=6
                    moving_label_plot[:,0,:,:,:][moving_label_plot[:,7,:,:,:]==1]=7
                    moving_label_plot[:,0,:,:,:][moving_label_plot[:,8,:,:,:]==1]=8
                    # moving_label_plot[:,0,:,:,:][moving_label_plot[:,9,:,:,:]==1]=9
                    moving_label_plot = moving_label_plot[0,0,:,:,:]
                    
                    fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,0,:,:,:]==1]=0
                    fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,1,:,:,:]==1]=1
                    fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,2,:,:,:]==1]=2
                    fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,3,:,:,:]==1]=3
                    fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,4,:,:,:]==1]=4
                    fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,5,:,:,:]==1]=5
                    fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,6,:,:,:]==1]=6
                    fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,7,:,:,:]==1]=7
                    fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,8,:,:,:]==1]=8
                    fixed_label_plot = fixed_label_plot[0,0,:,:,:]
                    
                    # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,0,:,:,:]==1]=0
                    # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,1,:,:,:]==1]=1
                    # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,2,:,:,:]==1]=2
                    # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,3,:,:,:]==1]=3
                    # moving_label_warped_plot = moving_label_warped_plot[0,0,:,:,:]
                    
                    for i in range(1,5): 
                        # i = 1, 1,2,3,4,5,6,7,8,9,...24
                        # (i-1)*24+1, (i-1)*18+2
                        depth = i * 25
                        plt.figure("0", (54, 72))
                        plt.subplot(12, 9, (i-1)*27+1)
                        plt.title(f"moving_image {idx} d={depth}")
                        plt.imshow(np.rot90(moving_image[:, :, depth],k=1), cmap="gray")
                        plt.subplot(12, 9, (i-1)*27+2)
                        plt.title(f"fixed_image {idx} d={depth}")
                        plt.imshow(np.rot90(fixed_image[:, :, depth],k=1), cmap="gray")
                        plt.subplot(12, 9, (i-1)*27+3)
                        plt.title(f"warp_image {idx} d={depth}")
                        plt.imshow(np.rot90(moving_warped[:, :, depth],k=1), cmap="gray")

                        plt.subplot(12, 9, (i-1)*27+4)
                        plt.title(f"moving_label {idx} d={depth}")
                        plt.imshow(np.rot90(moving_label_plot[:, :, depth],k=1), cmap="jet",vmin=0,vmax=9)
                        plt.subplot(12, 9, (i-1)*27+5)
                        plt.title(f"fixed_label {idx} d={depth}")
                        plt.imshow(np.rot90(fixed_label_plot[:, :, depth],k=1), cmap="jet",vmin=0,vmax=9)
                        plt.subplot(12, 9, (i-1)*27+6)
                        plt.title(f"warp_label {idx} d={depth}")
                        plt.imshow(np.rot90(moving_label_warped_plot[:, :, depth],k=1), cmap="jet",vmin=0,vmax=9)
                        plt.subplot(12, 9, (i-1)*27+7)
                        plt.title(f"fake 6month Seg {idx} d={depth}")
                        plt.imshow(np.rot90(Seg_out_argmax[:, :, depth],k=1), cmap="jet",vmin=0,vmax=9)
                        
                        plt.subplot(12, 9, (i-1)*27+8)
                        plt.title(f"deformation {idx} d={depth}")
                        plt.imshow(np.rot90(plot_2D_deformation(disp_field_pos.cpu().detach()[0,[0, 1],:,:,depth]),k=1), cmap="gist_gray")
                        plt.subplot(12, 9, (i-1)*27+9)
                        plt.title(f"jacobian {idx} d={depth}")
                        plt.imshow(np.rot90(jacobian[:, :, depth],k=1))
                        
                    
                    
                    
                    
                        plt.subplot(12, 9, (i-1)*27+10)
                        plt.title(f"moving_image {idx} d={depth}")
                        plt.imshow(np.rot90(moving_image[:, depth, :],k=1), cmap="gray")
                        plt.subplot(12, 9, (i-1)*27+11)
                        plt.title(f"fixed_image {idx} d={depth}")
                        plt.imshow(np.rot90(fixed_image[:, depth, :],k=1), cmap="gray")
                        plt.subplot(12, 9, (i-1)*27+12)
                        plt.title(f"pred_image {idx} d={depth}")
                        plt.imshow(np.rot90(moving_warped[:, depth, :],k=1), cmap="gray")
                        
                        plt.subplot(12, 9, (i-1)*27+13)
                        plt.title(f"moving_label {idx} d={depth}")
                        plt.imshow(np.rot90(moving_label_plot[:, depth, :],k=1), cmap="jet",vmin=0,vmax=9)
                        plt.subplot(12, 9, (i-1)*27+14)
                        plt.title(f"fixed_label {idx} d={depth}")
                        plt.imshow(np.rot90(fixed_label_plot[:, depth, :],k=1), cmap="jet",vmin=0,vmax=9)
                        plt.subplot(12, 9, (i-1)*27+15)
                        plt.title(f"warp_label {idx} d={depth}")
                        plt.imshow(np.rot90(moving_label_warped_plot[:, depth, :],k=1), cmap="jet",vmin=0,vmax=9)
                        plt.subplot(12, 9, (i-1)*27+16)
                        plt.title(f"fake 6month Seg {idx} d={depth}")
                        plt.imshow(np.rot90(Seg_out_argmax[:, depth, :],k=1), cmap="jet",vmin=0,vmax=9)
                        
                        plt.subplot(12, 9, (i-1)*27+17)
                        plt.title(f"deformation {idx} d={depth}")
                        plt.imshow(np.rot90(plot_2D_deformation(disp_field_pos.cpu().detach()[0,[0, 2],:,depth,:]),k=1), cmap="gist_gray")
                        plt.subplot(12, 9, (i-1)*27+18)
                        plt.title(f"jacobian {idx} d={depth}")
                        plt.imshow(np.rot90(jacobian[:, depth, :],k=1))
                        
                        
                        
        
                    
                        plt.subplot(12, 9, (i-1)*27+19)
                        plt.title(f"moving_image {idx} d={depth}")
                        plt.imshow(np.rot90(moving_image[depth, :, :],k=1), cmap="gray")
                        plt.subplot(12, 9, (i-1)*27+20)
                        plt.title(f"fixed_image {idx} d={depth}")
                        plt.imshow(np.rot90(fixed_image[depth, :, :],k=1), cmap="gray")
                        plt.subplot(12, 9, (i-1)*27+21)
                        plt.title(f"pred_image {idx} d={depth}")
                        plt.imshow(np.rot90(moving_warped[depth, :, :],k=1), cmap="gray")
                        
                        plt.subplot(12, 9, (i-1)*27+22)
                        plt.title(f"moving_label {idx} d={depth}")
                        plt.imshow(np.rot90(moving_label_plot[depth, :, :],k=1), cmap="jet",vmin=0,vmax=9)
                        plt.subplot(12, 9, (i-1)*27+23)
                        plt.title(f"fixed_label {idx} d={depth}")
                        plt.imshow(np.rot90(fixed_label_plot[depth, :, :],k=1), cmap="jet",vmin=0,vmax=9)
                        plt.subplot(12, 9, (i-1)*27+24)
                        plt.title(f"warp_label {idx} d={depth}")
                        plt.imshow(np.rot90(moving_label_warped_plot[depth, :, :],k=1), cmap="jet",vmin=0,vmax=9)
                        plt.subplot(12, 9, (i-1)*27+25)
                        plt.title(f"fake 6month Seg {idx} d={depth}")
                        plt.imshow(np.rot90(Seg_out_argmax[depth, :, :],k=1), cmap="jet",vmin=0,vmax=9)
                        
                        plt.subplot(12, 9, (i-1)*27+26)
                        plt.title(f"deformation {idx} d={depth}")
                        plt.imshow(np.rot90(plot_2D_deformation(disp_field_pos.cpu().detach()[0,[1, 2],depth,:,:]),k=1), cmap="gist_gray")
                        plt.subplot(12, 9, (i-1)*27+27)
                        plt.title(f"jacobian {idx} d={depth}")
                        plt.imshow(np.rot90(jacobian[depth, :, :],k=1))
                    
                    suptitle_name = str(epoch) +'  fix:' + b_real_name + "  move:" + a_real_name 
                    plt.suptitle(suptitle_name)
                    plt.savefig(root_dir + '/'+ str(epoch) +'.png')
                
            epoch_loss /= (idx+1)

            average_LNCC /= (idx+1)
            average_regularization /= (idx+1)
            average_label_loss /= (idx+1)
            average_seg_loss /= (idx+1)
            self.logger.add_scalar('Average_LNCC_loss', average_LNCC, epoch)  
            self.logger.add_scalar('Average_regularization_loss', average_regularization, epoch)  
            self.logger.add_scalar('Average_label_loss', average_label_loss, epoch) 
            self.logger.add_scalar('Average_Seg_loss', average_seg_loss, epoch)
            
            print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
            
            with open(root_dir+'/val_result/'+'/Train_results.txt',"a") as file:
                file.write('Epoch: ')
                file.write(str(epoch))
                file.write(', ')
                file.write('average loss = ')
                file.write(str(epoch_loss))
                file.write(', average image loss = ')
                file.write(str(average_LNCC))
                file.write(', average regularization dice loss = ')
                file.write(str(average_regularization))
                file.write(', average label dice loss = ')
                file.write(str(average_label_loss))
                file.write(', average Seg dice loss = ')
                file.write(str(average_seg_loss))
                file.write('\n')

                          
            if epoch == args.epochs or epoch == 100:
                self.model.eval()
                with torch.no_grad():
                    for idx, (dhcp_real, month_6_real) in enumerate(zip(self.train_loader_dhcp_test, self.train_loader_6month_test)):
                    
                        a_real = dhcp_real["image_T2"]
                        moving_label = dhcp_real["image_label"]
                        moving_name = dhcp_real["image_name"][0]
                        mask = torch.zeros_like(a_real)
                        mask[0,0,:,:,:] = 1- moving_label[0,0,:,:,:]
                        
                        

                
                        b_real = month_6_real["image_T2"] 
                        fixed_label = month_6_real["image_label"]               
                        fixed_name = month_6_real["image_name"][0]
                        fixed_affine = month_6_real["affine"]



                        # Generator Computations
                        ##################################################
                        set_grad([self.Gba], False)
                        a_real = Variable(a_real)
                        b_real = Variable(b_real)
                        a_real, b_real, moving_label, fixed_label, mask = cuda([a_real, b_real, moving_label, fixed_label, mask])
                        

                        # Forward pass through generators
                        ##################################################
                        b_fake = self.Gba(a_real)
                        moving_image = self.min_max01(b_fake)
                        moving_image = moving_image * mask
                        fixed_image = self.min_max01(b_real)
               
                        
                        (disp_half_pos, disp_half_neg), (disp_field_pos, disp_field_neg), (flow_pos, flow_neg) = self.model(torch.cat((moving_image, fixed_image), dim=1))

                        moving_warped = SpatialTransformer(size=args.im_size)(moving_image, disp_field_pos)
                        moving_label_warped = SpatialTransformer(size=args.im_size, mode='bilinear')(moving_label, disp_field_pos)
                        
                        val_loss = self.LNCC(moving_warped, fixed_image)
                        regularization_loss = self.regularization(disp_field_pos)
                        label_loss = self.diceceloss(moving_label_warped, fixed_label)
                        
           
                        moving_image = moving_image.detach().cpu().numpy()
                        moving_label_plot = moving_label.detach().cpu().numpy()
                        fixed_image = fixed_image.detach().cpu().numpy()
                        fixed_label_plot = fixed_label.detach().cpu().numpy()
                        moving_warped = moving_warped.detach().cpu().numpy()
                        moving_label_warped_plot = moving_label_warped.detach().cpu().numpy()
                        moving_label_warped_plot = np.argmax(moving_label_warped_plot, axis=1)
                        moving_label_warped_plot = moving_label_warped_plot[0,:,:,:]
                        
                        jacobian = jacobian_determinant(disp_field_pos.cpu().detach()[0,:,:,:,:])
                    
                        moving_image = moving_image[0,0,:,:,:]
                        fixed_image = fixed_image[0,0,:,:,:]
                        moving_warped = moving_warped[0,0,:,:,:]
                    
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,0,:,:,:]==1]=0
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,1,:,:,:]==1]=1
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,2,:,:,:]==1]=2
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,3,:,:,:]==1]=3
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,4,:,:,:]==1]=4
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,5,:,:,:]==1]=5
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,6,:,:,:]==1]=6
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,7,:,:,:]==1]=7
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,8,:,:,:]==1]=8
                        # moving_label_plot[:,0,:,:,:][moving_label_plot[:,9,:,:,:]==1]=9
                        moving_label_plot = moving_label_plot[0,0,:,:,:]
                    
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,0,:,:,:]==1]=0
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,1,:,:,:]==1]=1
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,2,:,:,:]==1]=2
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,3,:,:,:]==1]=3
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,4,:,:,:]==1]=4
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,5,:,:,:]==1]=5
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,6,:,:,:]==1]=6
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,7,:,:,:]==1]=7
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,8,:,:,:]==1]=8
                        fixed_label_plot = fixed_label_plot[0,0,:,:,:]
                    
                        # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,0,:,:,:]==1]=0
                        # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,1,:,:,:]==1]=1
                        # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,2,:,:,:]==1]=2
                        # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,3,:,:,:]==1]=3
                        # moving_label_warped_plot = moving_label_warped_plot[0,0,:,:,:]   
          
                        
                        img_save = nib.Nifti1Image(fixed_image, fixed_affine.detach().cpu().numpy()[0,:,:])
                        nib.save(img_save, root_dir +'/val_result/' + str(epoch) + '_' + str(idx+1) + '_fix_'+fixed_name + '_T2' + '.nii.gz')
                        img_save = nib.Nifti1Image(np.float64(fixed_label_plot), fixed_affine.detach().cpu().numpy()[0,:,:])
                        nib.save(img_save, root_dir +'/val_result/' + str(epoch) + '_' + str(idx+1) + '_fix_' + fixed_name + '_label' + '.nii.gz')     
                         
                        img_save = nib.Nifti1Image(moving_image, fixed_affine.detach().cpu().numpy()[0,:,:])
                        nib.save(img_save, root_dir +'/val_result/' + str(epoch) + '_' + str(idx+1) + '_move_'+moving_name + '_T2' + '.nii.gz')
                        img_save = nib.Nifti1Image(np.float64(moving_label_plot), fixed_affine.detach().cpu().numpy()[0,:,:])
                        nib.save(img_save, root_dir +'/val_result/' + str(epoch) + '_' + str(idx+1) + '_move_' + moving_name + '_label' + '.nii.gz')   
                        
                        img_save = nib.Nifti1Image(moving_warped, fixed_affine.detach().cpu().numpy()[0,:,:])
                        nib.save(img_save, root_dir +'/val_result/' + str(epoch) + '_' + str(idx+1) + '_wrap_'+moving_name + '_T2' + '.nii.gz')
                        img_save = nib.Nifti1Image(np.float64(moving_label_warped_plot), fixed_affine.detach().cpu().numpy()[0,:,:])
                        nib.save(img_save, root_dir +'/val_result/' + str(epoch) + '_' + str(idx+1) + '_wrap_' + moving_name + '_label' + '.nii.gz')  
                        
                        img_save = nib.Nifti1Image(jacobian, fixed_affine.detach().cpu().numpy()[0,:,:])
                        nib.save(img_save, root_dir +'/val_result/' + str(epoch) + '_' + str(idx+1) + '_jacobian' + '.nii.gz')
            
               
            if epoch % args.val_interval == 0 or epoch == 1:
                self.model.eval()
                self.Seg.eval()
                with torch.no_grad():
                    average_val_loss = 0
                    average_regularization_loss = 0
                    average_label_dice_loss = 0   
                    average_Seg_dice_loss = 0                                           
                    for idx, (dhcp_real, month_6_real) in enumerate(zip(self.train_loader_dhcp_test, self.train_loader_6month_test)):
                    
                        a_real = dhcp_real["image_T2"]
                        moving_label = dhcp_real["image_label"]
                        moving_name = dhcp_real["image_name"][0]
                        mask = torch.zeros_like(a_real)
                        mask[0,0,:,:,:] = 1-moving_label[0,0,:,:,:]
                 

                
                        b_real = month_6_real["image_T2"] 
                        fixed_label = month_6_real["image_label"]               
                        fixed_name = month_6_real["image_name"][0]
                        fixed_affine = month_6_real["affine"]   
                        
                        # Generator Computations
                        ##################################################
                        a_real = Variable(a_real)
                        b_real = Variable(b_real)
                        a_real, b_real, moving_label, fixed_label, mask = cuda([a_real, b_real, moving_label, fixed_label, mask])
                        

                        # Forward pass through generators
                        ##################################################
                        b_fake = self.Gba(a_real)
                        
                        moving_image = self.min_max01(b_fake)
                        moving_image = moving_image * mask
                        fixed_image = self.min_max01(b_real)
                       
                        (disp_half_pos, disp_half_neg), (disp_field_pos, disp_field_neg), (flow_pos, flow_neg) = self.model(torch.cat((moving_image, fixed_image), dim=1))

                        moving_warped = SpatialTransformer(size=args.im_size)(moving_image, disp_field_pos)
                        moving_label_warped = SpatialTransformer(size=args.im_size, mode='bilinear')(moving_label, disp_field_pos)
                        
                        val_loss = self.LNCC(moving_warped, fixed_image)
                        regularization_loss = self.regularization(disp_field_pos)
                        average_val_loss = average_val_loss + val_loss.item()
                        average_regularization_loss = average_regularization_loss + regularization_loss.item()
                        label_loss = self.diceceloss(moving_label_warped, fixed_label)
                        
                        # Segmentation losses
                        ###################################################                              
                        Seg_out = self.Seg(b_real)
                        seg_loss = self.diceceloss(Seg_out, fixed_label)
                        seg_dice= 1-seg_loss
 
                        
                        average_label_dice_loss = average_label_dice_loss + label_loss.item()
                        average_Seg_dice_loss = average_Seg_dice_loss + seg_loss.item()
                        moving_image = moving_image.detach().cpu().numpy()
                        moving_label_plot = moving_label.detach().cpu().numpy()
                        fixed_image = fixed_image.detach().cpu().numpy()
                        fixed_label_plot = fixed_label.detach().cpu().numpy()
                        moving_warped = moving_warped.detach().cpu().numpy()
                        moving_label_warped_plot = moving_label_warped.detach().cpu().numpy()
                        moving_label_warped_plot = np.argmax(moving_label_warped_plot, axis=1)
                        moving_label_warped_plot = moving_label_warped_plot[0,:,:,:]
                        
                        Seg_out_plot = Seg_out.detach().cpu().numpy()
                        Seg_out_argmax = np.argmax(Seg_out_plot, axis=1)   
                        Seg_out_argmax = Seg_out_argmax[0,:,:,:]
                        
                        
                        
                        jacobian = jacobian_determinant(disp_field_pos.cpu().detach()[0,:,:,:,:])
                    
                        moving_image = moving_image[0,0,:,:,:]
                        fixed_image = fixed_image[0,0,:,:,:]
                        moving_warped = moving_warped[0,0,:,:,:]
                    
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,0,:,:,:]==1]=0
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,1,:,:,:]==1]=1
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,2,:,:,:]==1]=2
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,3,:,:,:]==1]=3
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,4,:,:,:]==1]=4
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,5,:,:,:]==1]=5
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,6,:,:,:]==1]=6
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,7,:,:,:]==1]=7
                        moving_label_plot[:,0,:,:,:][moving_label_plot[:,8,:,:,:]==1]=8
                        # moving_label_plot[:,0,:,:,:][moving_label_plot[:,9,:,:,:]==1]=9
                        moving_label_plot = moving_label_plot[0,0,:,:,:]
                    
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,0,:,:,:]==1]=0
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,1,:,:,:]==1]=1
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,2,:,:,:]==1]=2
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,3,:,:,:]==1]=3
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,4,:,:,:]==1]=4
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,5,:,:,:]==1]=5
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,6,:,:,:]==1]=6
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,7,:,:,:]==1]=7
                        fixed_label_plot[:,0,:,:,:][fixed_label_plot[:,8,:,:,:]==1]=8
                        fixed_label_plot = fixed_label_plot[0,0,:,:,:]
                    
                        # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,0,:,:,:]==1]=0
                        # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,1,:,:,:]==1]=1
                        # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,2,:,:,:]==1]=2
                        # moving_label_warped_plot[:,0,:,:,:][moving_label_warped_plot[:,3,:,:,:]==1]=3
                        # moving_label_warped_plot = moving_label_warped_plot[0,0,:,:,:]
                        
                        for i in range(1,5): 
                            # i = 1, 1,2,3,4,5,6,7,8,24,
                            depth = i * 25
                            plt.figure("2", (54, 72))
                            plt.subplot(12, 9, (i-1)*27+1)
                            plt.title(f"moving_image {idx} d={depth}")
                            plt.imshow(np.rot90(moving_image[:, :, depth],k=1), cmap="gray")
                            plt.subplot(12, 9, (i-1)*27+2)
                            plt.title(f"fixed_image {idx} d={depth}")
                            plt.imshow(np.rot90(fixed_image[:, :, depth],k=1), cmap="gray")
                            plt.subplot(12, 9, (i-1)*27+3)
                            plt.title(f"warp_image {idx} d={depth}")
                            plt.imshow(np.rot90(moving_warped[:, :, depth],k=1), cmap="gray")

                            plt.subplot(12, 9, (i-1)*27+4)
                            plt.title(f"moving_label {idx} d={depth}")
                            plt.imshow(np.rot90(moving_label_plot[:, :, depth],k=1), cmap="jet",vmin=0,vmax=9)
                            plt.subplot(12, 9, (i-1)*27+5)
                            plt.title(f"fixed_label {idx} d={depth}")
                            plt.imshow(np.rot90(fixed_label_plot[:, :, depth],k=1), cmap="jet",vmin=0,vmax=9)
                            plt.subplot(12, 9, (i-1)*27+6)
                            plt.title(f"warp_label {idx} d={depth}")
                            plt.imshow(np.rot90(moving_label_warped_plot[:, :, depth],k=1), cmap="jet",vmin=0,vmax=9)
                            plt.subplot(12, 9, (i-1)*27+7)
                            plt.title(f"real 6month Seg {idx} d={depth}")
                            plt.imshow(np.rot90(Seg_out_argmax[:, :, depth],k=1), cmap="jet",vmin=0,vmax=9)
                        
                            plt.subplot(12, 9, (i-1)*27+8)
                            plt.title(f"deformation {idx} d={depth}")
                            plt.imshow(np.rot90(plot_2D_deformation(disp_field_pos.cpu().detach()[0,[0, 1],:,:,depth]),k=1), cmap="gist_gray")
                            plt.subplot(12, 9, (i-1)*27+9)
                            plt.title(f"jacobian {idx} d={depth}")
                            plt.imshow(np.rot90(jacobian[:, :, depth],k=1))
                        
                    
                    
                    
                    
                            plt.subplot(12, 9, (i-1)*27+10)
                            plt.title(f"moving_image {idx} d={depth}")
                            plt.imshow(np.rot90(moving_image[:, depth, :],k=1), cmap="gray")
                            plt.subplot(12, 9, (i-1)*27+11)
                            plt.title(f"fixed_image {idx} d={depth}")
                            plt.imshow(np.rot90(fixed_image[:, depth, :],k=1), cmap="gray")
                            plt.subplot(12, 9, (i-1)*27+12)
                            plt.title(f"pred_image {idx} d={depth}")
                            plt.imshow(np.rot90(moving_warped[:, depth, :],k=1), cmap="gray")
                        
                            plt.subplot(12, 9, (i-1)*27+13)
                            plt.title(f"moving_label {idx} d={depth}")
                            plt.imshow(np.rot90(moving_label_plot[:, depth, :],k=1), cmap="jet",vmin=0,vmax=9)
                            plt.subplot(12, 9, (i-1)*27+14)
                            plt.title(f"fixed_label {idx} d={depth}")
                            plt.imshow(np.rot90(fixed_label_plot[:, depth, :],k=1), cmap="jet",vmin=0,vmax=9)
                            plt.subplot(12, 9, (i-1)*27+15)
                            plt.title(f"warp_label {idx} d={depth}")
                            plt.imshow(np.rot90(moving_label_warped_plot[:, depth, :],k=1), cmap="jet",vmin=0,vmax=9)
                            plt.subplot(12, 9, (i-1)*27+16)
                            plt.title(f"real 6month Seg {idx} d={depth}")
                            plt.imshow(np.rot90(Seg_out_argmax[:, depth, :],k=1), cmap="jet",vmin=0,vmax=9)
                        
                            plt.subplot(12, 9, (i-1)*27+17)
                            plt.title(f"deformation {idx} d={depth}")
                            plt.imshow(np.rot90(plot_2D_deformation(disp_field_pos.cpu().detach()[0,[0, 2],:,depth,:]),k=1), cmap="gist_gray")
                            plt.subplot(12, 9, (i-1)*27+18)
                            plt.title(f"jacobian {idx} d={depth}")
                            plt.imshow(np.rot90(jacobian[:, depth, :],k=1))
                        
                        
                        
        
                    
                            plt.subplot(12, 9, (i-1)*27+19)
                            plt.title(f"moving_image {idx} d={depth}")
                            plt.imshow(np.rot90(moving_image[depth, :, :],k=1), cmap="gray")
                            plt.subplot(12, 9, (i-1)*27+20)
                            plt.title(f"fixed_image {idx} d={depth}")
                            plt.imshow(np.rot90(fixed_image[depth, :, :],k=1), cmap="gray")
                            plt.subplot(12, 9, (i-1)*27+21)
                            plt.title(f"pred_image {idx} d={depth}")
                            plt.imshow(np.rot90(moving_warped[depth, :, :],k=1), cmap="gray")
                        
                            plt.subplot(12, 9, (i-1)*27+22)
                            plt.title(f"moving_label {idx} d={depth}")
                            plt.imshow(np.rot90(moving_label_plot[depth, :, :],k=1), cmap="jet",vmin=0,vmax=9)
                            plt.subplot(12, 9, (i-1)*27+23)

                            plt.title(f"fixed_label {idx} d={depth}")
                            plt.imshow(np.rot90(fixed_label_plot[depth, :, :],k=1), cmap="jet",vmin=0,vmax=9)
                            plt.subplot(12, 9, (i-1)*27+24)
                            plt.title(f"warp_label {idx} d={depth}")
                            plt.imshow(np.rot90(moving_label_warped_plot[depth, :, :],k=1), cmap="jet",vmin=0,vmax=9)
                            plt.subplot(12, 9, (i-1)*27+25)
                            plt.title(f"real 6month Seg {idx} d={depth}")
                            plt.imshow(np.rot90(Seg_out_argmax[depth, :, :],k=1), cmap="jet",vmin=0,vmax=9)
                        
                            plt.subplot(12, 9, (i-1)*27+26)
                            plt.title(f"deformation {idx} d={depth}")
                            plt.imshow(np.rot90(plot_2D_deformation(disp_field_pos.cpu().detach()[0,[1, 2],depth,:,:]),k=1), cmap="gist_gray")
                            plt.subplot(12, 9, (i-1)*27+27)
                            plt.title(f"jacobian {idx} d={depth}")
                            plt.imshow(np.rot90(jacobian[depth, :, :],k=1))
                            
                        suptitle_name = str(epoch) +'  fix:' + fixed_name + "  move:" + moving_name + ' LNCC loss: ' + str(val_loss.item()) + 'regu loss: ' + str(regularization_loss.item()) + 'label loss: ' + str(label_loss.item())
                        plt.suptitle(suptitle_name)
                        plt.savefig(root_dir+'/val_result/'+ str(epoch) + '_' + str(idx+1) +'.png')
                        
                        
                    
               
                    average_val_loss = average_val_loss/(idx+1)
                    average_label_dice_loss = average_label_dice_loss/(idx+1)
                    average_regularization_loss = average_regularization_loss/(idx+1)
                    average_Seg_dice_loss = average_Seg_dice_loss/(idx+1)
                    average_loss = average_val_loss + args.lamuda *  average_regularization_loss
                    
                    self.logger.add_scalar('Val: average_image_loss', average_val_loss, epoch)  
                    self.logger.add_scalar('Val: average_regularization_loss', average_regularization_loss, epoch)  
                    self.logger.add_scalar('Val: average_label_dice_loss', average_label_dice_loss, epoch)  
                    self.logger.add_scalar('Val: average_seg_dice_loss', average_Seg_dice_loss, epoch)  
                    with open(root_dir+'/val_result/'+'/fake6month_results.txt',"a") as file:
                        file.write('Epoch: ')
                        file.write(str(epoch))
                        file.write(', ')
                        file.write('average loss = ')
                        file.write(str(average_loss))
                        file.write(', average image loss = ')
                        file.write(str(average_val_loss))
                        file.write(', average regularization dice loss = ')
                        file.write(str(average_regularization_loss))
                        file.write(', average label dice loss = ')
                        file.write(str(average_label_dice_loss))
                        file.write(', average Seg dice loss = ')
                        file.write(str(average_Seg_dice_loss))
                        file.write('\n')
                    
                    
                    # Real 6month validation set, validate the segmentation performance, save the best-performing model
                    
                    real_6month_val_dice = 0
                    for idx, month6_real in enumerate(self.val_loader_6month):
                        
                        b_real = month6_real["image_T2"] 
                        month6_label = month6_real["image_label"]               
                        month6_name = month6_real["image_name"][0]
                        fixed_affine = month6_real["affine"]   
                        b_real = Variable(b_real)
                        b_real, month6_label = cuda([b_real, month6_label])
                        # Segmentation losses
                        ###################################################                              
                        Seg_out = self.Seg(b_real)
                        seg_loss = self.diceceloss(Seg_out, month6_label)
                        seg_dice= 1-seg_loss
                        real_6month_val_dice = real_6month_val_dice + seg_dice.item()
                        
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
                        fig1.savefig(val_result_path+'/Seg_val_'+ str(epoch) + '_' + month6_name +'.png')
                        
                    real_6month_val_dice = real_6month_val_dice / (idx+1)
                    with open(val_result_path+'/'+'Seg_val_results.txt',"a") as file:
                        file.write('\n')
                        file.write(str(epoch))
                        file.write(' val 6month dice: ')
                        file.write(str(real_6month_val_dice))
                        file.write('\n')
                        
                    if real_6month_val_dice>=self.best_val_accuracy:
                        self.best_val_accuracy = real_6month_val_dice+0
                        save_best_model(self.Seg,epoch,best_acc=self.best_val_accuracy,dir_add = root_dir)
                        with open(val_result_path+'/'+'Seg_val_results.txt',"a") as file:
                            file.write("save model")
                            file.write('\n')
                        
                        
                        
                        
                        
                        
                    
                    
                        
            # Override the latest checkpoint
            #######################################################
            save_checkpoint({'epoch': epoch,
                             'total_number': self.total_number,
                             'best_val_accuracy': self.best_val_accuracy,
                             'model': self.model.state_dict(),
                             'Seg': self.Seg.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'Seg_optimizer': self.Seg_optimizer.state_dict(),                             
                             },
                             '%s/register_seg.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.lr_scheduler.step()  
            self.Seg_lr_scheduler.step()     
            self.logger.close()
            

    def test(self,args):
        print("####################### Doing the whole brain segmentation for 6month infants #######################")
        print("Loading the data")
        
        dice_loss = DiceLoss(softmax=True)
        
        all_seg_path = root_dir + '/do_seg_on_6month'       
        if not os.path.exists(all_seg_path):
            os.makedirs(all_seg_path)
        
        print("Start the segmentation for 6month test")
        self.Seg.load_state_dict(torch.load(os.path.join(root_dir, "long_cy_reg_apply_6month.pt"))["state_dict"])
        self.Seg.eval()
        
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        
        average_6month = 0   
        csv_record_data = []                               
        for idx, month6_real in enumerate(self.test_loader_6_month):
            b_real = month6_real["image_T2"] 
            month6_label = month6_real["image_label"]               
            month6_name = month6_real["image_name"][0]
            month6_affine = month6_real["affine"]   
            b_real = Variable(b_real)
            b_real, month6_label = cuda([b_real, month6_label])
            # Segmentation losses
            ###################################################                              
            Seg_out = self.Seg(b_real)
            csv_record_data.append([Seg_out.detach().cpu().numpy(),month6_label.detach().cpu().numpy()])
            seg_loss = dice_loss(Seg_out, month6_label)
            seg_dice= 1-seg_loss
            average_6month = average_6month + seg_dice.item()
            
            month6_label_plot = month6_label.detach().cpu().numpy()
            
            month6_label_plot[:,0,:,:,:][month6_label_plot[:,0,:,:,:]==1]=0
            month6_label_plot[:,0,:,:,:][month6_label_plot[:,1,:,:,:]==1]=1
            month6_label_plot[:,0,:,:,:][month6_label_plot[:,2,:,:,:]==1]=2
            month6_label_plot[:,0,:,:,:][month6_label_plot[:,3,:,:,:]==1]=3
            month6_label_plot[:,0,:,:,:][month6_label_plot[:,4,:,:,:]==1]=4
            month6_label_plot[:,0,:,:,:][month6_label_plot[:,5,:,:,:]==1]=5
            month6_label_plot[:,0,:,:,:][month6_label_plot[:,6,:,:,:]==1]=6
            month6_label_plot[:,0,:,:,:][month6_label_plot[:,7,:,:,:]==1]=7
            month6_label_plot[:,0,:,:,:][month6_label_plot[:,8,:,:,:]==1]=8
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)   
            img_save = nib.Nifti1Image(np.float64(Seg_out_argmax[0,:,:,:]), month6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month6_name + '_prediction' + '.nii.gz')

            b_real_plot = b_real.detach().cpu().numpy()
            # month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                         
            img_save = nib.Nifti1Image(b_real_plot[0,0,:,:,:], month6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month6_name + '_T2' + '.nii.gz') 
            
            img_save = nib.Nifti1Image(month6_label_plot[0,0,:,:,:], month6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month6_name + '_VK' + '.nii.gz')  
            
            fig3 = plt.figure("3", (18, 18))
            ax3 = fig3.subplots(3,3)
                    
            ax3[0,0].title.set_text('6_month_real_T2')
            ax3[0,0].imshow(np.rot90(b_real_plot[0 , 0, :,:, round(b_real_plot.shape[4]/2)],k=3),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax3[0,1].title.set_text('VK_labels')
            ax3[0,1].imshow(np.rot90(month6_label_plot[0 ,0, :, :, round(month6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax3[0,2].title.set_text('Prediction')
            ax3[0,2].imshow(np.rot90(Seg_out_argmax[0 , :,:, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)

            ax3[1,0].title.set_text('6_month_real_T2')
            ax3[1,0].imshow(np.rot90(b_real_plot[0, 0, :, round(b_real_plot.shape[3]/2),:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax3[1,1].title.set_text('VK_labels')
            ax3[1,1].imshow(np.rot90(month6_label_plot[0, 0, :, round(month6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax3[1,2].title.set_text('Prediction')
            ax3[1,2].imshow(np.rot90(Seg_out_argmax[0 , :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)

            ax3[2,0].title.set_text('6_month_real_T2')
            ax3[2,0].imshow(np.rot90(b_real_plot[0, 0, round(b_real_plot.shape[2]/2), :,:],k=1),cmap="gray",vmin=-0.8,vmax=-0.2)
            ax3[2,1].title.set_text('VK_labels')
            ax3[2,1].imshow(np.rot90(month6_label_plot[0, 0, round(month6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax3[2,2].title.set_text('Prediction')
            ax3[2,2].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[1]/2),:,:]), cmap="jet",vmin=0,vmax=9)

            suptitle_name = month6_name + ' dice score: ' + str(seg_dice.item())
            fig3.suptitle(suptitle_name)
            fig3.savefig(all_seg_path+'/'+ month6_name +'.png')
        
        average_6month = average_6month / (idx+1)
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The average dice score of 6month subjects are: ')
            file.write(str(average_6month))
            file.write(' \n ') 
                   
        record_metric(csv_record_data, all_seg_path+'/3_Cyc+Att+Vox.csv')
        

    def add_ibeat(self,args):
        print("####################### Add iBeat segmentation outputs to CycleGAN+Seg 8-tissue segmentation outputs #######################")
        print("Loading the data")
        
        dice_loss = DiceLoss()
        add_ibeat_path = root_dir + '/add_ibeat_6month_test'       
        if not os.path.exists(add_ibeat_path):
            os.makedirs(add_ibeat_path)
        
        print("Start the segmentation for 6month test")
        
        self.Seg.load_state_dict(torch.load(os.path.join(root_dir, "long_cy_reg_apply_6month.pt"))["state_dict"])
        self.Seg.eval()
        
        with open(add_ibeat_path+'/add_ibeat_test_6month_results.txt',"a") as file:
            file.write('The dice score of 6month subjects are:')
            file.write(' \n ')
        

        for idx, month6_real in enumerate(self.test_loader_6_month):
            b_real = month6_real["image_T2"] 
            month6_label = month6_real["image_label"]               
            month6_name = month6_real["image_name"][0]
            month6_affine = month6_real["affine"] 
            month6_ibeat = month6_real["ibeat"]
              
            b_real = Variable(b_real)
            month6_label = Variable(month6_label)
            month_6_ibeat = Variable(month6_ibeat)
            b_real, month6_label, month_6_ibeat = cuda([b_real, month6_label, month_6_ibeat])
            # Segmentation losses
            ###################################################                              
            Seg_out = self.Seg(b_real)

            
            month_6_label_plot = month6_label.detach().cpu().numpy()
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
            
            img_save = nib.Nifti1Image(np.float64(combination[0,:,:,:]), month6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month6_name + '_combined' + '.nii.gz')

            b_real_plot = b_real.detach().cpu().numpy()
            #month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                         
            img_save = nib.Nifti1Image(b_real_plot[0,0,:,:,:], month6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month6_name + '_T2' + '.nii.gz') 
            
            img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, add_ibeat_path +'/' + month6_name + '_VK' + '.nii.gz')
            
            
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
        
        record_metric(csv_record_data, add_ibeat_path+'/5_Cyc+Att+Vox+iBeat.csv')        
        
    def create_combined_trainset(self,args):
        print("####################### Add iBeat segmentation outputs to CycleGAN+Seg 8-tissue segmentation outputs #######################")
        print("Loading the data")
        
        
        train_path_6_month = '/nfs/home/ydong/dataset/6_month_long/va_train/'
        
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
        
        self.Seg.load_state_dict(torch.load(os.path.join(root_dir, "long_cy_reg_apply_6month.pt"))["state_dict"])
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
           





                                
 
#############################################################################
#############################################################################
#############################################################################

def get_args():
    parser = ArgumentParser(description='cycle_register PyTorch')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_g', type=float, default=.00008)
    parser.add_argument('--lr_d', type=float, default=.00008)
    parser.add_argument('--lr_seg', type=float, default=.0004)
    
    parser.add_argument('--LNCC_kernel_size', type=int, default=15)
    parser.add_argument('--val_interval', type=int, default=20)
    parser.add_argument('--im_size', type=int, default=(128,128,128))
    parser.add_argument('--lr_r', type=float, default=2e-3)
    parser.add_argument('--lamuda', type=float, default=0.6)
    parser.add_argument('--weight_decay_r', type=float, default=1e-5)
    parser.add_argument('--smooth_sigma', type=float, default=0.9)
    
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=0.5)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=True)
    parser.add_argument('--add_ibeat', type=bool, default=True)
    parser.add_argument('--create_com_train', type=bool, default=False)
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
    model = cycle_register(args)
    model.train(args)
  if args.testing:
    print("Testing")
    model = cycle_register(args)
    model.test(args)
  if args.add_ibeat:
    print("adding iBeat segmentation outputs")
    model = cycle_register(args)
    model.add_ibeat(args)
  if args.create_com_train:
    print("create the combined outputs for 6-month train")
    model = cycle_register(args)
    model.create_combined_trainset(args)

if __name__ == '__main__':
    main()






