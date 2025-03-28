import os
import numpy as np
import glob
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, Dataset, CacheDataset
# from monai.handlers.utils import from_engine
from monai.losses import DiceLoss, DiceCELoss, LocalNormalizedCrossCorrelationLoss, GeneralizedDiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
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
    RandAdjustContrastd,
    RandGaussianNoised,
)
from monai.utils import set_determinism, first
# import itk
from monai.data import PILReader, NumpyReader, NibabelReader
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
import torchvision.datasets as dsets
from torch.optim import lr_scheduler
from argparse import ArgumentParser
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance, DiceMetric
import csv


print_config()



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


################################### setting ###################################
################################### setting ###################################
################################### setting ###################################

root_dir = '/nfs/home/ydong/code/pippi/real_6month_seg/save_model'
train_path_6_month = '/nfs/home/ydong/dataset/6_month_long/va_train/'
train_path_6_month_combined_outputs = '/nfs/home/ydong/code/pippi/long_cy/save_model/create_combined_train_set/'
val_path_6_month = '/nfs/home/ydong/dataset/6_month_long/va_before_val_5/'
test_path_6_month = '/nfs/home/ydong/dataset/6_month_long/va_test_10_subjects/'
apply_path_6_month = '/nfs/home/ydong/dataset/apply/seg/'
val_result_path = root_dir + '/val_result'
logger_path = root_dir + '/log'

if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(val_result_path):
    os.makedirs(val_result_path)
if not os.path.exists(logger_path):
    os.makedirs(logger_path)

################################### 6_month ###################################
################################### 6_month ###################################
################################### 6_month ###################################

set_determinism(seed=0)
train_images_T1 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T1.nii.gz')))
train_images_T2 = sorted(glob.glob(os.path.join(train_path_6_month, '*_T2.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(train_path_6_month_combined_outputs, '*_combined.nii.gz')))

train_dicts_6_month = [
    {"image_T1": image_T1, "image_T2": image_T2, "image_label": image_label, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:12]}
    for image_T1, image_T2, image_label in zip(train_images_T1, train_images_T2, train_labels)
]

train_transform_6_month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T1", "image_T2","image_label"]),
        EnsureChannelFirstd(keys=["image_T1","image_T2"]),
        ConvertToMultiChannel_com(keys="image_label"),
        RandAffined(keys=["image_T1","image_T2","image_label"], 
              mode=("bilinear","bilinear",'nearest'), 
              prob=0.5,
              # translate_range=(3, 3, 3),
              rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
              scale_range=(0.1, 0.1, 0.1),
              padding_mode="zeros",
          ),
        ScaleIntensityd(keys=["image_T1","image_T2"],minv=0,maxv=1),
        RandAdjustContrastd(keys=["image_T1"],
                            prob=0.5,
                            gamma=(0.3,2),
        ),
        RandAdjustContrastd(keys=["image_T2"],
                            prob=0.5,
                            gamma=(0.3,2),
        ),

    ]
)

train_transform_6_month_withoutaff = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T1", "image_T2","image_label"]),
        EnsureChannelFirstd(keys=["image_T1", "image_T2"]),
        ConvertToMultiChannel_com(keys="image_label"),
        ScaleIntensityd(keys=["image_T1", "image_T2"],minv=0,maxv=1),

    ]
)



################################### 6_month val ###################################
################################### 6_month val ###################################
################################### 6_month val ###################################

set_determinism(seed=0)
val_images_T1 = sorted(glob.glob(os.path.join(val_path_6_month, '*_T1.nii.gz')))
val_images_T2 = sorted(glob.glob(os.path.join(val_path_6_month, '*_T2.nii.gz')))
val_images_labels = sorted(glob.glob(os.path.join(val_path_6_month, '*_fixed_basalganglia_prediction.nii.gz')))

val_dicts_6_month = [
    {"image_T1": image_T1, "image_T2": image_T2, "image_label": image_label, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:12]}
    for image_T1, image_T2, image_label in zip(val_images_T1, val_images_T2, val_images_labels)
]

val_transform_6_month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T1", "image_T2","image_label"]),
        EnsureChannelFirstd(keys=["image_T1","image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        ScaleIntensityd(keys=["image_T1","image_T2"],minv=0,maxv=1),
        # HistogramNormalized(keys=["image_T2"],min=-1,max=1),

    ]
)


################################### 6_month test ###################################
################################### 6_month test ###################################
################################### 6_month test ###################################

set_determinism(seed=0)
test_images_T1 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T1.nii.gz')))
test_images_T2 = sorted(glob.glob(os.path.join(test_path_6_month, '*_T2.nii.gz')))
test_images_labels = sorted(glob.glob(os.path.join(test_path_6_month, '*_VK.nii.gz')))

test_dicts_6_month = [
    {"image_T1": image_T1, "image_T2": image_T2, "image_label": image_label, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:12]}
    for image_T1, image_T2, image_label in zip(test_images_T1, test_images_T2, test_images_labels)
]

test_transform_6_month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T1", "image_T2","image_label"]),
        EnsureChannelFirstd(keys=["image_T1","image_T2"]),
        ConvertToMultiChannel_6month(keys="image_label"),
        ScaleIntensityd(keys=["image_T1","image_T2"],minv=0,maxv=1),
        # HistogramNormalized(keys=["image_T2"],min=-1,max=1),

    ]
)

################################### 6_month apply ###################################
################################### 6_month apply ###################################
################################### 6_month apply ###################################

set_determinism(seed=0)
apply_images_T1 = sorted(glob.glob(os.path.join(apply_path_6_month, '*_T1.nii.gz')))
apply_images_T2 = sorted(glob.glob(os.path.join(apply_path_6_month, '*_T2.nii.gz')))

apply_dicts_6_month = [
    {"image_T1": image_T1, "image_T2": image_T2, "affine": nib.load(image_T2).affine, "image_name": os.path.basename(image_T2)[0:4]}
    for image_T1, image_T2 in zip(apply_images_T1, apply_images_T2)
]

apply_transform_6_month = Compose(
    [
        LoadImaged(reader=NibabelReader, keys=["image_T1", "image_T2"]),
        EnsureChannelFirstd(keys=["image_T1","image_T2"]),
        ScaleIntensityd(keys=["image_T1","image_T2"],minv=0,maxv=1),
        # HistogramNormalized(keys=["image_T2"],min=-1,max=1),

    ]
)

#############################################################################
#############################################################################
#############################################################################


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


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad



#############################################################################
#############################################################################
#############################################################################


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


# To save the checkpoint 
def save_checkpoint(state, save_path):
    torch.save(state, save_path)
    
def save_best_model(model, epoch, filename="real_6month_seg.pt", best_acc=0, dir_add=''):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


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

class InfantSeg(object):
    def __init__(self,args):

        # Define the network 
        #####################################################

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Seg=AttentionUnet(
                               spatial_dims=3,
                               in_channels=2,
                               out_channels=9,
                               channels=(32, 64, 128, 256, 512),
                               strides=(2, 2, 2, 2),
                               ).to(device)
        

        print_networks([self.Seg], ['Seg'])

        # Define Loss criterias

        self.dice_loss = DiceLoss(to_onehot_y=False,include_background=True,softmax=True)
        # self.gen_dice_loss = GeneralizedDiceLoss(to_onehot_y=False,include_background=True,softmax=True)  DiceCELoss
        self.dicece_loss = DiceCELoss(to_onehot_y=False,include_background=True,softmax=True)

        # Optimizers
        #####################################################
        self.Seg_optimizer = torch.optim.Adam(self.Seg.parameters(), lr=args.lr_seg, betas=(0.5, 0.999))
        self.Seg_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.Seg_optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
        
        
        # Data
        #####################################################           
        train_ds_6month = CacheDataset(data=train_dicts_6_month, transform=train_transform_6_month)
        self.train_loader_6_month = DataLoader(train_ds_6month, batch_size=args.batch_size,shuffle=True, num_workers=0,pin_memory=True)
        
        val_ds_6month = CacheDataset(data=val_dicts_6_month, transform=val_transform_6_month)
        self.val_loader_6_month = DataLoader(val_ds_6month, batch_size=args.batch_size,shuffle=True, num_workers=0,pin_memory=True)
        
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
            self.Seg.load_state_dict(ckpt['Seg'])
            self.Seg_optimizer.load_state_dict(ckpt['Seg_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 1
            self.logger=SummaryWriter(log_dir=logger_path)
            self.total_number=0
            self.best_val_accuracy = -1
            



    def train(self,args):
                            
                
                       
        train_dice_loss = []
        train_dice_score = []
       
        test_dice_score = []
        test_dice_loss = []

        for epoch in range(self.start_epoch, args.epochs+1):

            lr_seg = self.Seg_optimizer.param_groups[0]['lr']

            print('learning rate of Gen = %.7f' % lr_seg)
            
            
            log_seg_dice_loss = 0
            log_seg_dice_score = 0
            log_background = 0
            log_csf = 0
            log_wm = 0
            log_gm = 0
            log_ventricle = 0
            log_cerebellum = 0
            log_basal_ganglia = 0
            log_brainstem = 0
            log_hippo = 0
            log_myelination = 0
            

            if epoch ==1:
                for idx, month_6_real in enumerate(self.val_loader_6_month):
                    
                    month_6_real_T1 = month_6_real["image_T1"]
                    month_6_real_T2 = month_6_real["image_T2"]
                    month_6_label = month_6_real["image_label"]
                    month_6_name = month_6_real["image_name"][0]
                    month_6_affine = month_6_real["affine"]

                    month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()
                    month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                    month_6_label = month_6_label.detach().cpu().numpy()
                    month_6_label[:,0,:,:,:][month_6_label[:,0,:,:,:]==1]=0
                    month_6_label[:,0,:,:,:][month_6_label[:,1,:,:,:]==1]=1
                    month_6_label[:,0,:,:,:][month_6_label[:,2,:,:,:]==1]=2
                    month_6_label[:,0,:,:,:][month_6_label[:,3,:,:,:]==1]=3
                    month_6_label[:,0,:,:,:][month_6_label[:,4,:,:,:]==1]=4
                    month_6_label[:,0,:,:,:][month_6_label[:,5,:,:,:]==1]=5
                    month_6_label[:,0,:,:,:][month_6_label[:,6,:,:,:]==1]=6
                    month_6_label[:,0,:,:,:][month_6_label[:,7,:,:,:]==1]=7
                    month_6_label[:,0,:,:,:][month_6_label[:,8,:,:,:]==1]=8
                    # month_6_label[:,0,:,:,:][month_6_label[:,9,:,:,:]==1]=9
                
                   
                    img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
                    nib.save(img_save, root_dir +'/val_result/' + month_6_name + '_T2' + '.nii.gz')
                    img_save = nib.Nifti1Image(month_6_real_T1[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
                    nib.save(img_save, root_dir +'/val_result/' + month_6_name + '_T1' + '.nii.gz')
                    
                    img_save = nib.Nifti1Image(np.float64(month_6_label[0,0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
                    nib.save(img_save, root_dir +'/val_result/' + month_6_name + '_fixed_basalganglia_prediction' + '.nii.gz')
           
            # for idx, batch_data in enumerate(train_loader):
            self.Seg.train()
            for idx, month_6_real in enumerate(self.train_loader_6_month):
                self.total_number = self.total_number + 1
                
                month_6_real_T1 = month_6_real["image_T1"]
                month_6_real_T2 = month_6_real["image_T2"]
                month_6_label = month_6_real["image_label"]

                #########################################################################################################
                ### apply brain mask again
                #########################################################################################################
                mask = torch.ones_like(month_6_real_T1)
                mask[0,0,:,:,:] = mask[0,0,:,:,:] - month_6_label[0,0,:,:,:]
                month_6_real_T1 = month_6_real_T1 * mask
                month_6_real_T2 = month_6_real_T2 * mask
                #########################################################################################################
                
                
                month_6_real_T1 = Variable(month_6_real_T1)
                month_6_real_T2 = Variable(month_6_real_T2)
                month_6_label = Variable(month_6_label)

                
                month_6_real_T1, month_6_real_T2, month_6_label = cuda([month_6_real_T1, month_6_real_T2, month_6_label])                              
                data_combine = torch.cat((month_6_real_T1, month_6_real_T2),1)
                
                set_grad([self.Seg], True)
                self.Seg_optimizer.zero_grad()

                Seg_out = self.Seg(data_combine)
                seg_loss = self.dicece_loss(Seg_out, month_6_label)
                seg_dice= 1-seg_loss
                
                          
                background_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,0,:,:,:],1), torch.unsqueeze(month_6_label[:,0,:,:,:],1))
                
                csf_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,1,:,:,:],1), torch.unsqueeze(month_6_label[:,1,:,:,:],1))
                GM_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,2,:,:,:],1), torch.unsqueeze(month_6_label[:,2,:,:,:],1))
                WM_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,3,:,:,:],1), torch.unsqueeze(month_6_label[:,3,:,:,:],1))
                
                ventricle_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,4,:,:,:],1), torch.unsqueeze(month_6_label[:,4,:,:,:],1))
                cerebellum_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,5,:,:,:],1), torch.unsqueeze(month_6_label[:,5,:,:,:],1))
                basal_ganglia_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,6,:,:,:],1), torch.unsqueeze(month_6_label[:,6,:,:,:],1))
                
                brainstem_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,7,:,:,:],1), torch.unsqueeze(month_6_label[:,7,:,:,:],1))
                hippo_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,8,:,:,:],1), torch.unsqueeze(month_6_label[:,8,:,:,:],1))
                # myelination_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,9,:,:,:],1), torch.unsqueeze(month_6_label[:,9,:,:,:],1))
                          
                seg_loss.backward()
                self.Seg_optimizer.step()


                
                
                #########################################################################################################
                ### Save training images and outputs
                #########################################################################################################
                if idx==2 :
                
                    month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()
                    month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()

                    Seg_out_plot = Seg_out.detach().cpu().numpy()
                    month_6_label_plot = month_6_label.detach().cpu().numpy()
                    Seg_out_argmax = np.argmax(Seg_out_plot, axis=1)
                    
                    
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
                    # month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,9,:,:,:]==1]=9
              
                    fig2 = plt.figure("2", (24, 18))
                    ax2 = fig2.subplots(3,4)
                    
                    
                    ax2[0,0].title.set_text('6month_real_T1')
                    ax2[0,0].imshow(np.rot90(month_6_real_T1[0, 0, :, :,round(month_6_real_T1.shape[4]/2)],k=3),cmap="gray")
                    ax2[0,1].title.set_text('6month_real_T2')
                    ax2[0,1].imshow(np.rot90(month_6_real_T2[0, 0, :, :, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray")
                    ax2[0,2].title.set_text('Combined_labels')
                    ax2[0,2].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
                    ax2[0,3].title.set_text('Seg_outputs')
                    ax2[0,3].imshow(np.rot90(Seg_out_argmax[0 , :, :, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)   
                    
                    

                    ax2[1,0].title.set_text('6month_real_T1')
                    ax2[1,0].imshow(np.rot90(month_6_real_T1[0, 0, :, round(month_6_real_T1.shape[3]/2),:],k=1),cmap="gray")
                    ax2[1,1].title.set_text('6month_real_T2')
                    ax2[1,1].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray")
                    ax2[1,2].title.set_text('Combined_labels')
                    ax2[1,2].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9)  
                    ax2[1,3].title.set_text('Seg_outputs')
                    ax2[1,3].imshow(np.rot90(Seg_out_argmax[0, :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)  
                     
                    
                    
                    ax2[2,0].title.set_text('6month_real_T1')
                    ax2[2,0].imshow(np.rot90(month_6_real_T1[0, 0, round(month_6_real_T1.shape[2]/2), :,:],k=1),cmap="gray")
                    ax2[2,1].title.set_text('6month_real_T2')
                    ax2[2,1].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray")
                    ax2[2,2].title.set_text('Combined_labels')
                    ax2[2,2].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
                    ax2[2,3].title.set_text('Seg_outputs')
                    ax2[2,3].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[2]/2), :,:]), cmap="jet",vmin=0,vmax=9)   
                    
                    
                    fig2.savefig(root_dir + '/'+ str(epoch) +'.png')
                                  
                
                
                print("Epoch: (%3d) (%5d/%5d) | CSF dice:%.2e | WM dice:%.2e | GM dice:%.2e | background dice:%.2e | total dice loss:%.2e | total dice score:%.2e" % 
                                            (epoch, idx + 1, len(self.train_loader_6_month), csf_dice, WM_dice, GM_dice, background_dice, seg_loss, seg_dice))
                                                            
                with open(root_dir+'/val_result/'+'/logs.txt',"a") as file:
                    file.write('Epoch: ')
                    file.write(str(epoch))
                    file.write(', ')
                    file.write('( ')
                    file.write(str(idx+1))
                    file.write(' / ')
                    file.write(str(len(self.train_loader_6_month)))
                    file.write(' )\n ')
                     

                self.logger.add_scalar('Dice loss', seg_loss.item(), self.total_number) 
                self.logger.add_scalar('Dice score', seg_dice.item(), self.total_number) 


                
                
                
                log_seg_dice_loss = log_seg_dice_loss + seg_loss.item()
                log_seg_dice_score = log_seg_dice_score + seg_dice.item()
                log_background = log_background + background_dice.item()
                log_csf = log_csf + csf_dice.item()
                log_wm = log_wm + WM_dice.item()
                log_gm = log_gm + GM_dice.item()
                log_ventricle = log_ventricle + ventricle_dice.item()
                log_cerebellum = log_cerebellum + cerebellum_dice.item()
                log_basal_ganglia = log_basal_ganglia + basal_ganglia_dice.item()
                log_brainstem = log_brainstem + brainstem_dice.item()
                log_hippo = log_hippo + hippo_dice.item()
                # log_myelination = log_myelination + myelination_dice.item()
                


            log_seg_dice_loss = log_seg_dice_loss / len(self.train_loader_6_month)
            log_seg_dice_score = log_seg_dice_score / len(self.train_loader_6_month)
            log_background = log_background / len(self.train_loader_6_month)
            log_csf = log_csf / len(self.train_loader_6_month)
            log_wm = log_wm / len(self.train_loader_6_month)
            log_gm = log_gm / len(self.train_loader_6_month)
            log_ventricle = log_ventricle / len(self.train_loader_6_month)
            log_cerebellum = log_cerebellum / len(self.train_loader_6_month)
            log_basal_ganglia = log_basal_ganglia / len(self.train_loader_6_month)
            log_brainstem = log_brainstem / len(self.train_loader_6_month)
            log_hippo = log_hippo / len(self.train_loader_6_month)
            # log_myelination = log_myelination / len(self.train_loader_6_month)
            
            train_dice_loss.append(log_seg_dice_loss)
            train_dice_score.append(log_seg_dice_score)
            

            self.logger.add_scalar('Dice score epoch', log_seg_dice_score, epoch)
            self.logger.add_scalar('Dice loss epoch', log_seg_dice_loss, epoch)
            
            self.logger.add_scalars('Tissues dice scores', {'CSF':log_csf,
                                                            'WM':log_wm,
                                                            'GM':log_gm,
                                                            'Ventricle':log_ventricle,
                                                            'Cerebellum':log_cerebellum,
                                                            'Basal ganglia':log_basal_ganglia,
                                                            'Brainstem':log_brainstem,
                                                            'Hippocampus / Amygdala':log_hippo,
                                                            #'Myelination':log_myelination,
                                                            'Background':log_background},
                                                             epoch)
                            
            #########################################################################################################
            ### Apply segmentation to the test set
            #########################################################################################################
            if epoch % 10 ==0:
                self.Seg.eval()
                average_test_loss = 0
                average_test_dice = 0
                
                log_background = 0
                log_csf = 0
                log_wm = 0
                log_gm = 0
                log_ventricle = 0
                log_cerebellum = 0
                log_basal_ganglia = 0
                log_brainstem = 0
                log_hippo = 0
                log_myelination = 0
                
                for idx, month_6_real in enumerate(self.val_loader_6_month):

                    month_6_real_T1 = month_6_real["image_T1"]
                    month_6_real_T2 = month_6_real["image_T2"]
                    month_6_label = month_6_real["image_label"]
                    month_6_name = month_6_real["image_name"][0]
                    month_6_affine = month_6_real["affine"]
                
                    month_6_real_T1 = Variable(month_6_real_T1)
                    month_6_real_T2 = Variable(month_6_real_T2)
                    month_6_label = Variable(month_6_label)
                
                    month_6_real_T1, month_6_real_T2, month_6_label = cuda([month_6_real_T1, month_6_real_T2, month_6_label])                
                    data_combine = torch.cat((month_6_real_T1, month_6_real_T2),1)
                

                    Seg_out = self.Seg(data_combine)
                    seg_loss = self.dicece_loss(Seg_out, month_6_label)
                    seg_dice= 1-seg_loss
                    
                    background_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,0,:,:,:],1), torch.unsqueeze(month_6_label[:,0,:,:,:],1))
                
                    csf_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,1,:,:,:],1), torch.unsqueeze(month_6_label[:,1,:,:,:],1))
                    GM_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,2,:,:,:],1), torch.unsqueeze(month_6_label[:,2,:,:,:],1))
                    WM_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,3,:,:,:],1), torch.unsqueeze(month_6_label[:,3,:,:,:],1))
                
                    ventricle_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,4,:,:,:],1), torch.unsqueeze(month_6_label[:,4,:,:,:],1))
                    cerebellum_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,5,:,:,:],1), torch.unsqueeze(month_6_label[:,5,:,:,:],1))
                    basal_ganglia_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,6,:,:,:],1), torch.unsqueeze(month_6_label[:,6,:,:,:],1))
                
                    brainstem_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,7,:,:,:],1), torch.unsqueeze(month_6_label[:,7,:,:,:],1))
                    hippo_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,8,:,:,:],1), torch.unsqueeze(month_6_label[:,8,:,:,:],1))
                    # myelination_dice = 1-self.dice_loss(torch.unsqueeze(Seg_out[:,9,:,:,:],1), torch.unsqueeze(month_6_label[:,9,:,:,:],1))

                
                    # Segmentation losses
                    ###################################################
                    
                    Seg_out_plot = Seg_out.detach().cpu().numpy()
                    month_6_label_plot = month_6_label.detach().cpu().numpy()
                    Seg_out_argmax = np.argmax(Seg_out_plot, axis=1)      
                    month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()
                    month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()
                    

                    #img_save = nib.Nifti1Image(np.float64(Seg_out_argmax[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
                    #nib.save(img_save, root_dir +'/val_result/' + str(epoch) + '_' + month_6_name + '_prediction' + '.nii.gz')
                   

                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,0,:,:,:]==1]=0
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,1,:,:,:]==1]=1
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,2,:,:,:]==1]=2
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,3,:,:,:]==1]=3
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,4,:,:,:]==1]=4
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,5,:,:,:]==1]=5
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,6,:,:,:]==1]=6
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,7,:,:,:]==1]=7
                    month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,8,:,:,:]==1]=8
                    #month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,9,:,:,:]==1]=9
                

                   
                    fig = plt.figure("1", (24, 18))
                    ax = fig.subplots(3,4)
                    
                    ax[0,0].title.set_text('6month_real_T1')
                    ax[0,0].imshow(np.rot90(month_6_real_T1[0, 0, :, :,round(month_6_real_T1.shape[4]/2)],k=3),cmap="gray")
                    ax[0,1].title.set_text('6month_real_T2')
                    ax[0,1].imshow(np.rot90(month_6_real_T2[0, 0, :, :, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray")
                    ax[0,2].title.set_text('Combined_labels')
                    ax[0,2].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
                    ax[0,3].title.set_text('Seg_outputs')
                    ax[0,3].imshow(np.rot90(Seg_out_argmax[0 , :, :, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)   
                    
                    

                    ax[1,0].title.set_text('6month_real_T1')
                    ax[1,0].imshow(np.rot90(month_6_real_T1[0, 0, :, round(month_6_real_T1.shape[3]/2),:],k=1),cmap="gray")
                    ax[1,1].title.set_text('6month_real_T2')
                    ax[1,1].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray")
                    ax[1,2].title.set_text('Combined_labels')
                    ax[1,2].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9)  
                    ax[1,3].title.set_text('Seg_outputs')
                    ax[1,3].imshow(np.rot90(Seg_out_argmax[0, :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)  
                     
                    
                    
                    ax[2,0].title.set_text('6month_real_T1')
                    ax[2,0].imshow(np.rot90(month_6_real_T1[0, 0, round(month_6_real_T1.shape[2]/2), :,:],k=1),cmap="gray")
                    ax[2,1].title.set_text('6month_real_T2')
                    ax[2,1].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray")
                    ax[2,2].title.set_text('Combined_labels')
                    ax[2,2].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
                    ax[2,3].title.set_text('Seg_outputs')
                    ax[2,3].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[2]/2), :,:]), cmap="jet",vmin=0,vmax=9)  
                    
                    suptitle_name = str(epoch) +' : ' + month_6_name + ' dice score: ' + str(seg_dice.item())
                    fig.suptitle(suptitle_name)
                    fig.savefig(root_dir+'/val_result/'+ str(epoch) + '_' + month_6_name +'.png')                    
                    
                    average_test_loss = average_test_loss + seg_loss.item()
                    average_test_dice = average_test_dice + seg_dice.item() 
                     
                    log_background = log_background + background_dice.item()
                    log_csf = log_csf + csf_dice.item()
                    log_wm = log_wm + WM_dice.item()
                    log_gm = log_gm + GM_dice.item()
                    log_ventricle = log_ventricle + ventricle_dice.item()
                    log_cerebellum = log_cerebellum + cerebellum_dice.item()
                    log_basal_ganglia = log_basal_ganglia + basal_ganglia_dice.item()
                    log_brainstem = log_brainstem + brainstem_dice.item()
                    log_hippo = log_hippo + hippo_dice.item()
                    # log_myelination = log_myelination + myelination_dice.item()
                                  
                average_test_loss = average_test_loss / (idx+1) 
                average_test_dice = average_test_dice / (idx+1) 
                
                with open(root_dir+'/val_result/'+'results.txt',"a") as file:
                    file.write('\n')
                    file.write(str(epoch))
                    file.write(' val 6month dicece: ')
                    file.write(str(average_test_dice))
                    file.write('\n')
                
                test_dice_score.append(average_test_dice)
                test_dice_loss.append(average_test_loss)
                if average_test_dice>=self.best_val_accuracy:
                    self.best_val_accuracy = average_test_dice+0
                    save_best_model(self.Seg,epoch,best_acc=self.best_val_accuracy,dir_add = root_dir)
                    with open(root_dir+'/val_result/'+'results.txt',"a") as file:
                        file.write("save model")
                        file.write('\n')
                

                log_background = log_background / (idx+1) 
                log_csf = log_csf / (idx+1) 
                log_wm = log_wm / (idx+1) 
                log_gm = log_gm / (idx+1) 
                log_ventricle = log_ventricle / (idx+1) 
                log_cerebellum = log_cerebellum / (idx+1) 
                log_basal_ganglia = log_basal_ganglia / (idx+1) 
                log_brainstem = log_brainstem / (idx+1) 
                log_hippo = log_hippo / (idx+1) 
                #log_myelination = log_myelination / (idx+1) 
                
                self.logger.add_scalar('Val: Dice score epoch', average_test_dice, epoch)
                self.logger.add_scalar('Val: Dice loss epoch', average_test_loss, epoch)
            
                self.logger.add_scalars('Val: Tissues dice scores', {'CSF':log_csf,
                                                            'WM':log_wm,
                                                            'GM':log_gm,
                                                            'Ventricle':log_ventricle,
                                                            'Cerebellum':log_cerebellum,
                                                            'Basal ganglia':log_basal_ganglia,
                                                            'Brainstem':log_brainstem,
                                                            'Hippocampus / Amygdala':log_hippo,
                                                            #'Myelination':log_myelination,
                                                            'Background':log_background},
                                                             epoch)
    
                             
                
            # Override the latest checkpoint
            #######################################################
            save_checkpoint({'epoch': epoch + 1,
                             'total_number': self.total_number,
                             'best_val_accuracy': self.best_val_accuracy,
                                   'Seg': self.Seg.state_dict(),
                                   'Seg_optimizer': self.Seg_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################

            self.Seg_lr_scheduler.step()
        
    
        
        train_x_axis = list(range(1, len(train_dice_loss)+1))
        plt.figure(figsize=(15,10))
        plt.plot(train_x_axis, train_dice_loss, c='b',label='Dice loss')
        plt.plot(train_x_axis, train_dice_score, c='r',label='Dis score')
        plt.title("Train: Dice loss and score")
        plt.xlabel("epoch")
        plt.ylabel("Dice loss and score")
        plt.legend()
        plt.savefig(root_dir+'/Train Dice loss and score.png')
        
        test_x_axis = list(range(1, len(test_dice_score)+1))
        plt.figure(figsize=(15,10))
        plt.plot(test_x_axis, test_dice_loss, c='b',label='Dice loss')
        plt.plot(test_x_axis, test_dice_score, c='r',label='Dis score')
        plt.title("Val: Dice loss and score")
        plt.xlabel("epoch")
        plt.ylabel("Dice loss and score")
        plt.legend()
        plt.savefig(root_dir+'/Val Dice loss and score.png')
        

        
        self.logger.close()


    def test(self,args):
        print("####################### Doing the whole brain segmentation for 6month infants #######################")
        print("Loading the data")
        
        dice_loss = DiceLoss(softmax=True)
        
        
        test_ds_6_month = CacheDataset(data=test_dicts_6_month, transform=test_transform_6_month)
        self.test_loader_6_month = DataLoader(test_ds_6_month, batch_size=args.batch_size, num_workers=0,pin_memory=True)
        
        all_seg_path = root_dir + '/do_seg_on_6month'

        
        if not os.path.exists(all_seg_path):
            os.makedirs(all_seg_path)

        print("Start the segmentation for 6month test")
        self.Seg.load_state_dict(torch.load(os.path.join(root_dir, "real_6month_seg.pt"))["state_dict"])
        self.Seg.eval()
        
        average_6month = 0
        csv_record_data = []

        for idx, month_6_real in enumerate(self.test_loader_6_month):

            month_6_real_T1 = month_6_real["image_T1"]
            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]

            month_6_real_T1 = Variable(month_6_real_T1)
            month_6_real_T2 = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
                    

                
            month_6_real_T1, month_6_real_T2, month_6_label = cuda([month_6_real_T1, month_6_real_T2, month_6_label])                              
            data_combine = torch.cat((month_6_real_T1, month_6_real_T2),1)
                

            Seg_out = self.Seg(data_combine)
            csv_record_data.append([Seg_out.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
            seg_loss = dice_loss(Seg_out, month_6_label)
            seg_dice= 1-seg_loss
            average_6month = average_6month + seg_dice.item()
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)   

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
            # month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,9,:,:,:]==1]=9            
            
            
            img_save = nib.Nifti1Image(np.float64(Seg_out_argmax[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_prediction' + '.nii.gz')

            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()          
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_T2' + '.nii.gz')      
            
            month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()          
            img_save = nib.Nifti1Image(month_6_real_T1[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_T1' + '.nii.gz')   
                    
            img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_VK' + '.nii.gz')  
            
            
            
            fig5 = plt.figure("5", (24, 18))
            ax5 = fig5.subplots(3,4)
            
                    
            ax5[0,0].title.set_text('6month_real_T1')
            ax5[0,0].imshow(np.rot90(month_6_real_T1[0, 0, :, :,round(month_6_real_T1.shape[4]/2)],k=3),cmap="gray")
            ax5[0,1].title.set_text('6month_real_T2')
            ax5[0,1].imshow(np.rot90(month_6_real_T2[0, 0, :, :, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray")
            ax5[0,2].title.set_text('VK_labels')
            ax5[0,2].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax5[0,3].title.set_text('Seg_outputs')
            ax5[0,3].imshow(np.rot90(Seg_out_argmax[0 , :, :, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)   
                    
                    

            ax5[1,0].title.set_text('6month_real_T1')
            ax5[1,0].imshow(np.rot90(month_6_real_T1[0, 0, :, round(month_6_real_T1.shape[3]/2),:],k=1),cmap="gray")
            ax5[1,1].title.set_text('6month_real_T2')
            ax5[1,1].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray")
            ax5[1,2].title.set_text('VK_labels')
            ax5[1,2].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9)  
            ax5[1,3].title.set_text('Seg_outputs')
            ax5[1,3].imshow(np.rot90(Seg_out_argmax[0, :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)  
                     
                    
                    
            ax5[2,0].title.set_text('6month_real_T1')
            ax5[2,0].imshow(np.rot90(month_6_real_T1[0, 0, round(month_6_real_T1.shape[2]/2), :,:],k=1),cmap="gray")
            ax5[2,1].title.set_text('6month_real_T2')
            ax5[2,1].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray")
            ax5[2,2].title.set_text('VK_labels')
            ax5[2,2].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax5[2,3].title.set_text('Seg_outputs')
            ax5[2,3].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[2]/2), :,:]), cmap="jet",vmin=0,vmax=9)  

            suptitle_name = month_6_name + ' dice score: ' + str(seg_dice.item())
            fig5.suptitle(suptitle_name)
            fig5.savefig(all_seg_path+'/'+ month_6_name +'.png')   
        
        average_6month = average_6month / (idx+1)
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The average dice score of 6month subjects are: ')
            file.write(str(average_6month))
            file.write(' \n ')    

        record_metric(csv_record_data, all_seg_path+'/6_Cyc+Att+iBeat+Att.csv')


    def segment_all(self,args):
        print("####################### Doing the whole brain segmentation for 6month infants #######################")
        print("Loading the data")
        
        dice_loss = DiceLoss(softmax=True)
        
        train_ds_6month_withoutaff = CacheDataset(data=train_dicts_6_month, transform=train_transform_6_month_withoutaff)
        self.train_loader_6_month_withoutaff = DataLoader(train_ds_6month_withoutaff, batch_size=args.batch_size,shuffle=True, num_workers=0,pin_memory=True)
        
        test_ds_6_month = CacheDataset(data=test_dicts_6_month, transform=test_transform_6_month)
        self.test_loader_6_month = DataLoader(test_ds_6_month, batch_size=args.batch_size, num_workers=0,pin_memory=True)
        
        all_seg_path = root_dir + '/do_seg_on_all_6month_subjects'

        
        if not os.path.exists(all_seg_path):
            os.makedirs(all_seg_path)

        print("Start the segmentation for 6month test")
        self.Seg.load_state_dict(torch.load(os.path.join(root_dir, "real_6month_seg.pt"))["state_dict"])
        self.Seg.eval()
        
        average_6month = 0
        csv_record_data = []

        for idx, month_6_real in enumerate(self.test_loader_6_month):

            month_6_real_T1 = month_6_real["image_T1"]
            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]

            month_6_real_T1 = Variable(month_6_real_T1)
            month_6_real_T2 = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
                    

                
            month_6_real_T1, month_6_real_T2, month_6_label = cuda([month_6_real_T1, month_6_real_T2, month_6_label])                              
            data_combine = torch.cat((month_6_real_T1, month_6_real_T2),1)
                

            Seg_out = self.Seg(data_combine)
            csv_record_data.append([Seg_out.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
            seg_loss = dice_loss(Seg_out, month_6_label)
            seg_dice= 1-seg_loss
            average_6month = average_6month + seg_dice.item()
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)   

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
            # month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,9,:,:,:]==1]=9            
            
            
            img_save = nib.Nifti1Image(np.float64(Seg_out_argmax[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_prediction' + '.nii.gz')

            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()          
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_T2' + '.nii.gz')      
            
            month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()          
            img_save = nib.Nifti1Image(month_6_real_T1[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_T1' + '.nii.gz')   
                    
            img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_VK' + '.nii.gz')  
            
            
            
            fig5 = plt.figure("5", (24, 18))
            ax5 = fig5.subplots(3,4)
            
                    
            ax5[0,0].title.set_text('6month_real_T1')
            ax5[0,0].imshow(np.rot90(month_6_real_T1[0, 0, :, :,round(month_6_real_T1.shape[4]/2)],k=3),cmap="gray")
            ax5[0,1].title.set_text('6month_real_T2')
            ax5[0,1].imshow(np.rot90(month_6_real_T2[0, 0, :, :, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray")
            ax5[0,2].title.set_text('VK_labels')
            ax5[0,2].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax5[0,3].title.set_text('Seg_outputs')
            ax5[0,3].imshow(np.rot90(Seg_out_argmax[0 , :, :, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)   
                    
                    

            ax5[1,0].title.set_text('6month_real_T1')
            ax5[1,0].imshow(np.rot90(month_6_real_T1[0, 0, :, round(month_6_real_T1.shape[3]/2),:],k=1),cmap="gray")
            ax5[1,1].title.set_text('6month_real_T2')
            ax5[1,1].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray")
            ax5[1,2].title.set_text('VK_labels')
            ax5[1,2].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9)  
            ax5[1,3].title.set_text('Seg_outputs')
            ax5[1,3].imshow(np.rot90(Seg_out_argmax[0, :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)  
                     
                    
                    
            ax5[2,0].title.set_text('6month_real_T1')
            ax5[2,0].imshow(np.rot90(month_6_real_T1[0, 0, round(month_6_real_T1.shape[2]/2), :,:],k=1),cmap="gray")
            ax5[2,1].title.set_text('6month_real_T2')
            ax5[2,1].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray")
            ax5[2,2].title.set_text('VK_labels')
            ax5[2,2].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax5[2,3].title.set_text('Seg_outputs')
            ax5[2,3].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[2]/2), :,:]), cmap="jet",vmin=0,vmax=9)  

            suptitle_name = month_6_name + ' dice score: ' + str(seg_dice.item())
            fig5.suptitle(suptitle_name)
            fig5.savefig(all_seg_path+'/'+ month_6_name +'.png')   
        
        average_6month = average_6month / (idx+1)
        with open(all_seg_path+'/6month_results.txt',"a") as file:
            file.write('The average dice score of 6month subjects are: ')
            file.write(str(average_6month))
            file.write(' \n ')    

        record_metric(csv_record_data, all_seg_path+'/6_Cyc+Att+iBeat+Att.csv')

        for idx, month_6_real in enumerate(self.train_loader_6_month_withoutaff):

            month_6_real_T1 = month_6_real["image_T1"]
            month_6_real_T2 = month_6_real["image_T2"]
            month_6_label = month_6_real["image_label"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]

            month_6_real_T1 = Variable(month_6_real_T1)
            month_6_real_T2 = Variable(month_6_real_T2)
            month_6_label = Variable(month_6_label)
                    

                
            month_6_real_T1, month_6_real_T2, month_6_label = cuda([month_6_real_T1, month_6_real_T2, month_6_label])                              
            data_combine = torch.cat((month_6_real_T1, month_6_real_T2),1)
                

            Seg_out = self.Seg(data_combine)
            # csv_record_data.append([Seg_out.detach().cpu().numpy(),month_6_label.detach().cpu().numpy()])
            # seg_loss = dice_loss(Seg_out, month_6_label)
            # seg_dice= 1-seg_loss
            # average_6month = average_6month + seg_dice.item()
            
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)   

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
            # month_6_label_plot[:,0,:,:,:][month_6_label_plot[:,9,:,:,:]==1]=9            
            
            
            img_save = nib.Nifti1Image(np.float64(Seg_out_argmax[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_prediction' + '.nii.gz')

            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()          
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_T2' + '.nii.gz')      
            
            month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()          
            img_save = nib.Nifti1Image(month_6_real_T1[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, all_seg_path +'/' + month_6_name + '_T1' + '.nii.gz')   
                    
            # img_save = nib.Nifti1Image(month_6_label_plot[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            # nib.save(img_save, all_seg_path +'/' + month_6_name + '_VK' + '.nii.gz')  
            
            
            
            fig6 = plt.figure("5", (24, 18))
            ax6 = fig6.subplots(3,4)
            
                    
            ax6[0,0].title.set_text('6month_real_T1')
            ax6[0,0].imshow(np.rot90(month_6_real_T1[0, 0, :, :,round(month_6_real_T1.shape[4]/2)],k=3),cmap="gray")
            ax6[0,1].title.set_text('6month_real_T2')
            ax6[0,1].imshow(np.rot90(month_6_real_T2[0, 0, :, :, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray")
            ax6[0,2].title.set_text('combined_outputs')
            ax6[0,2].imshow(np.rot90(month_6_label_plot[0 ,0, :, :, round(month_6_label_plot.shape[4]/2)],k=3), cmap="jet",vmin=0,vmax=9)
            ax6[0,3].title.set_text('Seg_outputs')
            ax6[0,3].imshow(np.rot90(Seg_out_argmax[0 , :, :, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)   
                    
                    

            ax6[1,0].title.set_text('6month_real_T1')
            ax6[1,0].imshow(np.rot90(month_6_real_T1[0, 0, :, round(month_6_real_T1.shape[3]/2),:],k=1),cmap="gray")
            ax6[1,1].title.set_text('6month_real_T2')
            ax6[1,1].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray")
            ax6[1,2].title.set_text('combined_outputs')
            ax6[1,2].imshow(np.rot90(month_6_label_plot[0, 0, :, round(month_6_label_plot.shape[3]/2),:],k=1), cmap="jet",vmin=0,vmax=9)  
            ax6[1,3].title.set_text('Seg_outputs')
            ax6[1,3].imshow(np.rot90(Seg_out_argmax[0, :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)  
                     
                    
                    
            ax6[2,0].title.set_text('6month_real_T1')
            ax6[2,0].imshow(np.rot90(month_6_real_T1[0, 0, round(month_6_real_T1.shape[2]/2), :,:],k=1),cmap="gray")
            ax6[2,1].title.set_text('6month_real_T2')
            ax6[2,1].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray")
            ax6[2,2].title.set_text('combined_outputs')
            ax6[2,2].imshow(np.rot90(month_6_label_plot[0, 0, round(month_6_label_plot.shape[2]/2), :,:],k=1), cmap="jet",vmin=0,vmax=9) 
            ax6[2,3].title.set_text('Seg_outputs')
            ax6[2,3].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[2]/2), :,:]), cmap="jet",vmin=0,vmax=9)  

            suptitle_name = month_6_name
            fig6.suptitle(suptitle_name)
            fig6.savefig(all_seg_path+'/'+ month_6_name +'.png')  


















class Seg_apply(object):
    def __init__(self,args):

        # Define the network 
        #####################################################

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Seg=AttentionUnet(
                               spatial_dims=3,
                               in_channels=2,
                               out_channels=9,
                               channels=(32, 64, 128, 256, 512),
                               strides=(2, 2, 2, 2),
                               ).to(device)
        

        print_networks([self.Seg], ['Seg'])       
        
        # Data
        ##################################################### 
        print("Loading the data")          
        apply_ds_6month = CacheDataset(data=apply_dicts_6_month, transform=apply_transform_6_month)
        self.apply_loader_6_month = DataLoader(apply_ds_6month, batch_size=args.batch_size,shuffle=True, num_workers=0,pin_memory=True)
        
        
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
            self.Seg.load_state_dict(ckpt['Seg'])
            self.Seg_optimizer.load_state_dict(ckpt['Seg_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 1
            self.logger=SummaryWriter(log_dir=logger_path)
            self.total_number=0
            self.best_val_accuracy = 0
            
    def apply(self,args):
        print("####################### Doing the whole brain segmentation for 6month infants #######################")
        print("Loading the data")
        
        all_seg_path = root_dir + '/do_seg_all_subjects'
        apply_subjects_path = all_seg_path + '/apply_subjects'

        
        if not os.path.exists(all_seg_path):
            os.makedirs(all_seg_path)
        if not os.path.exists(apply_subjects_path):
            os.makedirs(apply_subjects_path)

        print("Start the segmentation for 6month apply set")
        self.Seg.load_state_dict(torch.load(os.path.join(root_dir, "real_6month_seg.pt"))["state_dict"])
        self.Seg.eval()
        
        for idx, month_6_real in enumerate(self.apply_loader_6_month):

            month_6_real_T1 = month_6_real["image_T1"]
            month_6_real_T2 = month_6_real["image_T2"]
            month_6_name = month_6_real["image_name"][0]
            month_6_affine = month_6_real["affine"]

            month_6_real_T1 = Variable(month_6_real_T1)
            month_6_real_T2 = Variable(month_6_real_T2)
                    

                
            month_6_real_T1, month_6_real_T2 = cuda([month_6_real_T1, month_6_real_T2])                              
            data_combine = torch.cat((month_6_real_T1, month_6_real_T2),1)
                

            Seg_out = self.Seg(data_combine)
            Seg_out = Seg_out.detach().cpu().numpy()
            Seg_out_argmax = np.argmax(Seg_out, axis=1)   
            
                      
            
            
            img_save = nib.Nifti1Image(np.float64(Seg_out_argmax[0,:,:,:]), month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, apply_subjects_path +'/' + month_6_name + '_prediction' + '.nii.gz')

            month_6_real_T2 = month_6_real_T2.detach().cpu().numpy()          
            img_save = nib.Nifti1Image(month_6_real_T2[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, apply_subjects_path +'/' + month_6_name + '_T2' + '.nii.gz')      
            
            month_6_real_T1 = month_6_real_T1.detach().cpu().numpy()          
            img_save = nib.Nifti1Image(month_6_real_T1[0,0,:,:,:], month_6_affine.detach().cpu().numpy()[0,:,:])
            nib.save(img_save, apply_subjects_path +'/' + month_6_name + '_T1' + '.nii.gz')   
                    
            
            
            
            fig6 = plt.figure("6", (18, 18))
            ax6 = fig6.subplots(3,3)
            
                    
            ax6[0,0].title.set_text('6month_real_T1')
            ax6[0,0].imshow(np.rot90(month_6_real_T1[0, 0, :, :,round(month_6_real_T1.shape[4]/2)],k=3),cmap="gray")
            ax6[0,1].title.set_text('6month_real_T2')
            ax6[0,1].imshow(np.rot90(month_6_real_T2[0, 0, :, :, round(month_6_real_T2.shape[4]/2)],k=3),cmap="gray")
            ax6[0,2].title.set_text('Seg_outputs')
            ax6[0,2].imshow(np.rot90(Seg_out_argmax[0 , :, :, round(Seg_out_argmax.shape[3]/2)],k=3), cmap="jet",vmin=0,vmax=9)   
                    
                    

            ax6[1,0].title.set_text('6month_real_T1')
            ax6[1,0].imshow(np.rot90(month_6_real_T1[0, 0, :, round(month_6_real_T1.shape[3]/2),:],k=1),cmap="gray")
            ax6[1,1].title.set_text('6month_real_T2')
            ax6[1,1].imshow(np.rot90(month_6_real_T2[0, 0, :, round(month_6_real_T2.shape[3]/2),:],k=1),cmap="gray")
            ax6[1,2].title.set_text('Seg_outputs')
            ax6[1,2].imshow(np.rot90(Seg_out_argmax[0, :, round(Seg_out_argmax.shape[2]/2),:]), cmap="jet",vmin=0,vmax=9)  
                     
                    
                    
            ax6[2,0].title.set_text('6month_real_T1')
            ax6[2,0].imshow(np.rot90(month_6_real_T1[0, 0, round(month_6_real_T1.shape[2]/2), :,:],k=1),cmap="gray")
            ax6[2,1].title.set_text('6month_real_T2')
            ax6[2,1].imshow(np.rot90(month_6_real_T2[0, 0, round(month_6_real_T2.shape[2]/2), :,:],k=1),cmap="gray")
            ax6[2,2].title.set_text('Seg_outputs')
            ax6[2,2].imshow(np.rot90(Seg_out_argmax[0, round(Seg_out_argmax.shape[2]/2), :,:]), cmap="jet",vmin=0,vmax=9)  

            suptitle_name = month_6_name
            fig6.suptitle(suptitle_name)
            fig6.savefig(apply_subjects_path+'/'+ month_6_name +'.png')    
        
        
        
        






        
#############################################################################
#############################################################################
#############################################################################

def get_args():
    parser = ArgumentParser(description='InfantSeg PyTorch')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_seg', type=float, default=.0004)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=True)
    parser.add_argument('--segment_all', type=bool, default=False)
    parser.add_argument('--applying', type=bool, default=False)
    parser.add_argument('--checkpoint_dir', type=str, default=root_dir)
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
  
  if args.training:
    print("Training")
    model = InfantSeg(args)
    model.train(args)
  if args.testing:
    print("Testing")
    model = InfantSeg(args)
    model.test(args)
  if args.segment_all:
    print("Segmenting all 6-month subjects")
    model = InfantSeg(args)
    model.segment_all(args)
  
  if args.applying:
    print("applying")
    model = Seg_apply(args)
    model.apply(args)
  
     

if __name__ == '__main__':
    main()






