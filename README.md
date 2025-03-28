# Automatic 8 Tissue Segmentation for 6-Month Infant Brains
The GitHub repository for the paper "Automatic 8-Tissue Segmentation for 6-Month Infant Brains" is available here.

In this Paper, we propose the first 8-tissue segmentation pipeline for 6-month-old infant brains. The segmented tissues include WM, gray matter (GM), cerebrospinal fluid (CSF), ventricles, cerebellum, basal ganglia, brainstem, and hippocampus/amygdala.

# Contribution
Our pipeline takes raw 6-month images as inputs and generates the 8-tissue segmentation as outputs, forming an end-to-end segmentation pipeline. Our evaluation with real 6-month images achieved a DICE score of 0.92, an HD95 of 1.6 mm, and an ASSD of 0.42 mm.

![image](https://github.com/YilanDong19/Automatic_8_Tissue_Segmentation_for_6_Month_Infant_Brains/blob/3af626cafa3a7eb473c2df85c5db51a30166044d/Graph/graph.png)

# Paper

Published in MICCAI PIPPI. [http://dx.doi.org/10.1002/hbm.70190](https://link.springer.com/chapter/10.1007/978-3-031-73260-7_6)

# Citation
If you use this code for your research, please cite our paper:

Dong Y, Kyriakopoulou V, Grigorescu I, et al. Automatic 8-Tissue Segmentation for 6-Month Infant Brains[C]//International Workshop on Preterm, Perinatal and Paediatric Image Analysis. Cham: Springer Nature Switzerland, 2024: 59-69.

# Introduction: 
To investigate the best solution for segmenting our target dataset (6-month data), we train and compare five different DL pipelines:

### 1. AUNet (baseline)

A MONAI Attention UNet [1] was trained on the neonatal T2w images and labels, then applied directly to real 6-month T2w images.

### 2. Cyc+AUNet

CycleGAN [2] was employed to transform neonatal T2w images into synthesized 6-month T2w images (neonatal images with 6-month intensity contrast). At the same time, an Attention UNet was trained on these synthesized 6-month images to predict their corresponding neonatal labels.

### 3. Cyc+AUNet+VM

Using the pre-trained Cyc+AUNet, we further employed VoxelMorph [3] to register synthesized 6-month T2w images to the real 6-month image space (paired), and keep training the Attention UNet on the warped synthesized 6-month images to predict warped neonatal labels.

### 4. Cyc+AUNet+iBEAT

First, we fed real 6-month T1w and T2w images into iBEAT to obtain the segmentation outputs. Second, we employed the same strategy as for Cyc+AUNet, but replaced the WM, GM and CSF segmentation outputs of Cyc+AUNet with iBEAT segmentation outputs.


### 5. Cyc+AUNet+iBEAT+AUNet

We trained a second Attention UNet on the real 6-month T1w and T2w images and their segmentation outputs from Cyc+AUNet+iBEAT.

![image](https://github.com/YilanDong19/Automatic_8_Tissue_Segmentation_for_6_Month_Infant_Brains/blob/33eb9812af6597e26ad48b20724a1b224234a305/Graph/pipelines.png)



**1. Download First_step_prepare.ipynb (this file prepares everything)**

**2. Upload First_step_prepare.ipynb to google drive, open with google colab**

**3. Run First_step_prepare.ipynb**

The functions of First_step_prepare.ipynb :
1. Create folders
2. Download fMRI and sMRI data to google drive
3. Download necessary scripts

I also prepared a copy of all the necessary scripts in folder "scripts" for those who want to have a quick look. For those who want to run these scripts, you still need to run First_step_prepare.ipynb at the beginning.


### Models: 
1. SVM 
2. FCN
3. AE-FCN [1] 
4. GCN [2]
5. EV-GCN [3]

### Dataset: 
ABIDE 870 samples, one subject failed with FreeSurfer pipeline

### Feature types: 
1. fMRI (solely CC200 atlas, you can perform experiments on other atlas from ABIDE database)
2. sMRI (our Freesurfer outputs, 7.0)
3. fMRI(CC200) + sMRI
               
### Ensemble methods: 
1. Max voting [4]
2. EMMA [5]
                  

###   Folder "scripts": 
I uploaded all the necessary scripts to this folder as a copy for those who want to have a quick look. Most of the scripts are named by "machine learning model" + "feature type", for example, AE_FCN_fMRI.ipynb is to train Auto-encoder + fully connected neural network (AE-FCN) on fMRI features. The scripts named "EMMA_*" can only be run after all the other scripts, since EMMA ensemble method is to combine the outputs of all the machine learning models. 

###  Folder "SmoothGrad_interpretation": 
The SmoothGrad application for FCN, AE-FCN, GCN, and EV-GCN has been uploaded to this folder. We utilized SmoothGrad interpretation methods on these models to assess their stability and identify the features influencing their decision-making.

###  References
[1] Rakić, Mladen, Mariano Cabezas, Kaisar Kushibar, Arnau Oliver, and Xavier Lladó. 2020. ‘Improving the Detection of Autism Spectrum Disorder by Combining Structural and Functional MRI Information’. NeuroImage: Clinical 25(November 2019). doi: 10.1016/j.nicl.2020.102181. 

[2] Parisot, Sarah, Sofia Ira Ktena, Enzo Ferrante, Matthew Lee, Ricardo Guerrero, Ben Glocker, and Daniel Rueckert. 2018. ‘Disease Prediction Using Graph Convolutional Networks: Application to Autism Spectrum Disorder and Alzheimer’s Disease’. Medical Image Analysis 48:117–30. doi: 10.1016/j.media.2018.06.001. 

[3] Huang, Yongxiang, and Albert C. S. Chung. 2020. ‘Edge-Variational Graph Convolutional Networks for Uncertainty-Aware Disease Prediction’. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) 12267 LNCS:562–72. doi: 10.1007/978-3-030-59728-3_55. 

[4] Thomas G. Dietterich. 2000. ‘Ensemble Methods in Machine Learning’. Multiple Classifier Systems. 

[5] Kamnitsas, Konstantinos, Wenjia Bai, Enzo Ferrante, Steven McDonagh, Matthew Sinclair, Nick Pawlowski, Martin Rajchl, Matthew Lee, Bernhard Kainz, Daniel Rueckert, and Ben Glocker. 2017. ‘Ensembles of Multiple Models and Architectures for Robust Brain Tumour Segmentation’.

[6] Jean-Philippe Fortin, Drew Parker, Birkan Tunc, Takanori Watanabe, Mark A Elliott, Kosha Ruparel, David R Roalf, Theodore D Satterthwaite, Ruben C Gur, Raquel E Gur, Robert T Schultz, Ragini Verma, Russell T Shinohara. Harmonization Of Multi-Site Diffusion Tensor Imaging Data. NeuroImage, 161, 149-170, 2017
