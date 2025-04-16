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


The **Cyc+AUNet+iBEAT+AUNet** pipeline demonstrated the highest performance, achieving a DICE score of 0.92, an HD95 of 1.6 𝑚𝑚, and an ASSD of 0.42 𝑚𝑚 in 6-month brain segmentation. More comparison details can be found in our paper.


We uploaded the final segmentation models to the folder "Saved_segmentation_models"





[5] Kamnitsas, Konstantinos, Wenjia Bai, Enzo Ferrante, Steven McDonagh, Matthew Sinclair, Nick Pawlowski, Martin Rajchl, Matthew Lee, Bernhard Kainz, Daniel Rueckert, and Ben Glocker. 2017. ‘Ensembles of Multiple Models and Architectures for Robust Brain Tumour Segmentation’.

[6] Jean-Philippe Fortin, Drew Parker, Birkan Tunc, Takanori Watanabe, Mark A Elliott, Kosha Ruparel, David R Roalf, Theodore D Satterthwaite, Ruben C Gur, Raquel E Gur, Robert T Schultz, Ragini Verma, Russell T Shinohara. Harmonization Of Multi-Site Diffusion Tensor Imaging Data. NeuroImage, 161, 149-170, 2017
