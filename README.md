# DSTUF for CASSI


This repo is the illustration and implementation of paper "Exploiting Frequency Correlation for Hyperspectral Image Reconstruction"

|                          *Scene 1*                           |                          *Scene 2*                           |                          *Scene 3*                           |                          *Scene 4*                           |                          *Scene 5*                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/ID1267/DST/blob/main/figures/woman.gif"  height="80%" width="80%"> | <img src="https://github.com/ID1267/DST/blob/main/figures/cube.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/DST/blob/main/figures/colorboard.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/DST/blob/main/figures/orange.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/DST/blob/main/figures/cup.gif"  height="80%" width="80%"> |
|                          *Scene 6*                           |                          *Scene 7*                           |                          *Scene 8*                           |                          *Scene 9*                           |                          *Scene 10*                           |
| <img src="https://github.com/ID1267/DST/blob/main/figures/person.gif"  height="80%" width="80%"> | <img src="https://github.com/ID1267/DST/blob/main/figures/flower.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/DST/blob/main/figures/duck.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/DST/blob/main/figures/pencil.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/DST/blob/main/figures/lion.gif"  height="80%" width="80%"> |



# Abstract

Deep priors have emerged as potent methods in hyperspectral image (HSI) reconstruction. While most methods emphasize space-domain learning using image space priors like non-local similarity, frequency-domain learning using image frequency priors remains neglected, limiting the reconstruction capability of networks. In this paper, we first propose a Hyperspectral Frequency Correlation (HFC) prior rooted in in-depth statistical frequency analyses of existent HSI datasets. Leveraging the HFC prior, we subsequently establish the frequency domain learning composed of a Spectral-wise self-Attention of Frequency (SAF) and a Spectral-spatial Interaction of Frequency (SIF) targeting low-frequency and high-frequency components, respectively. The outputs of SAF and SIF are adaptively merged by a learnable gating filter, thus achieving a thorough exploitation of image frequency priors. Integrating the frequency domain learning and the existing space domain learning, we finally develop the Correlation-driven Mixing Domains Transformer (CMDT) for HSI reconstruction. Extensive experiments highlight that our method surpasses various state-of-the-art (SOTA) methods in reconstruction quality and computational efficiency.

# Comparison with state-of-the-art methods

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/psnr-ssim-flops-final.png" height="50%" width="50%" alt="">
</div>

PSNR-Params-FLOPs comparisons of our DSTUF and SOTA methods. The scale of circles matches with parameters (M).

# Hyperspectral Frequency Correlation (HFC) Prior

## Rough Visualization

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/visualdct.jpg" height="80%" width="80%" alt="">
</div>

The first colomn is the $9^{th}$ Scene of KAIST. The rest columns are the spectrograms of different spectra with specific wavelengths of $Scene9$.

## HFC PartI

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/HFC.png" height="40%" width="40%" alt="">
</div>

Top: example HSIs from the KAIST dataset and remote datasets: Chikusei and Urban. Middle: the corresponding spectral correlation heatmap in the space domain. Bottom: the corresponding spectral correlation heatmap in the frequency domain.

## HFC PartII

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/patch.png" height="50%" width="50%" alt="">
</div>

Visualisation of the frequency-domain spectral correlation heatmap from patch-1 to patch-5 in spectrogram of HSI.

## Statistics of HFC

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/distribution_graph.png" height="30%" width="30%" alt="">
<img src="https://github.com/ID1267/DST/blob/main/figures/distribution_graph_whole.png" height="30%" width="30%" alt="">
</div>

(a) Histogram of the number of samples with spectral correlation in dual domains of all 1029 HSIs. (b) The probability distribution graph of spectral correlation in dual domains of all HSIs.

# Architecture

## Dual-domain Spectral-spatial Transformer-based Unfolding Framework

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/DFUF.png" height="80%" width="80%" alt="">
</div>

The architecture of our RDLUF with $K$ stages (iterations). RDLGD and PM denote the Residual Degradation Learning Gradient Descent module and the Proximal Mapping module in each stage. There is a stage interaction between stages.

## Frequency Process

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/FDLB.png" height="80%" width="80%" alt="">
</div>

Diagram of the Frequency Process. (a) The structure of the patched frequency-domain channel-wise self-attention module. (b) The structure of the spectral-spatial frequency fusion module.



# Usage 

## Prepare Dataset:

Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--DST
    |--real
    	|-- test_code
    	|-- train_code
    |--simulation
    	|-- test_code
    	|-- train_code
    |--visualization
    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask_3d_shift.mat
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask_3d_shift.mat
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
```

 We use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set.

## Simulation Experiement:

### Training


```
cd DST/simulation/train_code/

# DSTUF 2stage
python train.py --template dstuf_2stg --outf ./exp/dstuf_2stg/ --method dstuf_2stg

# DSTUF 3stage
python train.py --template dstuf_3stg --outf ./exp/dstuf_3stg/ --method dstuf_3stg

# DSTUF 5stage
python train.py --template dstuf_5stg --outf ./exp/dstuf_5stg/ --method dstuf_5stg

# DSTUF 9stage
python train.py --template dstuf_9stg --outf ./exp/dstuf_9stg/ --method dstuf_9stg
```

The training log, trained model, and reconstrcuted HSI will be available in `DSTUF/simulation/train_code/exp/` .

### Testing

Place the pretrained model path to "opt.pretrained_model_path" in "DST/simulation/test_code/option.py".

Run the following command to test the model on the simulation dataset.

```
cd DST/simulation/test_code/


# DSTUF 2stage
python test.py --template dstuf_2stg --outf ./exp/dstuf_2stg/ --method dstuf_2stg

# DSTUF 3stage
python test.py --template dstuf_3stg --outf ./exp/dstuf_3stg/ --method dstuf_3stg

# DSTUF 5stage
python test.py --template dstuf_5stg --outf ./exp/dstuf_5stg/ --method dstuf_5stg

# DSTUF 9stage
python test.py --template dstuf_9stg --outf ./exp/dstuf_9stg/ --method dstuf_9stg
```

- The reconstrcuted HSIs will be output into `DSTUF/simulation/test_code/exp/`
- Place the reconstructed results into `DSTUF/simulation/test_code/Quality_Metrics/results` and

```
Run cal_quality_assessment.m
```

to calculate the PSNR and SSIM of the reconstructed HSIs.

### Visualization

- Put the reconstruted HSI in `DSTUF/visualization/simulation_results/results` and rename it as method.mat, e.g., DSTUF_2stg.mat
- Generate the RGB images of the reconstructed HSIs

```
cd DSTUF/visualization/
Run show_simulation.m 
```

## Real Experiement:

### Training

```
cd DSTUF/real/train_code/

# DSTUF 2stage
python train.py --template DSTUF_2stg --outf ./exp/DSTUF_2stg/ --method DSTUF_2stg
```

The training log and trained model will be available in `DSTUF/real/train_code/exp/`

### Testing

```
Place the pretrained model path to "pretrained_model_path" in "DSTUF/real/test_code/test.py".

cd DSTUF/real/test_code/

# DSTUF 2stage
python test.py --template DSTUF_2stg --outf ./exp/DSTUF_2stg/ --method DSTUF_2stg
```

The reconstrcuted HSI will be output into `DSTUF/real/test_code/Results/`

### Visualization

- Put the reconstruted HSI in `DSTUF/visualization/real_results/results` and rename it as method.mat, e.g., DSTUF_2stg.mat.
- Generate the RGB images of the reconstructed HSI

```
cd DSTUF/visualization/
Run show_real.m
```

## Acknowledgements

Our code is heavily borrowed from [MST](https://github.com/caiyuanhao1998/MST)  and [DAUHST](https://github.com/caiyuanhao1998/MST), thanks for their generous open source.
