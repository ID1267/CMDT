# CMDT for CASSI


This repo is the illustration and implementation of paper "Exploiting Frequency Correlation for Hyperspectral Image Reconstruction"

|                          *Scene 1*                           |                          *Scene 2*                           |                          *Scene 3*                           |                          *Scene 4*                           |                          *Scene 5*                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/ID1267/CMDT/blob/main/figures/woman.gif"  height="80%" width="80%"> | <img src="https://github.com/ID1267/CMDT/blob/main/figures/cube.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/CMDT/blob/main/figures/colorboard.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/CMDT/blob/main/figures/orange.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/CMDT/blob/main/figures/cup.gif"  height="80%" width="80%"> |
|                          *Scene 6*                           |                          *Scene 7*                           |                          *Scene 8*                           |                          *Scene 9*                           |                          *Scene 10*                           |
| <img src="https://github.com/ID1267/CMDT/blob/main/figures/person.gif"  height="80%" width="80%"> | <img src="https://github.com/ID1267/CMDT/blob/main/figures/flower.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/CMDT/blob/main/figures/duck.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/CMDT/blob/main/figures/pencil.gif" width="80%" height="80%"> | <img src="https://github.com/ID1267/CMDT/blob/main/figures/lion.gif"  height="80%" width="80%"> |



# Abstract

Deep priors have emerged as potent methods in hyperspectral image (HSI) reconstruction. While most methods emphasize space-domain learning using image space priors like non-local similarity, frequency-domain learning using image frequency priors remains neglected, limiting the reconstruction capability of networks. In this paper, we first propose a Hyperspectral Frequency Correlation (HFC) prior rooted in in-depth statistical frequency analyses of existent HSI datasets. Leveraging the HFC prior, we subsequently establish the frequency domain learning composed of a Spectral-wise self-Attention of Frequency (SAF) and a Spectral-spatial Interaction of Frequency (SIF) targeting low-frequency and high-frequency components, respectively. The outputs of SAF and SIF are adaptively merged by a learnable gating filter, thus achieving a thorough exploitation of image frequency priors. Integrating the frequency domain learning and the existing space domain learning, we finally develop the Correlation-driven Mixing Domains Transformer (CMDT) for HSI reconstruction. Extensive experiments highlight that our method surpasses various state-of-the-art (SOTA) methods in reconstruction quality and computational efficiency.

# Comparison with state-of-the-art methods

<div align=center>
<img src="https://github.com/ID1267/CMDT/blob/main/figures/psnr-ssim-flops-final.png" height="50%" width="50%" alt="">
</div>

PSNR-Params-FLOPs comparisons of our method and SOTA methods. The scale of circles matches with parameters (M).

# Hyperspectral Frequency Correlation (HFC) Prior

## Initial Observation

<div align=center>
<img src="https://github.com/ID1267/CMDT/blob/main/figures/dctvisual.png" height="80%" width="80%" alt="">
</div>

The first column is the $9^{th}$ Scene of KAIST. The rest columns are the spectrograms of different spectra with specific wavelengths of $Scene9$.

## Visualisations in HFC

<div align=center>
<img src="https://github.com/ID1267/CMDT/blob/main/figures/HFC.png" height="40%" width="40%" alt="">
</div>

Top: example HSIs from the nature scenes and the remote sensing scenes. Bottom: the corresponding spectral-spatial correlation heatmap of frequency tokens and space tokens.

<div align=center>
<img src="https://github.com/ID1267/CMDT/blob/main/figures/patch.png" height="50%" width="50%" alt="">
</div>

Visualization of the spectral-spatial correlation heatmap from frequency token-1 to token-5 in the spectrogram of HSI.

## Statistics in HFC

<div align=center>
<img src="https://github.com/ID1267/CMDT/blob/main/figures/histogram.png" height="30%" width="30%" alt="">
<img src="https://github.com/ID1267/CMDT/blob/main/figures/probability.png" height="25%" width="30%" alt="">
</div>

(a) Histogram of the number of samples with spectral correlation in two domains of 1029 HSIs. (b) The probability distribution graph of spectral correlation in two domains of 1029 HSIs.

# Architecture

## Deep Frequency Unfolding Framework
<div align=center>
<img src="https://github.com/ID1267/CMDT/blob/main/figures/DFUF.png" height="80%" width="80%" alt="">
</div>

Diagram of the overall framework. (a) Deep Frequency Unfolding Framework. (b) The pipeline of the U-shaped prior module. (c) The correlation-driven mixing domains transformer. (d) The structure of space domain learning.

## The Frequency Domain Learning

<div align=center>
<img src="https://github.com/ID1267/CMDT/blob/main/figures/FDLB.png" height="80%" width="80%" alt="">
</div>

The pipeline of the frequency domain learning. (a) The structure of the spectral-wise self-attention of frequency. (b) The structure of the spectral-spatial interaction of frequency.



# Usage 

## Prepare Dataset:

Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--CMDT
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
cd CMDT/simulation/train_code/

# CMDT 2stage
python train.py --template CMDT_2stg --outf ./exp/CMDT_2stg/ --method CMDT_2stg

# CMDT 3stage
python train.py --template CMDT_3stg --outf ./exp/CMDT_3stg/ --method CMDT_3stg

# CMDT 5stage
python train.py --template CMDT_5stg --outf ./exp/CMDT_5stg/ --method CMDT_5stg

# CMDT 9stage
python train.py --template CMDT_9stg --outf ./exp/CMDT_9stg/ --method CMDT_9stg
```

The training log, trained model, and reconstructed HSI will be available in `CMDT/simulation/train_code/exp/`.

### Testing

Place the pre-trained model path to "opt.pretrained_model_path" in "CMDT/simulation/test_code/option.py".

Run the following command to test the model on the simulation dataset.

```
cd CMDT/simulation/test_code/


# CMDT 2stage
python test.py --template CMDT_2stg --outf ./exp/CMDT_2stg/ --method CMDT_2stg

# CMDT 3stage
python test.py --template CMDT_3stg --outf ./exp/CMDT_3stg/ --method CMDT_3stg

# CMDT 5stage
python test.py --template CMDT_5stg --outf ./exp/CMDT_5stg/ --method CMDT_5stg

# CMDT 9stage
python test.py --template CMDT_9stg --outf ./exp/CMDT_9stg/ --method CMDT_9stg
```

- The reconstructed HSIs will be output into `CMDT/simulation/test_code/exp/`
- Place the reconstructed results into `CMDT/simulation/test_code/Quality_Metrics/results` and

```
Run cal_quality_assessment.m
```

to calculate the PSNR and SSIM of the reconstructed HSIs.

### Visualization

- Put the reconstructed HSI in `CMDT/visualization/simulation_results/results` and rename it as method.mat, e.g., CMDT_2stg.mat
- Generate the RGB images of the reconstructed HSIs

```
cd CMDT/visualization/
Run show_simulation.m 
```

## Real Experiement:

### Training

```
cd CMDT/real/train_code/

# CMDT 2stage
python train.py --template CMDT_2stg --outf ./exp/CMDT_2stg/ --method CMDT_2stg
```

The training log and trained model will be available in `CMDT/real/train_code/exp/`

### Testing

```
Place the pre-trained model path to "pretrained_model_path" in "CMDT/real/test_code/test.py".

cd CMDT/real/test_code/

# CMDT 2stage
python test.py --template CMDT_2stg --outf ./exp/CMDT_2stg/ --method CMDT_2stg
```

The reconstructed HSI will be output into `CMDT/real/test_code/Results/`

### Visualization

- Put the reconstructed HSI in `CMDT/visualization/real_results/results` and rename it as method.mat, e.g., CMDT_2stg.mat.
- Generate the RGB images of the reconstructed HSI

```
cd CMDT/visualization/
Run show_real.m
```

## Acknowledgements

Our code is heavily borrowed from [MST](https://github.com/caiyuanhao1998/MST)  and [DAUHST](https://github.com/caiyuanhao1998/MST), thanks for their generous open source.
