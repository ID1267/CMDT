# DSTUF for CASSI


This repo is the illustration and implementation of paper "Exploiting Spectral Correlation in Frequency Domain for Hyperspectral Imaging"

|                          *Scene 1*                           |                          *Scene 6*                           |                          *Scene 8*                           |                          *Scene 10*                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/scene1.gif"  height=170 width=170> | <img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/scene6.gif" width=170 height=170> | <img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/scene8.gif" width=170 height=170> | <img src="https://github.com/ShawnDong98/RDLUF_MixS2/blob/master/figures/scene10.gif" width=170 height=170> |


# Abstract

Deep-prior-based unfolding networks are recently emerging as potent methods in hyperspectral image (HSI) reconstruction. While most methods prioritize space-domain learning with existing image priors like non-local similarity, frequency-domain exploitation often goes overlooked. Few methods try to seek solutions in the frequency domain via sole frequency loss or end-to-end brute-force mapping, lacking interpretability and efficiency due to the absence of a specific frequency prior for HSI. In this paper, we propose a hyperspectral frequency correlation (HFC) prior rooted in an in-depth statistical analysis of existent HSI datasets. Utilizing the HFC prior, we birth the frequency domain learning branch (FDLB) composed of a attention-based module and a convolution-based module, separately targeting low frequencies and high frequencies. The outputs of two modules are adaptively merged via a learnable frequency-level filter, ensuring a thorough frequency exploration. Additionally, an attention-driven space domain learning branch (SDLB) is incorporated to exploit pixel-level content as complementary to FDLB. Integrating FDLB and SDLB, we establish the dual-domain spectral-spatial transformer (DST) for HSI reconstruction. Extensive experiments highlight the superiority of our HFC-driven DST over various state-of-the-art (SOTA) methods in both reconstruction quality and computational efficiency.

# Comparison with state-of-the-art methods

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/psnr-ssim-flops.png" height="50%" width="50%" alt="">
</div>

# Hyperspectral Frequency Correlation (HFC) Prior

## Rough Visualization

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/visualdct.jpg" height="80%" width="80%" alt="">
</div>

The first colomn is the $9^{th}$ Scene of KAIST. The rest columns are the spectrograms of different spectra with specific wavelengths of $Scene9$.

## HFC PartI

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/overallfig.jpg" height="80%" width="80%" alt="">
</div>

Top: example HSIs from the KAIST dataset and remote datasets: Chikusei and Urban. Middle: the corresponding spectral correlation heatmap in the space domain. Bottom: the corresponding spectral correlation heatmap in the frequency domain.

## HFC PartII

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/fig2.jpg" height="80%" width="80%" alt="">
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
<img src="https://github.com/ID1267/DST/blob/main/figures/DSTUF.jpg" height="80%" width="80%" alt="">
</div>

The architecture of our RDLUF with $K$ stages (iterations). RDLGD and PM denote the Residual Degradation Learning Gradient Descent module and the Proximal Mapping module in each stage. There is a stage interaction between stages.

## Mixing priors across Spectral and Spatial Transformer(PM)

<div align=center>
<img src="https://github.com/ID1267/DST/blob/main/figures/FCSA.jpg" height="80%" width="80%" alt="">
</div>

Diagram of the Mix $S^2$ Transformer. (a) Mix $S^2$ Transformer adopts a U-shaped structure with block interactions. (b) The basic unit of the MixS2 Transformer, Mix $S^2$ block. (c) The structure of the spectral self-attention branch. (d) The structure of the lightweight inception branch. (e) The components of the gated-Dconv feed-forward network(GDFN)



# Usage 

## Prepare Dataset:

Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--RDLUF_MixS2
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
cd RDLUF_MixS2/simulation/train_code/

# RdLUF-MixS2 3stage
python train.py --template duf_mixs2 --outf ./exp/duf_mixs2_3stage/ --method duf_mixs2 --stage 3 --body_share_params 0  --clip_grad

# RdLUF-MixS2 5stage
python train.py --template duf_mixs2 --outf ./exp/duf_mixs2_5stage/ --method duf_mixs2 --stage 5 --body_share_params 1  --clip_grad

# RdLUF-MixS2 7stage
python train.py --template duf_mixs2 --outf ./exp/duf_mixs2_7stage/ --method duf_mixs2 --stage 7 --body_share_params 1  --clip_grad

# RdLUF-MixS2 9stage
python train.py --template duf_mixs2 --outf ./exp/duf_mixs2_9stage/ --method duf_mixs2 --stage 9 --body_share_params 1  --clip_grad
```

The training log, trained model, and reconstrcuted HSI will be available in `RDLUF_MixS2/simulation/train_code/exp/` .

### Testing

Place the pretrained model to `RDLUF_MixS2/simulation/test_code/checkpoints/`

Run the following command to test the model on the simulation dataset.

```
cd RDLUF_MixS2/simulation/test_code/


# RdLUF-MixS2 3stage
python test.py --template duf_mixs2 --stage 3 --body_share_params 0 --outf ./exp/duf_mixs2_3stage/ --method duf_mixs2 --pretrained_model_path ./checkpoints/RDLUF_MixS2_3stage.pth

# RdLUF-MixS2 5stage
python test.py --template duf_mixs2 --stage 5 --body_share_params 1 --outf ./exp/duf_mixs2_5stage/ --method duf_mixs2 --pretrained_model_path ./checkpoints/RDLUF_MixS2_5stage.pth

# RdLUF-MixS2 7stage
python test.py --template duf_mixs2 --stage 7 --body_share_params 1 --outf ./exp/duf_mixs2_7stage/ --method duf_mixs2 --pretrained_model_path ./checkpoints/RDLUF_MixS2_7stage.pth

# RdLUF-MixS2 9stage
python test.py --template duf_mixs2 --stage 9 --body_share_params 1 --outf ./exp/duf_mixs2_9stage/ --method duf_mixs2 --pretrained_model_path ./checkpoints/RDLUF_MixS2_9stage.pth
```

- The reconstrcuted HSIs will be output into `RDLUF_MixS2/simulation/test_code/exp/`
- Place the reconstructed results into `RDLUF_MixS2/simulation/test_code/Quality_Metrics/results` and

```
Run cal_quality_assessment.m
```

to calculate the PSNR and SSIM of the reconstructed HSIs.

### Visualization

- Put the reconstruted HSI in `RDLUF_MixS2/visualization/simulation_results/results` and rename it as method.mat, e.g., RDLUF_MixS2_9stage.mat
- Generate the RGB images of the reconstructed HSIs

```
cd RDLUF_MixS2/visualization/
Run show_simulation.m 
```

## Real Experiement:

### Training

```
cd RDLUF_MixS2/real/train_code/

# RDLUF-MixS2 3stage
python train.py --template duf_mixs2 --outf ./exp/rdluf_mixs2_3stage/ --method duf_mixs2 --stage 3 --body_share_params 1
```

The training log and trained model will be available in `RDLUF_MixS2/real/train_code/exp/`

### Testing

```
cd RDLUF_MixS2/real/test_code/

# RDLUF-MixS2 3stage
python test.py --template duf_mixs2 --outf ./exp/rdluf_mixs2_3stage/ --method duf_mixs2 --stage 3 --body_share_params 1 --pretrained_model_path ./checkpoints/RDLUF_MixS2_3stage.pth
```

The reconstrcuted HSI will be output into `RDLUF_MixS2/real/test_code/Results/`

### Visualization

- Put the reconstruted HSI in `RDLUF_MixS2/visualization/real_results/results` and rename it as method.mat, e.g., RDLUF_MixS2_3stage.mat.
- Generate the RGB images of the reconstructed HSI

```
cd RDLUF_MixS2/visualization/
Run show_real.m
```

## Acknowledgements

Our code is heavily borrowed from [MST](https://github.com/caiyuanhao1998/MST)  and [DGSMP](https://github.com/TaoHuang95/DGSMP), thanks for their generous open source.


## Citation

If this code helps you, please consider citing our works:

```shell
@inproceedings{dong2023residual,
  title={Residual Degradation Learning Unfolding Framework with Mixing Priors across Spectral and Spatial for Compressive Spectral Imaging},
  author={Dong, Yubo and Gao, Dahua and Qiu, Tian and Li, Yuyan and Yang, Minxi and Shi, Guangming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22262--22271},
  year={2023}
}
```
