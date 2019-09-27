# Power Line Corridor LiDAR Point Cloud Segmentation Using Convolutional Neural Network
Created by Jisheng Yang, Zijun Huang, Maochun Huang, Xianxina Zeng, Dong Li, Yun Zhang <br>
![](https://github.com/Prominem/Power-Line-Corridor-LiDAR-Point-Cloud-Segmentation/blob/master/figure6.jpg)
# introduction
This repository is code release of out PRCV 2019 paper (here). In this work, we propose a deep learning based method to segment power line corridor LiDAR point cloud. We design an effective channel presentation for LiDAR point cloud and adapt a point cloud segmentation network ([pointnet](https://github.com/charlesq34/pointnet)) as our basic network. To verify the generalization ability of our channel presentation, we also do experiments on Kitti dataset. Experiments show that our channel presentation not only works well on Power Line Corridor LiDAR Point Cloud dataset, but also generalizes well on Kitti dataset.<br>
For more details of our method, please refer to our paper.
# Citation
If you find our work useful in your research, please consider citing:

```
@ARTICLE{power line segmentation,
author={Jisheng Yang, Zijun Huang, Maochun Huang, Xianxina Zeng, Dong Li, Yun Zhang},
title = {Power Line Corridor LiDAR Point Cloud Segmentation Using Convolutional Neural Network},
journal={PRCV 2019},
year={2019}
}

```
# Installation
Codes of this release is implement with python3.6. Please install numpy==1.16, tensorflow==1.14.
# Usage
Power Line Corridor LiDAR Point Cloud dataset is classified, 
