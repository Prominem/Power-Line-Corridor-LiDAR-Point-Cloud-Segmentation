# Power Line Corridor LiDAR Point Cloud Segmentation Using Convolutional Neural Network
Created by Jisheng Yang, Zijun Huang, Maochun Huang, Xianxina Zeng, Dong Li, Yun Zhang <br>
![](https://github.com/Prominem/Power-Line-Corridor-LiDAR-Point-Cloud-Segmentation/blob/master/figure6.jpg)
# introduction
This repository is code release of out PRCV 2019 paper (here). In this work, we propose a deep learning based method to segment power line corridor LiDAR point cloud. We design an effective channel presentation for LiDAR point cloud and adapt a point cloud segmentation network ([pointnet](https://github.com/charlesq34/pointnet)) as our basic network. To verify the generalization ability of our channel presentation, we also do experiments on Kitti dataset. Experiments show that our channel presentation not only works well on Power Line Corridor LiDAR Point Cloud dataset, but also generalizes well on Kitti dataset.<br>
Since Kitti doesn't provide point-wise semantic labels, we obtain semantic labels with methods discribed in [squeezeseg](https://github.com/BichenWuUCB/SqueezeSeg), assigning same labels to points within a 3D bounding box.<br>
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
Power Line Corridor LiDAR Point Cloud dataset is classified, so only experiments on Kitti dataset is discribed. But codes on Power Line Corridor LiDAR Point Cloud dataset is also release.
# Prepare Training and Validation Data
Download Kitti 3D object([here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)), put them in ```/data```, unzip them and organize the folders as follows:
```
data/kitti/
  data_object_velodyne/
  label2/
```
Run ``` calculate_3d_bbox_corners.py ``` to extract point clouds from kitti and label them, you can use [CC](http://www.cloudcompare.org/) to visulize in ```/data/labeled_point_cloud/```. Run ```nine2four.py``` to delete some labels, since we only consider there class: Car, Pedestrian, Cyclist. You can also visulize in /data/input_point_cloud_dir.<br>
## train using pointnet
Run ```data_process_base_kitti.py``` and ```seg_codes/train_base.py``` in sequence. 
## validate using pointnet
Run ```seg_codes/batch_inference_base.py```. You can visualize results in /log_base/dump/. And then open ```seg_codes/statistics_mul.py```  and make sure ```results_dir = base_dir + '/../log_base/dump'``` and run it to see IoU, precision and recall.
## train using our method(p3)
Run ```data_process_local_kitti.py``` and ```seg_codes/train_local.py``` in sequence. 
## validate using our method(p3)
Run ```seg_codes/batch_inference_local.py```. You can visualize results in /log_local/dump/. And then open ```seg_codes/statistics_mul.py``` and make sure ```results_dir = base_dir + '/../log_local/dump'``` and run it  to see IoU, precision and recall.
## extended experiments
We found channel presentation ![](https://github.com/Prominem/Power-Line-Corridor-LiDAR-Point-Cloud-Segmentation/blob/master/p4.png) works better in Kitti dataset. I means the original intensity. You can try the following operation to see the results.
## train using our method(p4)
Run ```data_process_local_kitti_v2.py``` and ```seg_codes/train_local_v2.py``` in sequence. 
## validate using our method(p4)
Run ```seg_codes/batch_inference_local_v2.py```. You can visualize results in /log_local_v2/dump/. And then open ```seg_codes/statistics_mul.py``` and make sure ```results_dir = base_dir + '/../log_local_v2/dump'``` and run it  to see IoU, precision and recall.
# result
The results are different from those in paper and they are better. Because we didn't have time to train enough epoches before submitting the paper. Table 1 shows the IOU result of 20 epoches. 
![](https://github.com/Prominem/Power-Line-Corridor-LiDAR-Point-Cloud-Segmentation/blob/master/table1.png)
# References
[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://github.com/charlesq34/pointnet)<br>
[SqueezeSeg](https://github.com/BichenWuUCB/SqueezeSeg)



