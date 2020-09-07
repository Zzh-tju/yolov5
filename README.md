<img src="CIoU.png" width="800px"/>
## Complete-IoU Loss and Cluster-NMS for Improving Object Detection and Instance Segmentation. 

<a href="https://apps.apple.com/app/id1452689527" target="_blank">
<img src="https://user-images.githubusercontent.com/26833433/82944393-f7644d80-9f4f-11ea-8b87-1a5b04f555f1.jpg" width="1000"></a>
&nbsp

### This repo only focuses on NMS speed improvement based on https://github.com/ultralytics/yolov5.

### See `non_max_suppression` function of [utils/general.py](utils/general.py) for our Cluster-NMS implementation.

# Batch mode Cluster-NMS

Torchvision NMS has the fastest speed but fails to run in batch mode. And DIoU-NMS cannot be used.

Cluster-NMS is made for this.

## Pretrained Weights

| Model | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 37.0     | 37.0     | 56.2     | **2.4ms** | **416** || 7.5M   | 13.2B
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 44.3     | 44.3     | 63.2     | 3.4ms     | 294     || 21.8M  | 39.4B
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 47.7     | 47.7     | 66.5     | 4.4ms     | 227     || 47.8M  | 88.1B
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | **49.2** | **49.2** | **67.7** | 6.9ms     | 145     || 89.0M  | 166.4B
| | | | | | || |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0) + TTA|**50.8**| **50.8** | **68.9** | 25.5ms    | 39      || 89.0M  | 354.3B
| | | | | | || |
| [YOLOv3-SPP](https://github.com/ultralytics/yolov5/releases/tag/v3.0) | 45.6     | 45.5     | 65.2     | 4.5ms     | 222     || 63.0M  | 118.0B

** AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results in the table denote val2017 accuracy.  
** All AP numbers are for single-model single-scale without ensemble or test-time augmentation. **Reproduce** by `python test.py --data coco.yaml --img 640 --conf 0.001`  
** Speed<sub>GPU</sub> measures end-to-end time per image averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) instance with one V100 GPU, and includes image preprocessing, PyTorch FP16 image inference at --batch-size 32 --img-size 640, postprocessing and NMS. Average NMS time included in this chart is 1-2ms/img.  **Reproduce** by `python test.py --data coco.yaml --img 640 --conf 0.1`  
** All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
** Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) runs at 3 image sizes. **Reproduce** by `python test.py --data coco.yaml --img 832 --augment` 

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.6`. To install run:
```bash
$ pip install -r requirements.txt
```

#### Hardware
 - 2 RTX 2080 Ti
 
Evaluation command: `python test.py --weights yolov5s.pt --data coco.yaml --img 640 --augment --merge --batch-size 32`

YOLOv5s.pt

|   NMS  | TTA | max-box | weighted threshold |  time (ms) | AP | AP50 | AP75 | APs | APm | APl |
|:------------------------------------:|:----:|:----:|:----:|:----:|

|             Torchvision NMS            |  on  |   -  |  -   | 3.2 / 17.9 | 38.0 | 56.5 | 41.2 | 20.9 | 42.6 | 51.7 |
|         Merge + Torchvision NMS        |  on  |   -  | 0.65 | 3.2 / 18.6 | 38.0 | 56.5 | 41.4 | 20.9 | 42.7 | 51.8 |
|          Weighted Cluster-NMS          |  on  | 1000 | 0.8  | 3.2 / 6.6  | 38.0 | 55.7 | 41.6 | 20.5 | 42.8 | 51.9 |
|          Weighted Cluster-NMS          |  on  | 1500 | 0.65 | 3.2 / 10.2 | 38.1 | 56.1 | 41.9 | 20.9 | 42.7 | 51.8 |
|          Weighted Cluster-NMS          |  on  | 1500 | 0.8  | 3.2 / 10.2 | 38.3 | 56.2 | 41.8 | 21.1 | 43.0 | 52.0 |
|          Weighted Cluster-NMS          |  on  | 2000 | 0.8  | 3.2 / 14.5 | 38.4 | 56.4 | 41.9 | 21.3 | 43.1 | 52.1 |
|------------------------------------|
|             Torchvision NMS            |  off |   -  |  -   | 1.5 / 5.4  | 36.9 | 56.2 | 40.0 | 21.0 | 42.1 | 47.4 |
|         Merge + Torchvision NMS        |  off |   -  | 0.65 | 1.3 / 6.7  | 36.9 | 56.2 | 40.2 | 20.9 | 42.1 | 47.4 |
|          Weighted Cluster-NMS          |  off | 1000 | 0.65 | 1.3 / 6.5  | 36.9 | 56.0 | 40.2 | 20.9 | 42.0 | 47.3 |
|          Weighted Cluster-NMS          |  off | 1000 | 0.8  | 1.3 / 6.5  | 37.0 | 56.0 | 40.3 | 21.1 | 42.2 | 47.5 |
|          Weighted Cluster-NMS          |  off | 1000 | 0.8  | 3.2 / 10.2 | 38.1 | 56.1 | 41.9 | 20.9 | 42.7 | 51.8 |
|          Weighted Cluster-NMS          |  off | 2000 | 0.8  | 3.2 / 14.5 | 38.4 | 56.4 | 41.9 | 21.3 | 43.1 | 52.1 |

YOLOv5m.pt

|   NMS  | TTA | max-box | weighted threshold |  time (ms) | AP | AP50 | AP75 | APs | APm | APl |
|:------------------------------------:|:----:|:----:|:----:|:----:|

|             Torchvision NMS            |  on  |   -  |  -   | 6.4 / 10.4 | 45.1 | 63.2 | 49.0 | 27.0 | 50.2 | 60.5 |
|         Merge + Torchvision NMS        |  on  |   -  | 0.65 | 6.4 / 11.5 | 45.0 | 63.2 | 49.0 | 26.9 | 50.2 | 60.3 |
|          Weighted Cluster-NMS          |  on  | 1000 | 0.65 | 6.4 / 6.8  | 44.6 | 62.3 | 49.1 | 26.0 | 50.0 | 60.4 |
|          Weighted Cluster-NMS          |  on  | 1500 | 0.65 | 6.4 / 9.8  | 44.9 | 62.9 | 49.4 | 26.6 | 50.2 | 60.4 |
|          Weighted Cluster-NMS          |  on  | 1500 | 0.8  | 6.4 / 9.8  | 45.2 | 62.9 | 49.4 | 26.8 | 50.4 | 60.5 |
|------------------------------------|
|             Torchvision NMS            |  off |   -  |  -   | 2.7 / 4.5  | 44.3 | 63.2 | 48.2 | 27.4 | 50.0 | 56.4 |
|         Merge + Torchvision NMS        |  off |   -  | 0.65 | 2.7 / 6.1  | 44.2 | 63.1 | 48.4 | 27.4 | 50.1 | 56.2 |
|          Weighted Cluster-NMS          |  off | 1000 | 0.65 | 2.7 / 6.1  | 44.2 | 62.9 | 48.5 | 27.3 | 50.0 | 56.3 |
|          Weighted Cluster-NMS          |  off | 1000 | 0.8  | 2.7 / 6.1  | 44.3 | 62.9 | 48.5 | 27.4 | 50.1 | 56.4 |

YOLOv5x.pt `python test.py --weights yolov5s.pt --data coco.yaml --img 832 --augment --merge --batch-size 32`

|   NMS  | TTA | max-box | weighted threshold |  time (ms) | AP | AP50 | AP75 | APs | APm | APl |
|:------------------------------------:|:----:|:----:|:----:|:----:|

|         Merge + Torchvision NMS        |  on  |   -  | 0.65 | 31.7 / 10.7 | 50.2 | 68.5 | 55.2 | 34.2 | 54.9 | 64.0 |
|          Weighted Cluster-NMS          |  on  | 1500 | 0.8  | 31.8 / 9.9  | 50.3 | 68.0 | 55.4 | 33.9 | 55.1 | 64.6 |

** AP reports on `coco 2017val`. 
** `TTA` denotes Test-Time Augmentation.
** `max-box` denotes maximum number of boxes processed in Batch Mode Cluster-NMS.
** `weighted threshold` denotes the threshold used in weighted coordinates.
** time reports model inference / NMS.
** To avoid randomness, NMS runs three times here. see  [test.py](test.py)
```
# Run NMS
t = time_synchronized()
output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, max_box=max_box, merge=merge)
output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, max_box=max_box, merge=merge)
output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, max_box=max_box, merge=merge)
t1 += time_synchronized() - t
```
## Related issues

* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [INCREASING NMS SPEED](https://github.com/ultralytics/yolov3/issues/679)
* [TESTING/INFERENCE AUGMENTATION](https://github.com/ultralytics/yolov3/issues/931)

## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab Notebook** with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- **Kaggle Notebook** with free GPU: [https://www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)




## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


#### This is the code for our paper:
 - [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
 - [Enhancing Geometric Factors into Model Learning and Inference for Object Detection and Instance Segmentation](http://arxiv.org/abs/2005.03572)

```
@Inproceedings{zheng2020distance,
  author    = {Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
   year      = {2020},
}

@Article{zheng2020ciou,
  author= {Zhaohui Zheng, Ping Wang, Dongwei Ren, Wei Liu, Rongguang Ye, Qinghua Hu, Wangmeng Zuo},
  title={Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation},
  journal={arXiv:2005.03572},
  year={2020}
}
```
