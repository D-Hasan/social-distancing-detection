# Social Distancing Detection
**[Social Distancing Detection](https://github.com/D-Hasan/social-distancing-detection/raw/main/Social%20Distancing%20Detection.pdf)**<br>
Authors: Gabriel Deza, Danial Hasan, Mengyu Yang (Team GDM for CSC420)

A social distancing detection pipeline that automatically detects social distancing violations from stationary camera footage. Our method takes into account "social bubbles", where groups consisting of families, couples, etc. are not classified as violations. 

## Datasets
The Mall and Town Centre datasets: 
https://drive.google.com/drive/folders/19EXevD-ov5BVrE60InTAJEKtiXBtGtH8?usp=sharing

UCSD Pedestrian Dataset: 
http://www.svcl.ucsd.edu/projects/peoplecnt/

LSTN Fudan-ShanghaiTech Dataset:
https://github.com/sweetyy83/Lstn_fdst_dataset

The UCSD and LSTN datasets are included in the repos but the links are provided for reference. The Oxford Town Centre dataset must be downloaded since it's too large.

## Usage
The full pipeline consists of the following steps:
1. Object Detection
2. Tracking (Centroid or Optical Flow)
3. Grouping

### Preliminaries

 #### Requirements
 The specific versions of PyTorch, TorchVision, and CUDAToolKit used are listed below. Additional requirements are listed in the ```requirements.txt``` file.
```
 conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```

#### YOLO Weights
If you would like to use YOLOv3 for object detection, download the weights for yolo as follows:
```
cd weights/
bash download_weights.sh
```


### Object Detection
To run object detection (and tracking for the case of optical flow) on the various datasets using the various models run either ```detect.py```, for use in centroid tracking, or ```detect_optical_flow.py``` for directly tracking with optical flow:
 ```
 python detect.py [--vis] [--num_frames] model dataset
 python detect_optical_flow.py [--vis] [--num_frames] model dataset
 ``` 

**Note:** for the fastest start time, use ```python detect.py faster_rcnn lstn --num_frames 100```, as PyTorch will download the weights automatically if needed.

Where ```model``` is one of ```{hog, yolo, faster_rcnn, mask_rcnn}``` and ```dataset``` is one of ```{oxford_town, ucsd, lstn}```. The results will automatically be stored in the corresponding folder in the ```results``` directory. The ```--vis``` tag will designate if you want output segmentations to be saved into the results directory along with the segmentation/tracking results. The ```--num_frames``` limits the total number of frames that are analyzed.

The ```model_configs``` directory contains config files for all the object detection models. Here the threshold parameters for confidence per dataset (and non maximal suppression where applicable), along with a forced CPU inference, can be found. 

### Tracking
**Note:** *This section applies only for Centroid Tracking* Optical flow tracking is directly embdedded into object detection, so once ```detect_optical_flow.py``` has been run, the tracked pedestrians are saved into the corresponding folder already. 

To run centroid tracking after object detection takes place:
```
python centroid_tracking.py [--vis] [--num_frames] model dataset
```
Specifying the model and dataset that have already been used at the object detection stage.


### Grouping
This will output a comparison visual of social distancing detection as compared to the [Yang & Yurtsever](https://github.com/dongfang-steven-yang/social-distancing-monitoring) implementation to show the efficacy of grouping to eliminate false positives.

```
python grouping.py [--vis] [--num_frames] model dataset tracking
```
The ```tracking``` argument denotes which tracking method was used from ```{optical_flow, centroid_tracking}```. The ```vis``` argument will plot a gif of groups by themselves on a 2D plot. 

## Homography
To approximate the camera to world matrices, look at the Jupyter Notebooks in the ```calibration``` directory. The notebooks showcase the method of finding keypoints in the image and the world using a bicycle for reference length.

## Credit
We draw inspiration from an implementation of social distancing by [Yang and Yurtsever](https://github.com/dongfang-steven-yang/social-distancing-monitoring). The implementation and weights from [torchvision](https://pytorch.org/docs/1.5.0/torchvision/models.html#faster-r-cnn) are used for Faster-RCNN, the implementation of YOLOv3 is slightly modified from [Erik Lindernoren's version](https://github.com/eriklindernoren/PyTorch-YOLOv3) and we use his pretrained weights, and finally the OpenCV implementation of the HoG descriptor + SVM people detector is used.


