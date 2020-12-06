# Dataset
The Mall and Town Centre datasets: https://drive.google.com/drive/folders/19EXevD-ov5BVrE60InTAJEKtiXBtGtH8?usp=sharing

# Usage
To run segmentation on videos and output graphics of the bounding boxes for people and a top-down graph of their respective locations and distances, run:
 ```
 python detect.py 
 ``` 
 Modify the arguments in the main function to select for dataset and segmentation model. The results are stored in the corresponding results directory.

 ## Requirements
 The specific versions of PyTorch, TorchVision, and CUDAToolKit used are listed below. Additional requirements are listed in the ```requirements.txt``` file.
 ```
 conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```

# About
We draw inspiration from an implementation of social distancing by [Yang and Yurtsever](https://github.com/dongfang-steven-yang/social-distancing-monitoring). The implementation and weights from [torchvision](https://pytorch.org/docs/1.5.0/torchvision/models.html#faster-r-cnn) are used for Faster-RCNN, the implementation of YOLOv3 is slightly modified from [Erik Lindernoren's version](https://github.com/eriklindernoren/PyTorch-YOLOv3) and we use his pretrained weights, and finally the OpenCV implementation of the HoG descriptor + SVM people detector is used.


