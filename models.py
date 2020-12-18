import cv2
import numpy as np 
import torch 
import torchvision


from yolo import Darknet
from utils_yolo import non_max_suppression, rescale_boxes, resize, pad_to_square


def make_masks(shape, boxes, mask_weight):
    masks = []
    for i in range(len(boxes)):
    # extract the bounding box coordinates
        (x1, y1) = (boxes[i][0], boxes[i][1]) #represents 1 corner
        (x2, y2) = (boxes[i][2], boxes[i][3])
        

        mask = np.zeros(shape, dtype='uint8')
        mask_h = y2-y1
        mask_w = x2-x1
        mask[round(y1 + mask_weight*mask_h):round(y2-mask_weight*mask_h),
                round(x1+1.3*mask_weight*mask_w):round(x2-1.3*mask_weight*mask_w)] = 255

        masks.append(mask)

    return masks 

class HOG:
    def __init__(self, model_config, data_config):
        super().__init__()
        
        self.model_config = model_config 
        self.data_config = data_config

        self.model = cv2.HOGDescriptor()
        self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.thres = float(data_config['conf_thres'])
        self.win_stride = int(data_config['win_stride'])
        self.mask_weight = float(model_config['mask_weight'])

        if 'scale' in model_config:
            self.scale = float(model_config['scale'])
        else:
            self.scale = 1.05

    def predict(self, img, mask=False):
        boxes, scores = self.model.detectMultiScale(img, winStride=(self.win_stride, self.win_stride), scale=self.scale)
        
        # OpenCV format is (x, y, w, h) so need to offset to make box coords

        try:
            boxes[:,2:] = boxes[:,:2] + boxes[:,2:]   
        except:
            return boxes, scores, [] 

        humans_idx = np.where(scores > self.thres)[0]

        if mask:
            masks = make_masks(img.shape[:2], boxes, self.mask_weight)
            return boxes, masks, scores, humans_idx
        else:
            return boxes, scores, humans_idx 

        return boxes, scores, humans_idx


class YOLO:
    def __init__(self, model_config, data_config):
        super().__init__()

        self.model_config = model_config 
        self.data_config = data_config
        
        if model_config.getboolean('force_cpu'):
            self.device = torch.device("cpu") 
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.yolo_cfg = model_config['yolo_cfg']
        self.img_size = int(model_config['yolo_img_size'])
        self.weights = model_config['yolo_weights']

        self.conf_thres = float(data_config['conf_thres'])
        self.nms_thres = float(data_config['nms_thres'])
        self.mask_weight = float(model_config['mask_weight'])

        self.model = Darknet(self.yolo_cfg, img_size=self.img_size).to(self.device)
        self.model.load_darknet_weights(self.weights)

        self.model.to(device=self.device)
        self.model.eval()

    def predict(self, img, mask=False):
        img_t = np.moveaxis(img, -1, 0) / 255
        img_t = torch.tensor(img_t, device=self.device).float()

        img_t, _ = pad_to_square(img_t, 0)
        img_t = resize(img_t, self.img_size)

        detections = self.model(img_t.unsqueeze(0))

        detections = non_max_suppression(detections, conf_thres=self.conf_thres ,nms_thres=self.nms_thres)
        detections = rescale_boxes(detections[0], self.img_size, img.shape[:2])
        boxes = []
        classIDs = []
        scores = []

        for detection in detections:
            detection = detection.cpu().data.numpy()
            x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
            boxes.append(np.array([x1, y1, x2, y2]))
            classIDs.append(cls_pred)
            scores.append(conf)

        boxes = np.array(boxes)
        classIDs = np.array(classIDs)
        scores = np.array(scores)

        humans_idx = np.where(classIDs == 0)[0]

        if mask:
            masks = make_masks(img.shape[:2], boxes, self.mask_weight)
            return boxes, masks, scores, humans_idx
        else:
            return boxes, scores, humans_idx 


class Faster_RCNN:
    def __init__(self, model_config, data_config):
        super().__init__()

        self.model_config = model_config
        self.data_config = data_config 

        if model_config.getboolean('force_cpu'):
            self.device = torch.device("cpu") 
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        self.model.to(device=self.device)
        self.model.eval()

        self.conf_thres = float(data_config['conf_thres'])
        self.mask_weight = float(model_config['mask_weight'])

    def predict(self, img, mask=False):
        img_t = np.moveaxis(img, -1, 0) / 255
        img_t = torch.tensor(img_t, device=self.device).float()

        predictions = self.model([img_t])
        boxes = predictions[0]['boxes'].cpu().data.numpy()
        classIDs = predictions[0]['labels'].cpu().data.numpy()
        scores = predictions[0]['scores'].cpu().data.numpy()

        humans_idx = np.intersect1d(np.where(classIDs == 1), np.where(scores > self.conf_thres))

        if mask:
            masks = make_masks(img.shape[:2], boxes, self.mask_weight)
            return boxes, masks, scores, humans_idx
        else:
            return boxes, scores, humans_idx 


    

class Mask_RCNN:
    def __init__(self, model_config, data_config):
        super().__init__()

        self.model_config = model_config
        self.data_config = data_config 

        if model_config.getboolean('force_cpu'):
            self.device = torch.device("cpu") 
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        self.model.to(device=self.device)
        self.model.eval()

        self.conf_thres = float(data_config['conf_thres'])

    def predict(self, img, mask=False):
        img_t = np.moveaxis(img, -1, 0) / 255
        img_t = torch.tensor(img_t, device=self.device).float()

        predictions = self.model([img_t])
        boxes = predictions[0]['boxes'].cpu().data.numpy()
        classIDs = predictions[0]['labels'].cpu().data.numpy()
        scores = predictions[0]['scores'].cpu().data.numpy()

        humans_idx = np.intersect1d(np.where(classIDs == 1), np.where(scores > self.conf_thres))

        new_masks = []
        if mask:
            masks = predictions[0]['masks'].cpu().data.numpy()
            for i in range(len(masks)):
                mask = masks[i, ...].squeeze()
                mask[mask >= 0.5] = 255
                mask[mask < 0.5] = 0
                mask = mask.astype('uint8')
                new_masks.append(mask)
            return boxes, new_masks, scores, humans_idx
        else:
            return boxes, scores, humans_idx
