import torch
import torchvision
import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from plotting import plot_frame, plot_frame_one_row, get_roi_pts
from utils import ROIs, find_violation, make_gif

from utils import COCO_INSTANCE_CATEGORY_NAMES as LABELS
from utils_yolo import non_max_suppression, rescale_boxes, resize, pad_to_square

import cv2

from models import Darknet


def main(dataset, data_time, detector,stop_plotting,stop_running):
    #Create folder for results
    path_result = os.path.join('results', data_time + '_' + detector, dataset)
    #only creates the folders if they don't exist
    os.makedirs(path_result, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #GPU
    # device = torch.device("cpu")
    # initialize detector
    if 'cnn' in detector:
        if detector == 'faster_rcnn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
        # elif detector == 'yolo_cnn':
            model = Darknet(yolo_cfg, img_size=yolo_img_size).to(device)
            model.load_darknet_weights(yolo_weights)

        
        model.to(device=device)
        model.eval()

    else:
        model = cv2.HOGDescriptor()
        model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    # load background and rotation matrix
    img_bkgd_bev = cv2.imread('calibration/' + dataset + '_background_calibrated.png')
    transform_cam2world = np.loadtxt('calibration/' + dataset + '_matrix_cam2world.txt')

    # open video of dataset
    if dataset == 'oxford_town':
        cap = cv2.VideoCapture(os.path.join('datasets', 'TownCentreXVID.avi'))
        frame_skip = 10  # oxford town dataset has fps of 25
        thr_score = 0.9
    elif dataset == 'mall':
        cap = cv2.VideoCapture(os.path.join('datasets', 'mall.mp4'))
        frame_skip = 1
        thr_score = 0.9
    elif dataset == 'grand_central':
        cap = cv2.VideoCapture(os.path.join('datasets', 'grandcentral.avi'))
        frame_skip = 25  # grand central dataset has fps of 25
        thr_score = 0.5
    else:
        raise Exception('Invalid Dataset')

    statistic_data = []
    i_frame = 0
    while cap.isOpened(): #loops over frames in the video
        #reads frame and 'ret' is True if frame is read correctly
        ret, img = cap.read()
        if ret is False:
            break
        if i_frame > stop_running:
            break

        if i_frame < stop_plotting:
            vis = True
        else:
            vis = False

        t0 = time.time()
        # convert image from OpenCV format to PyTorch tensor format
        img_t = np.moveaxis(img, -1, 0) / 255
        img_t = torch.tensor(img_t, device=device).float()

        # pedestrian detection
        if detector == 'faster_rcnn':
            predictions = model([img_t])
            boxes = predictions[0]['boxes'].cpu().data.numpy()
            classIDs = predictions[0]['labels'].cpu().data.numpy()
            scores = predictions[0]['scores'].cpu().data.numpy()
        elif detector == 'yolo_cnn':
            img_t, _ = pad_to_square(img_t, 0)
            img_t = resize(img_t, yolo_img_size)

            detections = model(img_t.unsqueeze(0))
            # import pdb; pdb.set_trace()
            detections = non_max_suppression(detections, conf_thres=yolo_conf_thres ,nms_thres=yolo_nms_thres)
            detections = rescale_boxes(detections[0], yolo_img_size, img.shape[:2])
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

        else:
            boxes, scores = model.detectMultiScale(img, winStride=(4,4))
            # OpenCV format is (x, y, w, h) so need to offset to make box coords
            boxes[:,2:] = boxes[:,:2] + boxes[:,2:]   


        # import pdb; pdb.set_trace()
        # get positions and plot on raw image
        pts_world = []

        if detector == 'faster_rcnn':
            humans_idx = np.intersect1d(np.where(classIDs == 1), np.where(scores > thr_score))
        elif detector == 'yolo_cnn':
            humans_idx = np.where(classIDs == 0)[0]
        else:
            humans_idx = np.where(scores > 0.5)[0]

        for i in humans_idx:
        # extract the bounding box coordinates
            (x1, y1) = (boxes[i][0], boxes[i][1]) #represents 1 corner
            (x2, y2) = (boxes[i][2], boxes[i][3]) #represents the next corner

            if vis:
                # draw a bounding box rectangle and label on the image
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 0, 255], 2)
            # find the bottom center position and convert it to world coordinate
            p_c = np.array([[(x1 + x2)/2], [y2], [1]])
            p_w = np.matmul(transform_cam2world,p_c)
            p_w = p_w / p_w[2]
            pts_world.append([p_w[0][0], p_w[1][0]])
        

        t1 = time.time()

        pts_world = np.array(pts_world)
        if dataset == 'oxford_town':
            pts_world[:, [0, 1]] = pts_world[:, [1, 0]]

        statistic_data.append((i_frame, t1 - t0, pts_world))

        # visualize
        if vis:
            violation_pairs = find_violation(pts_world)
            pts_roi_world, pts_roi_cam = get_roi_pts(dataset=dataset, roi_raw=ROIs[dataset], matrix_c2w=transform_cam2world)
            #print('hold on1',len(pts_world),pts_roi_world.shape)
            #print('hold on2', len(pts_world), pts_roi_cam.shape)
            # print('-----begin------')
            # print('non roi', pts_world[0])
            # print('world',pts_roi_world[0,:])
            # print('cam', pts_roi_cam[0, :])
            # print('\n')
            fig = plot_frame_one_row(dataset=dataset,img_raw=img,pts_roi_cam=pts_roi_cam,
                pts_roi_world=pts_roi_world,pts_w=pts_world,pairs=violation_pairs)
            fig.savefig(os.path.join(path_result, 'frame%04d.png' % i_frame))
            plt.close(fig)
        # update loop info
        if i_frame % 5 == 0:
            print('Frame %d - Inference Time: %.2f' % (i_frame, t1 - t0))
        i_frame += 1

    # save statistics
    # f.close()
    cap.release()
    cv2.destroyAllWindows()
    pickle.dump(statistic_data, open(os.path.join(path_result, 'statistic_data.p'), 'wb'))
    make_gif(path_result)



if __name__ == '__main__':
    yolo_cfg = 'config/yolov3.cfg'
    yolo_img_size = 416
    yolo_weights = 'weights/yolov3.weights'
    yolo_conf_thres = 0.4
    yolo_nms_thres = 0.4

    data_time = 'test'
    dataset = 'grand_central'
    detector = 'hog'
    # detector = 'faster_rcnn'
    detector = 'yolo_cnn'
    

    print('=========== %s ===========' % dataset)
    np.set_printoptions(precision=4) #Ie: print floats to 4 decimals
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    stop_plotting = 100
    stop_running = 100
    main(dataset=dataset, data_time=data_time, detector=detector, stop_plotting = stop_plotting, stop_running = stop_running)

