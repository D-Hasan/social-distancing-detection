import torch
import torchvision
import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

import cv2
from test import TrackedObject

np.set_printoptions(precision=4)


class TrackedObject:
    def __init__(self, objectID, trackedFeatures, centroid):
        self.objectID = objectID
        self.trackedFeatures = trackedFeatures
        self.path = [centroid]
        self.lastFrame = None


def get_centroid(feature_points):
    n = feature_points.shape[0]
    return np.array([[feature_points[..., 0].sum() / n, feature_points[..., 1].sum() / n]])


def main(dataset_path, detector, transform_cam2world, pickle_path, vis, vis_dir):
    # initialize detector
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if detector == 'mask_rcnn':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif detector == 'faster_rcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        print("No such model")
        return

    model.to(device=device)
    model.eval()


    feature_params = dict(maxCorners=6,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    curr_frame = 0
    objID = 1
    detect_interval = 6
    msk_wgt = 0.1
    tracked_objs = []
    thr_score = 0.9
    dist_thrs = 100
    time_thrs = 35  # In # of frames
    potential_phase = []

    images = sorted([image for image in os.listdir(dataset_path) if image[-3:] == 'png' or image[-3:] == 'jpg'])
    num_frames = len(images)
    pts_world = [{}] * num_frames

    while curr_frame < num_frames:
        # Read current and next frame
        img = cv2.imread(os.path.join(dataset_path, images[curr_frame]))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, cols = gray_img.shape

        next_img = cv2.imread(os.path.join(dataset_path, images[curr_frame+1]))
        gray_next = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        if vis:
            plt.imshow(gray_next, cmap='gray')

        # counting process time
        t0 = time.time()

        # For initial frame, get all people
        if curr_frame == 0:
            # convert image from OpenCV format to PyTorch tensor format
            img_t = np.moveaxis(img, -1, 0) / 255
            img_t = torch.tensor(img_t, device=device).float()

            # pedestrian detection
            predictions = model([img_t])
            boxes = predictions[0]['boxes'].cpu().data.numpy()
            if detector == 'mask_rcnn':
                masks = predictions[0]['masks'].cpu().data.numpy()
            classIDs = predictions[0]['labels'].cpu().data.numpy()
            scores = predictions[0]['scores'].cpu().data.numpy()

            for i in range(len(boxes)):
                if classIDs[i] == 1 and scores[i] > thr_score:
                    # extract the bounding box coordinates
                    (x1, y1) = (boxes[i][0], boxes[i][1])
                    (x2, y2) = (boxes[i][2], boxes[i][3])

                    # Create mask of bounding box region
                    if detector == 'mask_rcnn':
                        mask = masks[i, ...].squeeze()
                        mask[mask >= 0.5] = 255
                        mask[mask < 0.5] = 0
                        mask = mask.astype('uint8')

                    if detector == 'faster_rcnn':
                        mask = np.zeros_like(gray_img)
                        mask_h = y2-y1
                        mask_w = x2-x1
                        mask[round(y1 + msk_wgt*mask_h):round(y2-msk_wgt*mask_h),
                             round(x1+1.3*msk_wgt*mask_w):round(x2-1.3*msk_wgt*mask_w)] = 255

                    # Get feature to track with optical flow
                    feature_points = cv2.goodFeaturesToTrack(gray_img, mask=mask, **feature_params)

                    # Calculate centroid
                    centroid = get_centroid(feature_points)

                    # Initialize tracker object for person
                    t = TrackedObject(objID, feature_points, centroid)

                    objID += 1
                    tracked_objs.append(t)


        # Update people detection by rerunning neural net
        elif curr_frame % detect_interval == 0:
            # convert image from OpenCV format to PyTorch tensor format
            img_t = np.moveaxis(img, -1, 0) / 255
            img_t = torch.tensor(img_t, device=device).float()

            # pedestrian detection
            predictions = model([img_t])
            boxes = predictions[0]['boxes'].cpu().data.numpy()
            if detector == 'mask_rcnn':
                masks = predictions[0]['masks'].cpu().data.numpy()
            classIDs = predictions[0]['labels'].cpu().data.numpy()
            scores = predictions[0]['scores'].cpu().data.numpy()

            # Get list of currently tracked people and create centroid map
            trackedIDs = []
            centroid_map = np.zeros_like(gray_img)
            for person in tracked_objs:
                trackedIDs.append(person.objectID)
                x, y = person.path[-1][0]
                centroid_map[round(y), round(x)] = person.objectID

            # Iterate through each bounding box
            for i in range(len(boxes)):
                if classIDs[i] == 1 and scores[i] > thr_score:
                    # extract the bounding box coordinates
                    (x1, y1) = (boxes[i][0], boxes[i][1])
                    (x2, y2) = (boxes[i][2], boxes[i][3])

                    # Create mask of bounding box region
                    if detector == 'faster_rcnn':
                        mask = np.zeros_like(gray_img)
                        mask[round(y1):round(y2), round(x1):round(x2)] = 1

                    if detector == 'mask_rcnn':
                        mask = masks[i, ...].squeeze()
                        mask[mask >= 0.5] = 1
                        mask[mask < 0.5] = 0
                        mask = mask.astype('uint8')

                    # Multiply centroid map from previous frame with mask to determine new/old people
                    masked_centroid = mask * centroid_map
                    unique_val = np.unique(masked_centroid)
                    if unique_val.shape[0] > 1 and unique_val[1] in trackedIDs:
                        existID = unique_val[1]
                        trackedIDs.remove(existID)

                        # Create mask of bounding box region
                        if detector == 'faster_rcnn':
                            mask = np.zeros_like(gray_img)
                            mask_h = y2 - y1
                            mask_w = x2 - x1
                            mask[round(y1 + msk_wgt * mask_h):round(y2 - msk_wgt * mask_h),
                                 round(x1 + 1.3 * msk_wgt * mask_w):round(x2 - 1.3 * msk_wgt * mask_w)] = 255

                        if detector == 'mask_rcnn':
                            mask = masks[i, ...].squeeze()
                            mask[mask >= 0.5] = 255
                            mask[mask < 0.5] = 0
                            mask = mask.astype('uint8')

                        # Get feature to track with optical flow
                        feature_points = cv2.goodFeaturesToTrack(gray_img, mask=mask, **feature_params)

                        # Calculate and update tracked features
                        for person in tracked_objs:
                            if person.objectID == existID:
                                person.trackedFeatures = feature_points

                    # This is a new person
                    else:
                        flag = True
                        # Create mask of bounding box region
                        if detector == 'faster_rcnn':
                            mask = np.zeros_like(gray_img)
                            mask_h = y2 - y1
                            mask_w = x2 - x1
                            mask[round(y1 + msk_wgt * mask_h):round(y2 - msk_wgt * mask_h),
                            round(x1 + 1.3 * msk_wgt * mask_w):round(x2 - 1.3 * msk_wgt * mask_w)] = 255

                        if detector == 'mask_rcnn':
                            mask = masks[i, ...].squeeze()
                            mask[mask >= 0.5] = 255
                            mask[mask < 0.5] = 0
                            mask = mask.astype('uint8')

                        # Get feature to track with optical flow
                        feature_points = cv2.goodFeaturesToTrack(gray_img, mask=mask, **feature_params)

                        # Calculate centroid
                        centroid = get_centroid(feature_points)

                        if len(potential_phase) > 0:
                            # Check if person just phased out for a few frames
                            min_dist = 1000
                            min_dist_person = potential_phase[0]
                            min_time = 1000
                            min_time_person = potential_phase[0]
                            for person in potential_phase:
                                dist = np.linalg.norm(person.path[-1] - centroid)
                                times = curr_frame - person.lastFrame
                                if dist < min_dist and times < time_thrs:
                                    min_dist = dist
                                    min_dist_person = person
                                if times < min_time and dist < dist_thrs:
                                    min_time = times
                                    min_time_person = person

                            if np.linalg.norm(min_time_person.path[-1] - centroid) < dist_thrs and \
                                    curr_frame - min_time_person.lastFrame < time_thrs:
                                for person in potential_phase:
                                    if person.objectID == min_time_person.objectID:
                                        print('Put back id(time): ', min_time_person.objectID)
                                        person.trackedFeatures = feature_points
                                        tracked_objs.append(person)
                                        potential_phase.remove(person)
                                        flag = False

                            elif np.linalg.norm(min_dist_person.path[-1] - centroid) < dist_thrs and \
                                    curr_frame - min_dist_person.lastFrame < time_thrs:
                                for person in potential_phase:
                                    if person.objectID == min_dist_person.objectID:
                                        print('Put back id(dist): ', min_dist_person.objectID)
                                        person.trackedFeatures = feature_points
                                        tracked_objs.append(person)
                                        potential_phase.remove(person)
                                        flag = False

                        if flag:
                            # Initialize tracker object for person
                            t = TrackedObject(objID, feature_points, centroid)
                            objID += 1
                            tracked_objs.append(t)

            # Remove any remaining floating points
            for personID in trackedIDs:
                for person in tracked_objs:
                    if person.objectID == personID:
                        if len(person.path) > 2 * detect_interval:
                            print('Potential phase: ', personID)
                            person.lastFrame = curr_frame
                            potential_phase.append(person)
                        tracked_objs.remove(person)

        # Object tracking only. Use optical flow to estimate position of people in next frame
        for person in tracked_objs:
            x, y = person.path[-1][0]
            if not (10 <= round(x) < cols - 10) or not (10 <= round(y) < rows - 10):
                print(f"Remove ID {person.objectID}")
                tracked_objs.remove(person)
                continue

            feature_points = person.trackedFeatures
            if feature_points is None:
                if len(person.path) > 2 * detect_interval:
                    print('Potential phase: ', person.objectID)
                    person.lastFrame = curr_frame
                    potential_phase.append(person)
                tracked_objs.remove(person)
                continue

            # First store points from current frame
            curr_centroid = person.path[-1]
            x, y = curr_centroid[0, 0], curr_centroid[0, 1]
            p_c = np.array([[x], [y], [1]])
            p_w = np.matmul(transform_cam2world, p_c)
            p_w = p_w / p_w[2]
            pts_world[curr_frame][person.objectID] = [p_w[0][0], p_w[1][0]]

            # Compute points for next frame
            feature_points = feature_points.astype('float32')
            next_points, st, err = cv2.calcOpticalFlowPyrLK(gray_img, gray_next, feature_points, None)
            centroid = get_centroid(next_points)
            if vis:
                plt.plot(centroid[0, 0], centroid[0, 1], 'rs', markersize=3)
                plt.annotate(person.objectID,  # this is the text
                             (centroid[0, 0], centroid[0, 1]),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 4),  # distance from text to points (x,y)
                             color='r',
                             ha='center')  # horizontal alignment can be left, right or center
            person.path.append(centroid)
            person.trackedFeatures = next_points

        if vis:
            plt.savefig(os.path.join(vis_dir, 'img' + str(curr_frame).zfill(3)))
            plt.close()

        t1 = time.time()

        # update loop info
        print('Frame %d - Inference Time: %.2f' % (curr_frame, t1 - t0))
        print('=======================')
        curr_frame += 1

    # Save file
    pickle.dump(pts_world, open(pickle_path, "wb"))


if __name__ == '__main__':

    # Specify from ['faster_rcnn', 'mask_rcnn']
    detector = 'mask_rcnn'

    # Specify path of dataset
    DATASET_PATH = 'lstn_24'

    # Specify transform matrix
    MATRIX_PATH = 'calibration/lstn_matrix_cam2world.txt'
    transform_cam2world = np.loadtxt(MATRIX_PATH)

    # Specify path of output pickle file
    pickle_path = './lstn_track_frames.p'

    # Specify output image directory. This is only used if vis=True.
    # If ture, it will save each frame with visualized trackers to create gifs.
    VIS_DIR = 'output_img'
    vis = False

    # Run
    main(dataset_path=DATASET_PATH, detector=detector, transform_cam2world=transform_cam2world,
         pickle_path=pickle_path, vis=vis, vis_dir=VIS_DIR)

