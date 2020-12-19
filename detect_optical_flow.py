import os
import time
import pickle
import argparse
import configparser


import torch
import numpy as np
import matplotlib.pyplot as plt

import cv2

from utils import init_model, init_loader, parse_dataset_config


import warnings
warnings.filterwarnings("ignore")
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


# def main(cap, detector, transform_cam2world, num_frames, pickle_path, vis, vis_dir):

start_time = time.time()

def main(dataset, data_time, detector, config, dataset_config, vis, stop_running=-1):

    #Create folder for results
    vis_dir = os.path.join('results', data_time + '_' + detector, dataset)
    #only creates the folders if they don't exist
    os.makedirs(vis_dir, exist_ok=True)

    pickle_path = os.path.join(vis_dir, 'opt_flow_data.p')

     # Init model
    model = init_model(detector, config['MODEL_CONFIG'], config[DATASET_KEY])

    # Init dataloader
    loader = init_loader(dataset)

    # load background and rotation matrix
    transform_cam2world = np.loadtxt('calibration/' + dataset + '_matrix_cam2world.txt')

    stop_running = int(config['DEFAULT']['stop_running'])

    feature_params = dict(maxCorners=6,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    curr_frame = 0
    objID = 1
    detect_interval = 6
    tracked_objs = []
    dist_thrs = 100
    time_thrs = 35  # In # of frames
    potential_phase = []

    pts_world = [] 

    for img in loader:
        if curr_frame > stop_running:
            break 

        
        # Read current and next frame
        ret, next_img = loader.peek()
        if not ret:
            break 
        
        pts_world.append({})

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, cols = gray_img.shape

        gray_next = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        if vis:
            plt.imshow(gray_next, cmap='gray')
        

        # counting process time
        t0 = time.time()

        # For initial frame, get all people
        if curr_frame == 0:
            boxes, masks, scores, humans_idx = model.predict(img, mask=True)
            
            for i in humans_idx:
                mask = masks[i]
                # Get feature to track with optical flow
                feature_points = cv2.goodFeaturesToTrack(gray_img, mask=mask, **feature_params)

                # import pdb; pdb.set_trace()
                # Calculate centroid
                if feature_points is None :
                    continue 
                    # import pdb; pdb.set_trace()
                centroid = get_centroid(feature_points)

                # Initialize tracker object for person
                t = TrackedObject(objID, feature_points, centroid)

                objID += 1
                tracked_objs.append(t)


        # Update people detection by rerunning neural net
        elif curr_frame % detect_interval == 0:
            # convert image from OpenCV format to PyTorch tensor format
            boxes, masks, scores, humans_idx = model.predict(img, mask=True)

            # Get list of currently tracked people and create centroid map
            trackedIDs = []
            centroid_map = np.zeros_like(gray_img)
            for person in tracked_objs:
                trackedIDs.append(person.objectID)
                x, y = person.path[-1][0]
                try:
                    centroid_map[round(y), round(x)] = person.objectID
                except:
                    continue 

            # Iterate through each bounding box
            for i in humans_idx:
                # extract the bounding box coordinates
                (x1, y1) = (boxes[i][0], boxes[i][1])
                (x2, y2) = (boxes[i][2], boxes[i][3])

                # Create mask of bounding box region
                if detector == 'mask_rcnn':
                    mask = np.zeros_like(masks[i])
                    mask[masks[i] == 255] = 1
                    mask[masks[i] == 0] = 0
                    mask = mask.astype('uint8')
                else:
                    mask = np.zeros_like(gray_img)
                    mask[round(y1):round(y2), round(x1):round(x2)] = 1

                # Multiply centroid map from previous frame with mask to determine new/old people
                masked_centroid = mask * centroid_map
                unique_val = np.unique(masked_centroid)
                if unique_val.shape[0] > 1 and unique_val[1] in trackedIDs:
                    existID = unique_val[1]
                    trackedIDs.remove(existID)

                    mask = masks[i]

                    # Get feature to track with optical flow
                    feature_points = cv2.goodFeaturesToTrack(gray_img, mask=mask, **feature_params)

                    # Calculate and update tracked features
                    for person in tracked_objs:
                        if person.objectID == existID:
                            person.trackedFeatures = feature_points

                # This is a new person
                else:
                    flag = True

                    mask = masks[i]

                    # Get feature to track with optical flow
                    feature_points = cv2.goodFeaturesToTrack(gray_img, mask=mask, **feature_params)

                    if feature_points is None :
                        continue 
                    
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
        if curr_frame % 5 == 0:
            avg_time_per_frame = (t1 - start_time)/(curr_frame+1)
            print('Frame %d - Avg. Inference Time: %.2f' % (curr_frame, avg_time_per_frame))
            print('=======================')
        curr_frame += 1

    # Save file
    pickle.dump(pts_world, open(pickle_path, "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Object detection model to use, one of {yolo, hog, faster_rcnn")
    parser.add_argument("dataset", help="Dataset to use, one of {oxford_town, lstn, ucsd}")
    parser.add_argument("--vis", help="Visualize bounding boxes and bird's eye view.", dest='vis', action='store_true')
    parser.add_argument('--num_frames', help="number of frames to track", default=-1)

    parser.set_defaults(vis=False)
    args = parser.parse_args()

    vis = args.vis 
    num_frames = int(args.num_frames)


    detector = args.model 
    config_path = os.path.join('model_configs', detector + '.cfg')

    config = configparser.ConfigParser()
    config.read(config_path)

    dataset = args.dataset
    DATASET_KEY = dataset.upper() + "_CONFIG"

    dataset_config = configparser.ConfigParser()
    dataset_config.read('datasets/datasets.cfg')
    dataset_config = dataset_config[dataset.upper()]
    dataset_config = parse_dataset_config(dataset_config)


    data_time = 'optical_flow'
    

    print('=========== %s ===========' % dataset)
    np.set_printoptions(precision=4) #Ie: print floats to 4 decimals


    main(dataset, data_time, detector, config, dataset_config, vis, stop_running=num_frames)


