import pickle
import os 
import time
import math
import argparse
import configparser

import numpy as np
import matplotlib.pyplot as plt

from plotting import group_plotting, plot_grouping_frame_one_row, get_roi_pts
from utils import convert_optical_flow_tracking, init_loader, find_group_violations, find_violation, parse_dataset_config



def grouping(data, threshold=1.7):
    '''

    :param data: The input is the list of dictionaries from tracking.
    :return: if there is N distinct people across all frames, its a N-long list, where each element represents the group that that index is with.
    '''
    humans  = []
    for i in range(0,len(data)):
        for key,value in data[i].items():
            if key not in humans:
                humans.append(key)
    #sanity check
    humans = set(humans)
    num_humans = len(humans)
    #Create a (# Frames) x (# Humans) x 2 tensor and populate it with positions.
    #this represents a easy structure to store the positions across frames of each.
    positions = np.full((len(data),num_humans,2),-20.0) #(500,40,2) 2 for x,y
    for i in range(0,len(data)):
        for key,value in data[i].items():
            positions[i,key,:] = value
    #Create a (# Frames) x (# Humans) x (# Humans) 3D tensor.
    #this represents the structure where we store the distance between two points
    t0 = time.time()
    distances = np.zeros((len(data),num_humans,num_humans),dtype = np.float64) #(500,40,40)

    for i in range(0,len(data)):
        for j in range(0,num_humans):
            x_j = float(positions[i, j, 0])
            y_j = float(positions[i, j, 1])
            for k in range(0,num_humans):
                x_k = float(positions[i, k, 0])
                y_k = float(positions[i, k, 1])
                if (x_k !=-20) or  (y_k != -20) or (x_j != -20) or (y_j != -20):
                    distances[i,j,k] = np.sqrt((x_j - x_k)**2 + (y_j-y_k)**2)
                else:
                    distances[i,j,k] = -1

    t1 = time.time()

    groups = np.full((num_humans, num_humans), False, dtype= bool)
    t0 = time.time()
    for human in range(num_humans):
        #if human == 0:
        human_path = positions[:,human,0] #x positions over time for that human (500,1)
        frames = np.where(human_path > -15.0)[0]
        dx = positions[frames[-1],human,0] -positions[frames[0],human,0]
        dy = positions[frames[-1],human,1] -positions[frames[0],human,1]
        dir_human = math.degrees(math.atan(dx/(dy+0.00001)))
        #dir_human = (positions[frames[-1],human,1] -positions[frames[0],human,1]) /(positions[frames[-1],human,0] -positions[frames[0],human,0] + 0.000001) #direction of human
        for candidate in range(num_humans):
            candidate_path = positions[:,candidate,0]
            candidate_frames = np.where(candidate_path >-15.0)[0]
            common_frames = np.intersect1d(frames, candidate_frames,assume_unique = True)
            if common_frames.shape[0] != 0:
                if (common_frames.shape[0] / max(frames.shape[0],candidate_frames.shape[0])) >= 0.35: #Ie: the humans are together for at least 70
                    common_dist = distances[common_frames,human,candidate]
                    dx_c = positions[candidate_frames[-1],candidate,0] -positions[candidate_frames[0],candidate,0]
                    dy_c = positions[candidate_frames[-1],candidate,1] -positions[candidate_frames[0],candidate,1]
                    cand_dir = math.degrees(math.atan(dx_c/(dy_c+0.00001)))
                    if np.average(common_dist) < threshold and np.max(common_dist) < 4: #and np.abs(cand_dir - dir_human) <100:
                        groups[human,candidate] = True
                        #groups[candidate,human] = True

    t1 = time.time()
    group_list = [[] for i in range(0,num_humans)]
    #print('grouping Time: %.2f' % (t1 - t0))
    for human in range(num_humans):
        row = groups[human, :]
        #print('Human: ', human, 'group: ',np.where(row == True)[0])
        group_list[human] = np.where(row == True)[0]

    group_final_list = []
    for i in range(len(group_list)):
        if len(group_list[i]) != 1:
            if len(group_final_list) == 0:
                group_final_list.append(group_list[i])
            else:
                added = False
                for j in range(len(group_final_list)): #iterate over current groups
                    for human in group_list[i]:
                        if human in group_final_list[j] and added == False:
                            group_final_list[j] = list(set().union(group_final_list[j],group_list[i]))
                            added = True
                if added == False:
                    group_final_list.append(list(group_list[i]))


    return group_list, group_final_list


def final_gif_plotting(dataset, path_result, labeled_positions, group_list, labels, dataset_config, limit=-1):
    # Init loader
    loader = init_loader(dataset)

    # load background and rotation matrix
    transform_cam2world = np.loadtxt('calibration/' + dataset + '_matrix_cam2world.txt')

    if limit < 0:
        limit = len(labeled_positions)

    i_frame = 0
    for img in loader: #loops over frames in the video or image folder
        if i_frame >= limit-1:
            break


        t0 = time.time()
        
        pts_world = []
        for label in labeled_positions[i_frame]:
            pts_world.append(labeled_positions[i_frame][label])

        pts_world = np.array(pts_world)

        if dataset == 'oxford_town':
            pts_world[:, [0, 1]] = pts_world[:, [1, 0]]

        pts_cam = []
        for pt_world in pts_world:
            pt_cam = np.linalg.inv(transform_cam2world) @ np.array([[pt_world[0]], [pt_world[1]], [1]]).reshape(3)
            pts_cam.append(pt_cam / pt_cam[-1])

        if dataset == 'oxford_town':
            pts_world[:, [0, 1]] = pts_world[:, [1, 0]]

        pts_cam = np.array(pts_cam)


        violation_pairs = find_violation(pts_world) 
        non_group_violation_pairs, group_pairs = find_group_violations(
                            labeled_positions[i_frame], 
                            group_list, 
                            labels[i_frame]
                            )
        
        pts_roi_world, pts_roi_cam = get_roi_pts(
                            dataset=dataset,
                            roi_raw=dataset_config['roi'],
                            matrix_c2w=transform_cam2world
                            )
        fig = plot_grouping_frame_one_row(
                            dataset=dataset,
                            img_raw=img,
                            pts_roi_cam=pts_roi_cam,
                            pts_roi_world=pts_roi_world,
                            pts_w=pts_world,
                            pts_cam=pts_cam,
                            pairs=violation_pairs,
                            pts_w_groups=pts_world, 
                            non_group_pairs =non_group_violation_pairs,
                            group_pairs=group_pairs,
                            pts_dict=labeled_positions[i_frame],
                            config=dataset_config
                            )
        fig.savefig(os.path.join(path_result, 'frame%04d.png' % i_frame))
        plt.close(fig)

        t1 = time.time()

        # update loop info
        print('Frame %d - Inference Time: %.2f' % (i_frame, t1 - t0))
        i_frame += 1



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Object detection model to use, one of {yolo, hog, faster_rcnn")
    parser.add_argument("dataset", help="Dataset to use, one of {oxford_town, lstn, ucsd}")
    parser.add_argument('tracking', help="tracking method used, one of {optical_flow, centroid_tracking}")
    parser.add_argument('--vis', help="visualize groups in bird's eye view",dest='vis', action='store_true')
    parser.add_argument('--num_frames', help="number of frames to track", default=-1)

    parser.set_defaults(vis=False)

    args = parser.parse_args()
    vis = args.vis 
    detector = args.model 
    dataset = args.dataset
    tracking = args.tracking
    num_frames = int(args.num_frames)

    dataset = args.dataset

    dataset_config = configparser.ConfigParser()
    dataset_config.read('datasets/datasets.cfg')
    dataset_config = dataset_config[dataset.upper()]
    dataset_config = parse_dataset_config(dataset_config)

    path_result = os.path.join('results', tracking + '_' + detector, dataset)


    if tracking == 'optical_flow':
        threshold = 2.5

        data_path = os.path.join(path_result, 'opt_flow_data.p')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            data = data

        dictionaries, labels, data = convert_optical_flow_tracking(data)

    elif tracking == 'centroid_tracking':
        threshold = 1.7

        data_path = os.path.join(path_result, 'statistic_data.p')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            data = data

        with open(os.path.join(path_result, 'tracking_data.p'), 'rb') as f:
            dictionaries, labels = pickle.load(f)

    start = time.time()
    groups,groups_condensed = grouping(dictionaries, threshold=threshold)
    end = time.time()
    print('Grouping time: {:.2f}s / {} frames'.format(end-start, len(dictionaries)))

    #Create folder for results
    path_result = os.path.join('results', 'grouping', tracking + '_' + detector, dataset)
    #only creates the folders if they don't exist
    os.makedirs(path_result, exist_ok=True)

    if vis:
        group_gif_path = os.path.join(path_result, 'group_output.gif')
        group_plotting(data,labels,groups_condensed,group_gif_path)

    final_gif_plotting(dataset, path_result, dictionaries, groups, labels, dataset_config, limit=num_frames)


