import configparser
import argparse 
import os
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cv2

from plotting import plot_frame_one_row, get_roi_pts
from utils import find_violation, make_gif, parse_dataset_config
from utils import COCO_INSTANCE_CATEGORY_NAMES as LABELS
from utils import init_model, init_loader


import warnings
warnings.filterwarnings("ignore")


def main(dataset, data_time, detector, config, dataset_config, vis=False, stop_running=-1):

    #Create folder for results
    path_result = os.path.join('results', data_time + '_' + detector, dataset)
    #only creates the folders if they don't exist
    os.makedirs(path_result, exist_ok=True)
    
    # Init model
    model = init_model(detector, config['MODEL_CONFIG'], config[DATASET_KEY])

    # Init loader
    loader = init_loader(dataset)

    # load background and rotation matrix
    transform_cam2world = np.loadtxt('calibration/' + dataset + '_matrix_cam2world.txt')


    statistic_data = []
    i_frame = 0

    for img in loader: #loops over frames in the video or image folder
        if i_frame > stop_running:
            break

        t0 = time.time()
        
        boxes, scores, humans_idx = model.predict(img)        
        
        pts_world = []
           
        for i in humans_idx:
        # extract the bounding box coordinates
            (x1, y1) = (boxes[i][0], boxes[i][1]) #represents 1 corner
            (x2, y2) = (boxes[i][2], boxes[i][3]) #represents the next corner

            if vis:
                # draw a bounding box rectangle and label on the image
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 0, 255], 2)

                cv2.imwrite(os.path.join(path_result, 'bb%04d.png' % i_frame), img)
            
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
            pts_roi_world, pts_roi_cam = get_roi_pts(
                                    dataset, 
                                    roi_raw=dataset_config['roi'], 
                                    matrix_c2w=transform_cam2world
                                    )
            if dataset == 'oxford_town':
                pts_roi_world[:, [0, 1]] = pts_roi_world[:, [1, 0]]

            fig = plot_frame_one_row(
                        dataset=dataset,
                        img_raw=img,
                        pts_roi_cam=pts_roi_cam,
                        pts_roi_world=pts_roi_world,
                        pts_w=pts_world,
                        pairs=violation_pairs,
                        config=dataset_config
                        )
            fig.savefig(os.path.join(path_result, 'frame%04d.png' % i_frame))
            plt.close(fig)

        # update loop info
        if i_frame % 5 == 0:
            print('Frame %d - Inference Time: %.2f' % (i_frame, t1 - t0))
        i_frame += 1

    cv2.destroyAllWindows()
    pickle.dump(statistic_data, open(os.path.join(path_result, 'statistic_data.p'), 'wb'))

    # if vis:
    #     make_gif(path_result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Object detection model to use, one of {yolo, hog, faster_rcnn")
    parser.add_argument("dataset", help="Dataset to use, one of {oxford_town, lstn, ucsd}")
    parser.add_argument("--vis", help="Visualize bounding boxes and bird's eye view.", dest='vis', action='store_true')
    parser.add_argument('--num_frames', help="number of frames to track", default=-1)
    parser.set_defaults(vis=False)

    args = parser.parse_args()

    vis = args.vis 
    detector = args.model 
    num_frames = int(args.num_frames)
    config_path = os.path.join('model_configs', detector + '.cfg')

    config = configparser.ConfigParser()
    config.read(config_path)

    dataset = args.dataset
    DATASET_KEY = dataset.upper() + "_CONFIG"

    dataset_config = configparser.ConfigParser()
    dataset_config.read('datasets/datasets.cfg')
    dataset_config = dataset_config[dataset.upper()]
    dataset_config = parse_dataset_config(dataset_config)


    data_time = 'centroid_tracking'
    

    print('=========== %s ===========' % dataset)
    np.set_printoptions(precision=4) #Ie: print floats to 4 decimals
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


    main(dataset, data_time, detector, config, dataset_config, vis=vis, stop_running=num_frames)

