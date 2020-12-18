import numpy as np
from scipy import stats
import glob
from PIL import Image
import os 

from models import HOG, YOLO, Faster_RCNN, Mask_RCNN
from dataloaders import VideoLoader, ImageLoader



COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def init_model(detector, model_config, data_config):
    if detector == 'faster_rcnn':
        model = Faster_RCNN(model_config, data_config)
    elif detector == 'mask_rcnn':
        model = Mask_RCNN(model_config, data_config)
    elif detector == 'yolo':
        model = YOLO(model_config, data_config)     
    elif detector == 'hog':
        model = HOG(model_config, data_config)
    else:
        raise Exception('invalid model, pick one of {yolo, hog, faster_rcnn')

    return model 


def init_loader(dataset):
    if dataset == 'oxford_town':
        path = os.path.join('datasets', 'TownCentreXVID.avi')
        loader = VideoLoader(path)

    elif dataset == 'lstn':
        path = os.path.join('datasets', 'lstn')
        loader = ImageLoader(path)

    elif dataset == 'ucsd':
        path = os.path.join('datasets', 'ucsdpeds', 'vidf', 'vidf1_33_000.y')
        loader = ImageLoader(path)
    
    else:
        raise Exception('invalid dataset, pick one of {oxford_town, lstn, ucsd}')

    return loader 


def parse_dataset_config(config):
    new_config = {}
    new_config['name'] = config['name']
    new_config['place'] = config['place']

    new_config['roi'] = [float(item) for item in config['roi'].split(',')]
    new_config['sub_3_lim'] = [int(item) for item in config['sub_3_lim'].split(',')]

    return new_config

def convert_optical_flow_tracking(data):
    dictionaries = []
    new_dictionaries = []
    labels = []
    new_data = []
    for i, frame in enumerate(data):
        new_data.append([i, 0.2, frame])
        dictionary = {}
        for j, key in enumerate(frame):
            try:
                dictionary[key-1] = np.array([frame[key][1], frame[key][0]])
            except:
                import pdb; pdb.set_trace()

        labels.append(list(dictionary.keys()))
        dictionaries.append(dictionary)

    return dictionaries, labels, new_data



def make_gif(results_dir):
    fp_in = results_dir + "/frame*.png"
    fp_out = results_dir + "/frames.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=100, loop=0)


def decode_data(data, roi):
    """
    - decode the raw data w.r.t. the defined roi.

    :param data:
    :param roi:
    :return:
    """
    x_min, x_max, y_min, y_max = roi
    area = (x_max - x_min) * (y_max - y_min)

    density = []
    ts_inference = []
    pts_roi_all_frame = []
    inds_frame = []
    nums_ped = []

    for i_frame, t_inference, pts in data:
        count_in = 0
        count_out = 0
        pts_roi = []
        for pt in pts:
            if x_min < pt[0] < x_max and y_min < pt[1] < y_max:
                count_in += 1
                pts_roi.append(pt)
            else:
                count_out += 1
        pts_roi_all_frame.append(np.array(pts_roi))
        density.append(count_in / area)
        ts_inference.append(t_inference)
        inds_frame.append(i_frame)
        nums_ped.append(count_in)

        # print('frame %d - num. of ped inside roi: %d, outside: %d' % (i_frame, count_in, count_out))

    return np.array(inds_frame), np.array(ts_inference), pts_roi_all_frame, np.array(density), nums_ped


def count_violation_pairs(pts_all_frames, dist=2.0):
    counts = []
    for pts in pts_all_frames:
        pairs = find_violation(pts, dist)
        counts.append(len(pairs))
    return np.array(counts)



def find_violation(pts, dist=2.0):
    """

    :param pts: positions of all pedestrians in a single frame
    :param dist: social distance
    :return: a list of index pairs indicating two pedestrians who are violating social distancing
    """
    n = len(pts)  # number of pedestrians
    pairs = []
    for i in np.arange(0, n, 1):
        for j in np.arange(i+1, n, 1):
            if np.linalg.norm(pts[i] - pts[j]) < dist:
                pairs.append((i, j))
    return pairs

def find_group_violations(pts, group_list, labels, dist=2.0):
    """

    :param pts: positions of all pedestrians in a single frame
    :param dist: social distance
    :return: a list of index pairs indicating two pedestrians who are violating social distancing
    """
    n = len(pts)  # number of pedestrians
    pairs = []
    group_pairs = []
    for i,(key1,value1) in enumerate(pts.items()):
        for j,(key2,value2) in enumerate(pts.items()):
            if np.linalg.norm(value1 - value2) < dist:
                # pairs.append((labels.index(key1), labels.index(key2)))
                if key2 not in group_list[key1]:
                    pairs.append((key1,key2))
                elif key1 != key2:
                    group_pairs.append((key1,key2))
                
    return pairs, group_pairs

