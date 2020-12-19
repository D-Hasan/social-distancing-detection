import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import numpy as np
import time
import os
import imageio


color = [ "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
            "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
            "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
            "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
            "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
            "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
            "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
            "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
            "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
            "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
            "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
            "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
            "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
            "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
            "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
            "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


def get_labels(data,inter_frame_dist, vis, vis_path, num_frames=-1):
    '''

    :param filename: Path to the pickle from a detect.py code. Should be a list of lists, specifically N lists if we have N frames. Each sub-list contains
    3 elements: first element is the frame number, 2nd element is the time taken to compute the frame, and 3rd element is a list of (x,y) positions.
    :param inter_frame_dist: The distance that you believe humans move between frames. For example, for oxford, you can sent this to 1 meter.
    :return:
    It returns 3 things: the first is called Result, which is a list of N dictionaries. Each dictionary has a key which is the human ID, and the value is the x,y
    position. It also returns a list called labels. each element is a list  themself, which just has the human ID for the index. For example, if at frame Q we have
    10 humans, labels[Q] would be a list of 10 elements, where if label[Q][0] is 14, this means that the first human in this frame is actually human 14.
    last return is the original data itself
    '''
    #data is a list of tuples, specifically N tuples, where n = number of frames
    if num_frames == -1:
        num_frames = len(data)
    labels = [[] for i in range(num_frames)]

    for i in range(0,num_frames):                          # iterate over every frame
        num_humans_i = len(data[i][2])                       # number of humans in frame I
    
        if i == 0:
            labels[0] = [a for a in range(0,num_humans_i)] # initialize the labels if its the first frame as human 1, human 2, ...., human N.
    
        if i != 0:
            num_humans_i_minus_1 = len(data[i - 1][2])  # number of humans in frame I-1
            labels[i] = [-1 for _ in range(num_humans_i)]     # otherwise, initialize them as zero.
            distances = np.zeros((num_humans_i, num_humans_i_minus_1)) #distance matrix
    
            #Populate distance matrix
            for j in range(0,num_humans_i):                      # iterate over people found in frame I
                x_j, y_j = data[i][2][j]                         # position of human j in frame I
                for k in range(0,num_humans_i_minus_1):           # loop over humans in previous layer
                    x_k, y_k = data[i-1][2][k]
                    distances[j,k] = np.sqrt((x_j-x_k)**2 + (y_j-y_k)**2)
    
            #Determine most likely pairs within matrix:
            threshold = inter_frame_dist
            done = []
            for x in range(0,num_humans_i):
                dist = distances[x,:]
                index_list = np.argpartition(dist, len(dist)-1) #gets indices of top 5 smallest distances
                found = False
                for idx_ in index_list:
                    if labels[i-1][idx_] not in done and dist[idx_] < threshold and found == False:
                        done.append(labels[i-1][idx_])
                        labels[i][x] = labels[i-1][idx_]
                        found = True
    
            for q in range(len(labels[i])):
                if labels[i][q] == -1: #these are the people that are confusing. we should probably check if they just phased.
                    #Checking for phase
                    phased = False
                    num_look_back = min(15,i)
    
                    if i >0:
                        best_dist = 10000
                        best_frame = None
                        best_label = None
                        threshold = inter_frame_dist/1.5
    
                        for A in range(1, num_look_back+1): #iterate over number of times we went back
                            for u in range(len(data[i - A][2])):
                                cand_x, cand_y = data[i - A][2][u]
                                dist = (np.sqrt((data[i][2][q][0] - cand_x) ** 2 + (data[i][2][q][1] - cand_y) ** 2))
    
                                if dist/A < threshold and dist/A < best_dist and labels[i-A][u] not in done:
                                    best_dist = dist/A
                                    best_frame = i-A
                                    best_label = u
    
                        if best_frame != None:
                            labels[i][q] = labels[best_frame][best_label]
                            done.append(labels[best_frame][best_label])
                            phased = True
    
                        if phased == False:
                            max_term_prev = max(labels[i - 1])
                            max_term_curr = max(labels[i])
                            result = max(max_term_prev, max_term_curr)
                            labels[i][q] = result + 1
                            done.append(result+1)


    results = []
    for i in range(num_frames - 1):
        # Make dictionary
        pos = {}
        for j in range(len(data[i][2])):
            pos[labels[i][j]] = data[i][2][j]
        results.append(pos)

    if vis == True and vis_path is not None:
        ppp = [plott(labels, data, i) for i in range(0, len(data))]
        imageio.mimsave(vis_path, ppp, fps=len(data) / 25)
    
    return results, labels


def plott(labels,data,curr_frame):
    unique_humans = set([item for sublist in labels for item in sublist])
    sub_3_lim = (20, -10, 0, 32)  # the boundaries of Oxford
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set(xlim=(sub_3_lim[0], sub_3_lim[1]), ylim=(sub_3_lim[2], sub_3_lim[3]))  # boundaries
    # blue box
    rect = patches.Rectangle((14, 5), -14, 23, linestyle='--', linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    ax.set_xlabel('x position (meter)')
    ax.set_ylabel('y position (meter)')
    leg = ['ROI']

    for human in unique_humans:
        x_list = []
        y_list = []
        frames = []
    
        for i in range(curr_frame+1):
            try:
                x, y = data[i][2][labels[i].index(human)]
                x_list.append(x)
                y_list.append(y)
                frames.append(i)
            except:
                x_list.append(-100)
                y_list.append(-100)
    
        x_list = x_list[-1:]
        y_list = y_list[-1:]
        corr_x_list = []
        corr_y_list = []
        for i in range(len(x_list)):
            if x_list[i] != -100:
                corr_x_list.append(x_list[i])
                corr_y_list.append(y_list[i])
    
        if corr_x_list != []:
            leg.append(human)
            plt.scatter(corr_x_list,corr_y_list,s=20,color = color[human])
    
    textstr = 'frame number: %.2d' % (curr_frame)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)
    ax.legend(leg)
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return image



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Object detection model to use, one of {yolo, hog, faster_rcnn")
    parser.add_argument("dataset", help="Dataset to use, one of {oxford_town, lstn, ucsd}")
    parser.add_argument('--vis', help="visualize groups in bird's eye view",dest='vis', action='store_true')
    parser.add_argument('--num_frames', help="number of frames to track", default=-1)
    parser.set_defaults(vis=False)

    args = parser.parse_args()
    vis = args.vis 
    detector = args.model 
    dataset = args.dataset
    num_frames = int(args.num_frames)

    data_time = 'centroid_tracking'
    path_result = os.path.join('results', data_time + '_' + detector, dataset)

    max_dist_between_frame = 1.3

    data_path = os.path.join(path_result, 'statistic_data.p')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        data = data


    tracking_gif_path = os.path.join(path_result, 'tracking_output.gif')

    start = time.time()
    dictionaries, labels = get_labels(data, max_dist_between_frame, vis, tracking_gif_path, num_frames=num_frames)
    end = time.time()
    
    if num_frames == -1:
        print('Centroid Tracking time: {:.2f}s / {} frames'.format(end-start, len(data)))
    else:
        print('Centroid Tracking time: {:.2f}s / {} frames'.format(end-start, min(num_frames,len(data))))

    
    with open(os.path.join(path_result, 'tracking_data.p'), 'wb') as f:
        pickle.dump([dictionaries, labels], f)






