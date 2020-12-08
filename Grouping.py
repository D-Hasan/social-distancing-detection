import pickle
import numpy as np
import time
import math

filename = '/Users/gabriel/PycharmProjects/420Project/social-distancing-monitoring/tracked_data.p'

def grouping(data):
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
    #print('Inference Time: %.2f' % (t1 - t0))
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
                    if np.average(common_dist) <1.5 and np.max(common_dist) < 4: #and np.abs(cand_dir - dir_human) <100:
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


    #file_path = '/Users/gabriel/PycharmProjects/420Project/social-distancing-monitoring/group_list.p'
    #pickle.dump(group_list, open(file_path, 'wb'))
    return group_list




