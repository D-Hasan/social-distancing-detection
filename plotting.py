import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import imageio


colors = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
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

def get_roi_pts(dataset, roi_raw, matrix_c2w):
    y1, y2, x1, x2 = roi_raw
    pts_world = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])
    
    pts_cam = []
    for pt_world in pts_world:
        pt_cam = np.linalg.inv(matrix_c2w) @ np.array([[pt_world[0]], [pt_world[1]], [1]]).reshape(3)
        pts_cam.append(pt_cam / pt_cam[-1])

    pts_cam = np.array(pts_cam)
    return pts_world, pts_cam[:, :2]


def plot_frame_one_row(dataset, img_raw, pts_roi_cam, pts_roi_world, pts_w, pairs, config):
    b, g, r = cv2.split(img_raw)  # get b,g,r
    img_raw = cv2.merge([r, g, b])  # switch it to rgb

    place = config['place']
    name = config['name']
    sub_3_lim = config['sub_3_lim']

    # plot
    fig = plt.figure(figsize=(8.77, 3.06))
    fig.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.90, wspace=0.3)
    fig.suptitle('%s (%s)' % (place, name))

    # subplot 1 - camera view
    a = fig.add_subplot(1, 3, (1, 2))
    plt.imshow(img_raw)
    a.plot(pts_roi_cam[:, 0], pts_roi_cam[:, 1], '--b')
    # a.set_title('Video')
    a.set_xlabel('x position (pixel)')
    a.set_ylabel('y position (pixel)')

    # subplot 2 - bird eye view social distancing
    a = fig.add_subplot(1, 3, 3)
    # a.set_title('BEV - social distancing')
    a.plot(pts_roi_world[:, 0], pts_roi_world[:, 1], '--b')
    a.plot(pts_w[:, 0], pts_w[:, 1], 'og', alpha=0.5)

    for pair in pairs:
        data = np.array([pts_w[pair[0]], pts_w[pair[1]]])
        a.plot(data[:, 0], data[:, 1], '-r')

    a.axis('equal')
    a.grid()
    a.set_xlabel('x position (meter)')
    a.set_ylabel('y position (meter)')
    a.set(xlim=(sub_3_lim[0], sub_3_lim[1]), ylim=(sub_3_lim[2], sub_3_lim[3]))

    return fig


def plot_grouping_frame_one_row(dataset, img_raw, pts_roi_cam, pts_roi_world, pts_w, pts_cam, pairs, pts_w_groups, non_group_pairs, group_pairs, pts_dict, config):
    b, g, r = cv2.split(img_raw)  # get b,g,r
    img_raw = cv2.merge([r, g, b])  # switch it to rgb

    place = config['place']
    name = config['name']
    sub_3_lim = config['sub_3_lim']

    fig = plt.figure(figsize=(9.77, 3.06))
    fig.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.90, wspace=0.3)
    fig.suptitle('%s (%s)' % (place, name))

    # subplot 1 - camera view
    a = fig.add_subplot(1, 4, (1, 2))
    plt.imshow(img_raw)

    a.plot(pts_roi_cam[:, 0], pts_roi_cam[:, 1], '--b')

    for point in pts_cam:
        a.plot(point[0], point[1], 'or')
    # a.set_title('Video')
    a.set_xlabel('x position (pixel)')
    a.set_ylabel('y position (pixel)')

    # subplot 2 - bird eye view social distancing
    a = fig.add_subplot(1, 4, 3)
    # a.set_title('BEV - social distancing')

    if dataset == 'oxford_town':
        a.plot(pts_roi_world[:, 1], pts_roi_world[:, 0], '--b')
    else:
        a.plot(pts_roi_world[:, 0], pts_roi_world[:, 1], '--b')
    # a.plot(pts_w[:, 0], pts_w[:, 1], 'og', alpha=0.5)
    for point in pts_dict.values():
        a.plot(point[0], point[1], 'og', alpha=0.5)

    for pair in pairs:
        data = np.array([pts_w[pair[0]], pts_w[pair[1]]])
        a.plot(data[:, 0], data[:, 1], '-r')

    a.axis('equal')
    a.grid()
    a.set_xlabel('x position (meter)')
    a.set_ylabel('y position (meter)')
    a.title.set_text('Non Grouping Birds Eye View')
    a.set(xlim=(sub_3_lim[0], sub_3_lim[1]), ylim=(sub_3_lim[2], sub_3_lim[3]))

    # subplot 3 - our grouped bird eye view social distancing
    a = fig.add_subplot(1, 4, 4)

    if dataset == 'oxford_town':
        a.plot(pts_roi_world[:, 1], pts_roi_world[:, 0], '--b')
    else:
        a.plot(pts_roi_world[:, 0], pts_roi_world[:, 1], '--b')

    for point in pts_dict.values():
        a.plot(point[0], point[1], 'og', alpha=0.5)

    for pair in non_group_pairs:
        data = np.array([pts_dict[pair[0]], pts_dict[pair[1]]])
        a.plot(data[:, 0], data[:, 1], '-r')

    for i, pair in enumerate(group_pairs):
        data = np.array([pts_dict[pair[0]], pts_dict[pair[1]]])
        color = colors[20 + i]
        a.plot(data[:, 0], data[:, 1], color='tab:blue', marker='o')

    a.axis('equal')
    a.grid()
    a.set_xlabel('x position (meter)')
    a.set_ylabel('y position (meter)')
    a.title.set_text('Grouping Birds Eye View')
    a.set(xlim=(sub_3_lim[0], sub_3_lim[1]), ylim=(sub_3_lim[2], sub_3_lim[3]))

    fig.tight_layout()

    return fig


def make_gif(results_dir):
    fp_in = results_dir + "/frame*.png"
    fp_out = results_dir + "/frames.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=20, loop=0)


def group_plotting(data,labels,group_final_list,gif_path):
    
    num_groups = len(group_final_list)
    num_colors = len(color)
    colors_group = color[20::(int(num_colors/num_groups)-2)]

    ppp = [plot(labels,data,i,colors_group, group_final_list) for i in range(0,len(data))]

    imageio.mimsave(gif_path, ppp, fps=len(data)/5)


def plot(labels,data,curr_frame,colors_group, group_final_list):
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
    
        x_list = x_list[-3:]
        y_list = y_list[-3:]
        corr_x_list = []
        corr_y_list = []
    
        for i in range(len(x_list)):
            if x_list[i] != -100:
                corr_x_list.append(x_list[i])
                corr_y_list.append(y_list[i])
    
        if corr_x_list != []:
            leg.append(human)
            in_group = False
    
            for qq in range(len(group_final_list)):
                if human in group_final_list[qq] and in_group ==False:
                    plt.scatter(corr_x_list,corr_y_list,s=20,color = colors_group[qq])
                    in_group = True
    
            if in_group == False:
                plt.scatter(corr_x_list, corr_y_list, s=20, color='black')
    
    textstr = 'frame number: %.2d' % (curr_frame)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)
    ax.legend(leg)
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return image