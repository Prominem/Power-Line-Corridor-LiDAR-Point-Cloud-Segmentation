import os
import numpy as np
from math import sin, cos
from multiprocessing import Pool


def extract_3d_label(label_path):
    label_dict = {}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            if len(line) > 3:
                key, value = line.split(' ', 1)
                if key in label_dict.keys():
                    label_dict[key].append([float(x) for x in value.split()])
                else:
                    label_dict[key] = [[float(x) for x in value.split()]]
    for key in label_dict.keys():
        label_dict[key] = np.array(label_dict[key])
    return label_dict


def add_points(velo, corners):   # visualize xyz in 3d label
    labels = np.zeros((velo.shape[0],1), dtype=np.int64)
    velo = np.concatenate((velo, labels), axis=-1)
    labels = np.ones((corners.shape[0], 2), dtype=np.int64)
    print(labels.shape)
    print(corners.shape)
    corners = np.concatenate((corners, labels), axis=-1)
    velo = np.concatenate((velo, corners), axis=0)

    return velo


def corner_3d(label_dict):
    corner_dict = {}
    for key in label_dict.keys():
        if key != 'DontCare':
            labels = label_dict[key]

            for i in range(labels.shape[0]):
                one_label = labels[i]
                if key == 'Car':
                    w = one_label[7]
                    h = one_label[8]
                    l = one_label[9]
                    w += 0.4
                    x = one_label[10]
                    y = one_label[11]
                    z = one_label[12]
                    ry = one_label[13]
                else:
                    w = one_label[7]
                    h = one_label[8]
                    l = one_label[9]
                    x = one_label[10]
                    y = one_label[11]
                    z = one_label[12]
                    ry = one_label[13]

                R = np.array([[+cos(ry), 0, +sin(ry)],
                              [0, 1, 0],
                              [-sin(ry), 0, +cos(ry)]])

                # 3D bounding box corners

                x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
                y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
                z_corners = [0, 0, 0, w, w, w, w, 0]  # --w/2

                x_corners += -l / 2
                y_corners += -h
                z_corners += -w / 2

                # bounding box in object co-ordinate
                corners_3D = np.array([x_corners, y_corners, z_corners])
                # print ( 'corners_3d', corners_3D.shape, corners_3D)

                # rotate
                corners_3D = R.dot(corners_3D)
                corners_3D = np.array([corners_3D[2,:],-corners_3D[0,:],-corners_3D[1,:]])
                # print ( 'corners_3d', corners_3D.shape, corners_3D)

                # translate
                corners_3D += np.array([z, -x, -y]).reshape((3, 1))

                corners_3d = np.transpose(corners_3D)

                if key in corner_dict.keys():
                    corner_dict[key].append(corners_3d)
                else:
                    corner_dict[key] = [corners_3d]

    for key in corner_dict.keys():
        corner_dict[key] = np.array(corner_dict[key])

    return corner_dict

def label_one_box(velo, box, label_index):
    x_sort_list = np.sort(box[:,0])
    xt1 = x_sort_list[0]
    xt2 = x_sort_list[2]
    xt3 = x_sort_list[4]
    xt4 = x_sort_list[6]
    # print(int(np.argwhere(box[:,0] == xt1)[0]))
    yt1 = box[int(np.argwhere(box[:,0] == xt1)[0]), 1]
    yt2 = box[int(np.argwhere(box[:, 0] == xt2)[0]), 1]
    yt3 = box[int(np.argwhere(box[:, 0] == xt3)[0]), 1]
    yt4 = box[int(np.argwhere(box[:, 0] == xt4)[0]), 1]
    if yt1 > yt2:
        x1 = xt2
        y1 = yt2
        x2 = xt1
        y2 = yt1
    else:
        x1 = xt1
        y1 = yt1
        x2 = xt2
        y2 = yt2

    if yt3 > yt4:
        x3 = xt3
        y3 = yt3
        x4 = xt4
        y4 = yt4
    else:
        x3 = xt4
        y3 = yt4
        x4 = xt3
        y4 = yt3

    v12 = np.array([x2 - x1, y2 - y1], dtype=np.float)
    v23 = np.array([x3 - x2, y3 - y2], dtype=np.float)
    v34 = np.array([x4 - x3, y4 - y3], dtype=np.float)
    v41 = np.array([x1 - x4, y1 - y4], dtype=np.float)
    v1p = np.array(velo[:, 0:2] - [x1, y1], dtype=np.float)
    v2p = np.array(velo[:, 0:2] - [x2, y2], dtype=np.float)
    v3p = np.array(velo[:, 0:2] - [x3, y3], dtype=np.float)
    v4p = np.array(velo[:, 0:2] - [x4, y4], dtype=np.float)

    cond1 = np.sum(v12 * v1p, axis=-1) >= 0
    cond2 = np.sum(v23 * v2p, axis=-1) >= 0
    cond3 = np.sum(v34 * v3p, axis=-1) >= 0
    cond4 = np.sum(v41 * v4p, axis=-1) >= 0
    cond = cond1 & cond2
    cond = cond & cond3
    cond = cond & cond4

    z_min = np.min(box[:, 2])
    z_cond = velo[:, 2] >= z_min
    cond = cond & z_cond
    velo[cond, -1] = int(label_index)

    return velo

def add_labels(velo, corner_dict):
    class2label_dict = {'Car': 0, 'Van': 1, 'Truck': 2,
                        'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5,
                        'Tram': 6, 'Misc': 7, 'DontCare': 8}
    # initialize labels
    labels = 8*np.ones((velo.shape[0], 1), dtype=np.int64)
    velo = np.concatenate((velo, labels), axis=-1)

    # add labels
    for key in corner_dict.keys():
        for i_box in range(corner_dict[key].shape[0]):
            box = corner_dict[key][i_box]
            label_index = class2label_dict[key]
            velo = label_one_box(velo, box, label_index)

    return velo


def add_points(velo, corners):   # visualize xyz in 3d label
    corners = corners.reshape(corners.shape[0]*corners.shape[1], corners.shape[2])
    labels = 9*np.ones((corners.shape[0], 2), dtype=np.int64)
    corners = np.concatenate((corners, labels), axis=-1)
    velo = np.concatenate((velo, corners), axis=0)
    return velo

def label_one_file(lidar_path, label_path, out_path):
    velo = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    label_dict = extract_3d_label(label_path)
    corner_dict = corner_3d(label_dict)
    velo = add_labels(velo, corner_dict)
    np.savetxt(out_path, velo)


# *****************************main**************************
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lidar_folder = BASE_DIR + '/data/kitti/data_object_velodyne/training/velodyne'
label_folder = BASE_DIR + '/data/kitti/label_2'
out_folder_path = BASE_DIR + '/data/labeled_point_cloud'  # output path of the labeled point cloud
# out_folder_path = '/media/jp/disk/temp/kitti/labeled_point_cloud'
if not os.path.exists(out_folder_path):
    os.makedirs(out_folder_path)
lidar_file_list = os.listdir(lidar_folder)
p = Pool(12)

for lidar_name in lidar_file_list:
    print(lidar_name)
    out_file_path = out_folder_path + '/' + lidar_name.split('.')[0] + '.txt'
    if not os.path.exists(out_file_path):
        lidar_file_path = lidar_folder + '/' + lidar_name
        label_path = label_folder + '/' + lidar_name.split('.')[0] + '.txt'
        p.apply_async(label_one_file, args=(lidar_file_path, label_path, out_file_path,))
print('Waiting for all subprocessed done...')
p.close()
p.join()
print('All subprecessed done.')
