import numpy as np
import os
from multiprocessing import Pool


def progress(percent, width=50):
    '''进度打印功能'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %.2f%%' % (show_str, percent), end='')


def process_one_file(data_dir, file_name, output_dir):
    file_path = data_dir + '/' + file_name
    data_label = np.loadtxt(file_path)  # shape=(n,5) columns:x,y,z,I,L

    # label nine2four
    label_dict = {'0': 1, '3': 2, '5': 3}  # car:1, pedestrian:2, cyclist:3,others:0
    new_data_label = np.zeros(data_label.shape)
    new_data_label[:,0:4] = data_label[:,0:4]
    for key in label_dict.keys():
        int_key = int(key)
        label_mask = data_label[:,-1] == int_key
        new_data_label[label_mask, -1] = label_dict[key]

    output_file_path = output_dir + '/' + file_name
    np.savetxt(output_file_path, new_data_label)


# ********************main***********************
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = base_dir + '/data/labeled_point_cloud'
# data_dir = '/media/jp/disk/temp/kitti/labeled_point_cloud'
input_point_cloud_dir = base_dir + '/data/input_point_cloud_dir'
# input_point_cloud_dir = '/media/jp/disk/temp/kitti/input_point_cloud_dir'
if not os.path.exists(input_point_cloud_dir):
    os.makedirs(input_point_cloud_dir)

p = Pool(12)
file_name_list = os.listdir(data_dir)
i_progress = 0
total_progress = len(file_name_list)
for file_name in file_name_list:
    i_progress += 1
    progress(i_progress*100.0/total_progress)
    out_file_path = input_point_cloud_dir + '/' + file_name
    if not os.path.exists(out_file_path):
    # process_one_file(data_dir, file_name, input_point_cloud_dir)
        p.apply_async(process_one_file, args=(data_dir, file_name, input_point_cloud_dir,))
print('Waiting for all subprocessed done...')
p.close()
p.join()
print('All subprecessed done.')