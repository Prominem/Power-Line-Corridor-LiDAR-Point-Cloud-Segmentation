import indoor3d_util_local
import os
import numpy as np
from multiprocessing import Pool
import time


def create_block(in_file_path, out_file_path):
    in_data_label = np.loadtxt(in_file_path)

    #  get rid of nan
    f_index = np.sum(in_data_label, axis=1)
    mask1 = np.isnan(f_index)
    mask2 = np.isinf(f_index)
    mask = np.logical_or(mask1, mask2)
    mask = (1 - mask).astype(np.bool)
    in_data_label = in_data_label[mask, :]

    current_data, current_label = indoor3d_util_local.room2blocks_plus_normalized(in_data_label, num_point)
    current_label = np.expand_dims(current_label, axis=-1)
    current_data_label = np.concatenate((current_data, current_label), axis=-1)

    np.save(out_file_path, current_data_label)

def progress(percent, width=50):
    '''进度打印功能'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %.2f%%' % (show_str, percent), end='')


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_point_cloud_dir = base_dir + '/data/input_point_cloud_dir'
    # input_point_cloud_dir = '/media/jp/disk/temp/kitti/input_point_cloud_dir'
    block = base_dir + '/data/block_local'
    # block = '/media/jp/disk/temp/kitti/block_local'
    if not os.path.exists(block):
        os.makedirs(block)

    num_point = 2048

    file_list = os.listdir(input_point_cloud_dir)
    num_validation = 1000
    np.random.shuffle(file_list)
    train_list = file_list[num_validation:]
    validation_list = file_list[:num_validation]

    # process training set
    train_block = block + '/train'
    if not os.path.exists(train_block):
        os.makedirs(train_block)
    p = Pool(12)
    for file_name in train_list:
        in_file_path = input_point_cloud_dir + '/' + file_name
        out_file_path = train_block + '/' + file_name.split('.')[0]+'_block.npy'
        if not os.path.exists(out_file_path):
            p.apply_async(create_block, args=(in_file_path,out_file_path,))

    print('Waiting for all subprocessed done...')
    p.close()
    p.join()
    print('Create blocks done.')

    # process validation set
    validation_block = block + '/validation'
    if not os.path.exists(validation_block):
        os.makedirs(validation_block)
    p = Pool(12)
    for file_name in validation_list:
        in_file_path = input_point_cloud_dir + '/' + file_name
        out_file_path = validation_block + '/' + file_name.split('.')[0] + '_block.npy'
        if not os.path.exists(out_file_path):
            p.apply_async(create_block, args=(in_file_path, out_file_path,))

    print('Waiting for all subprocessed done...')
    p.close()
    p.join()
    print('Create blocks done.')







