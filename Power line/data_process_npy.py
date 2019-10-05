import indoor3d_util_v8
import os
import numpy as np
import block_statistics
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

    current_data, current_label = indoor3d_util_v8.room2blocks_plus_normalized(in_data_label, num_point)
    current_label = np.expand_dims(current_label, axis=-1)
    current_data_label = np.concatenate((current_data, current_label), axis=-1)

    np.save(out_file_path, current_data_label)

def progress(percent, width=50):
    '''进度打印功能'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %.2f%%' % (show_str, percent), end='')

if __name__=='__main__':


    processed = '/media/jp/disk/kitti/lidar_labeled3'
    block = '/media/jp/disk/kitti/block9'
    if not os.path.exists(block):
        os.makedirs(block)

    if not os.path.exists(block):
        os.makedirs(block)

    num_point = 2048

    file_list = os.listdir(processed)
    num_file = len(file_list)

    p = Pool(12)

    for i in range(num_file):
        print(i)
        in_file_path = processed + '/' + file_list[i]
        out_file_path = block + '/' + file_list[i].split('.')[0]+'_block.npy'
        if not os.path.exists(out_file_path):
            p.apply_async(create_block, args=(in_file_path,out_file_path,))

    print('Waiting for all subprocessed done...')
    cond = 1
    while cond:
        num = len(os.listdir(block))
        recv_per = 100.0 * num / num_file
        progress(recv_per, width=30)
        if num == num_file:
            cond = 0
        time.sleep(1)
    p.close()
    p.join()
    print('Create blocks done.')







