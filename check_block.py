import numpy as np
import os
from multiprocessing import Pool

# *****************main***********************
base_dir = os.path.dirname(os.path.abspath(__file__))
block_dir = base_dir + '/data/block/train'
check_dir = base_dir + '/data/check'
if not os.path.exists(check_dir):
    os.makedirs(check_dir)
input_file_list = os.listdir(block_dir)
np.random.shuffle(input_file_list)
file_name = input_file_list[0]
file_path = block_dir + '/' + file_name
check_folder = check_dir + '/' + file_name.split('.')[0]
if not os.path.exists(check_folder):
    os.makedirs(check_folder)

data_label = np.load(file_path)  # shape = (n,npoints,8), x,y,z,i,x,y,z,l
save_data_label = np.zeros((data_label.shape[0], data_label.shape[1], 5))
save_data_label[:,:,0:3] = data_label[:,:,4:7]*[140.0, 100.0, 5.0]
save_data_label[:,:,3] = data_label[:,:,3]
save_data_label[:,:,4] = data_label[:,:,-1]

p = Pool(12)
for i_block in range(data_label.shape[0]):
    check_file_path = check_folder + '/' + str(i_block) + '.txt'
    np.savetxt(check_file_path, save_data_label[i_block])
    # p.apply_async(np.savetxt, args=(check_file_path, save_data_label[i_block], ))
