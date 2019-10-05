import numpy as np
import os


# a = np.array([[1,2,1,3],[4,1,2,5],[3,1,4,4]])
# b = np.array([1,2,3,4])
# b = np.expand_dims(b, axis=1)
# b = np.concatenate((b,b), axis=-1)
# c = a.dot(b).T
# print(c)

backup_folder_path = '/media/jp/disk/powernet_backup/kitti/2/local_v2'
if not os.path.exists(backup_folder_path):
    os.makedirs(backup_folder_path)
training_name_list_dir = backup_folder_path + '/train_name_list.txt'
validation_name_list_dir = backup_folder_path + '/validation_name_list.txt'
fout = open(training_name_list_dir, 'w')
training_folder_path = '/media/jp/disk/temp/kitti/block_local_v2/train'
validation_folder_path = '/media/jp/disk/temp/kitti/block_local_v2/validation'
training_list = os.listdir(training_folder_path)
validation_list = os.listdir(validation_folder_path)
for training in training_list:
    fout.write(training + '\n')
fout.close()

fout = open(validation_name_list_dir, 'w')
for validation in validation_list:
    fout.write(validation + '\n')
fout.close()


