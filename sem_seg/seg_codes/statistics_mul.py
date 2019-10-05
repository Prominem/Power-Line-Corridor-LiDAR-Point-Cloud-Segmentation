import numpy as np
import os
from multiprocessing import Pool
from multiprocessing import cpu_count
import datetime
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))
statistics_dir = base_dir + '/../log_statistics'
temp_folder_dir = statistics_dir + '/temp'
if not os.path.exists(statistics_dir):
    os.makedirs(statistics_dir)
if not os.path.exists(temp_folder_dir):
    os.makedirs(temp_folder_dir)

log_path = statistics_dir + '/log.txt'
f = open(log_path, 'a')


def log_string(out_str):
    f.write(out_str+'\n')
    f.flush()
    print(out_str)

def iou_pre_rec_per_cls(pred, gt, cls):
    # pred, shape=(n, )
    # gt, shape = (n, )
    # cls, int
    # return: np.array([iou, precision, recall]) or np.nan
    true_positive = np.sum((pred==cls)&(gt==cls))
    pred_positive = np.sum(pred==cls)
    gt_positive = np.sum(gt==cls)
    if gt_positive==0:
        return np.nan
    else:
        if(true_positive==0):
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float)
        else:
            iou = true_positive*100.0/(pred_positive+gt_positive-true_positive)
            precision = true_positive*100.0/pred_positive
            recall = true_positive*100.0/gt_positive
            return np.array([[iou, precision, recall]], dtype=np.float)

def process_one_scan(i_index, index_list, temp_folder_dir):
    # i_index: 0, 1, ... int
    # index_list: ['000010',...]
    pred_file_path = results_dir + '/' + index_list[i_index] + '_pred.txt'
    gt_file_path = results_dir + '/' + index_list[i_index] + '_gt.txt'
    pred_file = np.loadtxt(pred_file_path)  # shape=(n,5), x,y,z,i,l
    gt_file = np.loadtxt(gt_file_path)
    pred = pred_file[:, -1]
    gt = gt_file[:, -1]
    scan_name = index_list[i_index]

    car_iou_pre_rec_t = iou_pre_rec_per_cls(pred, gt, 1)
    if car_iou_pre_rec_t is not np.nan:
        car_path = temp_folder_dir + '/' + scan_name + '_car.npy'
        np.save(car_path, car_iou_pre_rec_t)


    pedestrian_iou_pre_rec_t = iou_pre_rec_per_cls(pred, gt, 2)
    if pedestrian_iou_pre_rec_t is not np.nan:
        pedestrian_path = temp_folder_dir + '/' + scan_name + '_pedestrian.npy'
        np.save(pedestrian_path, pedestrian_iou_pre_rec_t)

    cyclist_iou_pre_rec_t = iou_pre_rec_per_cls(pred, gt, 3)
    if cyclist_iou_pre_rec_t is not np.nan:
        cyclist_path = temp_folder_dir + '/' + scan_name + '_cyclist.npy'
        np.save(cyclist_path, cyclist_iou_pre_rec_t)

# **********************main*********************
now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
log_string('Start time:' + now_time)
results_dir = base_dir + '/../log_local_v2/dump'
file_list = os.listdir(results_dir)
print(file_list)
index_list = [name.split('_')[0] for name in file_list if name.endswith('_pred.txt')]

num_index = len(index_list)
car_iou_pre_rec = np.array([])
pedestrian_iou_pre_rec = np.array([])
cyclist_iou_pre_rec = np.array([])

p = Pool(cpu_count())
for i_index in range(num_index):
    print(i_index)
    p.apply_async(process_one_scan, args=(i_index, index_list, temp_folder_dir,))

print('Waiting for all subprocessed done...')
p.close()
p.join()
print('Part calculation done.')

car_result_list = [name for name in os.listdir(temp_folder_dir) if name.endswith('_car.npy')]
print(car_result_list)
pedestrian_result_list = [name for name in os.listdir(temp_folder_dir) if name.endswith('_pedestrian.npy')]
print(pedestrian_result_list)
cyclist_result_list = [name for name in os.listdir(temp_folder_dir) if name.endswith('_cyclist.npy')]
print(cyclist_result_list)

for result in car_result_list:
    car_r_path = temp_folder_dir + '/' + result
    car_r = np.load(car_r_path)
    if len(car_iou_pre_rec)==0:
        car_iou_pre_rec = car_r
    else:
        car_iou_pre_rec = np.concatenate((car_iou_pre_rec, car_r), axis=0)

for result in pedestrian_result_list:
    pedestrian_r_path = temp_folder_dir + '/' + result
    pedestrian_r = np.load(pedestrian_r_path)
    if len(pedestrian_iou_pre_rec) == 0:
        pedestrian_iou_pre_rec = pedestrian_r
    else:
        pedestrian_iou_pre_rec = np.concatenate((pedestrian_iou_pre_rec, pedestrian_r), axis=0)

for result in cyclist_result_list:
    cyclist_r_path = temp_folder_dir + '/' + result
    cyclist_r = np.load(cyclist_r_path)
    if len(cyclist_iou_pre_rec)==0:
        cyclist_iou_pre_rec = cyclist_r
    else:
        cyclist_iou_pre_rec = np.concatenate((cyclist_iou_pre_rec, cyclist_r), axis=0)

car_mean_iou_pre_rec = np.mean(car_iou_pre_rec, axis=0)
pedestrian_mean_iou_pre_rec = np.mean(pedestrian_iou_pre_rec, axis=0)
cylist_mean_iou_pre_rec = np.mean(cyclist_iou_pre_rec, axis=0)




log_string('class: car, pedestrian, cyclist')
log_string('IoU: %f, %f, %f' % (car_mean_iou_pre_rec[0], pedestrian_mean_iou_pre_rec[0], cylist_mean_iou_pre_rec[0]))
log_string('precision: %f, %f, %f' % (car_mean_iou_pre_rec[1], pedestrian_mean_iou_pre_rec[1], cylist_mean_iou_pre_rec[1]))
log_string('recall: %f, %f, %f' % (car_mean_iou_pre_rec[2], pedestrian_mean_iou_pre_rec[2], cylist_mean_iou_pre_rec[2]))

now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
log_string('End time:' + now_time)
f.close()
shutil.rmtree(temp_folder_dir)