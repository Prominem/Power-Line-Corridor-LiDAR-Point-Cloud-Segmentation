import argparse
import os
import sys
import tensorflow as tf
from model_local import *
# import indoor3d_util
import numpy as np
from multiprocessing import Pool
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=2048, help='Point number [default: 4096]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
channel_len = 8
NUM_CLASSES = 4
MODEL_PATH = BASE_DIR + '/../log_local/model.ckpt-20'

GPU_INDEX = FLAGS.gpu
DUMP_DIR = BASE_DIR + '/../log_local/dump'
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)

def evaluate():
    is_training = False
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)
 
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # restore model
    model_path = MODEL_PATH
    if model_path:
        saver.restore(sess, model_path)
        print('%s model restored...' % str(model_path))

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}

    #  load data
    data_label_folder_path = BASE_DIR + '/../../data/block_local/validation'
    # data_label_folder_path = '/media/jp/disk/temp/kitti/block_local/validation'
    out_folder_path = DUMP_DIR
    original_folder_path = BASE_DIR + '/../../data/input_point_cloud_dir'
    # original_folder_path = '/media/jp/disk/temp/kitti/input_point_cloud_dir'

    data_label_list = os.listdir(data_label_folder_path)

    p = Pool(12)
    for block_name in data_label_list:
        pred_out_path = out_folder_path + '/' + block_name.split('_')[0] + '_pred.txt'
        if not os.path.exists(pred_out_path):
            gt_out_path = out_folder_path + '/' + block_name.split('_')[0] + '_gt.txt'
            data_label_path = data_label_folder_path + '/' + block_name
            original_file_path = original_folder_path + '/' + block_name.split('_')[0] + '.txt'
            p.apply_async(eval_one_epoch(sess, ops, original_file_path, data_label_path, pred_out_path, gt_out_path))
    print('Waiting for all subprocessed done...')
    p.close()
    p.join()
    print('Create blocks done.')


def eval_one_epoch(sess, ops, original_file_path, data_label_path, path_pred_out, path_gt_out):
    is_training = False

    original_file = np.loadtxt(original_file_path)  # shape = (n,5), xyzil
    filter_mask = original_file[:, 2] > -2.2
    original_file = original_file[filter_mask, :]
    min_xyz = np.min(original_file[:, 0:3], axis=0)
    original_file[:, 0:3] -= min_xyz
    max_xyzi = np.max(original_file[:, 0:4], axis=0)

    fout = open(path_pred_out, 'w')
    fout_gt = open(path_gt_out, 'w')


    current_data_label = np.load(data_label_path)

    current_data = current_data_label[:,:,0:channel_len]
    current_label = np.array(current_data_label[:,:,-1], dtype=np.int)
    current_data = current_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size

    
    for batch_idx in range(num_batches):
        start_idx = batch_idx
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:start_idx+1, :, :],
                     ops['labels_pl']: current_label[start_idx:start_idx+1],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)

        pred_label = np.argmax(pred_val, axis=-1)

        pts = current_data[start_idx, :, :]
        l = current_label[start_idx,:]
        pts[:,4] = pts[:,4]*max_xyzi[0]
        pts[:,5] = pts[:,5]*max_xyzi[1]
        pts[:,6] = pts[:,6]*max_xyzi[2]
        pred = pred_label[0, :]
        for i in range(NUM_POINT):
            fout.write('%f %f %f %f %d \n' % (pts[i,4], pts[i,5], pts[i,6],pts[i,7]*max_xyzi[3], pred[i]))
            fout_gt.write('%f %f %f %f %d \n' % (pts[i,4], pts[i,5], pts[i,6], pts[i,7]*max_xyzi[3], current_label[start_idx, i]))

    fout.close()
    fout_gt.close()


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
