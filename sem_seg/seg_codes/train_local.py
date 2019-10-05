import argparse
import numpy as np
import tensorflow as tf
import time
import random

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base2_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(base2_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from model_local import *
from multiprocessing import Pool
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log_local', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=21, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=60, help='Batch Size during training [default: 48]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
# parser.add_argument('--data_dir', default='/media/jp/disk/temp/kitti/block_local', help='data directory')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
DATA_DIR = ROOT_DIR + '/data/block_local'

LOG_DIR = base2_DIR + '/log_local'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 4
len_channel = 8
Every_num_batch_show = 1000
num_load_sub_epoch = 100

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99



def progress(percent, width=50):
    '''进度打印功能'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %.2f%%' % (show_str, percent), end='')

#  load data
def load_data(train_data_folder_path, train_data_name_list):
    data_label = np.load(train_data_folder_path + '/' + train_data_name_list[0])
    for train_data_name in train_data_name_list[1:]:
        train_data_path = train_data_folder_path + '/' + train_data_name
        new_data_label = np.load(train_data_path)
        data_label = np.concatenate((data_label, new_data_label), axis=0)
        del new_data_label
        gc.collect()

    train_data = data_label[:, :, 0:-1]
    train_label = np.array(data_label[:, :, -1], dtype=np.int)
    del data_label
    gc.collect()
    return train_data, train_label


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            print('pred.shape')
            print(pred.shape)
            print('labels_pl')
            print(labels_pl.shape)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=20)
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print('ckpt.model_checkpoint_path')
            print(ckpt.model_checkpoint_path)
            start_epoch = int(ckpt.model_checkpoint_path.split('-')[1]) + 1
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('%s model restored...' % str(ckpt.model_checkpoint_path))
            log_string('%s model restored...' % str(ckpt.model_checkpoint_path))
        else:
            start_epoch = 0

        data_test = DATA_DIR + '/validation'
        file_name_list = os.listdir(data_test)
        path_test = data_test + '/' + os.listdir(data_test)[0]
        data_label_test = np.load(path_test)
        for file_name in file_name_list[1:]:
            t_path_test = data_test + '/' + file_name
            t_data_label_test = np.load(t_path_test)
            np.concatenate((data_label_test, t_data_label_test), axis = 0)


        train_data_folder_path = DATA_DIR + '/' + 'train'
        train_data_name_list = os.listdir(train_data_folder_path)
        np.random.shuffle(train_data_name_list)
        len_train_data = len(train_data_name_list)
        num_sub_epoch = len_train_data // num_load_sub_epoch
        if num_load_sub_epoch*num_sub_epoch < len_train_data:
            num_sub_epoch += 1

        for epoch in range(start_epoch, MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            # one training epoch
            for i_sub_epoch in range(num_sub_epoch):
                print('current sub epoch/num sub epoch: %d/%d' % (i_sub_epoch, num_sub_epoch-1))
                index_start = i_sub_epoch*num_load_sub_epoch
                index_end = (i_sub_epoch+1)*num_load_sub_epoch
                if i_sub_epoch == 0:
                    train_data, train_label = load_data(train_data_folder_path,
                                                        train_data_name_list[index_start:index_end])
                    rest_train_data, rest_train_label = train_one_sub_epoch(sess, ops, train_writer, train_data,
                                                                            train_label)
                    del train_data, train_label
                    gc.collect()
                elif (i_sub_epoch > 0) and (i_sub_epoch < num_sub_epoch - 1):
                    train_data, train_label = load_data(train_data_folder_path,
                                                        train_data_name_list[index_start:index_end])
                    train_data = np.concatenate((train_data, rest_train_data), axis=0)
                    train_label = np.concatenate((train_label, rest_train_label), axis=0)
                    del rest_train_data, rest_train_label
                    gc.collect()
                    rest_train_data, rest_train_label = train_one_sub_epoch(sess, ops, train_writer, train_data,
                                                                            train_label)
                    del train_data, train_label
                    gc.collect()
                else:
                    train_data, train_label = load_data(train_data_folder_path,
                                                        train_data_name_list[index_start:])
                    train_data = np.concatenate((train_data, rest_train_data), axis=0)
                    train_label = np.concatenate((train_label, rest_train_label), axis=0)
                    del rest_train_data, rest_train_label
                    gc.collect()
                    rest_train_data, rest_train_label = train_one_sub_epoch(sess, ops, train_writer, train_data,
                                                                            train_label)
                    del train_data, train_label, rest_train_data, rest_train_label
                    gc.collect()

            eval_one_epoch(sess, ops, test_writer, data_label_test)
            
            # Save the variables to disk.
            if epoch % 2 == 0:
                current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print(current_time)
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), epoch)
                log_string("Model saved in file: %s" % save_path)
                log_string('Current time is %s' % str(current_time))



def train_one_sub_epoch(sess, ops, train_writer, train_data, train_label):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        loss_path = '/home/jp/project/test/powernet/sem_seg/loss'
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)

        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

    if BATCH_SIZE*num_batches < file_size:
        start_idx = num_batches * BATCH_SIZE
        return current_data[start_idx:, :, :], current_label[start_idx:]
    else:
        return np.expand_dims(current_data[-1, :, :], axis=0), np.expand_dims(current_label[-1], axis=0)



        
def eval_one_epoch(sess, ops, test_writer,data_label_test):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    test_data = data_label_test[:, :, 0:len_channel]
    test_label = np.array(data_label_test[:, :, -1], dtype=np.int)
    
    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
    statistics_seen_class = []
    statistics_correct_class = []
    for i_class in range(len(total_seen_class)):
        if total_seen_class[i_class] != 0:
            statistics_seen_class.append(total_seen_class[i_class])
            statistics_correct_class.append(total_correct_class[i_class])
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(statistics_correct_class)/np.array(statistics_seen_class,dtype=np.float))))
         


if __name__ == "__main__":
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(current_time)
    log_string('Current time is %s' % str(current_time))
    train()
    LOG_FOUT.close()
