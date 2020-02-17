#!/usr/bin/env python
# coding: utf-8

# Federated Learning for Spatio-Temporal Predictions

import nest_asyncio
nest_asyncio.apply()

from __future__ import absolute_import, division, print_function

import collections
import functools
import os
import six
import time
import glob
from datetime import datetime
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

tf.compat.v1.enable_v2_behavior()

np.random.seed(0)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import datetime

import argparse

# # 0.  

parser = argparse.ArgumentParser(description='')
parser.add_argument('path', metavar='path', type=str, nargs='?',
                   help='path to data file')

PARTITION_SIZE = 30 


# # 1. Data Preprocessing

# ## 1.1 Samples (indexed by experiment)

samples_central = {}

for filename in glob.iglob(path):
    print(filename)
    exp_id = filename[-10:-4]
    count = 0
    samples_central.update({exp_id : {}})
    samples_central[exp_id].update({count : []})
    for line in open(filename):
        data = line.split(',')
        if len(data) == 7:
            x1 = float(data[3])
            x2 = float(data[4])
            samples_central[exp_id][count].append((x1, x2))
            # assumption that trajectories are 99 steps
            if len(samples_central[exp_id][count]) == 99:
                del samples_central[exp_id][count][98] # easier with even length
                count+=1
                samples_central[exp_id].update({count : []})
    # delete last empty trajectory
    del samples_central[exp_id][count]


# ## 1.2 Samples (indexed by experiment, client and collection time)

samples = {} #samples{ <exp_id> : {<rid>: { <traj_id> : { 'traj' : [], 'end' : <time_collected> }}}

for filename in glob.iglob(path):
    print(filename)
    exp_id = filename[-10:-4]
    last_sample_keys = {}
    samples.update({exp_id : {}})
    for line in open(filename):
        data = line.split(',')
        if len(data) == 7:
            rid = int(data[0])
            t = int(data[2])
            x1 = float(data[3])
            x2 = float(data[4])
            if rid in samples[exp_id].keys():
                last_key = last_sample_keys[rid]
                if(last_key not in samples[exp_id][rid]):
                    samples[exp_id][rid][last_key] = {'traj': [], 'end' : 0}
                samples[exp_id][rid][last_key]['traj'].append((x1, x2))
                samples[exp_id][rid][last_key]['end'] = t
                if len(samples[exp_id][rid][last_key]['traj']) == 99:
                    del samples[exp_id][rid][last_key]['traj'][98]
                    samples[exp_id][rid][last_key]['end'] = t
                    last_sample_keys[rid]+=1
            else:
                samples[exp_id].update({rid: {}})
                samples[exp_id][rid].update({0: {'traj': [(x1, x2)], 'end': 0}})
                last_sample_keys.update({rid: 0})
        else:
            last_key = last_sample_keys[rid] 
            if(last_key in samples[exp_id][rid] and len(samples[exp_id][rid][last_key]) != 0):
                last_sample_keys[rid]+=1


# ## 1.3 Communication graph 

# path = '../data/G_avoidance_20200131_204454.dat'

# neighbors = {}  #neigbors{ <exp_id> : {<rid>: { <t> : [<neighbors>]}}

# for filename in glob.iglob(path):
#     print(filename)    
#     exp_id = filename[-10:-4]
#     neighbors.update({exp_id : {}})
#     for line in open(filename):
#         data = line.split(',')
#         rid = int(data[0])
#         t = int(data[1])
#         nid = int(data[2])
#         if(t != 0):
#             if(rid not in neighbors[exp_id].keys()):
#                 neighbors[exp_id].update({rid: {}})
#             if(t not in neighbors[exp_id][rid].keys()):
#                 neighbors[exp_id][rid].update({t:[]})
#             neighbors[exp_id][rid][t].append(nid)


# # 2. Model Definition

# ## 2.1 Parameters

EMBEDDING_SIZE = 32
DROP_RATE = 0.3
NUM_OUTPUTS = 2

SEQ_LENGTH = 49
DIM_INPUT = 2

sample_shape = (SEQ_LENGTH, DIM_INPUT)


# ## 2.2 Simple LSTM 


def create_model():
    return tf.keras.models.Sequential([
    tf.keras.layers.Masking(mask_value=-10.,input_shape=sample_shape),
    tf.keras.layers.LSTM(EMBEDDING_SIZE,
                         return_sequences=True,
                         input_shape=sample_shape),
    tf.keras.layers.Dense(NUM_OUTPUTS)
    ])
#     tf.keras.layers.Dropout(DROP_RATE),


# # 3. Datasets 

TRAIN_RATIO = 0.8
VAL_RATIO =  1 - TRAIN_RATIO
PAST_HISTORY = 49
TIME_STEP = 0.1

BATCH_SIZE = 256
BUFFER_SIZE = 10000


# ## 3.1 Utilities


def _create_series_examples_from_dict(data_dict, start_index, end_index, history_size):
    data = []
    labels = []
    list_dataset = list(data_dict.values())
    array_dataset = np.asarray(list_dataset)
    for i in range(start_index, end_index):
        data.append(array_dataset[i][:history_size])
        labels.append(array_dataset[i][history_size:])
    data = np.asarray(data).reshape(end_index-start_index, history_size, 2)
    labels = np.asarray(labels).reshape(end_index-start_index, len(list_dataset[0]) - history_size , 2)
    
    return data, labels


def create_training_and_val_sets(data_dict, past_history=PAST_HISTORY, future_target=PAST_HISTORY):
    
    x_train = np.zeros((1, past_history, DIM_INPUT))
    y_train = np.zeros((1, future_target, DIM_INPUT))
    x_val = np.zeros((1, past_history, DIM_INPUT))
    y_val = np.zeros((1, future_target, DIM_INPUT))

    for v in data_dict.values():
        tot_samples = len(v)
        train_split = round(TRAIN_RATIO * tot_samples)
        x_train_tmp, y_train_tmp = _create_series_examples_from_dict(v, 0, train_split, past_history)
        x_val_tmp, y_val_tmp = _create_series_examples_from_dict(v, train_split, tot_samples, past_history)
        x_train = np.concatenate([x_train, x_train_tmp], axis=0)
        y_train = np.concatenate([y_train, y_train_tmp], axis=0)
        x_val = np.concatenate([x_val, x_val_tmp], axis=0)
        y_val = np.concatenate([y_val, y_val_tmp], axis=0)
        
    return x_train, x_val, y_train, y_val


def create_datasets(x_train, x_val, y_train, y_val):
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_set = train_set.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_set = val_set.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    return train_set, val_set


# ## 3.2 Centralized

x_train_C, x_val_C, y_train_C, y_val_C = create_training_and_val_sets(samples_central)


train_set_C, val_set_C = create_datasets(x_train_C, x_val_C, y_train_C, y_val_C)


# # 4. Training

# ## 4.1 Centralized 

# ### 4.1.1 Utilities

# Create new callback
class MyHistory(tf.keras.callbacks.Callback):
    """Adapted from https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L614"""

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.times = []
        self.history = {}
        self.start = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        delta = float((datetime.datetime.now() - self.start).total_seconds())
        self.times.append(delta)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


# ### 4.1.2 Parameters

EVALUATION_INTERVAL = 50
EPOCHS = 150  

# ### 4.1.3 Training loop

# Instantiate callback
myHistory = MyHistory()

simple_lstm = create_model()


logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

simple_lstm.compile(optimizer='SGD',
        loss='mean_squared_error')
simple_lstm.fit(train_set_C, epochs=EPOCHS,
              steps_per_epoch=EVALUATION_INTERVAL,
              validation_data=val_set_C, validation_steps=50,
              # callbacks=[tensorboard_callback, myHistory])
              callbacks=[myHistory])


# ### 4.1.4 Save Training History

filehandler = open('Centralized_' + path[-10:-4] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'wb') 
data_C = {'losses' : myHistory.history, 'times' : myHistory.times}
pickle.dump(data_C, filehandler)
filehandler.close()

# del data
# del samples_central
# del simple_lstm
# del myHistory


# ## 4.2 Federated Averaging in Server Setting

# ### 4.2.1 Parameters

EXP_DURATION = 100000
LOCAL_EPOCHS = 1 # 10 100
TRAIN_FRAC = 0.8
NUM_ROUNDS = 100
ROUND_DURATION = int(EXP_DURATION / NUM_ROUNDS)

TRAIN_RATIO = 0.8
VAL_RATIO = 1 - TRAIN_RATIO
PAST_HISTORY = 49
FUTURE_TARGET = 49

# ### 4.2.2 Utilities

def _create_series_examples_from_batch(dataset, start_index, end_index, history_size):
   data = []
   labels = []
   list_dataset = list(dataset)
   array_dataset = np.asarray(list_dataset)
   for i in range(start_index, end_index):
       data.append(array_dataset[i][:history_size])
       labels.append(array_dataset[i][history_size:])
       
   data = np.asarray(data).reshape(end_index-start_index, history_size, 2)
   labels = np.asarray(labels).reshape(end_index-start_index, len(list_dataset[0]) - history_size , 2)
   
   return data, labels


def create_training_and_val_batch(batch, past_history=PAST_HISTORY, future_target=PAST_HISTORY):
    
    x_train = np.zeros((1,PAST_HISTORY,2))
    y_train = np.zeros((1,FUTURE_TARGET,2))
    x_val = np.zeros((1,PAST_HISTORY,2))
    y_val = np.zeros((1,FUTURE_TARGET,2))
    for v in batch:
        tot_samples = len(v)
        train_split = round(TRAIN_RATIO * tot_samples)
        x_train_tmp, y_train_tmp = _create_series_examples_from_batch(v, 0, train_split, PAST_HISTORY)
        x_val_tmp, y_val_tmp = _create_series_examples_from_batch(v, train_split, tot_samples, PAST_HISTORY)
        x_train = np.concatenate([x_train, x_train_tmp], axis=0)
        y_train = np.concatenate([y_train, y_train_tmp], axis=0)
        x_val = np.concatenate([x_val, x_val_tmp], axis=0)
        y_val = np.concatenate([y_val, y_val_tmp], axis=0)
        
    return x_train, x_val, y_train, y_val


def create_datasets_FA(x_train, x_val, y_train, y_val):
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_set = train_set.cache().batch(PARTITION_SIZE).repeat()
    val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_set = val_set.cache().batch(PARTITION_SIZE).repeat()
    return train_set, val_set


def average_weights(weights):
    avg = np.zeros_like(trainable_weights[1][1])
    count = 0
    for k,v in weights.items():
        if(len(v) != 0):
            avg = np.add(avg, v)
            count += 1
    if(count != 0):
        avg = avg/count
    avg = np.squeeze(avg)
    return avg           


def weighted_average_weights(weights, num_samples):
    _sum = np.zeros_like(trainable_weights[1][1])
    count = 0
    for k,v in weights.items():
        if(len(v) != 0):
            _sum = np.add(_sum, np.multiply(num_samples[k], v))
            count += num_samples[k]
    if(count != 0):
        _sum = _sum/count
    avg = np.squeeze(_sum)
    return avg        


# ### 4.2.3 Rounds by global clock and data 

from tensorflow.keras.models import load_model

history = {}
local_lstm = create_model()

# For each experiment
for exp in samples.keys():
    
    history.update({exp : {}})
    
    # Per experiment settings 
    num_robots = len(samples[exp].keys())
    
    for i in range(1, num_robots + 1):
        history[exp].update({i : {}})

    # Set global clock to 0
    global_clock = 0
    round_num = 1
    
    # Initialize weights
    trainable_weights = {}
    arr_num_samples = {}
    w  = [v.numpy() for v in local_lstm.trainable_weights]
    w_list = [[w] for i in range(num_robots)]
    trainable_weights[0] = {}
    arr_num_samples[0] = {}
    trainable_weights[round_num] = {k : v for (k, v) in zip(range(1, num_robots + 1), w_list)}
    arr_num_samples[round_num] = {k : 1 for (k, v) in zip(range(1, num_robots + 1), w_list)}
    leftover_samples_buffer = {}
    
    # buffer of last data index of previous round for each robot
    last_idx_previous_round = np.zeros(num_robots, dtype=int)
    
    for t in range(ROUND_DURATION, EXP_DURATION, ROUND_DURATION):
        
        print("_______________________")
        print("Round ", round_num)
        print("_______________________")
        
        # One round for each robot: data collection, local training and global update
        for i in samples[exp].keys():
            
            print("Robot ", i)
            
            batch = []
            
            # Get data index at time of round
            cond = True
            current_idx =  last_idx_previous_round[i-1]
            while(cond):
                current_idx+=1
                if(samples[exp][i][current_idx]['end'] > t):
                    cond = False
            
            num_samples = current_idx - last_idx_previous_round[i-1]
            
            print("Num samples", num_samples)
            
             # Check that we have enough data collected to participate in the round
            if(num_samples >= PARTITION_SIZE):
                # Take extra data collected before end of round
                tmp = [samples[exp][i][j]['traj'] for j in range(last_idx_previous_round[i-1], current_idx)]
                batch.append(tmp)
                get_ipython().run_line_magic('xdel', 'tmp')
                # Take remaining data from previous round
                if(i in leftover_samples_buffer.keys() and len(leftover_samples_buffer[i]) > 0):
                    batch.append(leftover_samples_buffer[i])
                    leftover_samples_buffer[i] = []
            # Check that we have enough previous data      
            elif(i in leftover_samples_buffer.keys() 
                 and (len(leftover_samples_buffer[i]) + num_samples) >= PARTITION_SIZE): 
                batch.append(leftover_samples_buffer[i])
                leftover_samples_buffer[i] = []
            else:
                if(i not in leftover_samples_buffer.keys()):
                    leftover_samples_buffer.update({i: []})
                leftover_samples_buffer[i] = [samples[exp][i][j]['traj'] for j in range(last_idx_previous_round[i-1], current_idx)]
                last_idx_previous_round[i-1] = current_idx
                print('Robot not participating', i)
                continue
            last_idx_previous_round[i-1] = current_idx
            
            # Get weights
            current_weights = weighted_average_weights(trainable_weights[round_num], arr_num_samples[round_num])
#             current_weights = average_weights(trainable_weights[round_num])
            
            # Perform local training
            # Create datasets
            x_train_FA, x_val_FA, y_train_FA, y_val_FA = create_training_and_val_batch(batch)
            train_batch, val_batch = create_datasets_FA(x_train_FA, x_val_FA, y_train_FA, y_val_FA)
            # Clone simple_lstm and initialize it with newest weights
            local_lstm = tf.keras.models.load_model('lstm.h5', compile=False)
            keras_model_clone = tf.keras.models.clone_model(local_lstm)
            keras_model_clone.compile(optimizer='SGD', loss='mean_absolute_error')
            keras_model_clone.set_weights(current_weights)
            robot_history = keras_model_clone.fit(train_batch, epochs=LOCAL_EPOCHS,
              steps_per_epoch=len(x_train_FA),
              validation_data=val_batch, 
              validation_steps=len(x_val_FA))#,
#              callbacks=[l_history])
            # Write weights 
            if((round_num+1) not in trainable_weights.keys()):
                trainable_weights.update({(round_num+1): {}})
                arr_num_samples.update({(round_num+1): {}})
            trainable_weights[round_num+1].update({i: keras_model_clone.get_weights()})
            arr_num_samples[round_num+1].update({i: num_samples})
            # Write metrics
            history[exp][i].update({round_num : { 'losses': robot_history.history, 'num_samples': num_samples}})
            del current_weights
            del robot_history
            del train_batch
            del val_batch
            del batch
            del x_train_FA, y_train_FA, x_val_FA, y_val_FA
            del keras_model_clone
            del local_lstm
            tf.keras.backend.clear_session()
        round_num+=1

# ### 4.2.4 Save training data

data_FA = {}
for exp in history:
    data_FA.update({exp : {} })
    for round_num in range(1, NUM_ROUNDS):
        data_FA[exp].update({round_num: {}})
        for robot in history[exp]:
            if (round_num in history[exp][robot].keys()):
                data_FA[exp][round_num].update({robot : {}})
                data_FA[exp][round_num][robot] = {'losses' : history[exp][robot][round_num]['losses'], 'num_samples' : history[exp][robot][round_num]['num_samples']}

filehandler = open('FA_history_' + str(num_robots) + '_' + str(BATCH_SIZE) + '_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'wb') 
pickle.dump(data_FA, filehandler)
filehandler.close()


# ## 4.3 Serverless Federated Averaging


# class VSElem():
#     def __init__(self, data, t, i):
#         self.data = data
#         self.timestamp = t
#         self.rid = i

# class VirtualStigmery():
#     def __init__(self, vid):
#         self.id = vid
#         self.dict = {}
        
#     def put():
    
    
#     def __call__():

# 1) vstig : key = round_num, data = < num_writes, weights_average > and store last 
#    round participated in (but conflicts problematic)
# 2) vstig : key = round_num, data = < rid_list, weights > and last round participated in (but increased load)
# 3) vstig : id = round_num, key = rid, data = < weights > and last round participated in (but delay in getting
# the newest round? or possible to start a wrong round)
# Furthermore, to avoid conflicts, do we stop at exactly a certain number of participants for the weight average?
# one vstig (1-2) or one per round (3) ? or CRDT after all ?
        

# Un-synch clocks by random integer (todo)
# time_offsets = np.random.randint(0, 10, size=(num_robots,))


# # 5. Inference and Evaluation

# ## 5.0 Datasets


# samples{ <exp_id> : {<rid>: { <traj_id> : { 'traj' : [], 'end' : <time_collected> }}}

plt.figure(figsize=(12, 12))
for v in samples.keys():
    colors = cm.rainbow(np.linspace(0, 1, len(samples[v].keys())))
    cnt = 0
    for r in samples[v].keys():        
        for i in samples[v][r].keys():
            traj = samples[v][r][i]['traj']
            if (len(traj) > 0):
                x, y = zip(*traj)
                ax, = plt.plot(x, y, '-.', color=colors[cnt])
        cnt += 1
        ax.set_label(str(cnt))

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",fontsize=16)
plt.xlabel('x (m)', FontSize=16)
plt.ylabel('y (m)', FontSize=16)
plt.save()


# ## 5.1 Validation Loss Curve

# ### 5.1.1 Pre-Processing

avg_val_loss = np.zeros(NUM_ROUNDS)

for exp in data_FA.keys():
    for round_num in range(1, NUM_ROUNDS):
        count = 0.
        sum_loss = 0.
        for robot in data_FA[exp].keys():
            if (robot in data_FA[exp][round_num].keys()):
                num_samples = data_FA[exp][round_num][robot]['num_samples']
                count += num_samples
                sum_loss += num_samples * data_FA[exp][round_num][robot]['losses']['val_loss'][0]
        if(count != 0):
            avg_val_loss[round_num - 1] = sum_loss/count
        

# ### 5.1.2 Plot

fig = plt.figure(figsize=(12, 6))
ax = plt.gca()
loss_C = data_C['losses']['val_loss']
plt.plot(loss_C[0:100], 'bv-', label='Centralized')
plt.plot(avg_val_loss[avg_val_loss!=0], 'r*-', label='FA')
plt.legend(loc='upper right', fontsize=16)
plt.xlabel('Epochs (Iterations or Communication rounds)', FontSize=16)
plt.ylabel('Validation Loss', FontSize=16)
plt.save()

# ## 5.2 Runtime Curve

fig = plt.figure(figsize=(12, 6))
runtimes_C_s = data_C['times']
epochs_C = data_C.epoch[1:]
plt.plot(epochs_C, runtimes_C_s, 'b.-', label='Centralized')
plt.legend(loc='upper right')
plt.xlabel('Epochs (Iterations or Communication rounds)')
plt.ylabel('Duration (s)')
plt.save()

# # ## 5.3 Predictions

# from mpl_toolkits.mplot3d import Axes3D

# def create_time_steps(length):
#     return list(range(-length, 0))

# def multi_step_plot(history, true_future, prediction):

#     history=np.array(history[history!=-10]).reshape([-1, 2])
#     prediction=np.array(prediction[true_future!=-10]).reshape([-1,2])
#     true_future=np.array(true_future[true_future!=-10]).reshape([-1, 2])
    
#     fig = plt.figure(figsize=(12, 6))
#     ax = plt.gca()
#     ax.set_xlim([-10,10])
#     ax.set_ylim([-2,2])
    
#     num_in = create_time_steps(len(history))
#     num_out = len(true_future)
#     plt.plot(num_in, history[:, 0], 'b', label='History_x')
#     plt.plot(num_in, history[:, 1], 'b', label='History_y')
#     plt.plot(np.arange(num_out), true_future, 'bo-',
#            label='True Future')
#     if prediction.any():
#         plt.plot(np.arange(num_out), prediction, 'ro',
#                  label='Predicted Future')
#         plt.legend(loc='upper left')
#         plt.show()


# def multi_step_plot_3D(history, true_future, prediction):
#     history=np.array(history[history!=-10]).reshape([-1, 2])
#     prediction=np.array(prediction[true_future!=-10]).reshape([-1,2])
#     true_future=np.array(true_future[true_future!=-10]).reshape([-1, 2])
    
#     fig = plt.figure(figsize=(12, 6))
#     ax = Axes3D(fig)
#     ax.set_xlim([-2,2])
#     ax.set_ylim([-2,2])
#     ax.set_zlim([-10,10])
#     num_in = create_time_steps(len(history))
#     num_out = len(true_future)
#     ax.plot3D(np.array(history[:, 0]), np.array(history[:, 1]), num_in, 'bo-', label='History')
#     ax.plot3D(np.array(true_future[:,0]), np.array(true_future[:,1]), np.arange(num_out), 'bo',
#            label='True Future')
#     if prediction.any():
#         ax.plot3D(np.array(prediction[:,0]), np.array(prediction[:,1]), np.arange(num_out), 'ro',
#                  label='Predicted Future')
#         plt.legend(loc='upper left')
#         plt.show()


# for x, y in val_set.take(3):
#     multi_step_plot_3D(history=x[0], true_future=y[0], prediction=simple_lstm_model.predict(x)[0])
#     multi_step_plot(history=x[0], true_future=y[0], prediction=simple_lstm_model.predict(x)[0])