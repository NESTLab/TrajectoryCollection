#!/usr/bin/env python
# coding: utf-8

# Federated Learning for Spatio-Temporal Predictions

from __future__ import absolute_import, division, print_function

import collections
import functools
import os
import time
import glob
from datetime import datetime
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.compat.v1.enable_v2_behavior()
tf.compat.v1.enable_eager_execution()

np.random.seed(0)

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# matplotlib.use("Pdf")

import datetime

from barrier import Barrier
from vsweights import VSWeights

import sys

# # 0. Parse Script Arguments

BEHAVIOR=str(sys.argv[1])

path_string=BEHAVIOR.split('_')[0]
path=path_string+'*.dat'
PERCENT_QUORUM = float(sys.argv[2])
QUOTA = int(sys.argv[3])


# Global constants and variables

# Neural Network parameters
EMBEDDING_SIZE = 32
NUM_OUTPUTS = 2
SEQ_LENGTH = 32
DIM_INPUT = 2
DROPOUT=0.0
sample_shape = (SEQ_LENGTH, DIM_INPUT)

# Generic training parameters
TRAIN_RATIO = 0.8
VAL_RATIO =  1 - TRAIN_RATIO
PAST_HISTORY = 32
FUTURE_TARGET = 48

# Centralized training parameters
BATCH_SIZE_C = 256
BUFFER_SIZE_C = 10000
EVALUATION_INTERVAL_C = 50
EPOCHS_C = 150  #going to be overridden
VALIDATION_STEPS_C = 50

# FL training parameters
# Generic
EXP_DURATION = 50000
LOCAL_EPOCHS = 1

samples_central = {}
samples = {} # samples{ <exp_id> : {<rid>: { <traj_id> : { 'traj' : [], 'end' : <time_collected> }}}
history_FL = {}
summary_FL = {}
history_DFL = {}
summary_DFL = {}
neighbors = {}  #neigbors{ <exp_id> : {<t>: { <rid> : [<neighbors>]}}

# # 1. Data Preprocessing

# ## 1.2 Samples (indexed by experiment, client and collection time)

for filename in glob.iglob(path):
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
                if len(samples[exp_id][rid][last_key]['traj']) == 100:
                    # del samples[exp_id][rid][last_key]['traj'][98]
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
    num_robots = len(samples[exp_id].keys())
    # Communication graph
    filename = "G_" + filename[:]
    print(filename)
    exp_id = filename[-10:-4]
    neighbors.update({exp_id : {}})
    for line in open(filename):
        data = line.split(',')
        rid = int(data[0])
        t = int(data[1])
        nid = int(data[2])
        if(t != 0):
            if(t not in neighbors[exp_id].keys()):
                print(t)
                neighbors[exp_id].update({t: {}})
                for i in range(1, num_robots+1):
                    neighbors[exp_id][t].update({i:[]})
            neighbors[exp_id][t][rid].append(nid)

# # 2. Model Definition

# ## 2.1 Simple LSTM

def create_model():
    return tf.keras.models.Sequential([
    tf.keras.layers.Masking(mask_value=-10.,input_shape=sample_shape),
    tf.keras.layers.LSTM(EMBEDDING_SIZE,
                         return_sequences=False,
                         input_shape=sample_shape),
    tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(FUTURE_TARGET*NUM_OUTPUTS),
    tf.keras.layers.Reshape([FUTURE_TARGET,NUM_OUTPUTS])
    ])
sample_model=create_model()
sample_model.summary()

# # 5. Distributed Federated Learning

# ## 4.1 Utilities

def _create_series_examples_from_batch(dataset, start_index, end_index, history_size, target_size):
   data = []
   labels = []
   list_dataset = list(dataset)
   array_dataset = np.asarray(list_dataset)
   for i in range(start_index, end_index):
       data.append(array_dataset[i][:history_size])
       labels.append(array_dataset[i][history_size:history_size+target_size])

   data = np.asarray(data).reshape(end_index-start_index, history_size, 2)
   labels = np.asarray(labels).reshape(end_index-start_index, target_size , 2)
   return data, labels


def create_training_and_val_batch(batch, past_history=PAST_HISTORY, future_target=FUTURE_TARGET):

    x_train = np.zeros((1,PAST_HISTORY,2))
    y_train = np.zeros((1,FUTURE_TARGET,2))
    x_val = np.zeros((1,PAST_HISTORY,2))
    y_val = np.zeros((1,FUTURE_TARGET,2))
    for v in batch:
        tot_samples = len(v)
        train_split = round(TRAIN_RATIO * tot_samples)
        x_train_tmp, y_train_tmp = _create_series_examples_from_batch(v, 0, train_split, past_history,future_target)
        x_val_tmp, y_val_tmp = _create_series_examples_from_batch(v, train_split, tot_samples, past_history,future_target)
        x_train = np.concatenate([x_train, x_train_tmp], axis=0)
        y_train = np.concatenate([y_train, y_train_tmp], axis=0)
        x_val = np.concatenate([x_val, x_val_tmp], axis=0)
        y_val = np.concatenate([y_val, y_val_tmp], axis=0)

    return x_train, x_val, y_train, y_val


def create_datasets_FL(x_train, x_val, y_train, y_val):
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_set = train_set.cache().batch(QUOTA).repeat()
    val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_set = val_set.cache().batch(QUOTA).repeat()
    return train_set, val_set


def average_weights(weights):
    avg = np.zeros((1,5))#np.zeros_like()#trainable_weights[1][1])
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
    _sum = np.zeros((1,5))#trainable_weights[1][1])
    count = 0
    for k,v in weights.items():
        if(len(v) != 0):
            _sum = np.add(_sum, np.multiply(num_samples[k], v))
            count += num_samples[k]
    if(count != 0):
        _sum = _sum/count
    avg = np.squeeze(_sum)
    return avg

# ## 5.1 Utilities
#
# def train_locally(rid, batch, current_weights):
#     # Perform local training
#     # Create datasets
#     x_train_DFL, x_val_DFL, y_train_DFL, y_val_DFL = create_training_and_val_batch(batch)
#     train_batch, val_batch = create_datasets_FL(x_train_DFL, x_val_DFL, y_train_DFL, y_val_DFL)
#     # Clone simple_lstm and initialize it with newest weights
#     # local_lstm = tf.keras.models.load_model('lstm.h5', compile=False)
#     keras_model_clone = tf.keras.models.clone_model(sample_model)
#     keras_model_clone.compile(optimizer='rmsprop', loss='mean_squared_error')
#     keras_model_clone.set_weights(current_weights)
#     start = datetime.datetime.now()
#     robot_history = keras_model_clone.fit(train_batch, epochs=LOCAL_EPOCHS,
#       steps_per_epoch=len(x_train_DFL),
#       validation_data=val_batch,
#       validation_steps=len(x_val_DFL))
#     stop = datetime.datetime.now()
#     # Write weights
#     trainable_weights = keras_model_clone.get_weights()
#     # Compute learning duration in timesteps (100ms)
#     duration = round((stop - start).total_seconds() * 10)
#     # Write metrics (GLOBAL VARIABLE)
#     history_DFL[exp][rid].update({round_num : { 'losses': robot_history.history,
#                                                 'num_samples': len(x_train_DFL) + len(x_val_DFL),
#                                                 'time': duration} })
#     # Not sure if needed
#     tf.keras.backend.clear_session()
#     return trainable_weights, duration

# ## 4.2 Training loop

local_lstm = create_model()

# For each experiment
for exp in samples.keys():

    history_FL.update({exp : {}})
    summary_FL.update({exp : {}})

    summary_FL[exp].update({'num_participants' : {}})
    summary_FL[exp].update({'round_time' : {}})

    # Per experiment settings
    num_robots = len(samples[exp].keys())

    for i in range(1, num_robots + 1):
        history_FL[exp].update({i : {}})

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

    # buffer of last data index of previous round for each robot
    last_idx_previous_round = np.zeros(num_robots, dtype=int)

    t = 0

    while (t < EXP_DURATION - 10000):

        num_participants = 0

        # Find time of round start (resume from indices at previous round)
        current_indices = last_idx_previous_round
        tmp_idx = np.add(current_indices, QUOTA)
        min_learners = round(PERCENT_QUORUM * num_robots)
        times_at_quota = []
        for i in samples[exp].keys():
            if tmp_idx[i-1] in samples[exp][i].keys():
                times_at_quota.append(samples[exp][i][tmp_idx[i-1]]['end'])
        times_at_quota.sort()
        t = times_at_quota[min_learners - 1]

        summary_FL[exp]['round_time'].update({round_num : t})

        print("FL round ", round_num, "at t ", t)

        # One round for each robot: data collection, local training and global update
        for i in samples[exp].keys():

            batch = []

            current_idx =  last_idx_previous_round[i-1]
            while(samples[exp][i][current_idx]['end'] <= t):
                current_idx+=1
                if(current_idx not in samples[exp][i].keys()):
                    current_idx -=1
                    break

            num_samples = current_idx - last_idx_previous_round[i-1]

             # Check that we have enough data collected to participate in the round
            if(num_samples >= QUOTA):
                # Take extra data collected before end of round
                tmp = [samples[exp][i][j]['traj'] for j in range(last_idx_previous_round[i-1], current_idx)]
                batch.append(tmp)
            else:
                continue
            num_participants += 1
            last_idx_previous_round[i-1] = current_idx

            # Get weights
            current_weights = weighted_average_weights(trainable_weights[round_num], arr_num_samples[round_num])

            # Perform local training
            # Create datasets
            x_train_FL, x_val_FL, y_train_FL, y_val_FL = create_training_and_val_batch(batch)
            train_batch, val_batch = create_datasets_FL(x_train_FL, x_val_FL, y_train_FL, y_val_FL)
            # Clone simple_lstm and initialize it with newest weights
            # local_lstm = tf.keras.models.load_model('lstm.h5', compile=False)
            local_lstm=create_model()
            keras_model_clone = tf.keras.models.clone_model(local_lstm)
            keras_model_clone.compile(optimizer='rmsprop', loss='mean_squared_error')
            keras_model_clone.set_weights(current_weights)
            start = datetime.datetime.now()
            robot_history = keras_model_clone.fit(train_batch, epochs=LOCAL_EPOCHS,
              steps_per_epoch=len(x_train_FL),
              validation_data=val_batch,
              validation_steps=len(x_val_FL))
            stop = datetime.datetime.now()
            # Compute learning duration in timesteps (100ms)
            duration = round((stop - start).total_seconds() * 10)
            # Write weights
            if((round_num+1) not in trainable_weights.keys()):
                trainable_weights.update({(round_num+1): {}})
                arr_num_samples.update({(round_num+1): {}})
            trainable_weights[round_num+1].update({i: keras_model_clone.get_weights()})
            arr_num_samples[round_num+1].update({i: num_samples})
            # Write metrics
            history_FL[exp][i].update({round_num : { 'losses': robot_history.history, 'num_samples': num_samples,
                                                     'time': duration}})
            # del current_weights
            del robot_history
            del train_batch
            del val_batch
            del batch
            del x_train_FL, y_train_FL, x_val_FL, y_val_FL
            del keras_model_clone
            del local_lstm
            tf.keras.backend.clear_session()
        summary_FL[exp]['num_participants'].update({round_num : num_participants})

        if (num_participants == 0):
            trainable_weights[round_num+1] = trainable_weights[round_num]
            arr_num_samples[round_num+1] = trainable_weights[round_num]
            break
        round_num+=1
    summary_FL[exp].update({'last_weights' : current_weights})

# ### 4.3 Save training data

data_FL = {}
for exp in history_FL:
    data_FL.update({exp : {} })
    for r in range(1, round_num):
        data_FL[exp].update({r: {}})
        for robot in history_FL[exp]:
            if (r in history_FL[exp][robot].keys()):
                data_FL[exp][r].update({robot : {}})
                data_FL[exp][r][robot] = {'losses' : history_FL[exp][robot][r]['losses'],
                                                    'num_samples' : history_FL[exp][robot][r]['num_samples'],
                                                    'time': history_FL[exp][robot][r]['time']}

fname=BEHAVIOR+'_' + filename[-10:-4] + "_"  + str(PERCENT_QUORUM).replace(".","") + '_' + str(QUOTA) +"_" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

filehandler = open('FL_history_' + fname + '.pkl', 'wb')
pickle.dump(data_FL, filehandler)
filehandler.close()

filehandler = open('FL_summary_' + fname + '.pkl', 'wb')
pickle.dump(summary_FL, filehandler)
filehandler.close()
