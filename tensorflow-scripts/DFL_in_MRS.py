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

path = sys.argv[1]
PERCENT_QUORUM = float(sys.argv[2])
QUOTA = int(sys.argv[3])

# Global constants and variables

# Neural Network parameters
EMBEDDING_SIZE = 32
NUM_OUTPUTS = 2
SEQ_LENGTH = 49
DIM_INPUT = 2
sample_shape = (SEQ_LENGTH, DIM_INPUT)

# Generic training parameters
TRAIN_RATIO = 0.8
VAL_RATIO =  1 - TRAIN_RATIO
PAST_HISTORY = 49
FUTURE_TARGET = 49

# Centralized training parameters
BATCH_SIZE_C = 256
BUFFER_SIZE_C = 10000
EVALUATION_INTERVAL_C = 50
EPOCHS_C = 150  #going to be overridden
VALIDATION_STEPS_C = 50

# FL training parameters
# Generic
EXP_DURATION = 100000
LOCAL_EPOCHS = 1

samples_central = {}
samples = {} # samples{ <exp_id> : {<rid>: { <traj_id> : { 'traj' : [], 'end' : <time_collected> }}}
history_FL = {}
summary_FL = {}
history_DFL = {}
summary_DFL = {}
neighbors = {}  #neigbors{ <exp_id> : {<t>: { <rid> : [<neighbors>]}}

# # 1. Data Preprocessing

# ## 1.1 Samples (indexed by experiment)

for filename in glob.iglob(path):
    exp_id = filename[-10:-4]
    print(filename)
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
    num_robots = len(samples[exp_id].keys())
    # Communication graph 
    filename = filename[:8] + "G_" + filename[8:]
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
                         return_sequences=True,
                         input_shape=sample_shape),
    tf.keras.layers.Dense(NUM_OUTPUTS)
    ])

# # 5. Distributed Federated Learning

# ## 4.1 Utilities

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

def train_locally(rid, batch, current_weights):
    # Perform local training
    # Create datasets
    x_train_DFL, x_val_DFL, y_train_DFL, y_val_DFL = create_training_and_val_batch(batch)
    train_batch, val_batch = create_datasets_FL(x_train_DFL, x_val_DFL, y_train_DFL, y_val_DFL)
    # Clone simple_lstm and initialize it with newest weights
    local_lstm = tf.keras.models.load_model('lstm.h5', compile=False)
    keras_model_clone = tf.keras.models.clone_model(local_lstm)
    keras_model_clone.compile(optimizer='SGD', loss='mean_absolute_error')
    keras_model_clone.set_weights(current_weights)
    start = datetime.datetime.now()
    robot_history = keras_model_clone.fit(train_batch, epochs=LOCAL_EPOCHS,
      steps_per_epoch=len(x_train_DFL),
      validation_data=val_batch, 
      validation_steps=len(x_val_DFL))
    stop = datetime.datetime.now()
    # Write weights 
    trainable_weights = keras_model_clone.get_weights()
    # Compute learning duration in timesteps (100ms)
    duration = round((stop - start).total_seconds() * 10)
    # Write metrics (GLOBAL VARIABLE)
    history_DFL[exp][rid].update({round_num : { 'losses': robot_history.history, 
                                                'num_samples': len(x_train_DFL) + len(x_val_DFL),
                                                'time': duration} })
    # Not sure if needed
    tf.keras.backend.clear_session()
    return trainable_weights, duration

# ## 5.2 Training Loop 

local_lstm = create_model()

# For each experiment
for exp in samples.keys():
    
    history_DFL.update({exp : {}})
    summary_DFL.update({exp : {}})

    summary_DFL[exp].update({'entry_barrier_start' : []})
    summary_DFL[exp].update({'entry_barrier_end' : []})
    summary_DFL[exp].update({'exit_barrier_end' : []})
    
    summary_DFL[exp].update({'num_participants' : {}})
    
    # Per experiment settings 
    num_robots = len(samples[exp].keys())
    QUORUM = round(PERCENT_QUORUM * num_robots)
    
    # Initialize
    for i in range(1, num_robots + 1):
        history_DFL[exp].update({i : {}})

    # Set global clock to 0
    global_clock = 0
    round_num = 1
    
    # Initialize weights
    w_ini  = [v.numpy() for v in local_lstm.trainable_weights]
    w_list = [[w_ini] for i in range(num_robots)]
    w = VSWeights()
    w.ws.rs = {k: {k : (v,0,k) for (k, v) in zip(range(1, num_robots + 1), w_list)} for k in range(1,num_robots + 1)}
    w.ss.rs = {k: {k : (1,0,k) for (k, v) in zip(range(1, num_robots + 1), w_list)} for k in range(1, num_robots + 1) }

    
    # buffer of last data index of previous round for each robot
    last_idx_previous_round = np.zeros(num_robots, dtype=int)
    
    t = 0
    
    # Loop through simulation
    while(t < EXP_DURATION - 1000):
        
        # BARRIER
        
        # Find time of barrier start (resume from indices at previous barrier)
        current_indices = last_idx_previous_round
        tmp_idx = np.add(current_indices, QUOTA)
        # ready = {samples[exp][i][tmp_idx[i-1]]['end'] : i if tmp_idx[i-1] in samples[exp][i].keys() for i in samples[exp].keys()}
        ready = {}
        for i in samples[exp].keys():
            if tmp_idx[i-1] in samples[exp][i].keys():
                if samples[exp][i][tmp_idx[i-1]]['end'] not in ready:
                    ready.update({samples[exp][i][tmp_idx[i-1]]['end'] : [i]})
                else:
                    ready[samples[exp][i][tmp_idx[i-1]]['end']].append(i)
        # Find first robot to get enough data to trigger barrier
        t = min(list(ready.keys()))    
            
        b = Barrier()
        
        while(not b.quorum(QUORUM) and t < EXP_DURATION):
            b.update(neighbors[exp][t])
            if t in ready:
                for r in ready[t]:
                    b.put(r)
            t = t + 1
    
        summary_DFL[exp]['entry_barrier_end'].append(t)
    
        participants = b.ready()
        summary_DFL[exp]['num_participants'].update({round_num : len(participants)})
    
        # LEARNING ROUND
        
        print("DFL round ", round_num, "at t ", t)

        current_weights = weighted_average_weights(w.weights(), w.samples()) 
        
        round_data = {}        
        t = t + 1
        w = VSWeights()
        b = Barrier()
        
        # Mark non-participating robots as ready
        for rid in samples[exp].keys():
            if(rid not in participants):
                b.put(rid)
        
        # Pre-compute learning step and timestamp events
        for rid in participants:
            
            if (rid not in history_DFL[exp].keys()):
                history_DFl[exp].update({rid : {}})
                
            batch = []
            # Get data index at beginning of round
            current_idx = last_idx_previous_round[rid-1]
            while(samples[exp][rid][current_idx]['end'] <= t):
                current_idx+=1
                if current_idx not in samples[exp_id][rid].keys():
                    current_idx -= 1
                    break

            num_samples = current_idx - last_idx_previous_round[rid-1]
            

            # Take data collected 
            tmp = [samples[exp][rid][i]['traj'] for i in range(last_idx_previous_round[rid-1], current_idx)]
            batch.append(tmp)

            if len(batch[0]) < QUOTA:
                print("error in num_samples")
                print("num_samples ", num_samples)
                print(len(batch[0]))
                round_data.update({rid: { 't' : t , 'w' : current_weights, 'n' : num_samples }})
                last_idx_previous_round[rid-1] = current_idx
                continue
            
            last_idx_previous_round[rid-1] = current_idx
            
            weights, duration = train_locally(rid, batch, current_weights)
            round_data.update({rid: { 't' : t + duration, 'w' : weights, 'n' : num_samples }})
            
        while(not b.quorum(num_robots) and t < EXP_DURATION):
            b.update(neighbors[exp][t])
            for rid in participants:
                if(t == round_data[rid]['t']):
                    b.put(rid)
                    w.put_weights(rid, round_data[rid]['w'])
                    w.put_samples(rid, round_data[rid]['n'])
            t = t + 1
        
        summary_DFL[exp]['exit_barrier_end'].append(t)
        round_num = round_num + 1

# ### 4.3 Save training data

data_DFL = {}
for exp in history_DFL:
    data_DFL.update({exp : {} })
    for r in range(1, round_num):
        data_DFL[exp].update({r: {}})
        for robot in history_DFL[exp]:
            if (r in history_DFL[exp][robot].keys()):
                data_DFL[exp][r].update({robot : {}})
                data_DFL[exp][r][robot] = {'losses' : history_DFL[exp][robot][r]['losses'], 
                                                    'num_samples' : history_DFL[exp][robot][r]['num_samples'],
                                                    'time': history_DFL[exp][robot][r]['time']}

filehandler = open('DFL_history_' + filename[-10:-4] + '_' + str(QUOTA) + '_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'wb') 
pickle.dump(data_DFL, filehandler)
filehandler.close()

filehandler = open('DFL_summary_' + filename[-10:-4] + '_' + str(QUOTA) + '_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'wb') 
pickle.dump(summary_DFL, filehandler)
filehandler.close()

# Overwrite global constant

EPOCHS_C = round_num

# # 3. Centralized 

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
    train_set = train_set.cache().shuffle(BUFFER_SIZE_C).batch(BATCH_SIZE_C).repeat()
    val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_set = val_set.cache().shuffle(BUFFER_SIZE_C).batch(BATCH_SIZE_C).repeat()
    return train_set, val_set

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


# ## 3.2 Datasets

x_train_C, x_val_C, y_train_C, y_val_C = create_training_and_val_sets(samples_central)


train_set_C, val_set_C = create_datasets(x_train_C, x_val_C, y_train_C, y_val_C)

# # 3.3. Training loop

# Instantiate callback
myHistory = MyHistory()

simple_lstm = create_model()

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

simple_lstm.compile(optimizer='SGD',
        loss='mean_squared_error')
simple_lstm.fit(train_set_C, epochs=EPOCHS_C,
              steps_per_epoch=EVALUATION_INTERVAL_C,
              validation_data=val_set_C, validation_steps=VALIDATION_STEPS_C,
              callbacks=[myHistory])

# ##3.4 Save Training History

filehandler = open('Centralized_' + filename[-10:-4] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'wb') 
data_C = {'losses' : myHistory.history, 'times' : myHistory.times}
pickle.dump(data_C, filehandler)
filehandler.close()


# # 4 Federated Averaging in Server Setting


# # ## 4.2 Training loop 

# local_lstm = create_model()

# # For each experiment
# for exp in samples.keys():
    
#     history_FL.update({exp : {}})
#     summary_FL.update({exp : {}})
    
#     summary_FL[exp].update({'num_participants' : {}})

#     # Per experiment settings 
#     num_robots = len(samples[exp].keys())
    
#     for i in range(1, num_robots + 1):
#         history_FL[exp].update({i : {}})

#     round_num = 1
    
#     # Initialize weights
#     trainable_weights = {}
#     arr_num_samples = {}
#     w  = [v.numpy() for v in local_lstm.trainable_weights]
#     w_list = [[w] for i in range(num_robots)]
#     trainable_weights[0] = {}
#     arr_num_samples[0] = {}
#     trainable_weights[round_num] = {k : v for (k, v) in zip(range(1, num_robots + 1), w_list)}
#     arr_num_samples[round_num] = {k : 1 for (k, v) in zip(range(1, num_robots + 1), w_list)}

#     # buffer of last data index of previous round for each robot
#     last_idx_previous_round = np.zeros(num_robots, dtype=int)

#     t = 0
    
#     while (t < EXP_DURATION - 1000):
        
#         num_participants = 0

#         # Find time of round start (resume from indices at previous round)
#         current_indices = last_idx_previous_round
#         tmp_idx = np.add(current_indices, QUOTA)
#         min_learners = round(PERCENT_QUORUM * num_robots)
#         times_at_quota = []
#         for i in samples[exp].keys():
#             if tmp_idx[i-1] in samples[exp][i].keys():
#                 times_at_quota.append(samples[exp][i][tmp_idx[i-1]]['end'])
#         times_at_quota.sort()
#         t = times_at_quota[min_learners - 1]

#         print("FL round ", round_num, "at t ", t)

#         # One round for each robot: data collection, local training and global update
#         for i in samples[exp].keys():
            
#             batch = []
            
#             current_idx =  last_idx_previous_round[i-1]
#             while(samples[exp][i][current_idx]['end'] <= t):
#                 current_idx+=1
            
#             num_samples = current_idx - last_idx_previous_round[i-1]
            
#              # Check that we have enough data collected to participate in the round
#             if(num_samples >= QUOTA):
#                 # Take extra data collected before end of round
#                 tmp = [samples[exp][i][j]['traj'] for j in range(last_idx_previous_round[i-1], current_idx)]
#                 batch.append(tmp)
#             else:
#                 continue
#             num_participants += 1
#             last_idx_previous_round[i-1] = current_idx
            
#             # Get weights
#             current_weights = weighted_average_weights(trainable_weights[round_num], arr_num_samples[round_num])
            
#             # Perform local training
#             # Create datasets
#             x_train_FL, x_val_FL, y_train_FL, y_val_FL = create_training_and_val_batch(batch)
#             train_batch, val_batch = create_datasets_FL(x_train_FL, x_val_FL, y_train_FL, y_val_FL)
#             # Clone simple_lstm and initialize it with newest weights
#             local_lstm = tf.keras.models.load_model('lstm.h5', compile=False)
#             keras_model_clone = tf.keras.models.clone_model(local_lstm)
#             keras_model_clone.compile(optimizer='SGD', loss='mean_absolute_error')
#             keras_model_clone.set_weights(current_weights)
#             start = datetime.datetime.now()
#             robot_history = keras_model_clone.fit(train_batch, epochs=LOCAL_EPOCHS,
#               steps_per_epoch=len(x_train_FL),
#               validation_data=val_batch, 
#               validation_steps=len(x_val_FL))
#             stop = datetime.datetime.now()
#             # Compute learning duration in timesteps (100ms)
#             duration = round((stop - start).total_seconds() * 10)
#             # Write weights 
#             if((round_num+1) not in trainable_weights.keys()):
#                 trainable_weights.update({(round_num+1): {}})
#                 arr_num_samples.update({(round_num+1): {}})
#             trainable_weights[round_num+1].update({i: keras_model_clone.get_weights()})
#             arr_num_samples[round_num+1].update({i: num_samples})
#             # Write metrics
#             history_FL[exp][i].update({round_num : { 'losses': robot_history.history, 'num_samples': num_samples,
#                                                      'time': duration}})
#             del current_weights
#             del robot_history
#             del train_batch
#             del val_batch
#             del batch
#             del x_train_FL, y_train_FL, x_val_FL, y_val_FL
#             del keras_model_clone
#             del local_lstm
#             tf.keras.backend.clear_session()
#         summary_FL[exp]['num_participants'].update({round_num : num_participants})
#         if (num_participants == 0):
#             trainable_weights[round_num+1] = trainable_weights[round_num]
#             arr_num_samples[round_num+1] = trainable_weights[round_num]
#             break
#         round_num+=1

# # ### 4.3 Save training data

# data_FL = {}
# for exp in history_FL:
#     data_FL.update({exp : {} })
#     for r in range(1, round_num):
#         data_FL[exp].update({r: {}})
#         for robot in history_FL[exp]:
#             if (r in history_FL[exp][robot].keys()):
#                 data_FL[exp][r].update({robot : {}})
#                 data_FL[exp][r][robot] = {'losses' : history_FL[exp][robot][r]['losses'], 
#                                                     'num_samples' : history_FL[exp][robot][r]['num_samples'],
#                                                     'time': history_FL[exp][robot][r]['time']}

# filehandler = open('FL_history_' + filename[-10:-4] + '_' + str(QUOTA) + '_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'wb') 
# pickle.dump(data_FL, filehandler)
# filehandler.close()

# filehandler = open('FL_summary_' + filename[-10:-4] + '_' + str(QUOTA) + '_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'wb') 
# pickle.dump(summary_FL, filehandler)
# filehandler.close()

# ## 6.0 Datasets

# plt.figure(figsize=(12, 12))
# for v in samples.keys():
#     colors = cm.rainbow(np.linspace(0, 1, len(samples[v].keys())))
#     cnt = 0
#     for r in samples[v].keys():        
#         for i in samples[v][r].keys():
#             traj = samples[v][r][i]['traj']
#             if (len(traj) > 0):
#                 x, y = zip(*traj)
#                 ax, = plt.plot(x, y, '-.', color=colors[cnt])
#         cnt += 1
#         ax.set_label(str(cnt))

# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left",fontsize=16)
# plt.xlabel('x (m)', FontSize=16)
# plt.ylabel('y (m)', FontSize=16)
# plt.savefig("results/")

# ## 6.1 Validation Loss Curve

# ### 6.1.1 Pre-Processing

# avg_val_loss = np.zeros(NUM_ROUNDS)

# for exp in data_FA.keys():
#     for round_num in range(1, NUM_ROUNDS):
#         count = 0.
#         sum_loss = 0.
#         for robot in data_FA[exp].keys():
#             if (robot in data_FA[exp][round_num].keys()):
#                 num_samples = data_FA[exp][round_num][robot]['num_samples']
#                 count += num_samples
#                 sum_loss += num_samples * data_FA[exp][round_num][robot]['losses']['val_loss'][0]
#         if(count != 0):
#             avg_val_loss[round_num - 1] = sum_loss/count
        

# # ### 6.1.2 Plot

# fig = plt.figure(figsize=(12, 6))
# ax = plt.gca()
# loss_C = data_C['losses']['val_loss']
# plt.plot(loss_C[0:100], 'bv-', label='Centralized')
# plt.plot(avg_val_loss[avg_val_loss!=0], 'r*-', label='FA')
# plt.legend(loc='upper right', fontsize=16)
# plt.xlabel('Epochs (Iterations or Communication rounds)', FontSize=16)
# plt.ylabel('Validation Loss', FontSize=16)
# plt.savefig("results/")

# # ## 6.2 Runtime Curve

# fig = plt.figure(figsize=(12, 6))
# runtimes_C_s = data_C['times']
# epochs_C = data_C.epoch[1:]
# plt.plot(epochs_C, runtimes_C_s, 'b.-', label='Centralized')
# plt.legend(loc='upper right')
# plt.xlabel('Epochs (Iterations or Communication rounds)')
# plt.ylabel('Duration (s)')
# plt.savefig("results/")