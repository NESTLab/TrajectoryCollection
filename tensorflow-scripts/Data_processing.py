#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import collections
import functools
import os
import six
import time
import glob
from datetime import datetime
import pickle
import sys

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

tf.compat.v1.enable_v2_behavior()

np.random.seed(0)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import datetime

# # Process Results

experiment = sys.argv[1]

# experiment = 'foraging_K60'


# ## Load DFL files


path = 'results/' + experiment + '/DFL_history*'

results_DFL = {}
exp_DFL = []

for filename in glob.iglob(path): 
    print(filename)
    
    temp = filename.split('_')
    exp_id = temp[4] + temp[5]
    print(exp_id)
    exp_DFL.append(exp_id)
    
    results_DFL.update({exp_id : {'history' : [] , 'summary' : []}})
    
    filehandler = open(filename, 'rb') 
    results_DFL[exp_id]['history'] = pickle.load(filehandler)
    filehandler.close()
    
    filename = filename.replace("history", "summary")
    if os.path.isfile(filename):
        filehandler = open(filename, 'rb')
        results_DFL[exp_id]['summary'] = pickle.load(filehandler)  
        filehandler.close()


# ## Load FL files

path = 'results/' + experiment +'/FL_history*'

results_FL = {}
exp_FL = []

for filename in glob.iglob(path): 
    print(filename)
    
#     exp_id = filename[-6:]
    temp = filename.split('_')
    exp_id = temp[4] + temp[5]
    exp_FL.append(exp_id)
    
    results_FL.update({exp_id : {'history' : [] , 'summary' : []}})
    
    filehandler = open(filename, 'rb') 
    results_FL[exp_id]['history'] = pickle.load(filehandler)
    filehandler.close()
    
    filename = filename.replace("history", "summary")
    if os.path.isfile(filename):
        filehandler = open(filename, 'rb')
        results_FL[exp_id]['summary'] = pickle.load(filehandler)  
        filehandler.close()


# ## Load Centralized files

path = 'results/' + experiment +'/Centralized*'

results_C = {}
exp_C = []

for filename in glob.iglob(path): 
    
    print(filename)
#     exp_id = filename[-6:]
    temp = filename.split('_')
    exp_id = temp[3] + temp[4]
    exp_C.append(exp_id)
    
    results_C.update({exp_id : []})
    
    filehandler = open(filename, 'rb') 
    results_C[exp_id] = pickle.load(filehandler)
    filehandler.close()

exp_C.sort()
exp_DFL.sort()
exp_FL.sort()


print(exp_DFL)
print(exp_C)
print(exp_FL)


# ## Validation Loss


def get_average_loss(history):
    for exp in history.keys():
        avg_val_loss = np.zeros(len(history[exp]))
        for round_num in range(1, len(history[exp])+1):
            count = 0.
            sum_loss = 0.
            for robot in history[exp].keys():
                if (robot in history[exp][round_num].keys()):
                    num_samples = history[exp][round_num][robot]['num_samples']
                    count += num_samples
                    sum_loss += num_samples * history[exp][round_num][robot]['losses']['val_loss'][0]
            if(count != 0):
                avg_val_loss[round_num - 1] = sum_loss/count
    return avg_val_loss


def get_time_of_round(round_num, summary):
    for exp in summary.keys():
        return summary[exp]['entry_barrier_end'][round_num-1]


def get_time_of_round_FL(round_num, summary):
    for exp in summary.keys():
        return summary[exp]['round_time'][round_num-1]


def get_round_of_convergence(loss, threshold_p=0.1):
    thresh = threshold_p * min(loss) #loss[-1]
    finite_diff = abs(loss - min(loss)) #loss[-1]
    return list(map(lambda i: i< thresh, finite_diff)).index(True)+1 


for exp_c, exp_d, exp_f in zip(exp_C, exp_DFL, exp_FL):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    # Centralized
    loss_C = results_C[exp_c]['losses']['val_loss']
    plt.plot(loss_C, 'b.-', label='Centralized')
    ax.annotate(format(loss_C[-1], '.3f') , xy= (len(loss_C)-1, loss_C[-1] + 0.01), color='b',fontsize='x-large')
    
    # DFL
    loss_DFL = get_average_loss(results_DFL[exp_d]['history'])
    plt.plot(loss_DFL, 'g--', label='DFL')
    ax.annotate(format(loss_DFL[-1], '.3f') , xy= (len(loss_DFL)-1, loss_DFL[-1] + 0.02), color='g', fontsize='x-large')
    if len(results_DFL[exp_d]['summary']) != 0:
        conv_round = get_round_of_convergence(loss_DFL)
        t_conv = get_time_of_round(conv_round, results_DFL[exp_d]['summary'])
        plt.axvline(x=conv_round, color='g')
        ax.annotate("time: " + format(t_conv/10, '.0f') + "s", xy= (conv_round + 1, loss_DFL[-1] + 0.15), color='g', fontsize='x-large')
        ax.annotate("round: " + format(conv_round, '.0f') , xy= (conv_round + 1, loss_DFL[-1] + 0.1), color='g', fontsize='x-large')
        
    # FL
    loss_FL = get_average_loss(results_FL[exp_f]['history'])
    plt.plot(loss_FL, 'r-.', label='FL')
    ax.annotate(format(loss_FL[-1], '.3f') , xy= (len(loss_FL)-1, loss_FL[-1] + 0.04), color='r', fontsize='x-large')
    if len(results_FL[exp_f]['summary']) != 0:
        conv_round = get_round_of_convergence(loss_FL)
        plt.axvline(x=conv_round, color='r')
        t_conv = get_time_of_round_FL(conv_round, results_FL[exp_f]['summary'])
        ax.annotate("time: " + format(t_conv/10, '.0f') + "s", xy= (conv_round + 1, loss_FL[-1] + 0.25), color='r', fontsize='x-large')
        ax.annotate("round: " + format(conv_round, '.0f') , xy= (conv_round + 1, loss_FL[-1] + 0.2), color = 'r',fontsize='x-large')
        
    print(exp_c)
#     plt.plot(avg_val_loss[avg_val_loss!=0], 'r*-', label='FA')
    plt.legend(loc='upper right', fontsize='x-large')
    ax.tick_params(labelsize='x-large')
    plt.xlabel('Epochs (Iterations or Communication rounds)', fontsize='x-large')
    plt.ylabel('Validation Loss', fontsize='x-large')
    plt.savefig("results/" + experiment + "/val_loss" + exp_c + ".eps", dpi=600, format="eps")


# ## Inter-epoch time


def get_timing(history, summary):
    for exp in history.keys():
        timings = np.zeros((len(history[exp]), 4))
        # timings[r] = [entry_b_start, entry_b_end, training_min_duration, exit_b_end]
        for round_num in range(1, len(history[exp])+1):
            #print(summary[exp])
            ebs = 0#summary[exp]['entry_barrier_start'][round_num-1]
            ebe = summary[exp]['entry_barrier_end'][round_num-1]
            exbe = summary[exp]['exit_barrier_end'][round_num-1]
            durations = []
            for robot in history[exp].keys():
                if (robot in history[exp][round_num].keys()):
                    durations.append(history[exp][round_num][robot]['time'])
            exbs = min(durations) + ebe
            timings[round_num-1] = [ebs, ebe, exbs, exbs]
    return timings


for exp_d, exp_f in zip(exp_DFL, exp_FL):
    fig = plt.figure(figsize=(12, 6))
    if len(results_DFL[exp_d]['summary']) != 0:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()
        timings_DFL = get_timing(results_DFL[exp_d]['history'], results_DFL[exp_d]['summary'])
        inter_epoch = np.zeros(len(timings_DFL)-1)
        inter_epoch[0] = timings_DFL[0][1]
        for i in range(1, len(timings_DFL)-1):
            inter_epoch[i] = timings_DFL[i+1][1] - timings_DFL[i][2]
        plt.plot(inter_epoch, 'g.-', label='DFL')
        print(exp_d)
    #     plt.plot(avg_val_loss[avg_val_loss!=0], 'r*-', label='FA')
    if  len(results_FL[exp_f]['summary']) != 0:
        for elem in results_FL[exp_f]['summary'].values():
            inter_epoch = np.diff(list(elem['round_time'].values()))
            plt.plot(inter_epoch[inter_epoch>0], 'r-.', label='FL')
    plt.legend(loc='upper right', fontsize=16)
    ax.tick_params(labelsize='x-large')
    plt.xlabel('Epochs (Iterations or Communication rounds)', FontSize=16)
    plt.ylabel('Inter-epoch duration (0.1s)', FontSize=16)
    plt.savefig("results/" + experiment + "/inter_epoch_" + exp_d + ".eps", dpi=600, format="eps")


# ## Number of participants


def get_participants(summary):
    for exp in summary.keys():
        num_particip = np.zeros(len(summary[exp]['num_participants']))
        for round_num in range(len(summary[exp]['num_participants'])):
            num_particip[round_num] = summary[exp]['num_participants'][round_num+1]
        return num_particip


for exp_d, exp_f in zip(exp_DFL, exp_FL):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    if not isinstance(results_DFL[exp_d]['summary'], list):
        num_p = get_participants(results_DFL[exp_d]['summary'])
        plt.plot(num_p, 'go-', label='DFL')
    if not isinstance(results_FL[exp_d]['summary'], list):        
        num_p_FL = get_participants(results_FL[exp_f]['summary'])
        plt.plot(num_p_FL[:-1], 'r-.', label='FL')
    print(exp_d)
    plt.legend(loc='upper right', fontsize=16)
    plt.xlabel('Epochs (Iterations or Communication rounds)', FontSize=16)
    plt.ylabel('Number of participants', FontSize=16)
    plt.savefig("results/"+ experiment + "/num_participants_" + exp_d + ".eps", dpi=600, format="eps")
