#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 19:52:08 2020

@author: alikemalcelenk
"""

import gym
import numpy as np
import random 
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3').env

# Q Table
q_table = np.zeros([env.observation_space.n, env.action_space.n]) # row -> state, column -> action   
# bu taxi env sinde 500 state 6 action var. Detayları github documentation da var. gym in sitesinden bak.
# env.observation_space.n = 500
# env.action_space.n = 6

# hyperparameters
alpha = 0.1 # learning rate
gamma = 0.9 # discount rate
epsilon = 0.1 # exploit vs explore için

# Plotting Matrix
reward_list=[]
dropouts_list=[]

episode_number = 10000
for i in range(1, episode_number):
    
    #initialize environemnt
    state = env.reset()
    
    reward_count = 0
    dropouts = 0
    
    while True:
        
        # exploit vs explore to find action
        # %10 exploit %90 explore olmasını istiyorum
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
         
        
        # action process and take reward / observation=next state
        next_state, reward, done, _ = env.step(action)
        
        # Q learning func
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max) #q func
        
        # update Q table
        q_table[state, action] = next_value
        
        # update State 
        state = next_state
        
        #find wrong dropouts  - yolcuları ne kadar yanlış yere indirdiğini sayıcak
        if reward == -10: #-10 =  yanlış yerde yolcu indrmiş.
            dropouts += 1
            
        reward_count += reward
        
        if done: 
            break
    
    if i%10 == 0:
        dropouts_list.append(dropouts)
        reward_list.append(reward_count)    
        print('Episode {}, Reward: {}, Wrong Dropout: {}'.format(i, reward_count, dropouts))
    
    
# %% visualize

fig, axs = plt.subplots(1,2)

axs[0].plot(reward_list)
axs[0].set_xlabel('episode')
axs[0].set_ylabel('reward')

axs[1].plot(dropouts_list)
axs[1].set_xlabel('episode')
axs[1].set_ylabel('dropouts')

axs[0].grid(True)
axs[1].grid(True)
    
plt.show()
    
    
    
    
