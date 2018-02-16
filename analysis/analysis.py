# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:08:59 2018

@author: Jesse
"""

import sys
sys.path.append('../src/estimation')
sys.path.append('../src/utils')

import os 
import pickle as pkl
import numpy as np
#import matplotlib.pyplot as plt 
#from policy_utils import policyProbs
#from VL import thetaPi

ENV_NAME = 'Cartpole'

#Analyze  data 

#For computing thetaHat
obs_data = pkl.load(open('{}-obs-data.p'.format(ENV_NAME), 'rb'))

#Lists for data dictionaries 
none = [] 
bts = []
rs = [] 
bts_rs = [] 
eps_5 = []
eps_10 = []
eps_decay = []

def get_episode_lengths(data_dict):
  scores = []
  for ep in np.unique(data_dict['episode']):
    scores.append(len(np.array(data_dict['score'])[np.where(data_dict['episode']==ep)]))
  return scores

def get_mean_episode_lengths(data_dict_list):
  scores = []
  episodes = np.unique(data_dict_list[0]['episode'])
  for ep in episodes: 
    mean_for_ep = np.mean([len(np.array(d['score'])[np.where(d['episode']==ep)]) for d in data_dict_list])
    scores.append(mean_for_ep)
  return scores

def get_episode_totals(data_dict):
  scores = []
  for ep in np.unique(data_dict['episode']):
    scores.append(np.sum(np.array(data_dict['score'])[np.where(data_dict['episode']==ep)]))
  return scores

def get_mean_episode_totals(data_dict_list):
  scores = []
  episodes = np.unique(data_dict_list[0]['episode'])
  for ep in episodes: 
    mean_for_ep = np.mean([np.sum(np.array(d['score'])[np.where(d['episode']==ep)]) for d in data_dict_list])
    scores.append(mean_for_ep)
  return scores
  
for fname in os.listdir('../results/{}-winscp'.format(ENV_NAME)):
    try:
      d = pkl.load(open('../results/{}-winscp/'.format(ENV_NAME) + fname, 'rb'))
      if 'epsDecay' in d.keys():
          if not d['actorCritic']:
              if d['epsDecay']:
                  eps_decay.append(d)
              else:
                  if d['epsilon'] > 0:
                      if d['epsilon'] == 0.05:
                          eps_5.append(d)
                      else:
                          eps_10.append(d)
                  else:
                      if d['bts']:
                          if d['randomShrink']:
                              bts_rs.append(d)
                          else:
                              bts.append(d)
                      else:
                          if d['randomShrink']:
                              rs.append(d)
                          else:
                              none.append(d)
    except EOFError:
      pass
                           
