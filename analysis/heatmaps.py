# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:41:08 2018

@author: Jesse
"""

import os 
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from analysis import get_episode_totals

#eps_vals = [0, 0.05, 0.1, 0.2, 0.3]
#decay_vals = [0, 0.5, 1]
ENV_NAME = 'Glucose'

data_dict = {'eps':[], 'decay':[], 'score':[]}
bts_scores = []

for fname in os.listdir('../results/{}-heatmap'.format(ENV_NAME)):
   try:
     d = pkl.load(open('../results/{}-heatmap/'.format(ENV_NAME) + fname, 'rb'))
     if not d['bts']:
       score = np.mean(get_episode_totals(d))
       data_dict['eps'].append(d['epsilon'])
       data_dict['decay'].append(d['epsDecay'])
       data_dict['score'].append(score)
     else: 
       bts_scores.append(np.mean(get_episode_totals(d)))
   except EOFError:
     pass

df = pd.DataFrame(data_dict)
hm = df.groupby(['eps', 'decay'])['score'].mean().unstack()
#plt.imshow(hm.as_matrix(), cmap='hot', interpolation='nearest')
print(hm)
print('bts performance: {}'.format(np.mean(bts_scores)))