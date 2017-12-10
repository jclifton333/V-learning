# -*- coding: utf-8 -*-
'''Created on Sat Dec  9 19:42:11 2017

@author: Jesse

FlappyBird! 
Docs: http://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html
Example: http://pygame-learning-environment.readthedocs.io/en/latest/user/home.html#installation

game.getGameState() returns dictionary (E.G.)
{'next_next_pipe_bottom_y': 260,
 'next_next_pipe_dist_to_player': 427.0,
 'next_next_pipe_top_y': 160,
 'next_pipe_bottom_y': 244,
 'next_pipe_dist_to_player': 283,
 'next_pipe_top_y': 144,
 'player_vel': 0,
 'player_y': 256}
Values are in pixels (?), where screen is 512x288

Use only
  next_pipe_bottom_y
  next_pipe_dist_to_player
  next_pipe_top-y
  player_y
  player_vel (px/frame?)
'''

import numpy as np
import features
from ple.games.flappybird import FlappyBird
from ple import PLE
from functools import partial
from policyUtils import piBin, policyProbsBin
from VL import betaOpt

def getState(sDict):
  '''
  Takes state dictionary returned by getGameState()
  Returns [next_pipe_bottom_y, next_pipe_dist_to_player, next_pipe_top_y, player_y, player_vel]
  '''
  return np.array([sDict['next_pipe_bottom_y'], sDict['next_pipe_dist_to_player'], sDict['next_pipe_top_y'],
                   sDict['player_y'], sDict['player_vel']]) 

def getFeatures(sDict, psi):
  '''
  Returns features given dictionary representation of state
  
  Parameters
  ----------
  sDict: dictionary representation of state (ple output)
  psi: feature function that returns features given a state
  
  Returns
  -------
  psi(s): features of state contained in sDict
  '''
  s = getState(sDict)
  return psi(s)
  
  

#State space boundaries
max_ = np.array([280, 300,  170,  300, 10])
min_ = np.array([120, 50, 30, 0, -15])
gridpoints = 5
basis = getBasis(max_, min_, gridpoints)

def VLflappy(basis = basis):
  
  nV  = basis.shape[0] #V-function feature dimension
  nPi = basis.shape[1] #Policy feature dimension 
  
  #Simulation settings
  nEp = 1000
  nFrame = 10000 #Max episode length
  epsilon = 0.0
  gamma = 0.9 
  sigmaSq = 1 
  vFeatures = partial(features.gRBF, basis = basis, sigmaSq = sigmaSq) #Features for v-function 
  piFeatures = features.identity                        #Features for policy
  
  #Data arrays 
  FV  = np.zeros((0, nV))
  FPi = np.zeros((0, nPi))
  R = np.zeros(0)
  A = np.zeros(0)
  Mu = np.zeros(0)          
  M = np.zeros((0, nV, nV))
  
  #Initialize game parameters
  bHat = np.zeros(nPi)       #Warm start for beta_hat estimation 
  aList = [None, 119] #Codes for actions   
  iEp = 0 #           #Count episode 
  game = FlappyBird()
  env = PLE(game, fps=30, display_screen=True)
  
  while iEp < nEp: 
    #Get initial state and features
    sDict = game.getGameState()
    fV = getFeatures(sDict, vFeatures)
    fPi = getFeatures(sDict, piFeatures)
    FV = np.vstack((FV, fV))
    FPi = np.vstack((FPi, fPi))
    score = 0
    
    for i in range(nFrame):
      score += 1
      
      #Take action
      aProb = piBin(fPi, bHat)
      a = (np.random.random() < aProb*(1-epsilon))
      mu = policyProbsBin(a, fPi, bHat, eps = epsilon)      
      
      #Take action and get next obs
      print(a, aList[a])
      r = env.act(aList[a])
      sDict = game.getGameState()
      fVp1 = getFeatures(sDict, vFeatures)
      fPi = getFeatures(sDict, piFeatures)
      
      #Update data
      outerProd = np.outer(fV, fV) - gamma*np.outer(fV, fVp1)
      fV = fVp1
      FV = np.vstack((FV, fV))
      FPi = np.vstack((FPi, fPi))
      A, R, Mu = np.append(A, a), np.append(R, r), np.append(Mu, mu)
      M = np.concatenate((M, [outerProd]), axis=0)      
      
      #Update policy estimate
      bHat = betaOpt(policyProbsBin, epsilon, M, A, R, FPi, FV[:,:-1], Mu, wStart = bHat)   
    
      if env.game_over():
        print('score: {}'.format(score))
        iEp += 1
        env.reset_game() 
        break

VLflappy()
  
  