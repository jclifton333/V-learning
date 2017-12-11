# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:52:51 2017

@author: Jesse

Environment objects for V-learning.
"""
from abc import abstractmethod
from functools import partial
import numpy as np
import gym
from featureUtils import getBasis, gRBF, identity, intercept
from policyUtils import policyProbsBin
import pdb


class VLenv():
  '''
  Generic VL environment class.  Should be implemented to accomodate cartpole,
  flappy bird, and finite MDPs.  
  '''
  
  @abstractmethod 
  def __init__(self):
    self.vFeatures  = None #Feature function for v-function 
    self.piFeatures = None #Feature function for policy 
    
  @abstractmethod
  def reset(self):
    '''
    Starts a new simulation. 
    :return: The initial state. 
    '''
    pass 
  
  @abstractmethod
  def step(self, action, betaHat, epsilon, state = None):
    '''
    Takes a step from the current state, given action, and adds results to data matrices.
    
    Parameters
    ----------
    action : action taken at current state
    betahat : parameters of current policy
    epsilon : epsilon (greedy) value
    state : current state , or None 

    Return
    ------
    Next state, boolean for simulation having terminated, data matrices
    '''    
    pass
  
  @abstractmethod
  def _set_features(self, featureChoice):
    '''
    Sets feature function.
    :param featureChoice: string for feature choice in ['gRBF', 'identity']
    '''
    
class Cartpole(VLenv):
  MAX_STATE = np.array([0.75, 3.3]) #Set bounds of state space
  MIN_STATE = -MAX_STATE  
  NUM_STATE = 2
    
  def __init__(self, gamma = 0.9, defaultReward = True, vFeatureChoice = 'gRBF', piFeatureChoice = 'identity', vGridpoints = 5, vSigmaSq = 1, piGridpoints = None, piSigmaSq = None):
    '''
    Constructs the cartpole environment, and sets feature functions.  
    
    Parameters
    ----------
    gamma: discount factor
    defaultReward: boolean for using defaultReward, or r = -abs(state[2]) - abs(state[3])
    vFeatureChoice: string for v-function features
    piFeatureChoice: string for policy features 
    vGridpoints: integer for number of points per dimension, for constructing RBF grid (for v-function features)
    vSigmaSq: variance for Gaussian RBF kernel (for v-function features)
    piGridpoints: '' (for policy features)
    piSigmaSq: '' (for policy features)    
    '''
    self.env = gym.make('CartPole-v0')
    self.gamma = gamma
    self.defaultReward = defaultReward
    
    #Set feature functions 
    if vFeatureChoice == 'gRBF': 
      vBasis = getBasis(Cartpole.MAX_STATE, Cartpole.MIN_STATE, vGridpoints)
      self.vFeatures = partial(gRBF, basis = vBasis, sigmaSq = vSigmaSq)
      self.nV = vGridpoints**2 
    else: 
      self.vFeatures = intercept
      self.nV = Cartpole.NUM_STATE + 1
    if piFeatureChoice == 'gRBF': 
      piBasis = getBasis(Cartpole.MAX_STATE, Cartpole.MIN_STATE, piGridpoints)
      self.piFeatures = partial(gRBF, basis = piBasis, sigmaSq = piSigmaSq)
      self.nPi = piGridpoints**2
    else:
      self.piFeatures = identity
      self.nPi = Cartpole.NUM_STATE      
    
    #Initialize data arrays
    self.F_V  = np.zeros((0, self.nV))        #V-function features
    self.F_Pi = np.zeros((0, self.nPi))       #Policy features
    self.A    = np.zeros(0)                   #Actions
    self.R    = np.zeros(0)                   #Rewards
    self.Mu   = np.zeros(0)                   #Action probabilities
    self.M    = np.zeros((0, self.nV, self.nV)) #Outer products (for computing thetaHat)
    
  def reset(self):
    '''
    Starts a new simulation, adds initial state to data.
    '''
    s = self.env.reset()
    s = s[2:]
    self.fV = self.vFeatures(s)
    self.fPi = self.piFeatures(s)
    self.F_V = np.vstack((self.F_V, self.fV))
    self.F_Pi = np.vstack((self.F_Pi, self.fPi))
  
  def step(self, action, bHat, epsilon, state = None):
    '''
    Takes a step from the current state, given action. Updates data matrices.
    
    Parameters
    ----------
    state : current state
    action : action taken at current state
    betahat : parameters of current policy
    epsilon : epsilon (greedy) value
    
    Return
    ------
    fV: features of next state
    F_V: array of v-function features
    F_Pi: array of policy features
    A: array of actions
    R: array of rewards 
    Mu: array of action probabilities
    M:  3d array of outer products 
    '''    
    sNext, reward, done, _ = self.env.step(action)
    sNext = sNext[2:]
    fVNext, fPiNext  = self.vFeatures(sNext), self.piFeatures(sNext)
    if not self.defaultReward:
      reward = -np.abs(sNext[0]) - np.abs(sNext[1])
    
    mu = policyProbsBin(action, self.fPi, bHat, eps = epsilon)
    outerProd = np.outer(self.fV, self.fV) - self.gamma * np.outer(self.fV, fVNext)
    
    #Update data
    self.fV  = fVNext
    self.fPi = fPiNext
    self.F_V = np.vstack((self.F_V, self.fV))
    self.F_Pi = np.vstack((self.F_Pi, self.fPi))
    self.A = np.append(self.A, action)
    self.R = np.append(self.R, reward)
    self.Mu = np.append(self.Mu, mu)
    self.M = np.concatenate((self.M, [outerProd]), axis=0)
    
    return self.fV, self.F_V, self.F_Pi, self.A, self.R, self.Mu, self.M
    
    
    
    
    
    
    
    