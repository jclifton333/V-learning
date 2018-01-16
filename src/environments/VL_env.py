# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:52:51 2017

@author: Jesse

Environment objects for V-learning.
"""
import sys 
sys.path.append('../utils')
sys.path.append('../estimation')

from abc import abstractmethod 
from functools import partial
import numpy as np
from feature_utils import getBasis, gRBF, identity, intercept
from policy_utils import pi, policyProbs
from utils import onehot 
from policy_iteration import policy_iteration, compute_vpi
from policy_gradient import total_policy_gradient
from VL import betaOpt, thetaPi
import pdb

class VL_env(object):
  '''
  Generic VL environment class.  Should be implemented to accommodate cartpole,
  flappy bird, and finite MDPs.  
  '''
  
  def __init__(self, MAX_STATE, MIN_STATE, NUM_STATE, NUM_ACTION, gamma, epsilon, fixUpTo, vFeatureArgs, piFeatureArgs):
    self.MAX_STATE = MAX_STATE 
    self.MIN_STATE = MIN_STATE 
    self.NUM_STATE = NUM_STATE 
    self.NUM_ACTION = NUM_ACTION
    self.fixUpTo = fixUpTo    
    self.totalSteps = 0
    self.episode = -1 #Will be incremented to 0 at first reset  
    self.gamma = gamma 
    self.epsilon = epsilon
    
    self.betaOpt = betaOpt
    self.pi = pi 
    self.policyProbs = policyProbs 
    self.thetaPi = thetaPi 
    self.total_policy_gradient = total_policy_gradient

    
    #Set feature functions, feature dimensions, and betaOpt 
    self.vFeatures, self.nV   = self._set_features(vFeatureArgs)
    self.piFeatures, self.nPi = self._set_features(piFeatureArgs)
    
    #Initialize data arrays
    self.F_V  = np.zeros((0, self.nV))          #V-function features
    self.F_Pi = np.zeros((0, self.nPi))         #Policy features
    self.A    = np.zeros((0, self.NUM_ACTION))  #Actions
    self.R    = np.zeros(0)                     #Rewards
    self.Mu   = np.zeros(0)                     #Action probabilities
    self.M    = np.zeros((0, self.nV, self.nV)) #Outer products (for computing thetaHat)
    
  def _set_features(self, featureArgs):
    '''
    Returns feature function and feature dimension. 
    :param featureArgs: Dictionary {'featureChoice':featureChoice}, where featureChoice is a string in
                        ['gRBF', 'intercept', 'identity']. If featureChoice == 'gRBF', then items 'gridpoints' 
                        and 'sigmaSq' must also be provided. 
    :return featureFunction: feature function corresponding to featureChoice
    :return nF: feature dimension     
    '''
    featureChoice = featureArgs['featureChoice']
    if featureChoice == 'gRBF': 
      gridpoints, sigmaSq = featureArgs['gridpoints'], featureArgs['sigmaSq']
      basis = getBasis(self.MAX_STATE, self.MIN_STATE, gridpoints) 
      featureFunc = partial(gRBF, basis = basis, sigmaSq = sigmaSq) 
      nF = gridpoints**self.NUM_STATE
    elif featureChoice == 'identity': 
       featureFunc = identity 
       nF = self.NUM_STATE 
    elif featureChoice == 'intercept': 
       featureFunc = intercept 
       nF = self.NUM_STATE + 1
    return featureFunc, nF
      
  def _update_data(self, action, bHat, done, reward, sNext):
    '''
    Updates data matrices.
    '''
    
    #Update data    
    fVNext, fPiNext  = self.vFeatures(sNext), self.piFeatures(sNext)    
    mu = self.policyProbs(action, self.fPi, bHat, eps = self.epsilon)
    outerProd = np.outer(self.fV, self.fV) - self.gamma * np.outer(self.fV, fVNext)
    
    self.fV  = fVNext
    self.fPi = fPiNext
    if isinstance(action, int): 
      act_vec = onehot(action, self.NUM_ACTION)
    else: 
      act_vec = action 
    self.A = np.vstack((self.A, act_vec)) 
    self.R = np.append(self.R, reward)
    self.Mu = np.append(self.Mu, mu)
    self.M = np.concatenate((self.M, [outerProd]), axis=0)

    if not done: 
      self.F_V = np.vstack((self.F_V, self.fV))
      self.F_Pi = np.vstack((self.F_Pi, self.fPi))
      
    if self.fixUpTo is not None:
      refDist =  self.F_V[:self.fixUpTo,:]
    else:
      refDist = self.F_V    

    return self.fPi, self.F_V, self.F_Pi, self.A, self.R, self.Mu, self.M, refDist, done, reward
    
  def _get_action(self, fPi, betaHat):
    '''
    Returns random action at state with features fPi under policy with parameters betaHat.  
    
    :param fPi: policy state feature array 
    :param betaHat: 2d array of policy parameters 
    :return action: onehot action 
    '''
    aProbs = self.pi(fPi, betaHat)
    epsProb = np.random.random() 
    action = (epsProb > self.epsilon) * np.random.choice(self.NUM_ACTION, p=aProbs) + \
             (epsProb <= self.epsilon) * np.random.choice(self.NUM_ACTION, p=np.ones(self.NUM_ACTION) / self.NUM_ACTION) 
    return action   
  
  def _reset_super(self):
    '''
    Carries out reset functions common to all environments; called in self.reset.  
    '''
    self.episodeSteps = 0 
    self.episode += 1
    self.F_V = np.vstack((self.F_V, self.fV))
    self.F_Pi = np.vstack((self.F_Pi, self.fPi)) 
    
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
  def update_schedule(self):
    '''    
    Returns boolean for whether it's time to re-estimate policy parameters.
    '''
    pass 
    
  @abstractmethod 
  def report(self, betaHat):
    '''
    Prints information about the current state of the iterates. 
    '''
    pass 
  
