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
from VL import betaOptVL
from QL import betaOptQL
import pdb

class RL_env(object):
  '''
  Generic RL environment class.  Should be implemented to accommodate cartpole,
  flappy bird, and finite MDPs.  
  '''
  
  def __init__(self, method, hardmax, MAX_STATE, MIN_STATE, NUM_STATE, NUM_ACTION, gamma, epsilon, fixUpTo, vFeatureArgs, piFeatureArgs):
    '''
    :param method: string in ['VL', 'QL']
    :param hardmax: boolean for using hard or softmax policy
    '''
    self.method = method
    self.hardmax = hardmax
    self.MAX_STATE = MAX_STATE 
    self.MIN_STATE = MIN_STATE 
    self.NUM_STATE = NUM_STATE 
    self.NUM_ACTION = NUM_ACTION
    self.fixUpTo = fixUpTo    
    self.totalSteps = 0
    self.episode = -1 #Will be incremented to 0 at first reset  
    self.gamma = gamma 
    self.epsilon = epsilon
    
    #Set policy functions 
    self.pi = lambda s, beta: pi(s, beta, self.hardmax) 
    self.policyProbs = lambda a, s, beta: policyProbs(a, s, beta, self.epsilon, self.hardmax)
    
    #Set parameter estimation functions
    if self.method == 'VL':
      self.betaOpt = betaOptVL 
      self.total_policy_gradient = total_policy_gradient
    elif self.method == 'QL':
      self.betaOpt = betaOptQL   
    
    #Set feature functions, feature dimensions, and betaOpt 
    self.vFeatures, self.nV   = self._set_features(vFeatureArgs)
    self.piFeatures, self.nPi = self._set_features(piFeatureArgs)
      
    #Initialize data lists; these will hold arrays for each episode
    self.F_V_list = [] 
    self.F_Pi_list = [] 
    self.A_list = []
    self.R_list = [] 
    self.Mu_list = [] 
    self.M_list = []    
    
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
    mu = self.policyProbs(action, self.fPi, bHat)
    outerProd = np.outer(self.fV, self.fV) - self.gamma * np.outer(self.fV, fVNext)
    
    self.fV  = fVNext
    self.fPi = fPiNext
    if isinstance(action, int): 
      act_vec = onehot(action, self.NUM_ACTION)
    else: 
      act_vec = action 
      
    try:
      self.A_list[-1] = np.vstack((self.A_list[-1], act_vec)) 
    except:
      pdb.set_trace()
    self.R_list[-1] = np.append(self.R_list[-1], reward)
    self.Mu_list[-1] = np.append(self.Mu_list[-1], mu)
    self.M_list[-1] = np.concatenate((self.M_list[-1], [outerProd]), axis=0)
    self.F_V_list[-1] = np.vstack((self.F_V_list[-1], self.fV))
    self.F_Pi_list[-1] = np.vstack((self.F_Pi_list[-1], self.fPi))
      
    if self.fixUpTo is not None:
      refDist =  np.vstack(self.F_V_list)[:self.fixUpTo,:]
    else:
      refDist = np.vstack(self.F_V_list)
    return self.fPi, self.F_V_list, self.F_Pi_list, self.A_list, self.R_list, self.Mu_list, self.M_list, refDist, done, reward
    
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
    self.F_V_list.append(np.zeros((0, self.nV)))                   #value function (V or Q) features
    self.F_Pi_list.append(np.zeros((0, self.nPi)))                  #Policy features (in QL, S rather than SxA used to compute Qmax)
    self.A_list.append(np.zeros((0, self.NUM_ACTION)))          #Actions
    self.R_list.append(np.zeros(0))                        #Rewards
    self.Mu_list.append(np.zeros(0))                              #Action probabilities
    self.M_list.append(np.zeros((0, self.nV, self.nV)))         #Outer products (for computing thetaHat)  

    
    self.episodeSteps = 0 
    self.episode += 1
    self.F_V_list[-1] = np.vstack((self.F_V_list[-1], self.fV))
    self.F_Pi_list[-1] = np.vstack((self.F_Pi_list[-1], self.fPi)) 
    
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
  
