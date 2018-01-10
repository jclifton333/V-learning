# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:52:51 2017

@author: Jesse

Environment objects for V-learning.
"""

from abc import abstractmethod 
from functools import partial
import numpy as np
from featureUtils import getBasis, gRBF, identity, intercept
from policyUtils import pi, policyProbs
from utils import onehot 
from policy_iteration import policy_iteration, compute_vpi
from VL import betaOpt

class VL_env(object):
  '''
  Generic VL environment class.  Should be implemented to accommodate cartpole,
  flappy bird, and finite MDPs.  
  '''
  
  def __init__(self, MAX_STATE, MIN_STATE, NUM_STATE, NUM_ACTION, gamma, epsilon, vFeatureArgs, piFeatureArgs):
    self.MAX_STATE = MAX_STATE 
    self.MIN_STATE = MIN_STATE 
    self.NUM_STATE = NUM_STATE 
    self.NUM_ACTION = NUM_ACTION 
    self.gamma = gamma 
    self.epsilon = epsilon
    self.betaOpt = betaOpt
    self.pi = pi 
    self.policyProbs = policyProbs 
    
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
    self.A = np.vstack((self.A, onehot(action, self.NUM_ACTION))) 
    self.R = np.append(self.R, reward)
    self.Mu = np.append(self.Mu, mu)
    self.M = np.concatenate((self.M, [outerProd]), axis=0)

    if not done: 
      self.F_V = np.vstack((self.F_V, self.fV))
      self.F_Pi = np.vstack((self.F_Pi, self.fPi))

    return self.fPi, self.F_V, self.F_Pi, self.A, self.R, self.Mu, self.M, done, reward
    
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
        
  
class randomFiniteMDP(VLenv):
  MAX_STATE = 1
  MIN_STATE = 0
  
  def __init__(self, nS, maxT, gamma = 0.9, epsilon = 0.1, vFeatureArgs = {'featureChoice':'identity'}, piFeatureArgs = {'featureChoice':'identity'}):
    '''
    Initializes randomFiniteMDP object, including generating the transition distributions and rewards and storing in mdpDict attribute.
    Currently using a fixed MDP, rather than randomly generating. 
    
    Currently only handles binary action spaces!
    
    Parameters
    ----------
    nA: number of actions in MDP
    nS: number of states in MDP     
    maxT: max number of steps in episode
    gamma : discount factor
    epsilon : for epsilon - greedy 
    vFeatureArgs :  Dictionary {'featureChoice':featureChoice}, where featureChoice is a string in ['gRBF', 'intercept', 'identity'].
                   If featureChoice == 'gRBF', then items 'gridpoints' and 'sigmaSq' must also be provided. 
    piFeatureArgs : '' 
    '''
    self.NUM_STATE = 4
    self.maxT = maxT
    self.NUM_ACTION = 3
    VLenv.__init__(self, randomFiniteMDP.MAX_STATE, randomFiniteMDP.MIN_STATE, self.NUM_STATE, self.NUM_ACTION, gamma, epsilon, vFeatureArgs, piFeatureArgs)
       
    transitionMatrices = np.random.dirichlet(alpha=np.ones(self.NUM_STATE), size=(self.NUM_ACTION, self.NUM_STATE)) #nA x NUM_STATE x NUM_STATE array of NUM_STATE x NUM_STATE transition matrices, uniform on simplex
    rewardMatrices = np.random.uniform(low=-10, high=10, size=(self.NUM_ACTION, self.NUM_STATE, self.NUM_STATE))
   
    #The commented lines use a pre-specified MDP, rather than a randomly-generated one 
    #TODO: make finiteMDP class and subclasses for random MDPs and specified MDPs   
    #self.NUM_STATE = 4
    #self.NUM_ACTION = 2
    #transitionMatrices = np.array([[[0.1, 0.9, 0, 0], [0.1, 0, 0.9, 0], [0, 0.1, 0, 0.9], [0, 0, 0.1, 0.9]],
    #                               [[0.9, 0.1, 0, 0], [0.9, 0, 0.1, 0], [0, 0.9, 0, 0.1], [0, 0, 0.9, 0.1]]])
    #rewardMatrices = np.ones((self.NUM_ACTION, self.NUM_STATE, self.NUM_STATE)) * -0.1
    #rewardMatrices[[0,0],[2,3],[3,3]] = 1

    self.transitionMatrices = transitionMatrices
    self.mdpDict= {}
    
    #Create transition dictionary of form {s_0 : {a_0: [( P(s_0 -> s_0), s_0, reward), ( P(s_0 -> s_1), s_1, reward), ...], a_1:...}, s_1:{...}, ...}
    for s in range(self.NUM_STATE):
        self.mdpDict[s] = {} 
        for a in range(self.NUM_ACTION):
            self.mdpDict[s][a] = [(transitionMatrices[a, s, sp1], sp1, rewardMatrices[a, s, sp1]) for sp1 in range(self.NUM_STATE)]
    policy_iteration_results = policy_iteration(self)
    self.optimalPolicy = policy_iteration_results[1][-1]
    self.optimalPolicyValue = policy_iteration_results[0][-1]
    
  def onehot(self, s):
    '''
    :parameter s: integer for state 
    :return s_dummy: onehot encoding of s
    '''    
    
    #s_dummy = np.zeros(self.NUM_STATE)
    #s_dummy[s] = 1
    #return s_dummy
    return onehot(s, self.NUM_STATE)

  def reset(self):
    '''
    Starts a new simulation at a random state, adds initial state features to data.
    '''
    s = np.random.choice(self.NUM_STATE)
    s_dummy = self.onehot(s)
    self.s = s
    self.t = 0 
    self.fV = self.vFeatures(s_dummy)
    self.fPi = self.piFeatures(s_dummy) 
    self.F_V = np.vstack((self.F_V, self.fV))
    self.F_Pi = np.vstack((self.F_Pi, self.fPi)) 
    return self.fPi 
  
  def step(self, action, bHat, state = None):
    '''
    Takes a step from the current state, given action. Updates data matrices.
    
    Parameters
    ----------
    state : current state
    action : action taken at current state
    bHat : parameters of current policy
    
    Return
    ------
    data tuple, with elements
      fV: features of next state
      F_V: array of v-function features
      F_Pi: array of policy features
      A: array of actions
      R: array of rewards 
      Mu: array of action probabilities
      M:  3d array of outer products 
      done: boolean for end of episode
    '''    

    #Get next observation
    self.t += 1
    nextStateDistribution = self.transitionMatrices[int(action), self.s, :]
    sNext = np.random.choice(self.NUM_STATE, p=nextStateDistribution)
    reward = self.mdpDict[self.s][action][sNext][2]
    done = self.t == self.maxT          
    data = self._update_data(action, bHat, done, reward, self.onehot(sNext))    
    return data 
    
  def evaluatePolicies(self, beta):
    '''
    Display optimal policy and policy associated with parameters beta.  
    
    :parameter beta: policy parameters 
    '''
    
    #Compute pi_beta 
    pi_beta = np.zeros(self.NUM_STATE)
    for s in range(self.NUM_STATE): 
      pi_beta[s] = self.pi(self.onehot(s), beta) 
    pi_beta = np.round(pi_beta)
    v_beta = compute_vpi(pi_beta, self)      
    print('pi opt: {} v opt: {}\n pi beta: {} v beta: {} beta: {}'.format(self.optimalPolicy, 
          self.optimalPolicyValue, pi_beta, v_beta, beta))
       
    
    
    