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
from ple.games.flappybird import FlappyBird
from ple import PLE
from featureUtils import getBasis, gRBF, identity, intercept
from policyUtils import policyProbsBin
import pdb

class VLenv(object):
  '''
  Generic VL environment class.  Should be implemented to accommodate cartpole,
  flappy bird, and finite MDPs.  
  '''
  
  def __init__(self, MAX_STATE, MIN_STATE, NUM_STATE, gamma, epsilon, vFeatureArgs, piFeatureArgs):
    self.MAX_STATE = MAX_STATE 
    self.MIN_STATE = MIN_STATE 
    self.NUM_STATE = NUM_STATE 
    self.gamma = gamma 
    self.epsilon = epsilon
    
    #Set feature functions and feature dimensions 
    self.vFeatures, self.nV   = self._set_features(vFeatureArgs)
    self.piFeatures, self.nPi = self._set_features(piFeatureArgs)
    
    #Initialize data arrays
    self.F_V  = np.zeros((0, self.nV))        #V-function features
    self.F_Pi = np.zeros((0, self.nPi))       #Policy features
    self.A    = np.zeros(0)                   #Actions
    self.R    = np.zeros(0)                   #Rewards
    self.Mu   = np.zeros(0)                   #Action probabilities
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
    
class Cartpole(VLenv):
  MAX_STATE = np.array([0.75, 3.3]) #Set bounds of state space
  MIN_STATE = -MAX_STATE  
  NUM_STATE = 2
    
  def __init__(self, gamma = 0.9, epsilon = 0.1, defaultReward = True, vFeatureArgs = {'featureChoice':'gRBF', 'sigmaSq':1, 'gridpoints':5}, piFeatureArgs = {'featureChoice':'identity'}):
    '''
    Constructs the cartpole environment, and sets feature functions.  
    
    Parameters
    ----------
    gamma : discount factor
    epsilon : for epsilon-greedy 
    defaultReward : boolean for using defaultReward, or r = -abs(state[2]) - abs(state[3])
    vFeatureArgs :  Dictionary {'featureChoice':featureChoice}, where featureChoice is a string in ['gRBF', 'intercept', 'identity'].
                   If featureChoice == 'gRBF', then items 'gridpoints' and 'sigmaSq' must also be provided. 
    piFeatureArgs : '' 
    '''
    VLenv.__init__(self, Cartpole.MAX_STATE, Cartpole.MIN_STATE, Cartpole.NUM_STATE, gamma, epsilon, vFeatureArgs, piFeatureArgs)
    self.env = gym.make('CartPole-v0')
    self.defaultReward = defaultReward       
   
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
    
    mu = policyProbsBin(action, self.fPi, bHat, eps = self.epsilon)
    outerProd = np.outer(self.fV, self.fV) - self.gamma * np.outer(self.fV, fVNext)
    
    #Update data
    self.fV  = fVNext
    self.fPi = fPiNext
    self.A = np.append(self.A, action)
    self.R = np.append(self.R, reward)
    self.Mu = np.append(self.Mu, mu)
    self.M = np.concatenate((self.M, [outerProd]), axis=0)
    if not done: 
      self.F_V = np.vstack((self.F_V, self.fV))
      self.F_Pi = np.vstack((self.F_Pi, self.fPi))
      
    return self.fPi, self.F_V, self.F_Pi, self.A, self.R, self.Mu, self.M, done 
    
class FlappyBirdEnv(VLenv):
  STATE_NAMES = ['next_pipe_bottom_y', 'next_pipe_dist_to_player', 
                 'next_pipe_top_y', 'player_y', 'player_vel'] #Names of the states we want to keep 
  MAX_STATE = np.array([280, 300,  170,  300, 10])
  MIN_STATE = np.array([120, 50, 30, 0, -15])
  NUM_STATE = 5 
  ACTION_LIST = [None, 119] #Action codes accepted by the FlappyBird API 
  
  def __init__(self, gamma = 0.9, epsilon = 0.1, displayScreen = False, vFeatureArgs = {'featureChoice':'gRBF', 'sigmaSq':1, 'gridpoints':5}, piFeatureArgs = {'featureChoice':'identity'}):
    '''
    Constructs the cartpole environment, and sets feature functions.  

    Parameters
    ----------
    gamma : discount factor
    epsilon : for epsilon - greedy 
    displayScreen : boolean for displaying FlappyBird screen
    vFeatureArgs :  Dictionary {'featureChoice':featureChoice}, where featureChoice is a string in ['gRBF', 'intercept', 'identity'].
                   If featureChoice == 'gRBF', then items 'gridpoints' and 'sigmaSq' must also be provided. 
    piFeatureArgs : '' 
    '''
    VLenv.__init__(self, FlappyBirdEnv.MAX_STATE, FlappyBirdEnv.MIN_STATE, FlappyBirdEnv.NUM_STATE, gamma, epsilon, vFeatureArgs, piFeatureArgs)
    self.gameStateReturner = FlappyBird() #Use this to return state dictionaries 
    self.env = PLE(self.gameStateReturner, fps = 30, display_screen = displayScreen) #Use this to input actions and return rewards
    self.env.init()
  
  @classmethod
  def stateFromDict(cls, sDict):
    '''
    Grab the states we want from sDict and put into array.
    
    :parameter sDict: dictionary representation of state 
    :return sArray: array of states in sDict with keys in FlappyBird.STATE_NAMES 
    '''
    sArray = np.array([sDict[stateName] for stateName in cls.STATE_NAMES])
    return sArray

  def reset(self):
    '''
    Starts a new simulation, adds initial state to data.
    '''
    self.env.reset_game() 
    sDict = self.gameStateReturner.getGameState() 
    s = self.stateFromDict(sDict)
    self.fV = self.vFeatures(s)
    self.fPi = self.piFeatures(s) 
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
    fV: features of next state
    F_V: array of v-function features
    F_Pi: array of policy features
    A: array of actions
    R: array of rewards 
    Mu: array of action probabilities
    M:  3d array of outer products 
    '''    
    #Get next observation 
    reward = self.env.act(FlappyBirdEnv.ACTION_LIST[action])
    sDict  = self.gameStateReturner.getGameState() 
    sNext  = self.stateFromDict(sDict)   
    done = self.env.game_over() 
    
    fVNext, fPiNext  = self.vFeatures(sNext), self.piFeatures(sNext)
    
    mu = policyProbsBin(action, self.fPi, bHat, eps = self.epsilon)
    outerProd = np.outer(self.fV, self.fV) - self.gamma * np.outer(self.fV, fVNext)
    
    #Update data
    self.fV  = fVNext
    self.fPi = fPiNext
    self.A = np.append(self.A, action)
    self.R = np.append(self.R, reward)
    self.Mu = np.append(self.Mu, mu)
    self.M = np.concatenate((self.M, [outerProd]), axis=0)

    if not done: 
      self.F_V = np.vstack((self.F_V, self.fV))
      self.F_Pi = np.vstack((self.F_Pi, self.fPi))
    
    return self.fPi, self.F_V, self.F_Pi, self.A, self.R, self.Mu, self.M, done 
  

    

    
    
    
    