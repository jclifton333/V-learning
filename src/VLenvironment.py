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
from policyUtils import piBin, policyProbsBin, piMulti, policyProbsMulti
from policy_iteration import policy_iteration, compute_vpi
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
    if self.NUM_ACTION == 2: 
      self.A = np.append(self.A, action)
    else: 
      action_vec = np.zeros(self.NUM_ACTION)
      action_vec[action] = 1
      self.A = np.hstack((self.A, action_vec)) 
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
    :param betaHat: 1d (if binary action) or 2d (if multi action) array of policy parameters 
    :return action: binary or onehot action 
    '''
    if self.NUM_ACTION == 2: 
      aProb = self.pi(fPi, betaHat) 
      action = np.random.random() < aProb * (1 - self.epsilon) 
    else: 
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
    
class Cartpole(VLenv):
  MAX_STATE = np.array([0.75, 3.3]) #Set bounds of state space
  MIN_STATE = -MAX_STATE  
  NUM_STATE = 2
  NUM_ACTION = 2
    
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
    self.pi = piBin 
    self.policyProbs = policyProbsBin
    self.NUM_ACTION = Cartpole.NUM_ACTION
   
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
    data tuple, with elements
      fPi: features of next state
      F_V: array of v-function features
      F_Pi: array of policy features
      A: array of actions
      R: array of rewards 
      Mu: array of action probabilities
      M:  3d array of outer products 
      done: boolean for end of episode
    '''    
    sNext, reward, done, _ = self.env.step(action)
    sNext = sNext[2:]
    if not self.defaultReward:
      reward = -np.abs(sNext[0]) - np.abs(sNext[1])
    
    data = self._update_data(action, bHat, done, reward, sNext)    
    return data      
    
class FlappyBirdEnv(VLenv):
  STATE_NAMES = ['next_pipe_bottom_y', 'next_pipe_dist_to_player', 
                 'next_pipe_top_y', 'player_y', 'player_vel'] #Names of the states we want to keep 
  MAX_STATE = np.array([280, 300,  170,  300, 10])
  MIN_STATE = np.array([120, 50, 30, 0, -15])
  NUM_STATE = 5 
  NUM_ACTION = 2
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
    self.NUM_ACTION = FlappyBirdEnv.NUM_ACTION
    self.pi = piBin 
    self.policyProbs = policyProbsBin     
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
    data tuple, with elements
      fPi: features of next state
      F_V: array of v-function features
      F_Pi: array of policy features
      A: array of actions
      R: array of rewards 
      Mu: array of action probabilities
      M:  3d array of outer products 
      done: boolean for end of episode
    '''    
    
    #Get next observation 
    reward = self.env.act(FlappyBirdEnv.ACTION_LIST[action])
    sDict  = self.gameStateReturner.getGameState() 
    sNext  = self.stateFromDict(sDict)   
    done = self.env.game_over() 

    data = self._update_data(action, bHat, done, reward, sNext)    
    return data
  
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
    self.nS = 4
    self.NUM_ACTION = 3
    VLenv.__init__(self, randomFiniteMDP.MAX_STATE, randomFiniteMDP.MIN_STATE, self.nS, gamma, epsilon, vFeatureArgs, piFeatureArgs)
    
    if self.NUM_ACTION == 2: 
      self.pi = piBin 
      self.policyProbs = policyProbsBin
    elif self.NUM_ACTION > 2: 
      self.pi = piMulti
      self.policyProbs = policyProbsMulti
    else:
      raise ValueError('Number of actions must be integer greater than 1.')
    
    #transitionMatrices = np.random.dirichlet(alpha=np.ones(nS), size=(self.NUM_ACTION, nS)) #nA x nS x nS array of nS x nS transition matrices, uniform on simplex
    transitionMatrices = np.array([[[0.1, 0.9, 0, 0], [0.1, 0, 0.9, 0], [0, 0.1, 0, 0.9], [0, 0, 0.1, 0.9]],
                                   [[0.9, 0.1, 0, 0], [0.9, 0, 0.1, 0], [0, 0.9, 0, 0.1], [0, 0, 0.9, 0.1]]])
    rewardMatrices = np.ones((self.NUM_ACTION, self.nS, self.nS)) * -0.1
    rewardMatrices[[0,0],[2,3],[3,3]] = 1
    self.maxT = maxT
    self.transitionMatrices = transitionMatrices
    self.mdpDict= {}
    
    #Create transition dictionary of form {s_0 : {a_0: [( P(s_0 -> s_0), s_0, reward), ( P(s_0 -> s_1), s_1, reward), ...], a_1:...}, s_1:{...}, ...}
    for s in range(self.nS):
        self.mdpDict[s] = {} 
        for a in range(self.NUM_ACTION):
            #self.mdpDict[s][a] = [(transitionMatrices[a, s, sp1], sp1, np.random.uniform(low=-10, high=10)) for sp1 in range(nS)]
            self.mdpDict[s][a] = [(transitionMatrices[a, s, sp1], sp1, rewardMatrices[a, s, sp1]) for sp1 in range(self.nS)]
    policy_iteration_results = policy_iteration(self)
    self.optimalPolicy = policy_iteration_results[1][-1]
    self.optimalPolicyValue = policy_iteration_results[0][-1]
    
  def onehot(self, s):
    '''
    :parameter s: integer for state 
    :return s_dummy: onehot encoding of s
    '''
    s_dummy = np.zeros(self.nS)
    s_dummy[s] = 1
    return s_dummy

  def reset(self):
    '''
    Starts a new simulation at a random state, adds initial state features to data.
    '''
    s = np.random.choice(self.nS)
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
    sNext = np.random.choice(self.nS, p=nextStateDistribution)
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
    pi_beta = np.zeros(self.nS)
    for s in range(self.nS): 
      pi_beta[s] = self.pi(self.onehot(s), beta) 
    pi_beta = np.round(pi_beta)
    v_beta = compute_vpi(pi_beta, self)      
    print('pi opt: {} v opt: {}\n pi beta: {} v beta: {} beta: {}'.format(self.optimalPolicy, 
          self.optimalPolicyValue, pi_beta, v_beta, beta))
       
    
    
    