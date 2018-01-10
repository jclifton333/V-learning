from VL_env import VL_env
import gym
import numpy as np

class Cartpole(VL_env):
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
    VL_env.__init__(self, Cartpole.MAX_STATE, Cartpole.MIN_STATE, Cartpole.NUM_STATE, Cartpole.NUM_ACTION, 
                   gamma, epsilon, vFeatureArgs, piFeatureArgs)
    self.env = gym.make('CartPole-v0')
    self.defaultReward = defaultReward    
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
