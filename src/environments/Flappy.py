from ple.games.flappybird import FlappyBird
from ple import PLE
from VL_env import VL_env 
import numpy as np

class Flappy(VL_env):
  STATE_NAMES = ['next_pipe_bottom_y', 'next_pipe_dist_to_player', 
                 'next_pipe_top_y', 'player_y', 'player_vel'] #Names of the states we want to keep 
  MAX_STATE = np.array([280, 300,  170,  300, 10])
  MIN_STATE = np.array([120, 50, 30, 0, -15])
  NUM_STATE = 5 
  NUM_ACTION = 2
  ACTION_LIST = [None, 119] #Action codes accepted by the FlappyBird API, corresponding to [0, 1]
  
  def __init__(self, gamma = 0.9, epsilon = 0.1, displayScreen = False, fixUpTo = None, vFeatureArgs = {'featureChoice':'gRBF', 'sigmaSq':1, 'gridpoints':5}, piFeatureArgs = {'featureChoice':'identity'}):
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
    VL_env.__init__(self, Flappy.MAX_STATE, Flappy.MIN_STATE, Flappy.NUM_STATE, Flappy.NUM_ACTION,
                   gamma, epsilon, fixUpTo, vFeatureArgs, piFeatureArgs)
    self.gameStateReturner = FlappyBird() #Use this to return state dictionaries 
    self.env = PLE(self.gameStateReturner, fps = 30, display_screen = displayScreen) #Use this to input actions and return rewards
    self.NUM_ACTION = Flappy.NUM_ACTION
  
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
    self.episodeSteps = 0
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
      refDist: 2d array of v-function features to use as V-learning reference distribution
      done: boolean for end of episode
      reward: 1d array of scalar rewards
    '''    
    
    self.totalSteps += 1
    self.episodeSteps += 1
    #Get next observation 
    reward = self.env.act(Flappy.ACTION_LIST[action])
    sDict  = self.gameStateReturner.getGameState() 
    sNext  = self.stateFromDict(sDict)   
    done = self.env.game_over() 

    data = self._update_data(action, bHat, done, reward, sNext)    
    return data
