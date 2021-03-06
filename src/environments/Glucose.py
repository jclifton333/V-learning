import sys
sys.path.append('../utils')
sys.path.append('../estimation')

from policy_gradient import total_policy_gradient_multi
from VL import thetaPiMulti
from RL_env import RL_env
import numpy as np
import pdb

class Glucose(RL_env): 
  '''
  Implements an environment closely similar to the online generative glucose
  model in the Luckett et al. V-learning paper.  

  Currently implementing for a single patient only!  
  '''
  NUM_STATE = 8 
  MAX_STATE = 1000 * np.ones(NUM_STATE) 
  MIN_STATE = np.zeros(NUM_STATE)
  NUM_ACTION = 2  
  
  #Generative model parameters 
  MU_GLUCOSE = 250
  SIGMA_GLUCOSE = 25
  INS_PROB = 0.3 
  MU_FOOD = 0 
  SIGMA_FOOD = 10 
  MU_ACT = 0 
  SIGMA_ACT = 10 
  MU_ACT_MOD = 31 
  SIGMA_ACT_MOD = 5 
  SIGMA_ER = 5  
  COEF = np.array([10, 0.9, 0.1, 0.1, -0.01, -0.01, -10, -4]) 
  
  #Test states 
  HYPOGLYCEMIC = np.array([50, 0, 33, 50, 0, 0, 0, 0])
  HYPERGLYCEMIC = np.array([200, 0, 30, 200, 0, 0, 78, 0])

  def __init__(self, method, hardmax, maxT, gamma = 0.9, epsilon = 0.1, fixUpTo = None, vFeatureArgs = {'featureChoice':'intercept'}, piFeatureArgs = {'featureChoice':'intercept'}):
    '''
    Constructs the Glucose environment, and sets feature functions.  
    
    Parameters
    ----------
    maxT : maximum number of steps per episode 
    gamma : discount factor
    epsilon : for epsilon-greedy 
    fixUpTo : integer for max number of observations to include in reference distribution for v-learning 
    vFeatureArgs :  Dictionary {'featureChoice':featureChoice}, where featureChoice is a string in ['gRBF', 'intercept', 'identity'].
                   If featureChoice == 'gRBF', then items 'gridpoints' and 'sigmaSq' must also be provided. 
    piFeatureArgs : '' 
    '''
    RL_env.__init__(self, method, hardmax, Glucose.MAX_STATE, Glucose.MIN_STATE, Glucose.NUM_STATE, Glucose.NUM_ACTION, 
                   gamma, epsilon, fixUpTo, vFeatureArgs, piFeatureArgs)
    self.maxT = maxT               
    self.S = np.zeros((0, Glucose.NUM_STATE)) #Record raw states, too 
    #Experimenting with quadratic v-function 
    #s_less_100 = lambda s: (s[0] < 100) * np.concatenate(([1], [s[0]], [s[0]**2]))
    #s_gr_100 = lambda s: (s[0] >= 100) * np.concatenate(([1], [s[0]], [s[0]**2]))
    s_less_100 = lambda s: (s[0] < 100) * np.concatenate(([1], [s[0]], [s[0]**2]))
    s_gr_100 = lambda s: (s[0] >= 100) * np.concatenate(([1], [s[0]], [s[0]**2]))
    self.vFeatures = lambda s: np.concatenate((s_less_100(s), s_gr_100(s)))
    self.piFeatures = self.vFeatures 
    self.nV = 6
    self.nPi = self.nV 
    self.F_Pi = np.zeros((0, self.nPi))
    self.F_V = np.zeros((0, self.nV))
    self.M    = np.zeros((0, self.nV, self.nV)) #Outer products (for computing thetaHat)

  def reward_function(self, s):
    '''
    Get reward based on state s and previous state.
    '''
    newGlucose = s[0]
    lastGlucose = self.S[-1,0]
    #Weights associated with last glucose
    #r11 = -1*((lastGlucose > 70)*(lastGlucose < 80) + (lastGlucose > 120)*(lastGlucose < 150))
    #r12 = -2*(lastGlucose > 150)
    #r13 = -3*(lastGlucose < 70)
    #Weights associated with new glucose
    #r21 = -1*((newGlucose > 70)*(newGlucose < 80) + (newGlucose > 120)*(newGlucose < 150))
    #r22 = -2*(newGlucose > 150)
    #r23 = -3*(newGlucose < 70)    
    #return r11 + r12 + r13 + r21 + r22 + r23
    
    #Experimenting with continuous piecewise-quadratic version 
    r1 = (newGlucose < 70)*(-0.005*newGlucose**2 + 0.95*newGlucose - 45) + \
         (newGlucose >= 70)*(-0.00017*newGlucose**2 + 0.02167*newGlucose -0.5)
    r2 = (lastGlucose < 70)*(-0.005*lastGlucose**2 + 0.95*lastGlucose - 45) + \
         (lastGlucose >= 70)*(-0.00017*lastGlucose**2 + 0.02167*lastGlucose -0.5)
    return r1 + r2
    
  def gen_food_and_activity(self): 
    '''
    Generate food and activity (two scalars)
    '''    
    food = np.random.normal(Glucose.MU_FOOD, Glucose.SIGMA_FOOD) 
    food = np.multiply(np.random.random() < 0.2, food) 
    activity = np.random.normal(Glucose.MU_ACT, self.SIGMA_ACT)
    activity = np.multiply(np.random.random() < 0.2, activity) 
    return food, activity 

  def reset(self): 
    '''
    '''
    #Generate data 
    food, activity = self.gen_food_and_activity()
    glucose = np.random.normal(Glucose.MU_GLUCOSE, Glucose.SIGMA_GLUCOSE)
    #glucose = glucose * (glucose > 50) * (glucose < 250) + 250 * (glucose > 250) + \
    #          50 * (glucose < 50) #Force glucose in [50, 250] 
    s3 = 100 
    s4 = 0 
    s5 = 0 
    s6 = 0
    s7 = 0 
    s = np.array([glucose, food, activity, s3, s4, s5, s6, s7]) 
    
    #Get and return features 
    self.S = np.vstack((self.S, s))
    self.fV = self.vFeatures(s)
    self.fPi = self.piFeatures(s)
    self._reset_super()
    return self.fPi 
    
  def next_state_reward(self, action): 
    '''
    Generates next state and reward. 
    :param action: onehot action encoding     
    :return sNext: next state 
    :return reward: reward associated with next state 
    '''
    last_state = self.S[-1,:] 
    if self.episodeSteps > 1: 
      last_last_state = self.S[-2,:] 
    else: 
      last_last_state = last_state
    if self.episodeSteps > 1: 
      last_action = self.A_list[-1][-1,1] 
    else: 
      last_action = 0
    X = np.hstack(([1], last_state[:2], last_last_state[1], last_state[2], last_last_state[2],
                   action, last_action)) 
    glucose = np.dot(Glucose.COEF, X) + np.random.normal(0, Glucose.SIGMA_ER) 
    #glucose = glucose * (glucose > 50) * (glucose < 250) + 250 * (glucose > 250) + \
    #      50 * (glucose < 50) #Force glucose in [50, 250] 
    food, activity = self.gen_food_and_activity()
    s3 = self.S[-1, 0]
    s4 = self.S[-1, 1]
    s5 = self.S[-1, 1]
    s6 = self.S[-1, 1]
    s7 = self.S[-1, 2]    
    
    sNext = np.array([glucose, food, activity, s3, s4, s5, s6, s7])  
    reward = self.reward_function(sNext) 
    return sNext, reward 
    
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
    done = self.episodeSteps == self.maxT 
    sNext, reward = self.next_state_reward(action)  
    data = self._update_data(action, bHat, done, reward, sNext)    
    self.S = np.vstack((self.S, sNext))
    return data      

  def report(self, betaHat):
    '''
    Reports information about current policy estimate at test states. 
    '''
    REPORT = 'Episode {} Eps: {} Total Reward: {}\nHypoglycemic policy: {} Hyperglycemic policy: {}\nbetaHat: {}'
    print(REPORT.format(self.episode, self.epsilon, np.sum(self.R[-self.episodeSteps:]), self.pi(self.piFeatures(Glucose.HYPOGLYCEMIC), betaHat), 
          self.pi(self.piFeatures(Glucose.HYPERGLYCEMIC), betaHat), betaHat))
          
  def update_schedule(self):
    return self.episodeSteps % 5 == 0

class GlucoseMulti(object):
  thetaPi = thetaPiMulti 
  total_policy_gradient = total_policy_gradient_multi  
  NUM_ACTION = Glucose.NUM_ACTION
  
  def __init__(self, maxT, nRep, gamma, epsilon, fixUpTo, vFeatureArgs = {'featureChoice':'intercept'}, piFeatureArgs = {'featureChoice':'intercept'}):
    '''
    Constructs multiple glucose environments.  
    
    Parameters
    ----------
    maxT : maximum number of steps per episode 
    nRep : number of replicates
    gamma : discount factor
    epsilon : for epsilon-greedy 
    fixUpTo : integer for max number of observations to include in reference distribution for v-learning 
    vFeatureArgs :  Dictionary {'featureChoice':featureChoice}, where featureChoice is a string in ['gRBF', 'intercept', 'identity'].
                   If featureChoice == 'gRBF', then items 'gridpoints' and 'sigmaSq' must also be provided. 
    piFeatureArgs : '' 
    '''
    self.nRep = nRep
    self.EnvList = [Glucose(maxT, gamma, epsilon, fixUpTo, vFeatureArgs, piFeatureArgs) for rep in range(nRep)]
    self.nPi = self.EnvList[0].nPi 
    self.nV = self.EnvList[0].nV
    
  def reset(self):
    '''
    Reset each environment in EnvList.
    :return: List of state policy feature arrays for each environment.
    '''
    return np.array([env.reset() for env in self.EnvList])
     
  def _get_action(self, fPiArray, betaHat):
    '''
    Calls _get_action for each environment on respective rows of fPiArray.  
    
    :param fPiArray: nRep x Glucose.nPi - size array of policy features 
    :param betaHat: Glucose.NUM_ACTION x Glucose.nPi-size array of policy params
    :return actionArray: nRep x Glucose.NUM_ACTION - size array of action vectors 
    '''
    actionArray = [] 
    for i in range(self.nRep): 
      env = self.EnvList[i]
      action = env._get_action(fPiArray[i,:], betaHat)
      actionArray.append(action)
    return np.array(actionArray)    
      
  def step(self, actionArray, betaHat):
    '''
    Take a step in each environment in EnvList. 
    
    :param actionArray: self.nRep x Glucose.NUM_ACTION - size array of dummy-encodings of actions for each environment 
    :param betaHat: Glucose.nPi - size array of policy parameters 
    :return: Lists of all arrays returned by each replicate's env.step() 
    '''
    fPi_List = [] 
    F_V_List = [] 
    F_Pi_List = [] 
    A_List = [] 
    R_List = [] 
    Mu_List= []
    M_List = [] 

    for i in range(self.nRep):
      action = actionArray[i]    
      env = self.EnvList[i]
      fPi, F_V, F_Pi, A, R, Mu, M, refDist, done, reward = env.step(action, betaHat) 
      fPi_List.append(fPi)
      F_V_List.append(F_V)
      F_Pi_List.append(F_Pi) 
      A_List.append(A)
      R_List.append(R)
      Mu_List.append(Mu)
      M_List.append(M)
    
    return fPi_List, F_V_List, F_Pi_List, A_List, R_List, Mu_List, M_List
    
  def update_schedule(self):
    return self.EnvList[0].update_schedule()
 
  def report(self, betaHat):
    '''
    Reports information about current policy estimate at test states. 
    '''
    REPORT = 'Episode {} Eps: {} Average Total Reward: {}\nHypoglycemic policy: {} Hyperglycemic policy: {}\nbetaHat: {}'
    print(REPORT.format(self.episode, self.epsilon, np.mean([np.sum(self.R[i,-self.episodeSteps:]) for i in range(self.nRep)]), self.pi(self.piFeatures(Glucose.HYPOGLYCEMIC), betaHat), 
          self.pi(self.piFeatures(Glucose.HYPERGLYCEMIC), betaHat), betaHat))
  
      