from VL_env import VL_env
import numpy as np
import pdb

class Glucose(VL_env): 
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
  MU_GLUCOSE = 100 
  SIGMA_GLUCOSE = 25
  INS_PROB = 0.3 
  MU_FOOD = 78 
  SIGMA_FOOD = 10 
  MU_ACT = 819 
  SIGMA_ACT = 10 
  MU_ACT_MOD = 31 
  SIGMA_ACT_MOD = 5 
  SIGMA_ER = 0.001 
  COEF = np.array([10, 0.9, 0.1, 0.1, -0.01, -0.01, -2, -4]) 
  
  #Test states 
  HYPOGLYCEMIC = np.array([50, 0, 33, 50, 0, 0, 0, 0])
  HYPERGLYCEMIC = np.array([200, 0, 30, 200, 0, 0, 78, 0])

  def __init__(self, maxT, gamma = 0.9, epsilon = 0.1, fixUpTo = None, vFeatureArgs = {'featureChoice':'intercept'}, piFeatureArgs = {'featureChoice':'identity'}):
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
    VL_env.__init__(self, Glucose.MAX_STATE, Glucose.MIN_STATE, Glucose.NUM_STATE, Glucose.NUM_ACTION, 
                   gamma, epsilon, fixUpTo, vFeatureArgs, piFeatureArgs)
    self.maxT = maxT               
    self.S = np.zeros((0, Glucose.NUM_STATE)) #Record raw states, too 
  
  def reward_function(self, s):
    '''
    Get reward based on state s and previous state.
    '''
    newGlucose = s[0]
    lastGlucose = self.S[-1,0]
    #Weights associated with last glucose
    r11 = -1*((lastGlucose > 70)*(lastGlucose < 80) + (lastGlucose > 120)*(lastGlucose < 150))
    r12 = -2*(lastGlucose > 150)
    r13 = -3*(lastGlucose < 70)
    #Weights associated with new glucose
    r21 = -1*((newGlucose > 70)*(newGlucose < 80) + (newGlucose > 120)*(newGlucose < 150))
    r22 = -2*(newGlucose > 150)
    r23 = -3*(newGlucose < 70)    
    return r11 + r12 + r13 + r21 + r22 + r23
   
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
    glucose = glucose * (glucose > 50) * (glucose < 250) + 250 * (glucose > 250) + \
              50 * (glucose < 50) #Force glucose in [50, 250] 
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
    if self.S.shape[0] >= 2: 
      last_last_state = self.S[-2,:] 
    else: 
      last_last_state = last_state
    if self.A.shape[0] > 0: 
      last_action = self.A[-1,1] 
    else: 
      last_action = 0
    X = np.hstack(([1], last_state[:2], last_last_state[1], last_state[2], last_last_state[2],
                   action, last_action)) 
    glucose = np.dot(Glucose.COEF, X) + np.random.normal(0, Glucose.SIGMA_ER) 
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
    REPORT = 'Episode {} Total Reward: {}\nHypoglycemic policy: {} Hyperglycemic policy: {}\nbetaHat: {}'
    print(REPORT.format(self.episode, np.sum(self.R[-self.episodeSteps:]), self.pi(Glucose.HYPOGLYCEMIC, betaHat), 
          self.pi(Glucose.HYPERGLYCEMIC, betaHat), betaHat))
          
  def update_schedule(self):
    return self.episodeSteps % 5 == 0
     