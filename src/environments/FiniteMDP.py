from VL_env import VL_env
import numpy as np
from policy_iteration import compute_vpi, policy_iteration
from utils import onehot

class FiniteMDP(VL_env):
  '''
  Generic subclass for MDPs with discrete state and action spaces. 
  '''

  MAX_STATE = 1 
  MIN_STATE = 0 
  
  def __init_(self, maxT, gamma, epsilon, transitionMatrices, rewardMatrices)
    '''
    Initialize MDP object, including self.mdpDict to store transition and reward information. 
    
    :param maxT: maximum time horizon 
    :param gamma: discount factor
    :param epsilon: for epsilon-greedy exploration 
    :param transitionMatrices: NUM_ACTION x NUM_STATE x NUM_STATE array of state transition matrices 
    :param rewardMatrices: NUM_ACTION x NUM_STATE x NUM_STATE array of rewards corresponding to transitions 
    '''
    
    self.maxT = maxT
    NUM_ACTION, NUM_STATE = transitionMatrices.shape[0], transitionMatrices.shape[1]
    VL_env.__init__(self, FiniteMDP.MAX_STATE, FiniteMDP.MIN_STATE, NUM_STATE, NUM_ACTION, gamma, epsilon,  
                    vFeatureArgs = {'featureChoice':'identity'}, piFeatureArgs = {'featureChoice':'identity'})
    
    #Create transition dictionary of form {s_0 : {a_0: [( P(s_0 -> s_0), s_0, reward), ( P(s_0 -> s_1), s_1, reward), ...], a_1:...}, s_1:{...}, ...}
    self.mdpDict= {}
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

class RandomFiniteMDP(FiniteMDP):
  '''
  Creates a FiniteMDP object from a random set of transitions and rewards. 
  Transition distributions are generated from Dirichlet(1, ... , 1), and rewards at each 
  transition are generated Unif(-10, 10). 
  '''

  def __init__(self, nA = 3, nS = 4, maxT, gamma = 0.9, epsilon = 0.1):
    '''
    Initializes randomFiniteMDP object by passing randomly generated transition distributions and rewards to FiniteMDP.
    
    Parameters
    ----------
    nA: number of actions in MDP
    nS: number of states in MDP     
    maxT: max number of steps in episode
    gamma : discount factor
    epsilon : for epsilon - greedy 
    '''
    #Generate transition distributions 
    transitionMatrices = np.random.dirichlet(alpha=np.ones(nS), size=(nA, nS)) #nA x NUM_STATE x NUM_STATE array of NUM_STATE x NUM_STATE transition matrices, uniform on simplex
    rewardMatrices = np.random.uniform(low=-10, high=10, size=(nA, nS, nS))

    #Initialize as FiniteMDP subclass 
    FiniteMDP.__init__(self, maxT, gamma, epsilon, transitionMatrices, rewardMatrices)

class Gridworld(FiniteMDP): 
  '''
  A difficult gridworld task with 16 squares. 
  '''
  NUM_STATE = 16
  NUM_ACTION = 4  
  
  def __init__(self, maxT, gamma = 0.9, epsilon = 0.1):
    '''
   
    Parameters
    ----------
    maxT: max number of steps in episode
    gamma : discount factor
    epsilon : for epsilon - greedy 
    '''
    
    #Define transition and reward arrays
    transitionMatrices = np.array([[[0.1, 0.9, 0, 0], [0.1, 0, 0.9, 0], [0, 0.1, 0, 0.9], [0, 0, 0.1, 0.9]],
                                   [[0.9, 0.1, 0, 0], [0.9, 0, 0.1, 0], [0, 0.9, 0, 0.1], [0, 0, 0.9, 0.1]]])
    rewardMatrices = np.ones((SimpleMDP.NUM_ACTION, SimpleMDP.NUM_STATE, SimpleMDP.NUM_STATE)) * -0.1
    rewardMatrices[[0,0],[2,3],[3,3]] = 1
    
    #Initialize as FiniteMDP subclass
    FiniteMDP.__init__(self, maxT, gamma, epsilon, transitionMatrices, rewardMatrices)

    
class SimpleMDP(FiniteMDP):
  '''
  A simple MDP with 2 actions and 4 states.  This is for testing.  
  '''
  NUM_STATE = 4
  NUM_ACTION = 2 
  
  def __init__(self, maxT, gamma = 0.9, epsilon = 0.1):
    '''
   
    Parameters
    ----------
    maxT: max number of steps in episode
    gamma : discount factor
    epsilon : for epsilon - greedy 
    '''
    
    #Define transition and reward arrays
    transitionMatrices = np.array([[[0.1, 0.9, 0, 0], [0.1, 0, 0.9, 0], [0, 0.1, 0, 0.9], [0, 0, 0.1, 0.9]],
                                   [[0.9, 0.1, 0, 0], [0.9, 0, 0.1, 0], [0, 0.9, 0, 0.1], [0, 0, 0.9, 0.1]]])
    rewardMatrices = np.ones((SimpleMDP.NUM_ACTION, SimpleMDP.NUM_STATE, SimpleMDP.NUM_STATE)) * -0.1
    rewardMatrices[[0,0],[2,3],[3,3]] = 1
    
    #Initialize as FiniteMDP subclass
    FiniteMDP.__init__(self, maxT, gamma, epsilon, transitionMatrices, rewardMatrices)
    
