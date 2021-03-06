import sys 
sys.path.append('../utils')

from RL_env import RL_env
from policy_iteration import compute_vpi, policy_iteration
from utils import onehot
import numpy as np
import pdb

class FiniteMDP(RL_env):
  '''
  Generic subclass for MDPs with discrete state and action spaces. 
  '''

  MAX_STATE = 1 
  MIN_STATE = 0 
  
  def __init__(self, method, hardmax, maxT, gamma, epsilon, transitionMatrices, rewardMatrices, terminalStates = [], fixUpTo = None):
    '''
    Initialize MDP object, including self.mdpDict to store transition and reward information. 
    
    :param maxT: maximum time horizon 
    :param gamma: discount factor
    :param epsilon: for epsilon-greedy exploration 
    :param transitionMatrices: NUM_ACTION x NUM_STATE x NUM_STATE array of state transition matrices 
    :param rewardMatrices: NUM_ACTION x NUM_STATE x NUM_STATE array of rewards corresponding to transitions 
    :param terminalStates: list of states that will end episode if reached 
    '''
    
    self.maxT = maxT
    self.terminalStates = terminalStates
    NUM_ACTION, NUM_STATE = transitionMatrices.shape[0], transitionMatrices.shape[1]
    RL_env.__init__(self, method, hardmax, FiniteMDP.MAX_STATE, FiniteMDP.MIN_STATE, NUM_STATE, NUM_ACTION, gamma, epsilon,  
                    fixUpTo, vFeatureArgs = {'featureChoice':'identity'}, piFeatureArgs = {'featureChoice':'identity'})
    
    #Create transition dictionary of form {s_0 : {a_0: [( P(s_0 -> s_0), s_0, reward), ( P(s_0 -> s_1), s_1, reward), ...], a_1:...}, s_1:{...}, ...}
    self.mdpDict= {}
    for s in range(self.NUM_STATE):
        self.mdpDict[s] = {} 
        for a in range(self.NUM_ACTION):
            self.mdpDict[s][a] = [(transitionMatrices[a, s, sp1], sp1, rewardMatrices[a, s, sp1]) for sp1 in range(self.NUM_STATE)]
    policy_iteration_results = policy_iteration(self)
    self.transitionMatrices = transitionMatrices
    self.rewardMatrices = rewardMatrices
    self.optimalQ = policy_iteration_results[2].T
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
    s = 0
    s_dummy = self.onehot(s)
    self.s = s
    self.fV = self.vFeatures(s_dummy)
    self.fPi = self.piFeatures(s_dummy) 
    self._reset_super()
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
    nextStateDistribution = self.transitionMatrices[int(action), self.s, :]
    sNext = np.random.choice(self.NUM_STATE, p=nextStateDistribution)
    reward = self.rewardMatrices[action, self.s, sNext]
    self.s = sNext
    done = ((self.episodeSteps == self.maxT) or (sNext in self.terminalStates))  
    data = self._update_data(action, bHat, done, reward, self.onehot(sNext))     
    return data 
    
  def estimate_policy(self, beta):
    policy = [] 
    for s in range(self.NUM_STATE): 
      s_vec = self.onehot(s) 
      s_features = self.piFeatures(s_vec) 
      qvals = np.dot(beta, s_features)
      policy.append(np.argmax(qvals))
    return policy
    
  def report(self, beta):
    '''
    Display optimal policy and policy associated with parameters beta.  
    
    :parameter beta: policy parameters 
    '''
    
    #Compute pi_beta 
    pi_beta = np.zeros(self.NUM_STATE)
    for s in range(self.NUM_STATE): 
      pi_beta[s] = np.argmax(self.pi(self.onehot(s), beta))
    pi_beta = np.round(pi_beta)
    v_beta = compute_vpi(pi_beta, self)      
        
    REPORT = 'Episode {} Total Reward: {}\npi opt: {} v opt: {}\n pi beta: {} v beta: {} beta: {}'
    print(REPORT.format(self.episode, np.sum(self.R_list[-self.episodeSteps:]), self.optimalPolicy, 
          self.optimalPolicyValue, pi_beta, v_beta, beta))
          
class SimpleMDP(FiniteMDP):
  '''
  A simple MDP with 2 actions and 4 states.  This is for testing.  
  '''
  NUM_STATE = 2
  NUM_ACTION = 2 
  TERMINAL = [] 
  
  def __init__(self, method ,hardmax, maxT, gamma = 0.9, epsilon = 0.1, fixUpTo=None):
    '''
   
    Parameters
    ----------
    maxT: max number of steps in episode
    gamma : discount factor
    epsilon : for epsilon - greedy 
    '''
    
    transitionMatrices = np.array([[[0, 1], [0, 1]],
                                   [[1, 0], [1, 0]]])
    rewardMatrices = np.array([[[0, 1], [0, 1]],
                               [[0, 0], [0, 0]]])
    
    #Initialize as FiniteMDP subclass
    FiniteMDP.__init__(self, method, hardmax, maxT, gamma, epsilon, transitionMatrices, rewardMatrices, SimpleMDP.TERMINAL, fixUpTo = fixUpTo)

  def update_schedule(self): 
    '''    
    Returns boolean for whether it's time to re-estimate policy parameters.
    '''
    return True
    
class RandomFiniteMDP(FiniteMDP):
  '''
  Creates a FiniteMDP object from a random set of transitions and rewards. 
  Transition distributions are generated from Dirichlet(1, ... , 1), and rewards at each 
  transition are generated Unif(-10, 10). 
  '''
  TERMINAL = [] 

  def __init__(self, method, hardmax, maxT, nA = 3, nS = 4, gamma = 0.9, epsilon = 0.1, fixUpTo = None):
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
    transitionMatrices = np.random.dirichlet(alpha=[1,1,1,1,10], size=(nA, nS)) #nA x NUM_STATE x NUM_STATE array of NUM_STATE x NUM_STATE transition matrices, uniform on simplex
    rewardMatrices = np.random.uniform(low=-10, high=10, size=(nA, nS, nS))

    #Initialize as FiniteMDP subclass 
    FiniteMDP.__init__(self, method, hardmax, maxT, gamma, epsilon, transitionMatrices, rewardMatrices, RandomFiniteMDP.TERMINAL, fixUpTo=fixUpTo)
  
  def update_schedule(self): 
    '''    
    Returns boolean for whether it's time to re-estimate policy parameters.
    '''
    return True 
    
class Gridworld(FiniteMDP): 
  '''
  A difficult gridworld task with 16 squares. 
  
  Actions
  ------
  0 = N
  1 = E
  2 = S 
  3 = W
 
  State numbering  
  ---------------
  0  1  2  3 
  4  5  6  7 
  8  9  10 11
  12 13 14 15
  
  Transitions
  -----------
  15 is terminal.
  Transitions are deterministic in states [0, 1, 2, 3, 7, 11], 
  uniformly randomly everywhere else (for every action). 

  Rewards
  -------
  Reward is -1 for each transition except transitions to 15, which are positive.   
  '''
  NUM_STATE = 16
  NUM_ACTION = 4  
  PATH = [0, 1, 2, 3, 7, 11] #States for which transitions are deterministic
  TERMINAL = [15] 
  
  #These functions help construct the reward and transition matrices.
  @staticmethod
  def adjacent(s):
    '''
    Returns states adjacent to s in order [N, E, S, W]
    If s on boundary, s is adjacent to itself in that direction
      e.g. adjacent(0) = [0, 1, 4, 0]
    This results in double-counting boundary states when transition is uniformly random
    '''
    return [s - 4*(s > 3), s + 1*((s+1) % 4 != 0),  s + 4*(s < 12), s - 1*(s % 4 != 0)]
  
  @classmethod 
  def transition(cls, s, a):
    #Returns the normal deterministic transition from state s given a 
    return cls.adjacent(s)[a]
  
  @staticmethod 
  def reward(s):
    #Returns reward for transitioning to state s
    if s < 15: 
      return -1
    else: 
      return 1
      
  def __init__(self, method, hardmax, maxT, gamma = 0.9, epsilon = 0.1, fixUpTo = None):
    '''   
    Parameters
    ----------
    maxT: max number of steps in episode
    gamma : discount factor
    epsilon : for epsilon - greedy 
    '''
    
    #Construct transition and reward arrays
    transitionMatrices = np.zeros((Gridworld.NUM_ACTION, Gridworld.NUM_STATE, Gridworld.NUM_STATE))
    rewardMatrices = np.zeros((Gridworld.NUM_ACTION, Gridworld.NUM_STATE, Gridworld.NUM_STATE))
    
    for s in range(Gridworld.NUM_STATE):
      if s in Gridworld.PATH:    
        for a in range(Gridworld.NUM_ACTION):
          s_next = self.transition(s, a) 
          transitionMatrices[a, s, s_next] = 1 
          rewardMatrices[a, s, s_next] = self.reward(s_next)
      elif s in Gridworld.TERMINAL:
        for a in range(Gridworld.NUM_ACTION): 
          s_next = s 
          transitionMatrices[a, s, s_next] = 1 
          rewardMatrices[a, s, s_next] = self.reward(s_next) 
      else: 
        for a in range(Gridworld.NUM_ACTION): 
          adjacent_states = self.adjacent(s) 
          uniform_transition_prob = 1 / len(adjacent_states)
          for s_next in adjacent_states: 
            transitionMatrices[a, s, s_next] += uniform_transition_prob 
            rewardMatrices[a, s, s_next] = self.reward(s_next)
    
    #Initialize as FiniteMDP subclass
    FiniteMDP.__init__(self, method, hardmax, maxT, gamma, epsilon, transitionMatrices, rewardMatrices, Gridworld.TERMINAL)

  def update_schedule(self): 
    '''    
    Returns boolean for whether it's time to re-estimate policy parameters.
    
    Update only at deterministic-transition states (irrelevant elsewhere since transitions are 
    independent of policy).
    '''
    return self.s in self.PATH

class Chain(FiniteMDP):
  '''
  Implements chain MDP described in http://proceedings.mlr.press/v48/osband16.pdf.
  '''
  NUM_ACTION = 2  
  
  def __init__(self, method, hardmax, N, epsilon, fixUpTo = None):
    '''
    :param N: length of chain (number of states)
    
    Using radial basis features instead of tabular. 
    '''
    
    transitionMatrices = np.zeros((Chain.NUM_ACTION, N, N))
    rewardMatrices = np.zeros((Chain.NUM_ACTION, N, N))
    
    #Fill transition and reward matrices 
    #Action 0 = Left 
    transitionMatrices[0, 0, 0] = 1
    for n in range(1,N): 
      transitionMatrices[0, n, n-1] = 1 
    #Action 1 = Right
    transitionMatrices[1, 0, 1] = 1 - 1/N 
    transitionMatrices[1, 0, 0] = 1/N 
    transitionMatrices[1, N-1, N-1] = 1 - 1/N 
    transitionMatrices[1, N-1, N-2] = 1/N 
    for n in range(1, N-1):
      transitionMatrices[1, n, n+1] = 1 - 1/N 
      transitionMatrices[1, n, n-1] = 1/N 
    #Only nonzero reward is at state N 
    for n in range(N): 
      rewardMatrices[0, n, N-1] = 1 
      rewardMatrices[1, n, N-1] = 1      
    FiniteMDP.__init__(self, method, hardmax, N, 0.7, epsilon, transitionMatrices, rewardMatrices, [N-1])
    
    #Set RBF features, using odd states as basis points 
    #basis_states = [s for s in range(N) if (s + 1) % 2 == 0] 
    #onehot_to_int = lambda vec: np.flatnonzero(vec)[0]
    #self.vFeatures = lambda s_vec: np.array([np.exp(-np.abs(onehot_to_int(s_vec) - s0)) for s0 in basis_states])
    #self.piFeatures = self.vFeatures
    #self.nV = self.nPi = int(np.floor(N / 2))
    
    #Reset data arrays because we overwrote default feature functions 
    #self.F_V  = np.zeros((0, self.nV))          #V-function features
    #self.F_Pi = np.zeros((0, self.nPi))         #Policy features
    #self.M    = np.zeros((0, self.nV, self.nV)) #Outer products (for computing thetaHat)
    
  def update_schedule(self):
    return True

  
  
