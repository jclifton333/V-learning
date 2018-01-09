import numpy as np

def compute_vpi(pi, mdp):
    a = np.array([sum([l[0]*l[2] for l in mdp.mdpDict[s][pi[s]]]) for s in range(mdp.NUM_STATE)])
    B = np.zeros((mdp.NUM_STATE, mdp.NUM_STATE))
    for s in range(mdp.NUM_STATE):
        for l in mdp.mdpDict[s][pi[s]]:
            B[s,l[1]] += l[0]            
    V= np.linalg.solve(np.identity(mdp.NUM_STATE)-mdp.gamma*B,a)    
    return V

def compute_qpi(vpi, mdp):
    Qpi = np.array([[sum([l[0]*(l[2] + mdp.gamma*vpi[l[1]]) for l in mdp.mdpDict[s][a]]) for a in range(mdp.NUM_ACTION)] for s in range(mdp.NUM_STATE)])    
    return Qpi

def policy_iteration(mdp, nIt=20):
    '''
    :parameter mdp: MDP-like object (e.g. randomFiniteMDP object) 
    :return Vs: sequence of value functions generated by PI
    :return pis: sequence of policies generated by PI 
    :return Q: Q-function corresponding to final policy iterate      
    '''
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.NUM_STATE,dtype='int')
    pis.append(pi_prev)
    for it in range(nIt):   
        vpi = compute_vpi(pi_prev, mdp)
        qpi = compute_qpi(vpi, mdp)
        pi = qpi.argmax(axis=1)
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    Q = compute_qpi(Vs[-1], mdp)
    return Vs, pis, Q

