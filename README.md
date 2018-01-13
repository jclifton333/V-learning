# V-learning

This repo hosts code for a research project on using bootstrapping to approximate Thompson sampling in model-free
reinforcement learning. 

`src` implements the V-learning method described in [Luckett et al.](https://arxiv.org/pdf/1611.03531.pdf), with the 
additional option to use bootstrap Thompson sampling.  In this case, the terms in the sum in the estimating equation 
used to estimate the V-function are perturbed by Exponential(1) random variables.   

There is now also the option to use an advantage actor-critic variant of the V-learning method, in which 
the V-function is estimated for a given policy, and then used to take a stochastic advantage-estimate policy-gradient 
step on the policy parameters.  

### Running simulations 

```sh
bash simulate.sh 
```

from root will create a local `results` directory, run simulations with settings specified in `simulate.sh`, 
and save results if `WRITE` is set to `'True'`. 

### Simulation environments 

Code for the simulation environments is in ```src/environments```.  The ```VL_env``` class provides a generic 
RL API appropriate for V-learning, from which each environment inherits.

* ```VL_env``` 

  - ```Cartpole```: classic Cartpole control task 
  
  - ```Flappy```: Flappy Bird game; the size of the state space and sample complexity have so far 
                  prohibited convergence to a good policy in a reasonable amount of time 
  
  - ```FiniteMDP```: for MDPs with finite state spaces
  
    * ```SimpleMDP```: easy MDP with 4 states and 2 actions, for testing 
    
    * ```RandomFiniteMDP```: generate random finite MDP, for testing 
    
    * ```Gridworld```: difficult gridworld task, for paper simulations 

### ToDo 

* Ensure Glucose generative model is correct 
* Test Flappy environment 
* Make sure function parameter and return descriptions are up-to-date; improve documentation generally 