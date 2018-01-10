# V-learning

### Method 

Source code implements the V-learning method described in [Luckett et al.](https://arxiv.org/pdf/1611.03531.pdf), 
with the option to use bootstrap Thompson sampling (`bts`), in which the terms in the sum in the estimating equation 
that is solved to estimate the V-function are perturbed by Exponential(1) random variables.  

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

  - ```Cartpole```
  
  - ```Flappy``` 
  
  - ```FiniteMDP```: for MDPs with finite state spaces
  
    * ```SimpleMDP```: easy MDP with 4 states and 2 actions, for testing 
    
    * ```RandomFiniteMDP```: generate random finite MDP, for testing 
    
    * ```Gridworld```: difficult gridworld task, for paper simulations 

### ToDo 

1. Test new VL_env class structure.
2. Refactor directory structure - want a VLenvironment directory, with separate scripts for each environment;  
   a VL directory for the V-learning algorithm; and a utils directory. 
3. Address TODOs in simulate.py and randomFiniteMDP class in VLenvironment.py.