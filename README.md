# NERL - Neuroevolution by Reinforcement Learning

> A majority of the code within this project was written by an AI as a proof of concept.
> 
> I am slowly optimizing and adding to this project, as a lot of the logic is sloppy and difficault to follow.
---
One problem with Neuroevolution algorithms is that they can take a long time to converge to a neuron network structure that works best for an agent, as they often work on a genetic algorithm to mutate new, randomly generated network architectures and pick the best of the networks to mutate in the next iteration. 

A well-made reinforcement learning agent can be very good at finding and optimizing solutions for problems, but that can be its weakness as well - It only optimizes to the tasks given to it. 
This however, can be very useful for some applications.

For example, what would happen if a Neuroevolution algorithm was driven by Reinforcement Learning instead of random mutations and chance? 
Could Reinforcement Learning figure out how to build an optimal neural network for an agent given its inputs and outputs?

This is what this project aims to achieve. I am using multi-compartment neuron models with STDP for more biologically realistic neurons, but they are more performance heavy and not suitable for real-time applications.

This was specifically built in order to build connectomes for digital creatures. The MultiCompartmentNeuron class contains multiple different subtypes of neurons to mimic biological neurons. 

The RL algorithm currently uses an enforcement reward to learn if adding/removing/moving neurons and their connections gives the connectome's "body" more control in its environment.
This is achieved by nesting an environment within an environment. The outer environment is exclusively for the neural network that is being modified.

The RL agent has access to these action options within the action space:
 - action[0]: Neuron index to remove (-1 for no removal)
 - action[1]: Source neuron index for connection (-1 for no connection)
 - action[2]: Target neuron index for connection (-1 for no connection)
 - action[3]: Neuron index to move (-1 for no movement)
 - action[4]: New x-coordinate (only used if moving)
 - action[5]: New y-coordinate (only used if moving)
 - action[6]: New z-coordinate (only used if moving)
 - action[7]: Add a new neuron? (0 for not adding a neuron, 1 for adding a neuron)
 - action[8]: New neuron subtype (0 for pyramidal, 1 for interneuron, 2 for motor, 3 for cpg)
 - action[9]: CPG ONLY: Oscillation frequency
 - action[10]: CPG ONLY: Oscillation amplitude
 - action[11]: CPG ONLY: Oscillation phase
