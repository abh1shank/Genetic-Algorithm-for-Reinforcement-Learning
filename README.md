# Genetic-Algorithm-for-Reinforcement-Learning

Altered the reward function for mountain car V0 environment from openAI gym to make learning more convenient.

Random weights (hypotheses) for each layer of the neural network are generated.

Fitness for each of these sets of weights are calculated, which is the amount of reward that the agent earns in 30 timesteps of the episodes.

Initially, 5% of the population is chosen for mutation, where mutation occurs in two ways
1) randomly swapping weights for neurons 
2) changing the sign of some weights from negative to positive and vice versa

If the reward earned doesnot increase from the previous mutation phase, the mutation rate is increased by 5% of its current value, else it is reduced to 95% of its current value.




![image](https://github.com/abh1shank/Genetic-Algorithm-for-Reinforcement-Learning/assets/97939389/4163169c-a88a-4516-8f39-00d62e561522)
