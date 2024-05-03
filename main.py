from Genetic import Genetic
import matplotlib.pyplot as plt
import tensorflow as tf
from Mountain_Car import MountainCarEnv

env=MountainCarEnv()
print(env.action_space)
test=Genetic(env,3,(2,),generations=100)
test.initialize_random_hypothesis()

model=test.model

print("DONE")
