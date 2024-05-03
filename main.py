from Genetic import Genetic
import matplotlib.pyplot as plt
import tensorflow as tf
from Mountain_Car import MountainCarEnv

env=MountainCarEnv()
print(env.action_space)
test=Genetic(env,3,(2,),generations=100)
test.initialize_random_hypothesis()

model=test.model

with open("weights.txt", "w") as file:
    file.write(f"{test.final_weights}\n")

print("DONE")