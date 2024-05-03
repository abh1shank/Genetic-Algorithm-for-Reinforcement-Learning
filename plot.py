with open("scores_by_ep.txt", "r") as file:
    numbers = [float(line.strip()) for line in file]
import matplotlib.pyplot as plt
import numpy as np
numbers=np.array(numbers)
plt.plot(numbers)
plt.xlabel('Episode Number')
plt.ylabel('Maximum reward achieved till that episode')
plt.show()

