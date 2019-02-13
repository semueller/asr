import matplotlib.pyplot as plt
import numpy as np


data = np.load('./distances.npy')

# Display matrix
plt.matshow(data)

plt.show()
