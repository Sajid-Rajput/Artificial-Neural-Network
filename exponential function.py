# import math

# input_layer = [4.8, 1.21, 2.385]

# E = math.e

# exp_values = []

# for input in input_layer:
#     exp_values.append(E**input)

# print(exp_values)

# norm_sum = sum(exp_values)
# norm_values = []

# for value in exp_values:
#     norm_values.append(value / norm_sum)

# print(norm_values)
# print(sum(norm_values))



# Exponential Function and normalization implementation using numpy 
# import numpy as np

# input_layer = [4.8, 1.21, 2.385]

# exp_values = np.exp(input_layer)
# norm_values = exp_values / np.sum(exp_values)
# norm_sum = np.sum(norm_values)

# print(norm_values)
# print(norm_sum)



# Exponential Function and normalization implementation using numpy for batch of inputs 
from cmath import exp
import numpy as np

input_layers = np.array(([1, 2, 3, 2.5],
                [2.0, 5.0, -1.0, 2.0],
                [-1.5, 2.7, 3.3, -0.8]))

exp_values = np.exp(input_layers)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)


norm_sum = np.sum(norm_values, axis=1)

print(norm_sum)