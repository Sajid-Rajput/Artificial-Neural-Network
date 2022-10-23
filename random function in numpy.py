from traceback import print_tb
import numpy as np

# By using all the users who run the code get the same set of values every time 
# np.random.seed(0)
# arr = np.random.randint(1,100,10)

# .random and .rand both are same function but the only difference are brackets

# arr = np.random.random((3,3,3))
# print(arr)

# arr = np.random.rand(3,3)
# print(arr)

# randn print both negative and positive numbers
# arr = np.random.randn(3,3)
# print(arr)

# randint print only integer values 
arr = 0.10 * np.random.randn(4,5)
print(arr)