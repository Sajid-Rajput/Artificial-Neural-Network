# In this lecture, we implement the neural network using OOP 
import numpy as np

# We use seed() function because all the users who run the code get the same set of values every time 
np.random.seed(0)

# In machine learning the training dataset is denoted with "X" 
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

# Here, we implement the hidden layer module with the help of Object Oriented Programming
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Shape the weights matrix as number_of_inputs(rows) and number_of_neurons(column). By using this approach we don't need to transpose the matrix when we apply dot product operation like lecture4.py

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)


print(layer2.output)
