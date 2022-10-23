import numpy as np

X_Training = np.array([[55, 0, 0, 1, 99, 1, 4], 
                       [22, 0,	1, 0, 98, 1, 3],
                       [23, 1,	1, 0, 100, 1, 1],
                       [33, 1,	1, 1, 103, 1, 3],
                       [40, 1,	0, 1, 103, 1, 2],
                       [30, 1,	1, 1, 100, 1, 3],
                       [35, 1,	1, 1, 98, 1, 2],
                       [60, 0,	1, 1, 98, 1, 4],
                       [63, 0,	0, 1, 99, 0, 0],
                       [28, 1,	1, 1, 100, 1, 1],
                       [70, 0,	0, 1, 101, 0, 0],
                       [45, 0,	0, 1, 101, 0, 0],
                       [20, 1,	0, 0, 100, 1, 1],
                       [19, 1,	1, 0, 101, 1, 3],
                       [16, 1,	0, 0, 102, 1, 1],
                       [0, 0, 0, 1, 101, 0, 0],
                       [23, 0,	1, 0, 99, 1, 2],
                       [2, 1, 1, 0, 102, 1, 0],
                       [26, 1,	1, 1, 98, 1, 2],
                       [4, 1, 0, 1, 102, 0, 0],
                       [29, 1,	1, 1, 100, 0, 1],
                       [6, 1, 1, 1, 101, 0, 0],
                       [60, 0,	1, 1, 103, 1, 4],
                       [8, 0, 0, 1, 102, 0, 0],
                       [55, 1,	1, 1, 104, 1, 4],
                       [10, 0,	0, 1, 101, 0, 0],
                       [45, 0,	0, 1, 98, 1, 2],
                       [12, 1,	0, 0, 101, 0, 0],
                       [41, 1,	1, 0, 99, 0, 3],
                       [14, 1,	0, 0, 101, 1, 1],
                       [33, 0,	0, 1, 100, 0, 2],
                       [16, 0,	1, 0, 101, 1, 1],
                       [36, 1,	1, 0, 101, 0, 3],
                       [47, 1,	1, 1, 102, 0, 4],
                       [19, 1,	0, 1, 101, 1, 1],
                       [41, 1,	1, 1, 103, 1, 3],
                       [52, 1,	1, 1, 104, 1, 2],
                       [61, 0,	1, 1, 101, 0, 0],
                       [31, 0, 0, 1, 98, 1, 4],
                       [63, 1,	1, 1, 102, 1, 0],
                       [44, 0,	0, 1, 99, 0, 2],
                       [65, 0,	0, 1, 102, 1, 0],
                       [53, 1,	0, 0, 100, 1, 1],
                       [67, 1,	1, 0, 101, 1, 0],
                       [39, 1,	0, 0, 103, 1, 2],
                       [69, 0,	0, 1, 101, 1, 0],
                       [70, 0,	1, 0, 101, 0, 0],
                       [49, 1,	1, 0, 100, 1, 1],
                       [72, 1,	1, 1, 102, 1, 0],
                       [45, 1,	0, 1, 99, 1, 3],
                       [75, 1,	1, 1, 101, 0, 1],
                       [45, 1,	1, 1, 99, 0, 3]])

X_Training_Validation = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])

Y_Testing = np.array([[77, 0, 1, 1, 102, 1, 0],
                      [50, 0, 0, 1, 99, 1, 3],
                      [78, 1, 1, 1, 102, 0, 1],
                      [40, 0, 0, 1, 100, 1, 3],
                      [81, 0, 0, 1, 102, 0, 0],
                      [44, 1, 0, 0, 103, 1, 2],
                      [85, 1, 1, 0, 101, 1, 1],
                      [55, 1, 0, 0, 100, 0, 2],
                      [89, 0, 1, 0, 101, 1, 0],
                      [33, 1, 0, 1, 102, 0, 1],
                      [91, 0, 0, 1, 101, 0, 0],
                      [95, 0, 0, 1, 102, 0, 0],
                      [99, 1, 1, 1, 102, 0, 0],
                      [102, 1, 0, 1, 101, 0, 0]])


Y_Testing_Validation = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Heviside:
    def forward(self, inputs):
        self.output = np.heaviside(inputs, 0)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs) - np.max(inputs, axis=1, keepdims=True)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, model_output, target_values):
        sample_losses = self.forward(model_output, target_values)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_predict, y_true):
        samples = len(y_predict)
        y_predict_cliped = np.clip(y_predict, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidence = y_predict_cliped[range(samples), y_true]
        if len(y_true.shape) == 2:
            correct_confidence = np.sum(y_predict_cliped*y_true, axis=1)
            
        negative_loglikelihood = -np.log(correct_confidence)
        return(negative_loglikelihood)

layer1 = Layer_Dense(7,8)
layer2 = Layer_Dense(8,2)
activation1 = Activation_Heviside()
activation2 = Activation_Heviside()

layer1.forward(X_Training)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
loss_function = Loss_CategoricalCrossEntropy()

loss = loss_function.calculate(activation2.output, X_Training_Validation)



lowest_loss = 999999
best_layer1_weights = layer1.weights.copy()
best_layer1_biases = layer1.biases.copy()
best_layer2_weights = layer2.weights.copy()
best_layer2_biases = layer2.biases.copy()

for iteration in range(100000):
    layer1.weights += 0.05 * np.random.randn(7, 8)
    layer1.biases += 0.05 * np.random.randn(1, 8)
    layer2.weights += 0.05 * np.random.randn(8, 2)
    layer2.biases += 0.05 * np.random.randn(1, 2)

    layer1.forward(X_Training)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = loss_function.calculate(activation2.output, X_Training_Validation)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == X_Training_Validation) * 100

    if loss < lowest_loss:
        print("New set of weights founds, iteration:", iteration,
        "loss: ", loss, "accuracy: ", accuracy,"%")
        best_layer1_weights = layer1.weights.copy()
        best_layer1_biases = layer1.biases.copy()
        best_layer2_weights = layer2.weights.copy()
        best_layer2_biases = layer2.biases.copy()
        lowest_loss = loss
    else:
        layer1.weights = best_layer1_weights
        layer1.biases = best_layer1_biases
        layer2.weights = best_layer2_weights
        layer2.biases = best_layer2_biases