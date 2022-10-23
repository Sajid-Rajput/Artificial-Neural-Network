import numpy as np

# np.random.seed(0)

X = np.array(([[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]))

# Y = np.array(([[1, 0],
#                [0, 1],
#                [1, 0]]))

Y = np.array([0, 1, 0])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


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

        # 1e-7 = 0.0000001  1-1e-7 = 0.9999999 
        y_predict_cliped = np.clip(y_predict, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidence = y_predict_cliped[range(samples), y_true]
        if len(y_true.shape) == 2:
            correct_confidence = np.sum(y_predict_cliped*y_true, axis=1)
            
        negative_loglikelihood = -np.log(correct_confidence)
        return(negative_loglikelihood)


# class Precise_CategoricalCrossEntropyLoss():
#     def calculate(self, y_predict, y_true):
#         predictions = np.argmax(y_predict, axis=1)
#         if len(y_true.shape) == 2:
#             target = np.argmax(y_true, axis=1)
#             accuracy = np.mean(predictions == target)
#         else:    
#             accuracy = np.mean(predictions == y_true)
#         return accuracy

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)
activation1 = Activation_Relu()
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
loss_function = Loss_CategoricalCrossEntropy()
# precise_CCEL = Precise_CategoricalCrossEntropyLoss()

loss = loss_function.calculate(activation2.output, Y)
# precise_loss = precise_CCEL.calculate(activation2.output, Y)

# print(activation2.output)
# print(loss)
# print(precise_loss)

lowest_loss = 999999
best_layer1_weights = layer1.weights.copy()
best_layer1_biases = layer1.biases.copy()
best_layer2_weights = layer2.weights.copy()
best_layer2_biases = layer2.biases.copy()

for iteration in range(100000):
    layer1.weights += 0.05 * np.random.randn(4, 5)
    layer1.biases += 0.05 * np.random.randn(1, 5)
    layer2.weights += 0.05 * np.random.randn(5, 2)
    layer2.biases += 0.05 * np.random.randn(1, 2)

    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = loss_function.calculate(activation2.output, Y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == Y) * 100

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


print("Best Layer1 weights: ", best_layer1_weights)
print("Best Layer1 biases: ", best_layer1_biases)
print("Best Layer1 weights: ", best_layer2_weights)
print("Best Layer2 biases: ", best_layer2_biases)
print("Lowest loss: ", lowest_loss)
