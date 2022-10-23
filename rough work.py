import numpy as np

softmax_output = np.array([[0.7, 0.2, 0.1],
                           [0.5, 0.1, 0.4],
                           [0.02, 0.9, 0.08]])

y_true = np.array([0, 1, 1])

predictions = np.argmax(softmax_output, axis=1)
accuracy = np.mean(predictions == y_true)
print("Accuracy: ", accuracy)
print(len(y_true.shape))

# import imp


# import numpy as np

# matrix = np.array([[0, -1.0, 1.5],
#                    [1, 2, -0.00004]])

# print(np.heaviside(matrix, 0))