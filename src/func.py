import numpy as np

def relu(inputs):
    return np.maximum(inputs, 0)

def softmax(inputs):
    exp = np.exp(inputs)
    print(exp, "\n")
    tmp = np.sum(exp, axis = 1, keepdims = True)
    print(tmp)
    return exp/tmp

def sigmoid(inputs):
    return 1 / (1+ np.exp(-inputs))