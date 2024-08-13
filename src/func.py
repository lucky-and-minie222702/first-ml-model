import numpy as np
from scipy.special import softmax as sft

def relu(inputs):
    return np.maximum(inputs, 0)

def softmax(inputs):
    return sft(inputs)

def sigmoid(inputs):
    return 1 / (1+ np.exp(-inputs))

def L2_regularization(la, weight1, weight2):
    weight1_loss = 0.5 * la * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * la * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss