import numpy as np
from const import const
from model import neuralNetwork
import sys

trainFile = open(const["trainFile"], "r")
trainData = trainFile.readlines()[1::]
trainFile.close()
testFile = open(const["testFile"], "r")
testData = testFile.readlines()[1::]
testFile.close()

model = neuralNetwork(
    const["learningRate"],
    const["wihFile"],
    const["whoFile"]
)

if "restart" in sys.argv:
    model.save()

model.loadData()
dataSize = len(trainData)
testSize = len(testData)
print("Training data size: ", dataSize, "test size: ", testSize)

def train():
    print("Training...")
    count = 0
    limit = int(sys.argv[2]) if int(sys.argv[2]) != -1 else dataSize
    for record in trainData:
        if count == limit:
            break
        count += 1
        allVal = record.split(",")
        inp = (np.asfarray(allVal[1::]) / 255 * 0.99) + 0.01
        tar = np.zeros(const["onodes"]) + 0.01
        tar[int(allVal[0])] = 0.99
        
        model.train(inp, tar)
        print("Inputs:", count, "/", limit, end="\r")
        sys.stdout.flush()
        
    model.save()
    print("\nDONE")

def test():
    print("Testing...")
    correct = 0
    count = 0
    limit = int(sys.argv[2]) if int(sys.argv[2]) != -1 else testSize
    for record in testData:
        if count == limit:
            break
        count += 1
        allVal = record.split(',')
        correctLabel = int(allVal[0])
        inputs = (np.asfarray(allVal[1:]) / 255.0 * 0.99) + 0.01
        outputs = model.query(inputs)
        label = np.argmax(outputs)
        if (label == correctLabel):
            correct += 1
        print("Test cases:", count, "/", limit, "correct ones:", correct, end="\r")
        sys.stdout.flush()
    print("\nDONE")
            	
    print("Correct:", correct)
    print("Wrong:", limit - correct)
    print("Accuracy:", f"{np.trunc((correct / limit) * 10000) / 100}%")

if "train" in sys.argv:
    train()
elif "test" in sys.argv:
    test()