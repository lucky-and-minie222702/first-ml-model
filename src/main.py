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
    const["whoFile"],
    const["biasWihFile"],
    const["biasWhoFile"],
)

if "restart" in sys.argv:
    prompt = input("Restart destroys every current training !\nAre you sure you want to restart (yes=yes, no=everything else)?: ")
    if(prompt == "yes"):
        model.save()
    else:
        print("Restart cancelled!")

print("Loading link weights...")
model.loadData()
dataSize = len(trainData)
testSize = len(testData)
if "info" in sys.argv:
    print("Training data size: ", dataSize, "test size: ", testSize)

def train():
    print("Training...")
    count = 0
    limit = int(sys.argv[2]) if int(sys.argv[2]) != -1 else dataSize
    start = int(sys.argv[3]) if int(sys.argv[3]) != -1 else 1 
    for record in trainData:
        if count == limit:
            break
        count += 1
        if count < start:
            continue
        allVal = record.split(",")
        inp = (np.asarray(allVal[1::], dtype = np.float32) / 255 * 0.99) + 0.01
        tar = np.zeros(const["onodes"]) + 0.01
        tar[int(allVal[0])] = 0.99
        
        model.train(inp, tar)
        percen = int(count / limit * 100)
        print(f"{percen}% [{"="*(percen//20)}{" "*(20 - percen//20)}]", "Inputs:", count, "/", limit, end="\r")
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
        inputs = (np.asarray(allVal[1:], dtype = np.float32) / 255.0 * 0.99) + 0.01
        outputs = model.query(inputs)
        label = np.argmax(outputs)
        if (label == correctLabel):
            correct += 1
        print("Test cases:", count, "/", limit, "expected:", correctLabel, "actual:", label, "correct ones:", correct)
        sys.stdout.flush()
    print("DONE")
    print("Correct:", correct)
    print("Wrong:", limit - correct)
    print("Accuracy:", f"{np.trunc((correct / limit) * 10000) / 100}%")

if "train" in sys.argv:
    train()
elif "test" in sys.argv:
    test()