import numpy as np
from const import const
from model import neuralNetwork
import sys
import time
import os
from func import emptyDir, createFile, forcedApply, applyData
import random as rd

trainFile = open(const["trainFile"], "r")
trainData = trainFile.readlines()[1::]
trainFile.close()
testFile = open(const["testFile"], "r")
testData = testFile.readlines()[1::]
testFile.close()

if "apply" in sys.argv:
    print("Applying", sys.argv[sys.argv.index("apply")+1], "as data directory")
    applyData(sys.argv[sys.argv.index("apply")+1])
else:
    applyData(const["dataPath"])

if "init" in sys.argv:
    print("Initializing...")
    os.mkdir(const["savePath"])
    os.mkdir(const["pretrainedPath"])
    os.mkdir(const["dataRootPath"])
    print("Done!")

if "clear" in sys.argv:
    prompt = input("Clear destroys every stored training data!\nAre you sure you want to restart (yes=yes, no=everything else)?: ")
    if(prompt == "yes"):
        emptyDir(const["pretrainedPath"])
    else:
        print("Clear cancelled!")
model = neuralNetwork(
    const["learningRate"],
    const["wihFile"],
    const["whoFile"],
    const["biasWihFile"],
    const["biasWhoFile"],
)
print("Loading link weights...")
model.loadData()

if "store" in sys.argv:
    dir = const["pretrainedPath"]+str(int(time.time()*10))+"/"
    forcedApply(dir)
    model.wihFile = const["wihFile"]
    model.whoFile = const["whoFile"]
    model.wihBiasFile = const["biasWihFile"]
    model.whoBiasFile = const["biasWhoFile"]
    os.mkdir(dir)
    createFile()
    model.save()
    pass


if "restart" in sys.argv:
    prompt = input("Restart destroys every current training data!\nAre you sure you want to restart (yes=yes, no=everything else)?: ")
    if(prompt == "yes"):
        model.save()
    else:
        print("Restart cancelled!")

dataSize = len(trainData)
testSize = len(testData)
if "info" in sys.argv:
    print("Training data size: ", dataSize, "test size: ", testSize)

def train():
    print("Training...")
    count = 0
    limit = int(sys.argv[2]) if int(sys.argv[2]) != -1 else dataSize
    start = int(sys.argv[3]) if int(sys.argv[3]) != -1 else 1 
    for record in trainData[start::]:
        if count == limit:
            break
        count += 1
        allVal = record.split(",")
        inp = (np.asarray(allVal[1::], dtype = np.float32) / 255 * 0.99) + 0.01
        tar = np.zeros(const["onodes"]) + 0.01
        tar[int(allVal[0])] = 0.99
        
        model.train(inp, tar)
        percen = int(count / limit * 100)
        print(f"{percen}% [{"="*(percen//5)}{" "*(20 - percen//5)}]", "Inputs:", count, "/", limit, end="\r")
        sys.stdout.flush()
        
    model.save()
    print("\nDONE")

def test():
    print("Testing...")
    correct = 0
    count = 0
    limit = int(sys.argv[2]) if int(sys.argv[2]) != -1 else testSize
    savePath = sys.argv[3] if sys.argv[3] != "default" else const["savePath"] + const["saveFile"]
    randTest = rd.choices(testData, k=limit)
    f = open(savePath, 'w')
    for record in randTest:
        count += 1
        allVal = record.split(',')
        correctLabel = int(allVal[0])
        inputs = (np.asarray(allVal[1:], dtype = np.float32) / 255.0 * 0.99) + 0.01
        outputs = model.query(inputs)
        label = np.argmax(outputs)
        if (label == correctLabel):
            correct += 1
        print("Test cases:", count, "/", limit, "expected:", correctLabel, "actual:", label, file=f)
        percen = int(count / limit * 100)
        print(f"{percen}% [{"="*(percen//5)}{" "*(20 - percen//5)}]", "Tests:", count, "/", limit, end="\r")
        sys.stdout.flush()
    f.close()
    print("DONE")
    print("Correct:", correct)
    print("Wrong:", limit - correct)
    print("Accuracy:", f"{np.trunc((correct / limit) * 10000) / 100}%")

if "train" in sys.argv:
    train()
elif "test" in sys.argv:
    test()