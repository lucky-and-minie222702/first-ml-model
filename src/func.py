import numpy as np
from scipy.special import softmax as sft
import os
from os.path import dirname, basename
import shutil        
from const import const
import sys

def relu(inputs):
    return np.maximum(inputs, 0)

def softmax(inputs):
    return sft(inputs)

def emptyDir(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.isfile(file_path):
                os.remove(file_path)
            print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")
        
def createFile():
    f = open(const["wihFile"], 'w')
    f.close()
    f = open(const["whoFile"], 'w')
    f.close()
    f = open(const["biasWihFile"], 'w')
    f.close()
    f = open(const["biasWhoFile"], 'w')
    f.close()
    
def applyLoad(path):
    const["dataRootPath"] = path
    const["trainFile"] = path + "/" + const["trainFile"]
    const["testFile"] = path + "/" + const["testFile"]
    
def applyData(path):
    const["dataPath"] = path
    const["wihFile"] = path + "/" + const["wihFile"]
    const["whoFile"] = path + "/" + const["whoFile"] 
    const["biasWihFile"] = path + "/" + const["biasWihFile"]
    const["biasWhoFile"] = path + "/" + const["biasWhoFile"]    

def forcedApplySave(dir):
    const["dataPath"] = dir
    const["wihFile"] = dir + str(dirname(dirname(const["wihFile"]))) + "/" + basename(const["wihFile"])
    const["whoFile"] = dir + str(dirname(dirname(const["whoFile"]))) + "/" + basename(const["whoFile"])
    const["biasWihFile"] = dir + str(dirname(dirname(const["biasWihFile"]))) + "/" + basename(const["biasWihFile"])
    const["biasWhoFile"] = dir + str(dirname(dirname(const["biasWhoFile"]))) + "/" + basename(const["biasWhoFile"])
    
def printProgress(percent, count, limit):
    loaded = "=" * (percent//2)
    unloaded = " " * (50 - (percent//2))
    print(f"{percent:3d}% [{loaded}{unloaded}]", "Inputs:", count, "/", limit, end="\r")
    sys.stdout.flush()