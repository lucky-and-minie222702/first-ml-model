import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sci
import sys
import datetime

def splitNum(n, p, dig):
    kq = ""
    for i in range(1, n+1):
        if i == 

class neuralNetwork:
    def save(self):
        np.savetxt(self.wihFile, self.wih)
        np.savetxt(self.whoFile, self.who)
        
    def loadData(self):
        self.wih=np.loadtxt(self.wihFile)
        self.who=np.loadtxt(self.whoFile)
        
    def __init__(self, learningRate, mxdigit, wihFile, whoFile, testFile, logFile, trainFile, init=True):
        self.inodes = 10*mxdigit*2
        self.onodes = 10*(mxdigit+1)
        self.hnodes = self.onodes + self.inodes
        self.lr = learningRate
        self.sep = self.inodes
        # file paths
        self.wihFile = wihFile
        self.whoFile = whoFile
        self.testFile = testFile
        self.logFile = logFile
        self.trainFile = trainFile
        # create link weights
        if init:
            print("Initializing link weights")
            f = open(self.logFile, "w")
            self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        else:
            print("Loading link weights")
            self.loadData()
        
        self.acFunc = lambda x : sci.expit(x)
        self.dig=mxdigit
        
        self.save()
        
    
    def train(self, inputLists, targetLists):
        inp = np.array(inputLists, ndmin=2).T
        tar = np.array(targetLists, ndmin=2).T
        
        hinp = np.dot(self.wih, inp)
        hout = self.acFunc(hinp)
        
        finalInp = np.dot(self.who, hout)
        finalOut = self.acFunc(finalInp)
        
        outErr = tar - finalOut
        herr = np.dot(self.who.T, outErr)
        
        self.who += self.lr * np.dot((outErr * finalOut * (1 - finalOut)), hout.T)
        self.wih += self.lr * np.dot((herr * hout * (1 - hout)), inp.T)
    
    def query(self, inputLists):
        inp = np.array(inputLists, ndmin=2).T
        hinp = np.dot(self.wih, inp)
        hout = self.acFunc(hinp)
        finalInp = np.dot(self.who, hout)
        finalOut = self.acFunc(finalInp)
        
        return finalOut
    
    def createData(self):
        pass
        

# net.test()
