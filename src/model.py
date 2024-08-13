import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import func
from const import const

class neuralNetwork:
    def save(self):
        np.savetxt(self.wihFile, self.wih)
        np.savetxt(self.whoFile, self.who)
        
    def loadData(self):
        self.wih=np.loadtxt(self.wihFile)
        self.who=np.loadtxt(self.whoFile)
        
    def __init__(self, learningRate, wihFile, whoFile, init=True):
        self.inodes = const["inodes"]
        self.onodes = const["onodes"]
        self.hnodes = int(2/3 * self.inodes) + self.onodes
        self.lr = learningRate
        self.sep = self.inodes
        # file paths
        self.wihFile = wihFile
        self.whoFile = whoFile
        # create link weights
        if init:
            print("Initializing link weights")
            self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        else:
            print("Loading link weights")
            self.loadData()
        
        self.acFunc = func.relu
        self.classifyFunc = func.softmax
        
    
    def train(self, inputLists, targetLists, autoSave = False):
        inp = np.array(inputLists, ndmin=2).T
        tar = np.array(targetLists, ndmin=2).T
        
        hinp = np.dot(self.wih, inp)
        hout = self.acFunc(hinp)
        
        finalInp = np.dot(self.who, hout)
        finalOut = self.classifyFunc(finalInp)
        
        deltaY = finalOut - tar
        deltaHerr = np.dot(self.who.T, deltaY)
        deltaHerr[hout <= 0] = 0
        
        wihGradient = np.dot(deltaHerr, inp.T)
        whoGradient = np.dot(deltaY, hout.T)
        
        self.who -= self.lr * whoGradient
        self.wih -= self.lr * wihGradient
        
        if autoSave:
            self.save()
    
    def query(self, inputLists):
        inp = np.array(inputLists, ndmin=2).T
        hinp = np.dot(self.wih, inp)
        hout = self.acFunc(hinp)
        finalInp = np.dot(self.who, hout)
        finalOut = self.classifyFunc(finalInp)
        
        return finalOut

# net.test()