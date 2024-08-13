import numpy as np
import sys
import datetime
import func
from const import const

class neuralNetwork:
    def save(self):
        np.savetxt(self.wihFile, self.wih)
        np.savetxt(self.whoFile, self.who)
        np.savetxt(self.wihBiasFile, self.wihBias)
        np.savetxt(self.whoBiasFile, self.whoBias)
        
    def loadData(self):
        self.wih = np.loadtxt(self.wihFile)
        self.who = np.loadtxt(self.whoFile)
        self.wihBias = np.loadtxt(self.wihBiasFile)
        self.whoBias = np.loadtxt(self.whoBiasFile)
        
    def __init__(self, learningRate, wihFile, whoFile, wihBiasFile, whoBiasFile, init=True):
        self.inodes = const["inodes"]
        self.onodes = const["onodes"]
        self.hnodes = int(2/3 * self.inodes) + self.onodes
        self.lr = learningRate
        self.sep = self.inodes
        # file paths
        self.wihFile = wihFile
        self.whoFile = whoFile
        self.wihBiasFile = wihBiasFile
        self.whoBiasFile = whoBiasFile
        # create link weights
        if init:
            print("Initializing link weights")
            self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
            self.wihBias = np.zeros((self.hnodes, 1))
            self.whoBias = np.zeros((self.onodes, 1))
        else:
            print("Loading link weights...")
            self.loadData()
        
        self.acFunc = func.relu
        self.classifyFunc = func.softmax
        
    
    def train(self, inputLists, targetLists, autoSave = False):
        inp = np.array(inputLists, ndmin=2).T
        tar = np.array(targetLists, ndmin=2).T
        
        self.wihBias = self.wihBias.reshape((self.hnodes, 1))
        hinp = np.dot(self.wih, inp) + self.wihBias
        hout = self.acFunc(hinp)
        
        # print(self.who.shape, np.dot(self.wih, inp).shape, np.dot(self.who, hout).shape, self.whoBias.shape)
        self.whoBias = self.whoBias.reshape((self.onodes, 1))
        finalInp = np.dot(self.who, hout) + self.whoBias
        finalOut = self.classifyFunc(finalInp)
        
        deltaY = (finalOut - tar)
        deltaHerr = np.dot(self.who.T, deltaY)
        deltaHerr[hout <= 0] = 0
        
        wihGradient = np.dot(deltaHerr, inp.T)
        wihBias = np.sum(deltaHerr, axis=0, keepdims=True)
        whoGradient = np.dot(deltaY, hout.T)
        whoBias = np.sum(deltaY, axis=0, keepdims=True)
        
        wihGradient += 0.01 * self.wih
        whoGradient += 0.01 * self.who
        
        self.who -= self.lr * whoGradient
        self.wih -= self.lr * wihGradient
        self.wihBias -= self.lr * wihBias
        self.whoBias -= self.lr * whoBias
        
        if autoSave:
            self.save()
    
    def query(self, inputLists):
        inp = np.array(inputLists, ndmin=2).T
        self.wihBias = self.wihBias.reshape((self.hnodes, 1))
        hinp = np.dot(self.wih, inp) + self.wihBias
        hout = self.acFunc(hinp)
        self.whoBias = self.whoBias.reshape((self.onodes, 1))
        finalInp = np.dot(self.who, hout) + self.whoBias
        finalOut = self.classifyFunc(finalInp)
        return finalOut

# net.test()