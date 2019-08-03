#!/usr/bin/python
# -*- coding: utf-8 -*-

import fcntl
import termios, os
from datetime import datetime
import time
import sys
import math

import io
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
from mutagen.mp3 import MP3 as mp3

def loadCsv(fileName):
    """
    Parameters
    -------
    fileName : string
        csv file name
    
    Returns
    -------
    npInputData : numpy.ndarray
        csv file values
    """
    inputData   = pd.read_csv(fileName)
    npInputData = inputData.values

    return npInputData

class PreProcessor(object):
    def __init__(self,datas):
        """
        Parameters
        -----------
        datas[0]:viconVoltageData
        datas[1]:viconPoseData
        datas[2]:rosPoseData
        """
        self.inputPoseRosObserved   = datas[0]

        self.beforeTimeGap = 0.0

    def preProcessing(self):
        """
        preprocessing
        """
        errorObserved              = self.getError(self.inputPoseRosObserved)
        
        return errorObserved

    def getError(self,rosData):
        """
        reduction sampling of vicon data based on ros data  
        Parameters
        -------
        viconData : numpy.ndarray
        rosData : numpy.ndarray
       
        Returns
        -------
        npErrorData : numpy.ndarray
        [time,x,y,ditance,theta]
        """
        aveX=0
        aveY=0
        aveTheta=0
        dataNum = 0
        for rosDataNum in range(len(rosData)):
            if rosDataNum % 1==0:
                aveX += rosData[rosDataNum,2]
                aveY += rosData[rosDataNum,3]
                aveTheta += rosData[rosDataNum,4]
                dataNum +=1
        print(dataNum)
        aveX /= dataNum
        aveY /= dataNum
        aveTheta /= dataNum
        print([aveX,aveY,aveTheta])
        errorData =[]
        #for rosDataNum in range(len(rosData)):
        for rosDataNum in range(len(rosData)):
            if rosDataNum % 1==0:
                xError          = rosData[rosDataNum,2]-aveX
                yError          = rosData[rosDataNum,3]-aveY
                thetaError      = rosData[rosDataNum,4]-aveTheta
                if thetaError>=math.pi:
                    thetaError -= 2*math.pi
                elif thetaError<=-math.pi:
                    thetaError += 2*math.pi

                error = [rosData[rosDataNum,1], xError, yError,thetaError]
                errorData.append(error)
        npErrorData = np.array(errorData)
        return npErrorData

class Drawer(object):
    def __init__(self,datas):
        self.errorObserved          = datas
    
    def draw(self):
        self.drawError(self.errorObserved)
        self.drawHistgram(self.errorObserved)
        self.figShow()

  
    
    def drawError(self,errorData):
        
        maxTime = np.max(errorData[:,0])
        
        fig1 = plt.figure()
        x_plot = fig1.add_subplot(311)
        x_plot.plot(errorData[:,0],errorData[:,1])
        x_plot.set_xlim(0,maxTime)
        x_plot.set_title("x Error")
        x_plot.set_xlabel("Time [s]")
        x_plot.set_ylabel("Error [mm]")
        
        y_plot = fig1.add_subplot(312)
        y_plot.plot(errorData[:,0],errorData[:,2])
        y_plot.set_xlim(0,maxTime)
        y_plot.set_title("y Error")
        y_plot.set_xlabel("Time [s]")
        y_plot.set_ylabel("Error [mm]")

        theta_plot = fig1.add_subplot(313)
        theta_plot.plot(errorData[:,0],errorData[:,3])
        theta_plot.set_xlim(0,maxTime)
        theta_plot.set_title("theta Error")
        theta_plot.set_xlabel("Time [s]")
        theta_plot.set_ylabel("Error [mm]")

      
    def drawHistgram(self,errorData):
        
        meanX = np.mean(errorData[:,1])
        stdX = np.std(errorData[:,1])
        print(len(errorData[:,0]))
        fig1 = plt.figure()
        x_plot = fig1.add_subplot(311)
        x_plot.hist(errorData[:,1],bins=40,density=None)
        gaussianXX = np.linspace(errorData[:,1].min(), errorData[:,1].max(), 40)
        gaussianXY = norm.pdf(gaussianXX, meanX, stdX)
        x_plot.plot(gaussianXX, gaussianXY, linewidth=3)
        print("x")
        print(stats.shapiro(errorData[:,1]))
        print('p-value:' + str(stats.kstest(errorData[:,1], "norm")[1]))


        meanY = np.mean(errorData[:,2])
        stdY = np.std(errorData[:,2])
        y_plot = fig1.add_subplot(312)
        y_plot.hist(errorData[:,2],bins=40,density=True)
        gaussianYX = np.linspace(errorData[:,2].min(), errorData[:,2].max(), 40)
        gaussianYY = norm.pdf(gaussianYX, meanX, stdX)
        y_plot.plot(gaussianYX, gaussianYY, linewidth=3)
        print("y")
        print(stats.shapiro(errorData[:,2]))
        print('p-value:' + str(stats.kstest(errorData[:,2], "norm")[1]))

        meanT = np.mean(errorData[:,3])
        stdT = np.std(errorData[:,3])
        theta_plot = fig1.add_subplot(313)
        theta_plot.hist(errorData[:,3],bins=40,density=True)
        gaussianTX = np.linspace(errorData[:,3].min(), errorData[:,3].max(), 40)
        gaussianTY = norm.pdf(gaussianTX, meanX, stdX)
        theta_plot.plot(gaussianTX, gaussianTY, linewidth=3)
        print("theta")
        print(stats.shapiro(errorData[:,3]))
        print('p-value:' + str(stats.kstest(errorData[:,3], "norm")[1]))

    def figShow(self):
        plt.show()

    def figClose(self):
        plt.close('all')

class Saver():
    def __init__(self):
        pass
    def savePoseData(self,inputData,timeData):
        self.wagonPosesX=[]
        self.wagonPosesY=[]
        self.wagonPosesTheta=[]
        for poseNum in range(len(inputData)):
            
            self.wagonPosesX.append(inputData[poseNum][0])
            self.wagonPosesY.append(inputData[poseNum][1])
            self.wagonPosesTheta.append(inputData[poseNum][2])

        dataArray = {"time":timeData, "x":self.wagonPosesX,"y":self.wagonPosesY,"theta":self.wagonPosesTheta}
        self.data = pd.DataFrame(dataArray,columns=["time","x","y","theta"])
        self.saveData(self.data,"poseXYT.csv")
    def saveData(self,inputData,fileName):
        inputData.to_csv(fileName)
        print("saved")

def main():
    print("Please input trial name")
    trialName = str(input())
    relativePath            = ("../data/") 
    
    pathInputRosObserved    = relativePath + trialName + "RosObserve.csv"  
  
    print("load RosObserve")
    inputPoseRosObserved    = loadCsv(pathInputRosObserved)
    
    preProcessor = PreProcessor([inputPoseRosObserved])
    datas = preProcessor.preProcessing()
    #print(datas)
    # draw
    drawer = Drawer(datas )
    drawer.draw()
        


if __name__ == '__main__':
    main()