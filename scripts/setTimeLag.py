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
from mutagen.mp3 import MP3 as mp3
import pygame

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
        self.inputV                 = datas[0]
        self.inputPose              = datas[1]
        self.inputPoseRos           = datas[2]
        self.inputPoseRosObserved   = datas[3]
        self.inputTimeFrame         = datas[4]

    def preProcessing(self):
        """
        preprocessing
        """
        timeInfo                   = self.getStartAndEndIndex(self.inputV, self.inputTimeFrame)
        poseDataFromVicon          = self.getPoseFromViconData(self.inputPose,timeInfo)
        poseDataFromRos            = self.processRosData(self.inputPoseRos)
        poseDataFromViconDminished = self.reductSampling(poseDataFromVicon,poseDataFromRos)
        error                      = self.getError(poseDataFromViconDminished,poseDataFromRos)

        print("poseDataFromRos:"+str(len(poseDataFromRos)))
        print("poseDataFromVicon:"+str(len(poseDataFromVicon)))
        print("poseDataFromViconDminished:"+str(len(poseDataFromViconDminished)))
        return poseDataFromRos,poseDataFromViconDminished,error

    def getStartAndEndIndex(self,inputData,timeFrame):
        """
        get start index and end index based on voltage data
        Parameters
        -------
        inputData : numpy.ndarray
            volatage time history
       
        Returns
        -------
        timeInfo : array
            [startIndex ,endIndex]
        """
        if len(timeFrame) == 2:
            timeInfo = [timeFrame[0],timeFrame[1]]
            print(timeInfo)
        else:
            start = False
            for columNum in range(len(inputData)):
                voltage = inputData[columNum,2]
                if start != True and voltage >0.3:
                    startIndex = inputData[columNum,0]
                    start = True
                elif start == True and voltage <0.3:
                    endIndex = inputData[columNum,0]
                    break
            timeInfo = [startIndex,endIndex]
            print(timeInfo)
        
        return timeInfo

    def getPoseFromViconData(self,inputData,timeInfo):
        """
        get pose data from vicon data fron start to end
        Parameters
        -------
        inputData : numpy.ndarray
            vicon data
        timeInfo : array
            [start index, end index]
       
        Returns
        -------
        npPoses : numpy.ndarray
           pose data
        """
        #extract data from start index to end index
        print (len(inputData))
        processedData = []
        for columNum in range(len(inputData)):
            columData = inputData[columNum]
            if columData[0] >=timeInfo[0] and columData[0] <=timeInfo[1]:
                trueTime = columData[0]
                columData[0] =trueTime-timeInfo[0]
                processedData.append(columData)
        print (len(processedData))

        poses = []
        time = []
        for columData in processedData:
            wagon=[]
            for frameNum in range(3):
                framePos = [columData[2+3*frameNum],columData[2+3*frameNum+1]]
                wagon.append(framePos)
            
            pose = self.calcurateWagonPose(wagon)
            poseData = [columData[0]*10/1000,pose[0],pose[1],pose[2]]
            #print (poseData)
            poses.append(poseData)
        npPoses = np.array(poses)
        print (len(npPoses))
        return npPoses

    def processRosData(self,inputData):
        """
        repair pose data
        Parameters
        -------
        inputData : numpy.ndarray
       
        Returns
        -------
        inputData : numpy.ndarray
        """
        poseDatas =[]
        for poseNum in range(len(inputData)):
            time = inputData[poseNum,1]
            x = inputData[poseNum,2] * 1000
            y = inputData[poseNum,3] * 1000
            theta = inputData[poseNum,4] -math.pi/2
            if theta<-math.pi:
                theta += 2*math.pi
            elif theta>math.pi:
                theta -= 2*math.pi
            pose=[time,x,y,theta]
            poseDatas.append(pose)
        npPoseDatas = np.array(poseDatas)
        return npPoseDatas

    def calcurateWagonPose(self,wagon):
        #wagon 0:rightfront,1:leftfront,2:rightback
        pose =[0,0,0]
        pose[0] = 0.5*(wagon[1][0]+wagon[2][0])
        pose[1] = 0.5*(wagon[1][1]+wagon[2][1])
        
        angle1 = math.atan2(wagon[0][1]-wagon[1][1],wagon[0][0]-wagon[1][0])
        angleX = math.cos(angle1)
        angleY = math.sin(angle1)

        angle2 = math.atan2(wagon[0][1]-wagon[2][1],wagon[0][0]-wagon[2][0])-math.pi/2
        angleX = angleX+math.cos(angle2)
        angleY = angleY+math.sin(angle2)

        pose[2]=math.atan2(angleY,angleX)+math.pi/2
        if pose[2]<-math.pi:
            pose[2] += 2*math.pi
        elif pose[2]>math.pi:
            pose[2] -= 2*math.pi

        #print("nan:"+str(angle1)+" "+str(angle2))
        if pose[2]>math.pi or pose[2]<-math.pi :
           pass 
        return pose

    def reductSampling(self, viconData,rosData):
        """
        reduction sampling of vicon data based on ros data  
        Parameters
        -------
        viconData : numpy.ndarray
        rosData : numpy.ndarray
       
        Returns
        -------
        npOutputData : numpy.ndarray
        """
        outputData =[]
        for rosDataNum in range(len(rosData)):
            minDifference = 2000
            for viconDataNum in range(len(viconData)):
                timeDifference = rosData[rosDataNum,0]-viconData[viconDataNum,0]
                if abs(timeDifference) < minDifference:
                    minDifference = timeDifference
                    indexCandidate = viconDataNum
            print(str(minDifference)+" "+str(rosData[rosDataNum,0])+" "+str(viconData[indexCandidate,0]))
            outputData.append(viconData[indexCandidate,:])
            #print (viconData[indexCandidate,:])
        npOutputData = np.array(outputData)
        #print (viconData)
        return npOutputData

    def getError(self, viconData,rosData):
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
        errorData =[]
        for rosDataNum in range(len(rosData)):
            time            = rosData[rosDataNum,0]
            xError          = abs(rosData[rosDataNum,1]-viconData[rosDataNum,1])
            yError          = abs(rosData[rosDataNum,2]-viconData[rosDataNum,2])
            distanceError   = abs(math.sqrt(math.pow(xError,2)+math.pow(yError,2)))

            thetaError      = rosData[rosDataNum,3]-viconData[rosDataNum,3]
            if thetaError>=math.pi:
                thetaError -= 2*math.pi
            elif thetaError<=-math.pi:
                thetaError += 2*math.pi

            error = [time, xError, yError, distanceError, thetaError]
            errorData.append(error)

        npErrorData = np.array(errorData)
        return npErrorData

class Drawer(object):
    def __init__(self,datas):
        self.poseFromRos    = datas[0]
        self.poseFromVicon  = datas[1]
        self.error          = datas[2]
    
    def draw(self):
        self.drawPose(self.poseFromVicon,self.poseFromRos)
        self.drawEachAxis(self.poseFromVicon,self.poseFromRos)
        self.drawError(self.error)
        self.figShow()

    def drawPose(self,inputData,inputDataRos):
        print(len(inputData))
        print(len(inputData[0]))
        print(len(inputDataRos))
        print(len(inputDataRos[0]))
        print(inputDataRos[0])
        """
        for poseNum in range(len(inputData)):
            #

            self.wagonPosesX.append(inputData[poseNum][1])
            self.wagonPosesY.append(inputData[poseNum][2])
            self.wagonPosesTheta.append(inputData[poseNum][3])

        for poseNum in range(len(inputDataRos)):
            #
            self.wagonPosesXRos.append(inputDataRos.iat[poseNum,2]*1000)
            self.wagonPosesYRos.append(inputDataRos.iat[poseNum,3]*1000)
            self.wagonPosesThetaRos.append(inputDataRos.iat[poseNum,4])
        """
        fig1 = plt.figure()
        path_plot = fig1.add_subplot(1,1,1)
        path_plot.plot(inputData[:,2],inputData[:,1])
        path_plot.plot(inputDataRos[:,2],inputDataRos[:,1])
        path_plot.invert_xaxis()
        path_plot.set_xlabel("y [mm]")
        path_plot.set_ylabel("x [mm]")
        
        
        poseInterval = 0.5
        intervalCount=0
        """
        for index,time in enumerate(self.time):
            if time>intervalCount*poseInterval:
                intervalCount+=1
                self.drawRectangle(self.wagonPosesX[index],self.wagonPosesY[index],self.wagonPosesTheta[index],0.5,0.3)
        """
        path_plot.set_aspect('equal')


    def drawEachAxis(self,inputData,inputDataRos):

        print(len(inputData))
        print(len(inputData[0]))
        print(len(inputDataRos))
        print(len(inputDataRos[0]))

        maxTime = np.max(inputData[:,0])

        fig1 = plt.figure()
        x_plot = fig1.add_subplot(311)
        x_plot.plot(inputData[:,0],inputData[:,1])
        x_plot.plot(inputDataRos[:,0],inputDataRos[:,1])
        x_plot.set_xlim(0,maxTime)
        x_plot.set_xlabel("Time [s]")
        x_plot.set_ylabel("x [mm]")
        
        y_plot = fig1.add_subplot(312)
        y_plot.plot(inputData[:,0],inputData[:,2])
        y_plot.plot(inputDataRos[:,0],inputDataRos[:,2])
        y_plot.set_xlim(0,maxTime)
        y_plot.set_xlabel("Time [s]")
        y_plot.set_ylabel("y [mm]")

        theta_plot = fig1.add_subplot(313)
        theta_plot.plot(inputData[:,0],inputData[:,3])
        theta_plot.plot(inputDataRos[:,0],inputDataRos[:,3])
        theta_plot.set_xlim(0,maxTime)
        theta_plot.set_xlabel("Time [s]")
        theta_plot.set_ylabel("theta [rad]")
    
    def drawError(self,errorData):

        maxTime = np.max(errorData[:,0])
        
        fig1 = plt.figure()
        x_plot = fig1.add_subplot(411)
        x_plot.plot(errorData[:,0],errorData[:,1])
        x_plot.set_xlim(0,maxTime)
        x_plot.set_title("x Error")
        x_plot.set_xlabel("Time [s]")
        x_plot.set_ylabel("Error [mm]")
        
        y_plot = fig1.add_subplot(412)
        y_plot.plot(errorData[:,0],errorData[:,2])
        y_plot.set_xlim(0,maxTime)
        y_plot.set_title("y Error")
        y_plot.set_xlabel("Time [s]")
        y_plot.set_ylabel("Error [mm]")

        theta_plot = fig1.add_subplot(413)
        theta_plot.plot(errorData[:,0],errorData[:,4])
        theta_plot.set_xlim(0,maxTime)
        theta_plot.set_title("theta Error")
        theta_plot.set_xlabel("Time [s]")
        theta_plot.set_ylabel("Error [mm]")

        distance_plot = fig1.add_subplot(414)
        distance_plot.plot(errorData[:,0],errorData[:,3])
        distance_plot.set_xlim(0,maxTime)
        distance_plot.set_title("Distance Error")
        distance_plot.set_xlabel("Time [s]")
        distance_plot.set_ylabel("Error [mm]")

    def figShow(self):
        plt.show()

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
    trialName = str(raw_input())
    absolutePath            = ("/home/ytnpc2017c/wagon_vicon_log2/") 
    pathInputV              = absolutePath + trialName + "V.csv" 
    pathInputPose           = absolutePath + trialName + "Pose.csv" 
    pathInputRos            = absolutePath + trialName + "Ros.csv" 
    pathInputRosObserved    = absolutePath + trialName + "RosObserve.csv" 
    pathTimeFrame           = absolutePath + trialName + "TimeFrame.csv" 

    inputV                  = loadCsv(pathInputV)
    inputPose               = loadCsv(pathInputPose)
    inputPoseRos            = loadCsv(pathInputRos)
    inputPoseRosObserved    = loadCsv(pathInputRos)

    try:
        inputTimeFrame  = loadCsv(pathTimeFrame)
    except:
        inputTimeFrame =[]

    #
    preProcessor = PreProcessor([inputV,inputPose,inputPoseRos ,inputPoseRosObserved,inputTimeFrame])
    datas = preProcessor.preProcessing()

    # draw
    drawer = Drawer(datas )
    drawer.draw()

if __name__ == '__main__':
    main()