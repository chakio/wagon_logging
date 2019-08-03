#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import fcntl
import io
import math
import os
import sys
import termios
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pygame
from mutagen.mp3 import MP3 as mp3

# default setting of figures
plt.rcParams['font.family'] = 'Times New Roman' # Fonts
plt.rcParams["mathtext.fontset"] = 'stix' # math fonts
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams["font.size"] = 20
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = False # make grid
plt.rcParams['lines.linewidth'] = 2.0 # line width
plt.rcParams['xtick.labelsize'] = 15 # 軸サイズ
plt.rcParams['ytick.labelsize'] = 15 # 軸フォント

plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'black'
# plt.rcParams['figure.figsize'] = [9.45, 7.22] # figure size

relativePath            = ("../wagon_vicon_log3/") 

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

        self.beforeTimeGap = 0.0

    def preProcessing(self):
        """
        preprocessing
        """
        #vicon
        self.timeInfo                                = self.getStartAndEndIndex(self.inputV,self.inputTimeFrame)
        poseDataFromVicon, robotPoseDataFromVicon    = self.getPoseFromViconData(self.inputPose,self.timeInfo)

        #ros
        poseDataFromRos                 = self.getPoseFromRosData(self.inputPoseRos)
        poseDataFromRosObserved         = self.getPoseFromRosObservedData(self.inputPoseRosObserved)
        robotPoseDataFromRos            = self.getRobotPoseFromRosData(self.inputPoseRos)
        
        #combinate
        poseDataFromViconDminished      = self.reductSampling(poseDataFromVicon,poseDataFromRos)
        robotPoseDataFromViconDminished = self.reductSampling(robotPoseDataFromVicon,poseDataFromRos)
        calibratedPoseDataFromRos       = self.calibrateLocalizationError(poseDataFromRos, robotPoseDataFromRos,robotPoseDataFromViconDminished)
        error                           = self.getError(poseDataFromViconDminished,poseDataFromRos)
        errorLocalization               = self.getLocalizationError(robotPoseDataFromViconDminished,robotPoseDataFromRos)
        
        
        print("poseDataFromRos:"+str(len(poseDataFromRos)))
        print("poseDataFromVicon:"+str(len(poseDataFromVicon)))
        print("poseDataFromViconDminished:"+str(len(poseDataFromViconDminished)))
        return calibratedPoseDataFromRos, poseDataFromRos,poseDataFromRosObserved,poseDataFromViconDminished,error,errorLocalization

    def changeStartTime(self):
        """
        changeStartTime
        """
        #vicon
        self.timeInfo              = self.getStartAndEndIndexFromTimeGap(self.inputV)
        poseDataFromVicon, robotPoseDataFromVicon    = self.getPoseFromViconData(self.inputPose,self.timeInfo)

        #ros
        poseDataFromRos                 = self.getPoseFromRosData(self.inputPoseRos)
        poseDataFromRosObserved         = self.getPoseFromRosObservedData(self.inputPoseRosObserved)
        robotPoseDataFromRos            = self.getRobotPoseFromRosData(self.inputPoseRos)
        
        #combinate
        poseDataFromViconDminished      = self.reductSampling(poseDataFromVicon,poseDataFromRos)
        robotPoseDataFromViconDminished = self.reductSampling(robotPoseDataFromVicon,poseDataFromRos)
        calibratedPoseDataFromRos       = self.calibrateLocalizationError(poseDataFromRos, robotPoseDataFromRos,robotPoseDataFromViconDminished)
        error                           = self.getError(poseDataFromViconDminished,poseDataFromRos)

        print("poseDataFromRos:"+str(len(poseDataFromRos)))
        print("poseDataFromVicon:"+str(len(poseDataFromVicon)))
        print("poseDataFromViconDminished:"+str(len(poseDataFromViconDminished)))
        return calibratedPoseDataFromRos,poseDataFromRos,poseDataFromRosObserved,poseDataFromViconDminished,error
    
    def changeCropTime(self,fileName):
        """
        changeCropTime
        """
        #vicon
        self.setCropTime(fileName)
        self.timeInfo              = self.getStartAndEndIndex(self.inputV,self.inputTimeFrame)
        poseDataFromVicon, robotPoseDataFromVicon    = self.getPoseFromViconData(self.inputPose,self.timeInfo)

        #ros
        poseDataFromRos                 = self.getPoseFromRosData(self.inputPoseRos)
        poseDataFromRosObserved         = self.getPoseFromRosObservedData(self.inputPoseRosObserved)
        robotPoseDataFromRos            = self.getRobotPoseFromRosData(self.inputPoseRos)
        
        #combinate
        poseDataFromViconDminished      = self.reductSampling(poseDataFromVicon,poseDataFromRos)
        robotPoseDataFromViconDminished = self.reductSampling(robotPoseDataFromVicon,poseDataFromRos)
        calibratedPoseDataFromRos       = self.calibrateLocalizationError(poseDataFromRos, robotPoseDataFromRos,robotPoseDataFromViconDminished)
        error                           = self.getError(poseDataFromViconDminished,poseDataFromRos)

        print("poseDataFromRos:"+str(len(poseDataFromRos)))
        print("poseDataFromVicon:"+str(len(poseDataFromVicon)))
        print("poseDataFromViconDminished:"+str(len(poseDataFromViconDminished)))
        return calibratedPoseDataFromRos,poseDataFromRos,poseDataFromRosObserved,poseDataFromViconDminished,error
    
    def setCropTime(self,fileName):
        """
        setCropTime
        """
        print("please input start time[s]")
        startTime = float(str(raw_input()))
        print(startTime)
        print("please input end time[s]")
        endTime = float(str(raw_input()))
        print(endTime)
        cropTime =[startTime,endTime]
        df= pd.DataFrame({"cropFrame":cropTime })
        df.to_csv(fileName+"CropTime.csv")

    def cropData(self,datas,cropTimes):
        poseDataCropedFromRosCalibratedCroped = self.getCropedData(datas[0],cropTimes)
        poseDataCropedFromViconCroped         = self.getCropedData(datas[1],cropTimes)
        poseDataCropedFromRosCroped           = self.getCropedData(datas[2],cropTimes)
        poseDataCropedFromRosObservedCroped   = self.getCropedData(datas[3],cropTimes)
        errorDataCroped                       = self.getCropedData(datas[4],cropTimes)
        return poseDataCropedFromRosCalibratedCroped,poseDataCropedFromViconCroped,poseDataCropedFromRosCroped,poseDataCropedFromRosObservedCroped,errorDataCroped


    #########################################################################
    #                       process  Vicon data                             #
    #########################################################################
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
            timeInfo = [timeFrame[0,1],timeFrame[1,1]]
            print(timeInfo)
        else:
            start = False
            startIndex = 0
            endIndex = len(inputData)
            for columNum in range(len(inputData)):
                voltage = inputData[columNum,2]
                if start != True and voltage >0.3:
                    #print("a")
                    startIndex = inputData[columNum,0]
                    start = True
                elif start == True and voltage <0.3:
                    endIndex = inputData[columNum,0]
                    break
            if startIndex == np.nan:
                startIndex = 0
            #print(startIndex)
            timeInfo = [startIndex,endIndex]
            print(timeInfo)
        
        return timeInfo
    
    def getStartAndEndIndexFromTimeGap(self,inputData):
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
        print("please input time gap[s]")
        timeGap = float(str(raw_input()))*100
        print(timeGap)
        timeGap += self.beforeTimeGap
        start = False
        for columNum in range(len(inputData)):
            voltage = inputData[columNum,2]
            if start != True and voltage >0.3:
                startIndex = inputData[columNum,0]
                start = True
            elif start == True and voltage <0.3:
                endIndex = inputData[columNum,0]
                break
        interval = endIndex-startIndex
        timeInfo = [startIndex + timeGap,endIndex + timeGap]
        self.beforeTimeGap = timeGap

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
        print("getPoseFromViconData")
        #extract data from start index to end index
        print (len(inputData))
        processedData = []
        for columNum in range(len(inputData)):
            columData = inputData[columNum,:]
            if columData[0] >=timeInfo[0] and columData[0] <=timeInfo[1]:
                trueTime = columData[0]
                columData[0] =trueTime-timeInfo[0]
                processedData.append(columData)
        print (len(processedData))

        poses = []
        for columData in processedData:
            
            wagon=[]
            for frameNum in range(3):
                framePos = [columData[2+3*frameNum],columData[2+3*frameNum+1]]
                wagon.append(framePos)
            
            pose = self.calcurateWagonPose(wagon)
            poseData = [columData[0]*10/1000,pose[0],pose[1],pose[2]]
            #print (poseData)
            poses.append(poseData)

        robotPoses = []
        for columData in processedData:
            
            robot = []
            for frameNum in range(2):
                robotFramePos = [columData[14+3*frameNum],columData[14+3*frameNum+1]]
                robot.append(robotFramePos)
            
            robotPose = self.calcurateRobotPose(robot)
            robotPoseData = [columData[0]*10/1000,robotPose[0],robotPose[1],robotPose[2]]
            #print (poseData)
            robotPoses.append(robotPoseData)
       
        npPoses         = np.array(poses)
        npRobotPoses    = np.array(robotPoses)
        print (len(npPoses))
        return npPoses, npRobotPoses


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

    def calcurateRobotPose(self,robot):
        #wagon 0:leftfront,1:rightfront
        pose =[0,0,0]
        pose[0] = 0.5*(robot[0][0]+robot[1][0])
        pose[1] = 0.5*(robot[0][1]+robot[1][1])
        
        pose[2] = math.atan2(robot[0][1]-robot[1][1],robot[0][0]-robot[1][0])

        if pose[2]<-math.pi:
            pose[2] += 2*math.pi
        elif pose[2]>math.pi:
            pose[2] -= 2*math.pi

        #print("nan:"+str(angle1)+" "+str(angle2))
        if pose[2]>math.pi or pose[2]<-math.pi :
           pass 
        return pose

    #########################################################################
    #                        process  Ros data                              #
    #########################################################################
    def getPoseFromRosData(self,inputData):
        """
        repair pose data
        Parameters
        -------
        inputData : numpy.ndarray
       
        Returns
        -------
        npPoseData : numpy.ndarray
        [0:time,1:x,2:y,3:theta]
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

    def getPoseFromRosObservedData(self,inputData):
        """
        repair pose data
        Parameters
        -------
        inputData : numpy.ndarray
       
        Returns
        -------
        npPoseData : numpy.ndarray
        [0:time,1:x,2:y,3:theta]
        """
        poseDatas =[]
        for poseNum in range(len(inputData)):
            time = inputData[poseNum,1]
            x = inputData[poseNum,2] * 1000 +180
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

    def getRobotPoseFromRosData(self,inputData):
        """
        repair pose data
        Parameters
        -------
        inputData : numpy.ndarray
       
        Returns
        -------
        npPoseDatas : numpy.ndarray
        [0:time,1:x,2:y,3:theta]
        """    
        poseDatas =[]
        for poseNum in range(len(inputData)):
            time = inputData[poseNum,1]
            x = inputData[poseNum,6] * 1000
            y = inputData[poseNum,7] * 1000
            theta = inputData[poseNum,8] +math.pi/2
            if theta<-math.pi:
                theta += 2*math.pi
            elif theta>math.pi:
                theta -= 2*math.pi
            pose=[time,x,y,theta]
            poseDatas.append(pose)
        npPoseDatas = np.array(poseDatas)
        return npPoseDatas

    
    #########################################################################
    #                           Combine data                                #
    #########################################################################

    def reductSampling(self, viconData,rosData):
        """
        reduction sampling of vicon data based on ros data  
        Parameters
        -------
        viconData : numpy.ndarray
        rosData : numpy.ndarray
        [0:time,1:x,2:y,3:theta]

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
            #print(str(minDifference)+" "+str(rosData[rosDataNum,0])+" "+str(viconData[indexCandidate,0]))
            outputData.append(viconData[indexCandidate,:])
            #print (viconData[indexCandidate,:])
        npOutputData = np.array(outputData)
        #print (viconData)
        return npOutputData

    def calibrateLocalizationError(self,rosPose,rosRobotPose, viconRobotPose):
        """
        calibrate Localization Error
        Parameters
        -------
        rosPose : numpy.ndarray
        rosRobotPose : numpy.ndarray
        viconRobotPose : numpy.ndarray
        [0:time,1:x,2:y,3:theta]
       
        Returns
        -------
        npCaribratedPose : numpy.ndarray
        """
        caribratedPoses = []
        for dataNum in range(len(rosPose)):
            wagonPositionMatrix             = np.array([rosPose[dataNum,1],rosPose[dataNum,2]])
            wagonDirection                  = rosPose[dataNum,3]

            localizationError               = [rosRobotPose[dataNum,1]-viconRobotPose[dataNum,1],rosRobotPose[dataNum,2]-viconRobotPose[dataNum,2],rosRobotPose[dataNum,3]-viconRobotPose[dataNum,3]]
            localizationErrorPosition       = np.array([localizationError[0],localizationError[1]])
            
            calibratedWagonPositionMatrix   = wagonPositionMatrix- localizationErrorPosition#np.dot(wagonPositionMatrix- localizationErrorPosition,self.getRotationMatrix(localizationError[2])) 
            caribratedDirection             = (wagonDirection )

            caribratedPose = [rosRobotPose[dataNum,0],calibratedWagonPositionMatrix[0],calibratedWagonPositionMatrix[1],caribratedDirection] 
            caribratedPoses.append(caribratedPose)
            #print("ros")
            #print(rosRobotPose[dataNum,:])
            #print("vicon")
            #print(viconRobotPose[dataNum,:])
            #print("error")
            #print(localizationError)

        npCaribratedPose = np.array(caribratedPoses)
        return npCaribratedPose

    def getCropedData(self,data,cropTimes):
        """
        reduction sampling of vicon data based on ros data  
        Parameters
        -------
        data : numpy.ndarray
        cropTimes : array cropTimes[0]:startTime, cropTimes[1]:endTime
       
        Returns
        -------
        npPoseData : numpy.ndarray
        [time,x,y,ditance,theta]
        """
        poseDatas = []
        for dataNum in range(len(data)):
            if data[dataNum,0]>=cropTimes[0,1] and data[dataNum,0]<=cropTimes[1,1]:
                   #print(data[dataNum,0])
                   data[dataNum,0] = data[dataNum,0] - cropTimes[0,1]
                   poseDatas.append(data[dataNum,:])
        npPoseData = np.array(poseDatas)
        #print (viconData)
        return npPoseData

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
        errorData   = []
        dynamicMAE  = [0,0,0,0]
        dynamicRMSE = [0,0,0,0]
        dynamicNum  = 0
        staticMAE   = [0,0,0,0]
        staticRMSE  = [0,0,0,0]
        staticNum   = 0


        for rosDataNum in range(len(rosData)):
            time            = rosData[rosDataNum,0]
            xError          = abs(rosData[rosDataNum,1]-viconData[rosDataNum,1])
            yError          = abs(rosData[rosDataNum,2]-viconData[rosDataNum,2])
            distanceError   = abs(math.sqrt(math.pow(xError,2)+math.pow(yError,2)))
            
            if rosData[rosDataNum,3]-viconData[rosDataNum,3]>math.pi/2:
                thetaError = rosData[rosDataNum,3]-viconData[rosDataNum,3]

            thetaError      = (rosData[rosDataNum,3]-viconData[rosDataNum,3])
            if thetaError>=math.pi:
                thetaError -= 2*math.pi
            elif thetaError<=-math.pi:
                thetaError += 2*math.pi
            thetaError = abs(thetaError)

            error = [time, xError, yError, distanceError, thetaError]
            errorData.append(error)

            if time < rosData[0,0]+1.0 or time>rosData[len(rosData)-1,0]-1.0:
                staticMAE[0] += xError
                staticMAE[1] += yError
                staticMAE[2] += thetaError
                staticMAE[3] += distanceError

                staticRMSE[0] += pow(xError,2)
                staticRMSE[1] += pow(yError,2)
                staticRMSE[2] += pow(thetaError,2)
                staticRMSE[3] += pow(distanceError,2)

                staticNum +=1
            
            else:
                dynamicMAE[0] += xError
                dynamicMAE[1] += yError
                dynamicMAE[2] += thetaError
                dynamicMAE[3] += distanceError

                dynamicRMSE[0] += pow(xError,2)
                dynamicRMSE[1] += pow(yError,2)
                dynamicRMSE[2] += pow(thetaError,2)
                dynamicRMSE[3] += pow(distanceError,2)

                dynamicNum +=1

        staticMAE[0] /= staticNum 
        staticRMSE[0] /= staticNum 
        staticRMSE[0] = math.sqrt(staticRMSE[0])
        staticMAE[1] /= staticNum 
        staticRMSE[1] /= staticNum 
        staticRMSE[1] = math.sqrt(staticRMSE[1])
        staticMAE[2] /= staticNum 
        staticRMSE[2] /= staticNum 
        staticRMSE[2] = math.sqrt(staticRMSE[2])
        staticMAE[3] /= staticNum 
        staticRMSE[3] /= staticNum 
        staticRMSE[3] = math.sqrt(staticRMSE[3])

        dynamicMAE[0] /= dynamicNum 
        dynamicRMSE[0] /= dynamicNum 
        dynamicRMSE[0] = math.sqrt(dynamicRMSE[0])
        dynamicMAE[1] /= dynamicNum 
        dynamicRMSE[1] /= dynamicNum 
        dynamicRMSE[1] = math.sqrt(dynamicRMSE[1])
        dynamicMAE[2] /= dynamicNum 
        dynamicRMSE[2] /= dynamicNum 
        dynamicRMSE[2] = math.sqrt(dynamicRMSE[2])
        dynamicMAE[3] /= dynamicNum 
        dynamicRMSE[3] /= dynamicNum 
        dynamicRMSE[3] = math.sqrt(dynamicRMSE[3])

        print("staticMAE")
        print(staticMAE)
        print("staticRMSE")
        print(staticRMSE)
        print(staticNum)

        print("dynamicMAE")
        print(dynamicMAE)
        print("dynamicRMSE")
        print(dynamicRMSE)
        print(dynamicNum)
        print(len(rosData))


        npErrorData = np.array(errorData)
        return npErrorData

    def get(self, viconData,rosData):
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
        errorData   = []
        dynamicMAE  = [0,0,0,0]
        dynamicRMSE = [0,0,0,0]
        dynamicNum  = 0
        staticMAE   = [0,0,0,0]
        staticRMSE  = [0,0,0,0]
        staticNum   = 0


        for rosDataNum in range(len(rosData)):
            time            = rosData[rosDataNum,0]
            xError          = abs(rosData[rosDataNum,1]-viconData[rosDataNum,1])
            yError          = abs(rosData[rosDataNum,2]-viconData[rosDataNum,2])
            distanceError   = abs(math.sqrt(math.pow(xError,2)+math.pow(yError,2)))
            
            if rosData[rosDataNum,3]-viconData[rosDataNum,3]>math.pi/2:
                thetaError = rosData[rosDataNum,3]-viconData[rosDataNum,3]

            thetaError      = (rosData[rosDataNum,3]-viconData[rosDataNum,3])
            if thetaError>=math.pi:
                thetaError -= 2*math.pi
            elif thetaError<=-math.pi:
                thetaError += 2*math.pi
            thetaError = abs(thetaError)

            error = [time, xError, yError, distanceError, thetaError]
            errorData.append(error)

            if time < rosData[0,0]+1.0 or time>rosData[len(rosData)-1,0]-1.0:
                staticMAE[0] += xError
                staticMAE[1] += yError
                staticMAE[2] += thetaError
                staticMAE[3] += distanceError

                staticRMSE[0] += pow(xError,2)
                staticRMSE[1] += pow(yError,2)
                staticRMSE[2] += pow(thetaError,2)
                staticRMSE[3] += pow(distanceError,2)

                staticNum +=1
            
            else:
                dynamicMAE[0] += xError
                dynamicMAE[1] += yError
                dynamicMAE[2] += thetaError
                dynamicMAE[3] += distanceError

                dynamicRMSE[0] += pow(xError,2)
                dynamicRMSE[1] += pow(yError,2)
                dynamicRMSE[2] += pow(thetaError,2)
                dynamicRMSE[3] += pow(distanceError,2)

                dynamicNum +=1

        staticMAE[0] /= staticNum 
        staticRMSE[0] /= staticNum 
        staticRMSE[0] = math.sqrt(staticRMSE[0])
        staticMAE[1] /= staticNum 
        staticRMSE[1] /= staticNum 
        staticRMSE[1] = math.sqrt(staticRMSE[1])
        staticMAE[2] /= staticNum 
        staticRMSE[2] /= staticNum 
        staticRMSE[2] = math.sqrt(staticRMSE[2])
        staticMAE[3] /= staticNum 
        staticRMSE[3] /= staticNum 
        staticRMSE[3] = math.sqrt(staticRMSE[3])

        dynamicMAE[0] /= dynamicNum 
        dynamicRMSE[0] /= dynamicNum 
        dynamicRMSE[0] = math.sqrt(dynamicRMSE[0])
        dynamicMAE[1] /= dynamicNum 
        dynamicRMSE[1] /= dynamicNum 
        dynamicRMSE[1] = math.sqrt(dynamicRMSE[1])
        dynamicMAE[2] /= dynamicNum 
        dynamicRMSE[2] /= dynamicNum 
        dynamicRMSE[2] = math.sqrt(dynamicRMSE[2])
        dynamicMAE[3] /= dynamicNum 
        dynamicRMSE[3] /= dynamicNum 
        dynamicRMSE[3] = math.sqrt(dynamicRMSE[3])

        print("staticMAE")
        print(staticMAE)
        print("staticRMSE")
        print(staticRMSE)
        print(staticNum)

        print("dynamicMAE")
        print(dynamicMAE)
        print("dynamicRMSE")
        print(dynamicRMSE)
        print(dynamicNum)
        print(len(rosData))


        npErrorData = np.array(errorData)
        return npErrorData

    def getLocalizationError(self, viconData,rosData):
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
        errorData   = []


        for rosDataNum in range(len(rosData)):
            time            = rosData[rosDataNum,0]
            xError          = abs(rosData[rosDataNum,1]-viconData[rosDataNum,1])
            yError          = abs(rosData[rosDataNum,2]-viconData[rosDataNum,2])
            distanceError   = abs(math.sqrt(math.pow(xError,2)+math.pow(yError,2)))
            
            if rosData[rosDataNum,3]-viconData[rosDataNum,3]>math.pi/2:
                thetaError = rosData[rosDataNum,3]-viconData[rosDataNum,3]

            thetaError      = (rosData[rosDataNum,3]-viconData[rosDataNum,3])
            if thetaError>=math.pi:
                thetaError -= 2*math.pi
            elif thetaError<=-math.pi:
                thetaError += 2*math.pi
            thetaError = abs(thetaError)

            error = [time, xError, yError, distanceError, thetaError]
            errorData.append(error)
            print(error)
            print(rosData[rosDataNum,:])
            print(viconData[rosDataNum,:])
        npErrorData = np.array(errorData)
        return npErrorData

    def saveTimeInfo(self,fileName):
        df= pd.DataFrame({"timeFrame":self.timeInfo})
        df.to_csv(fileName+"TimeFrame.csv")

    def getRotationMatrix(self,rad):
        """ 
        get Rotation Matrix
        Parameters
        -------
        rad : dloat
       
        Returns
        -------
        rotateionMatrix : np.array
        """
        rotateionMatrix = np.array([[np.cos(rad), -np.sin(rad)],
                    [np.sin(rad), np.cos(rad)]])
        return rotateionMatrix


class Drawer(object):
    def __init__(self,datas):
        self.poseFromRosCaribrated  = datas[0]
        self.poseFromRos            = datas[1]
        self.poseFromRosObserved    = datas[2]
        self.poseFromVicon          = datas[3]
        self.error                  = datas[4]
        #self.errorLocalization      = datas[5]
    
    def draw(self):
        self.drawPose(self.poseFromVicon,self.poseFromRosCaribrated,"aa")
        self.drawEachAxis(self.poseFromVicon,self.poseFromRosCaribrated,"caribrated","aa")
        self.drawPose(self.poseFromVicon,self.poseFromRos,"bb")
        self.drawEachAxis(self.poseFromVicon,self.poseFromRos,"estimated","ee")
        #self.drawEachAxis(self.poseFromVicon,self.poseFromRosObserved,"observed")
        self.drawError(self.error,"aa")
        #self.drawError(self.errorLocalization,"aa")
        self.figShow()

    def drawPose(self,inputData,inputDataRos,name):
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
        path_plot.plot(inputDataRos[:,2],inputDataRos[:,1], linestyle="-",color="k",linewidth = 1,label = "Estimated")
        path_plot.plot(inputData[:,2],inputData[:,1], linestyle="--",color="k",linewidth = 1,label = "Ground truth")
        
        path_plot.invert_xaxis()
        path_plot.set_xlabel(r'$\it{y}\ \mathrm {[mm]}$')
        path_plot.set_ylabel(r'$\it{x}\ \mathrm {[mm]}$')
        path_plot.set_xlim(-1250,1250)
        path_plot.set_ylim(0,2500)
        path_plot.legend(fontsize=15,ncol=2)

        poseInterval = 0.5
        intervalCount=0
        for index,time in enumerate(inputDataRos[:,0]):
            if time>intervalCount*poseInterval:
                intervalCount+=1
                self.drawRectangle(path_plot,inputDataRos[index,1],inputDataRos[index,2],-(inputDataRos[index,3]-math.pi/2),500,300,True)
                self.drawRectangle(path_plot,inputData[index,1],inputData[index,2],-(inputData[index,3]-math.pi/2),500,300,False)
        
        
        path_plot.set_aspect('equal')
        fig1.savefig(relativePath+name+".eps", bbox_inches="tight", pad_inches=0.1)


    def drawEachAxis(self,inputData,inputDataRos,mode,name):

        print(len(inputData))
        print(len(inputData[0]))
        print(len(inputDataRos))
        print(len(inputDataRos[0]))

        maxTime = np.max(inputData[:,0])

        fig1 = plt.figure()

        x_plot = fig1.add_subplot(311)
        #x_plot.set_title(mode)
        
        x_plot.plot(inputDataRos[:,0],inputDataRos[:,1],linestyle="-",color="k",linewidth = 1,label = "Estimated")
        x_plot.plot(inputData[:,0],inputData[:,1],linestyle="--",color="k",linewidth = 1,label = "Ground truth")
        x_plot.set_xlim(0,maxTime)
        x_plot.set_ylabel(r"$\it{x}\ \rm{[mm]}$")
        x_plot.legend(fontsize=15,ncol=2,bbox_to_anchor=(0.5, 1.4),loc='upper center')
        
        y_plot = fig1.add_subplot(312)
        y_plot.plot(inputData[:,0],inputData[:,2],linestyle="--",color="k",linewidth = 1)
        y_plot.plot(inputDataRos[:,0],inputDataRos[:,2],linestyle="-",color="k",linewidth = 1)
        y_plot.set_xlim(0,maxTime)
        y_plot.set_ylabel(r"$\it{y}\ \rm{[mm]}$")

        theta_plot = fig1.add_subplot(313)
        theta_plot.plot(inputData[:,0],inputData[:,3],linestyle="--",color="k",linewidth = 1)
        theta_plot.plot(inputDataRos[:,0],inputDataRos[:,3],linestyle="-",color="k",linewidth = 1)
        theta_plot.set_xlim(0,maxTime)
        theta_plot.set_xlabel("Time [s]")
        theta_plot.set_ylabel(r"$\theta\ \rm{[rad]}$")
        fig1.savefig(relativePath+name+"EachAxis"+".eps", bbox_inches="tight", pad_inches=0.1)

    
    def drawError(self,errorData,name):

        maxTime = np.max(errorData[:,0])
        
        fig1 = plt.figure()
        x_plot = fig1.add_subplot(411)
        x_plot.plot(errorData[:,0],errorData[:,1],color="k",linewidth = 1)
        x_plot.set_xlim(0,maxTime)
        x_plot.set_ylim(0,100)
        x_plot.set_ylabel(r"$\varepsilon_{\it{x}} \ \rm{[mm]}$")
    
        
        y_plot = fig1.add_subplot(412)
        y_plot.plot(errorData[:,0],errorData[:,2],color="k",linewidth = 1)
        y_plot.set_xlim(0,maxTime)
        y_plot.set_ylim(0,100)
        y_plot.set_ylabel(r"$\varepsilon_{\it{y}}\ \ \rm{[mm]}$")

        distance_plot = fig1.add_subplot(413)
        distance_plot.plot(errorData[:,0],errorData[:,3],color="k",linewidth = 1)
        distance_plot.set_xlim(0,maxTime)
        distance_plot.set_ylim(0,100)
        distance_plot.set_ylabel(r"$\varepsilon_{\it{dist}} \ \rm{[mm]}$")
        


        theta_plot = fig1.add_subplot(414)
        theta_plot.plot(errorData[:,0],errorData[:,4],color="k",linewidth = 1)
        theta_plot.set_xlim(0,maxTime)
        theta_plot.set_ylim(0,0.25)
        theta_plot.set_ylabel(r"$\varepsilon_{\theta} \ \rm{[rad]}$")
        theta_plot.set_xlabel("Time [s]")

        fig1.savefig(relativePath+name+"error"+".eps", bbox_inches="tight", pad_inches=0.1)

    
        

    def drawRectangle(self,plot,x,y,theta,width,height,type):

        radius = math.sqrt(math.pow(width,2)+math.pow(height,2))/2
        angle  = math.atan2(height,width)

        vertex = []
        #point0
        point = [x + radius * math.cos(angle + theta),y + radius * math.sin(angle + theta)]
        vertex.append(point)
        #point1
        point = [x + radius * math.cos(theta + math.pi - angle), y + radius * math.sin(theta + math.pi - angle)]
        vertex.append(point)
        #point2
        point = [x + radius * math.cos(theta + math.pi + angle), y + radius * math.sin(theta + math.pi + angle)]
        vertex.append(point)
        #point3
        point = [x + radius * math.cos(theta + 2*math.pi - angle), y + radius * math.sin(theta + 2*math.pi - angle)]
        vertex.append(point)

        if type:

            plot.plot([vertex[0][1], vertex[1][1]] ,[vertex[0][0], vertex[1][0]],color='black',  linestyle='-', linewidth = 0.4)
            plot.plot([vertex[1][1], vertex[2][1]] ,[vertex[1][0], vertex[2][0]],color='black',  linestyle='-', linewidth = 0.4)
            plot.plot([vertex[3][1], vertex[2][1]] ,[vertex[3][0], vertex[2][0]],color='black',  linestyle='-', linewidth = 0.4)
            plot.plot([vertex[0][1], vertex[3][1]] ,[vertex[0][0], vertex[3][0]],color='black',  linestyle='-', linewidth = 0.4)

            plot.plot([vertex[0][1], vertex[2][1]] ,[vertex[0][0], vertex[2][0]],color='black',  linestyle='-', linewidth = 0.2)
            plot.plot([vertex[1][1], vertex[3][1]] ,[vertex[1][0], vertex[3][0]],color='black',  linestyle='-', linewidth = 0.2)
        else:
            plot.plot([vertex[0][1], vertex[1][1]] ,[vertex[0][0], vertex[1][0]],color='black',  linestyle='--', linewidth = 0.4)
            plot.plot([vertex[1][1], vertex[2][1]] ,[vertex[1][0], vertex[2][0]],color='black',  linestyle='--', linewidth = 0.4)
            plot.plot([vertex[3][1], vertex[2][1]] ,[vertex[3][0], vertex[2][0]],color='black',  linestyle='--', linewidth = 0.4)
            plot.plot([vertex[0][1], vertex[3][1]] ,[vertex[0][0], vertex[3][0]],color='black',  linestyle='--', linewidth = 0.4)

            plot.plot([vertex[0][1], vertex[2][1]] ,[vertex[0][0], vertex[2][0]],color='black',  linestyle='--', linewidth = 0.2)
            plot.plot([vertex[1][1], vertex[3][1]] ,[vertex[1][0], vertex[3][0]],color='black',  linestyle='--', linewidth = 0.2)

    def figShow(self):
        plt.pause(.01)

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
    trialName = str(raw_input())
    
    
    pathInputV              = relativePath + trialName + "V.csv" 
    pathInputPose           = relativePath + trialName + "Pose.csv" 
    pathInputRos            = relativePath + trialName + "Ros.csv"   
    pathInputRosObserved    = relativePath + trialName + "RosObserve.csv"  
    pathTimeFrame           = relativePath + trialName + "TimeFrame.csv"
    pathCropTimes           = relativePath + trialName + "CropTime.csv" 

    print("load V")
    inputV                  = loadCsv(pathInputV)
    print("load Pose")
    inputPose               = loadCsv(pathInputPose)
    print("load Ros")
    inputPoseRos            = loadCsv(pathInputRos)
    print("load RosObserve")
    inputPoseRosObserved    = loadCsv(pathInputRosObserved)
    try:
        inputTimeFrame  = loadCsv(pathTimeFrame)
    except:
        inputTimeFrame =[]
    try:
        inputCropTimes  = loadCsv(pathCropTimes)
    except:
        inputCropTimes =[]
    
    preProcessor = PreProcessor([inputV,inputPose,inputPoseRos ,inputPoseRosObserved,inputTimeFrame])
    datas = preProcessor.preProcessing()

    # draw
    drawer = Drawer(datas )
    drawer.draw()
    if len(inputCropTimes)>0:
        datas = preProcessor.cropData(datas,inputCropTimes)
        drawer = Drawer(datas )
        drawer.draw()

    while True:
        print("change start time? [y/n]" )
        answer = str(raw_input())
        if answer == "y":
            inputV                  = loadCsv(pathInputV)
            inputPose               = loadCsv(pathInputPose)
            inputPoseRos            = loadCsv(pathInputRos)
            inputPoseRosObserved    = loadCsv(pathInputRosObserved)
            
            preProcessor = PreProcessor([inputV,inputPose,inputPoseRos ,inputPoseRosObserved,inputTimeFrame])
            datas = preProcessor.changeStartTime()
            drawer.figClose()
            
            # draw
            drawer = Drawer(datas )
            drawer.draw()
            if len(inputCropTimes)>0:
                datas = preProcessor.cropData(datas,inputCropTimes)
                drawer = Drawer(datas )
                drawer.draw()

            print("ok? [y/n]" )
            answer = str(raw_input())
            if answer == "y":
                preProcessor.saveTimeInfo(relativePath + trialName)
                print("saved")
                break
            else:
                print("please retru")

        elif answer == "n":
            print("skip")
            break
        else:
            print("try again")

    while True:
        print("change crop time? [y/n]" )
        answer = str(raw_input())
        if answer == "y":

            inputV                  = loadCsv(pathInputV)
            inputPose               = loadCsv(pathInputPose)
            inputPoseRos            = loadCsv(pathInputRos)
            inputPoseRosObserved    = loadCsv(pathInputRosObserved)
            
            preProcessor = PreProcessor([inputV,inputPose,inputPoseRos ,inputPoseRosObserved,inputTimeFrame])
            datas = preProcessor.changeCropTime(relativePath + trialName)
            drawer.figClose()

            # draw
            drawer = Drawer(datas )
            
            drawer.draw()
            if len(inputCropTimes)>0:
                datas = preProcessor.cropData(datas,inputCropTimes)
                drawer = Drawer(datas )
                drawer.draw()

        elif answer == "n":
            print("skip")
            break
        else:
            print("try again")
        


if __name__ == '__main__':
    main()
