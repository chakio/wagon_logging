#!/usr/bin/python
# -*- coding: utf-8 -*-

import fcntl
import termios, os
from datetime import datetime
import time
import sys
import math

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mutagen.mp3 import MP3 as mp3
import pygame

import rospy
import tf
import tf.transformations
import sensor_msgs
import std_msgs
from std_msgs.msg import String
from std_msgs.msg import Int32
from sensor_msgs.msg    import LaserScan
from geometry_msgs.msg  import PoseStamped,Twist, Quaternion,Vector3

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

    
class logger():
    def __init__(self):
        #init publisher
        pathInput              = "../data/unnpann.csv"
        print("load Pose")
        inputPose               = loadCsv(pathInput)
        
        self.time = inputPose[:,1]
        self.wagonPosesX = inputPose[:,2]
        self.wagonPosesY = inputPose[:,3]
        self.wagonPosesTheta = inputPose[:,4]
        self.endLogging()
        print(inputPose    )
        print("init")

    def endLogging(self):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["font.size"] = 20
        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.wagonPosesX,self.wagonPosesY)
        self.ax.invert_xaxis()
        
        
        poseInterval = 1
        intervalCount=0
        for index,time in enumerate(self.time):
            if time>intervalCount*poseInterval:
                intervalCount+=1
                self.drawRectangle(self.wagonPosesY[index],self.wagonPosesX[index],-(self.wagonPosesTheta[index]-math.pi/2),0.5,0.3)

        self.ax.set_xlabel('$\it{x} \mathrm {[m]}$')
        self.ax.set_ylabel('$\it{y} \mathrm {[m]}$')
        self.ax.set_aspect('equal')
        
        self.ax.set_xlim(-0.70,3.8)
        self.ax.set_ylim(-3.1,-1.4)
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()
        plt.show()
        
        #print self.data

    def drawRectangle(self,x,y,theta,width,height):

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

        self.ax.plot([vertex[0][1], vertex[1][1]] ,[vertex[0][0], vertex[1][0]],color='black',  linestyle='solid', linewidth = 0.2)
        self.ax.plot([vertex[1][1], vertex[2][1]] ,[vertex[1][0], vertex[2][0]],color='black',  linestyle='solid', linewidth = 0.2)
        self.ax.plot([vertex[3][1], vertex[2][1]] ,[vertex[3][0], vertex[2][0]],color='black',  linestyle='solid', linewidth = 0.2)
        self.ax.plot([vertex[0][1], vertex[3][1]] ,[vertex[0][0], vertex[3][0]],color='black',  linestyle='solid', linewidth = 0.2)

        self.ax.plot([vertex[0][1], vertex[2][1]] ,[vertex[0][0], vertex[2][0]],color='black',  linestyle='solid', linewidth = 0.1)
        self.ax.plot([vertex[1][1], vertex[3][1]] ,[vertex[1][0], vertex[3][0]],color='black',  linestyle='solid', linewidth = 0.1)

    

if __name__ == '__main__':
    Logger = logger()