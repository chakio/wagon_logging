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
from geometry_msgs.msg  import PoseStamped,Twist, Quaternion,Vector3,PoseArray

def callback(data):
    rospy.loginfo(rospy.get_caller_id()+"I heard %s",data.data)
    
class logger():
    def __init__(self):
        #init publisher
        self.resetContainer()
        rospy.init_node('logger', anonymous=True)
        self.absoluteSaveFolderName = "/home/ytnpc2017c/catkin_ws/src/wagon_logging/logData/"
        
        self.soundName = '/home/ytnpc2017c/catkin_ws/src/wagon_logging/sound/se.mp3' #再生したいmp3ファイル
        pygame.mixer.init()
        pygame.mixer.music.load(self.soundName) #音源を読み込み
        self.soundlength = mp3(self.soundName).info.length #音源の長さ取得
        
        print("init")
    def launchServer(self):
        #init subscriber
        estimated_pose_subscriber         = rospy.Subscriber('/hsr_wagon_world', PoseStamped, self.wagonPoseCallback)
        observed_pose_subscriber         = rospy.Subscriber('/hsr_observe_wagon_world', PoseStamped, self.observedWagonPoseCallback)
        twist_subscriber        = rospy.Subscriber('/hsr_wagonVel', Twist, self.wagonVelCallback)
        command_subscriber      = rospy.Subscriber('/hsr_logging_command', String, self.loggingCommandCallback)
        candidate_subscriber    = rospy.Subscriber('/hsr_wagon_tracking_candidate', Int32, self.candidateCallback)
        robotpose_subscriber    = rospy.Subscriber('/hsrb/base_pose', PoseStamped, self.robotPoseCallback)
        frames_subscriber       = rospy.Subscriber('/hsr_frame_world', PoseArray, self.framesCallback)
        fromFrame_pose_subscriber   = rospy.Subscriber('/hsr_wagon_world_each', PoseStamped, self.fromFrameWagonPoseCallback)

        rospy.loginfo("Ready to serve command.")
        
        rospy.spin()

    def resetContainer(self):
        self.wagonPosesX        = []
        self.wagonPosesY        = []
        self.wagonPosesTheta    = []
        self.robotPosesX        = []
        self.robotPosesY        = []
        self.robotPosesTheta    = []
        self.candidates         = []
        self.time               = []
        self.candidate          = 0
        self.data               = []

        self.observedWagonPosesX= []
        self.observedWagonPosesY= []
        self.observedWagonPosesTheta= []
        self.robotPosesXForObserve        = []
        self.robotPosesYForObserve        = []
        self.robotPosesThetaForObserve    = []
        self.observeTime               = []
        self.observeData               = []


        self.frameWagonPosesX        = []
        self.frameWagonPosesY        = []
        self.frameWagonPosesTheta    = []
        self.framesPosesX        = [[],[],[],[]]
        self.framesPosesY        = [[],[],[],[]]
        self.framesPoseX=[0,0,0,0]
        self.framesPoseY=[0,0,0,0]
        self.robotPosesXForFrame        = []
        self.robotPosesYForFrame        = []
        self.robotPosesThetaForFrame    = []
        self.frameTime               = []
        self.frameData               = []
  
        self.isLogging          = False
        print("reset")

    def startLogging(self):
        self.isLogging          = True
        self.playMusic()
        self.startTime = rospy.get_rostime()
        self.startTimeName = self.getNowTimeString()

        print("logging start")

    def endLogging(self):
        self.isLogging = False
        print("logging end")
        dataArray = {"time":self.time,"x":self.wagonPosesX,"y":self.wagonPosesY,"theta":self.wagonPosesTheta,"candidateNum":self.candidates,"rx":self.robotPosesX,"ry":self.robotPosesY,"rtheta":self.robotPosesTheta}
        self.data = pd.DataFrame(dataArray,columns=["time","x","y","theta","candidateNum","rx","ry","rtheta","ox","oy","otheta"])
        self.saveData(self.data,0)
        observeDataArray = {"time":self.observeTime,"x":self.observedWagonPosesX,"y":self.observedWagonPosesY,"theta":self.observedWagonPosesTheta,"rx":self.robotPosesXForObserve,"ry":self.robotPosesYForObserve,"rtheta":self.robotPosesThetaForObserve}
        self.data = pd.DataFrame(observeDataArray,columns=["time","x","y","theta","rx","ry","rtheta"])
        self.saveData(self.data,1)
        fromFrameDataArray = {"time":self.frameTime,"x":self.frameWagonPosesX,"y":self.frameWagonPosesY,"theta":self.frameWagonPosesTheta,"rx":self.robotPosesXForFrame,"ry":self.robotPosesYForFrame,"rtheta":self.robotPosesThetaForFrame,"f1x":self.framesPosesX[0],"f1y":self.framesPosesY[0],"f2x":self.framesPosesX[1],"f2y":self.framesPosesY[1],"f3x":self.framesPosesX[2],"f3y":self.framesPosesY[2],"f4x":self.framesPosesX[3],"f4y":self.framesPosesY[3]}
        self.data = pd.DataFrame(fromFrameDataArray,columns=["time","x","y","theta","rx","ry","rtheta","f1x","f1y","f2x","f2y","f3x","f3y","f4x","f4y"])
        self.saveData(self.data,2)
        self.playMusic()

        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.wagonPosesY,self.wagonPosesX)
        self.ax.invert_xaxis()
        
        
        poseInterval = 0.5
        intervalCount=0
        for index,time in enumerate(self.time):
            if time>intervalCount*poseInterval:
                intervalCount+=1
                self.drawRectangle(self.wagonPosesX[index],self.wagonPosesY[index],self.wagonPosesTheta[index],0.5,0.3)

        self.ax.set_aspect('equal')
        self.fig2, self.ax2 = plt.subplots()
        self.ax2.plot(self.frameWagonPosesY,self.frameWagonPosesX)
        self.ax2.plot(self.framesPosesY[0],self.framesPosesX[0])
        self.ax2.plot(self.framesPosesY[1],self.framesPosesX[1])
        self.ax2.plot(self.framesPosesY[2],self.framesPosesX[2])
        self.ax2.plot(self.framesPosesY[3],self.framesPosesX[3])
        self.ax2.invert_xaxis()
        
        self.ax2.set_aspect('equal')
        plt.show()
        print("closed")
        #print self.data

    def drawRectangle(self,x,y,theta,width,height):

        radius = math.sqrt(math.pow(width,2)+math.pow(height,2))/2
        angle  = math.atan2(width,height)

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

        self.ax.plot([vertex[0][1], vertex[1][1]] ,[vertex[0][0], vertex[1][0]],color='black',  linestyle='solid', linewidth = 0.5)
        self.ax.plot([vertex[1][1], vertex[2][1]] ,[vertex[1][0], vertex[2][0]],color='black',  linestyle='solid', linewidth = 0.5)
        self.ax.plot([vertex[3][1], vertex[2][1]] ,[vertex[3][0], vertex[2][0]],color='black',  linestyle='solid', linewidth = 0.5)
        self.ax.plot([vertex[0][1], vertex[3][1]] ,[vertex[0][0], vertex[3][0]],color='black',  linestyle='solid', linewidth = 0.5)

        self.ax.plot([vertex[0][1], vertex[2][1]] ,[vertex[0][0], vertex[2][0]],color='black',  linestyle='solid', linewidth = 0.25)
        self.ax.plot([vertex[1][1], vertex[3][1]] ,[vertex[1][0], vertex[3][0]],color='black',  linestyle='solid', linewidth = 0.25)

    def saveData(self,data, mode):
        if mode==1:
            data.to_csv(self.absoluteSaveFolderName+self.startTimeName+"observe.csv")
        elif mode ==0:
            data.to_csv(self.absoluteSaveFolderName+self.startTimeName+".csv")
        elif mode ==2:
            data.to_csv(self.absoluteSaveFolderName+self.startTimeName+"EachFrame.csv")
        print("saved")

    def quaternionToEuler(self,quaternion):
        """
        Convert Quaternion to Euler Angles
        quarternion: geometry_msgs/Quaternion
        euler: geometry_msgs/Vector3
        """
        e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        return Vector3(x=e[0], y=e[1], z=e[2])

    def getNowTimeString(self):
        now = datetime.now()
        formatedNow = 'test_{0:%Y%m%d}_{0:%H%M}_{0:%S}'.format(now)
        return formatedNow

    def playMusic(self):
        pygame.mixer.music.play(1) #再生開始。1の部分を変えるとn回再生(その場合は次の行の秒数も×nすること)
        #pygame.mixer.music.stop() #音源の長さ待ったら再生停止

    def wagonPoseCallback(self,pose):
        if self.isLogging:
            self.wagonPosesX.append(pose.pose.position.x)
            self.wagonPosesY.append(pose.pose.position.y)
            self.wagonPosesTheta.append(self.quaternionToEuler(pose.pose.orientation).z)
            self.candidates.append(self.candidate)

            self.time.append((pose.header.stamp-self.startTime).to_sec())

            self.robotPosesX.append(self.robotPoseX)
            self.robotPosesY.append(self.robotPoseY)
            self.robotPosesTheta.append(self.robotPoseTheta)

            #print((pose.header.stamp-self.startTime).to_sec())

    def observedWagonPoseCallback(self,pose):
        if self.isLogging:
            self.observedWagonPosesX.append(pose.pose.position.x)
            self.observedWagonPosesY.append(pose.pose.position.y)
            self.observedWagonPosesTheta.append(self.quaternionToEuler(pose.pose.orientation).z)

            self.observeTime.append((pose.header.stamp-self.startTime).to_sec())
            
            self.robotPosesXForObserve.append(self.robotPoseX)
            self.robotPosesYForObserve.append(self.robotPoseY)
            self.robotPosesThetaForObserve.append(self.robotPoseTheta)
            print((pose.header.stamp-self.startTime).to_sec())
    def robotPoseCallback(self,pose):
       
        self.robotPoseX=pose.pose.position.x
        self.robotPoseY=pose.pose.position.y
        self.robotPoseTheta=self.quaternionToEuler(pose.pose.orientation).z

    def fromFrameWagonPoseCallback(self,pose):
        if self.isLogging:
            self.frameWagonPosesX.append(pose.pose.position.x)
            self.frameWagonPosesY.append(pose.pose.position.y)
            self.frameWagonPosesTheta.append(self.quaternionToEuler(pose.pose.orientation).z)
            for frameNum in range(4):
                self.framesPosesX[frameNum].append(self.framesPoseX[frameNum])
                self.framesPosesY[frameNum].append(self.framesPoseY[frameNum])

            self.frameTime.append((pose.header.stamp-self.startTime).to_sec())
            
            self.robotPosesXForFrame.append(self.robotPoseX)
            self.robotPosesYForFrame.append(self.robotPoseY)
            self.robotPosesThetaForFrame.append(self.robotPoseTheta)

    def framesCallback(self,poseArray):
        
        for frameNum in range(4):
            self.framesPoseX[frameNum]=poseArray.poses[frameNum].position.x
            self.framesPoseY[frameNum]=poseArray.poses[frameNum].position.y

            
    def wagonVelCallback(self,velocity):
        pass
    def candidateCallback(self,data):
        self.candidate = data.data
    def loggingCommandCallback(self,command):
        if command.data == "start":
            self.resetContainer()
            self.startLogging()
            
        elif command.data == "end":
            self.endLogging()

if __name__ == '__main__':
    Logger = logger()
    Logger.launchServer()