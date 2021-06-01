# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:51:01 2021

@author: Rapi
"""

#! /usr/bin/env python2

"""
I/O script to save and load the data coming with the MPI-Sintel low-level
computer vision benchmark.

For more details about the benchmark, please visit www.mpi-sintel.de

CHANGELOG:
v1.0 (2015/02/03): First release

Copyright (c) 2015 Jonas Wulff
Max Planck Institute for Intelligent Systems, Tuebingen, Germany

"""

# Requirements: Numpy as PIL/Pillow
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import math
import copy as cp
# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'
   
def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 1082261504, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth


def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def GetRotationMatrix(theta_rx, theta_ry, theta_rz):
    # calculate cos and sin of angles
    sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
    sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
    sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
    R_Mx = np.matrix([[1, 0, 0,0],
                      [0, cos_rx, -sin_rx,0],
                      [0, sin_rx, cos_rx,0],
                      [0,0,0,1]])

    R_My = np.matrix([[cos_ry, 0, sin_ry,0],
                      [0, 1, 0,0],
                      [-sin_ry, 0, cos_ry,0],
                      [0,0,0,1]])

    R_Mz = np.matrix([[cos_rz, -sin_rz, 0,0],
                      [sin_rz, cos_rz, 0,0],
                      [0, 0, 1,0],
                      [0,0,0,1]])

    return R_Mx , R_My , R_Mz 


def getExtrinsic(rotation = None, translation = None):
    ext = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    ext = np.array(ext)
    if rotation == None:
        rotation = np.array([0,0,0])
    if translation == None:
        translation = np.array([0,0,0])
     
        
    rotation[0] = (rotation[0])*np.pi/180#math.radians(rotation[0]) 
    rotation[1] = (rotation[1])*np.pi/180#math.radians(rotation[1]) 
    rotation[2] = (rotation[2] )*np.pi/180#math.radians(rotation[2]) 
    ext = [[1,0,0,translation[0]],[0,1,0,translation[1]],[0,0,1,translation[2]],[0,0,0,1]]
    ext = np.array(ext)
    x,y,z = GetRotationMatrix(rotation[0],rotation[1],rotation[2])
    ext_ = x@(y@(z@ext))
    # ext[0:3,0:3] = ext_
    
    return ext_
        
        


def getRotationFrames():
    rot = np.zeros(12)
    rot += 0.5
    rot[0], rot[11] = 0, 0
    frames = [rot,[-p for p in rot[::-1]],[-p for p in rot[::-1]],rot]
    return frames


def getTranslationFrames():
    t = np.zeros(12)
    t += 0.5
    t[0], t[11] = 0, 0
    frames = [t,[-p for p in t[::-1]],[-p for p in t[::-1]],t]
    return frames
rotationFrames, translationFrames = getRotationFrames(), getTranslationFrames()

cam = "Q3/ambush_6.cam"
depth = "Q3/ambush_6.dpt"
im = "Q3/ambush_6.png"
depth = depth_read(depth)
img1 = cv.imread(im)
newIm = np.zeros(img1.shape)
(M,N) = cam_read(cam)
intInvers = M
fx, fy, cx, cy = intInvers[0,0],intInvers[1,1],intInvers[0,2],intInvers[1,2]
intInvers = np.linalg.inv(M)
ext = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
ext = np.array(ext)
rot = np.zeros(12)
rot += 0.5
rot[0], rot[1] = 0, 0
frames = [rot,[-p for p in rot[::-1]],[-p for p in rot[::-1]],rot]
T = np.zeros(12)
T += 0.5
T[0] = 0
framesT = [T,[1*-p for p in T],[1*-p for p in T],T]
j=0
rotation = [0,0,0]
translation = [0,0,0]
for frame in translationFrames:
    
    for i in frame:
        print(i)
        rotation[0]+=10

        newIm = np.zeros(img1.shape)
        newEx = getExtrinsic(cp.deepcopy(rotation),cp.deepcopy(translation))
        newEx = np.array([newEx[0], newEx[1],newEx[2]])
        for u in range(depth.shape[0]):
            for v in range(depth.shape[1]):
                d = depth[u, v]
                x = (u-cx) * d / fx
                y = (v - cy) * d / fy
                camPoint = [x,y,d,1]
                newPoint = M@(newEx@camPoint)
                newPoint = newPoint/newPoint[2]
                if int(newPoint[0]) < img1.shape[0] and int(newPoint[1]) < img1.shape[1] and int(newPoint[0]) >=0 and int(newPoint[1])>=0:
                    newIm[int(newPoint[0]),int(newPoint[1])] = img1[u,v]
                
        
                
        cv.imwrite(str(j)+'sample_out_1.png', newIm)
        j+=1
        exit()


