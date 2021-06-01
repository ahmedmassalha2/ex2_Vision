import numpy as np
import cv2 as cv
import sys

from time import time

from matplotlib import pyplot as plt

def AvgErr(m1, m2):
    m1, m2 = np.array(m1), np.array(m2)
    return np.mean(np.absolute(m1-m2))

def MedErr(m1, m2):
    m1, m2 = np.array(m1), np.array(m2)
    return np.median(np.absolute(m1-m2))

def Bad05_04(m1, m2):
    res05, res04, total = 0, 0, 0
    newMat = np.absolute(m1 - m2)
    for v in newMat:
        for e in v:
            total += 1
            if e > 0.5:
                res05 += 1
            if e > 4:
                res04 += 1
    return res05/total, res04/total


def calculateSSD(left, right):
    ssdValue = (left - right)**2
    return np.sum(ssdValue)

def calculateNCC(left, right):
    left = left.flatten()
    leftNorm = np.linalg.norm(left)
    right = right.flatten()
    rightNorm = np.linalg.norm(right)
    return np.correlate(left/leftNorm , right / rightNorm)

def compute_disp(left_image, right_image, R,D, objectFunction = calculateSSD):
    H, W = left_image.shape[0], left_image.shape[1]
    mat    = np.zeros((H,W))
    window = int((R - 1)/2)
    for x in range(H):        
        for y in range(W):
            
            if y - window >=0 and y + window + 1 < W and x - window >=0 and x + window + 1 < H: 
                windowLeft = left_image[x - window : x + window + 1 , y - window : y + window + 1]
                windowLeft = np.array(windowLeft)
                # Loop over window
                bestIDX, bestVal = 0, -5646469
                if objectFunction == calculateSSD:
                    bestVal = 5646469
                for d in range(y-D, y+1):
                    if d - window >= 0 and  window + d < W:
                        windowRight = right_image[x - window  : x + window + 1  , d - window: d + window + 1 ]
                        windowRight = np.array(windowRight)
                        value = objectFunction(windowLeft, windowRight)
                        if (value < bestVal and objectFunction == calculateSSD) or (value > bestVal and objectFunction != calculateSSD):
                            bestVal = value
                            bestIDX = d
                mat[x,y] = abs(bestIDX - y)
    return mat

    
        
arg = (sys.argv)     
im1 = (arg[1]) #"Q2/Moebius/im_left.png"
im2 = (arg[2])#"Q2/Moebius/im_right.png"
disp = (arg[3])#"Q2/Moebius/disp_left.png"
window = int(arg[4])
function = None
if int(arg[5]) == 1:
    print("method used: SSD")
    function = calculateSSD
else:
    print("method used: NCC")
    function = calculateNCC
startTime = time()
print("Getting images")     
img1 = cv.imread(im1)
img2 = cv.imread(im2)
disp = cv.imread(disp)
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
disp = cv.cvtColor(disp, cv.COLOR_BGR2GRAY)
print("images loaded")
maxDisparity = int(disp.max()/3)

# print(disp[0])
disp = np.array(disp)/3
# print(disp[0])
print("computing map")
mat = compute_disp(img1, img2,window, maxDisparity, function)
print('AvgErr',AvgErr(mat,disp))
print('MedErr',MedErr(mat,disp))
res05, res04 = Bad05_04(mat,disp)
print('Bad05',res05 * 100,'precent')
print('Bad04',res04 * 100,'precent')
print("Run time = ", time() - startTime)
plt.imsave('mapDisp.jpg',mat )
img1 = cv.imread('mapDisp.jpg')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imwrite('mapDisp.jpg',img1 )
f = plt.figure()
plt.imshow(img1)