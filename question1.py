
import cv2 as cv
import math
import random
import numpy as np
import copy as cp
from matplotlib import pyplot as plt
import sys



def getLine(F, p):
    return F.dot(p)


def drawlines(img,pts1,pts2,F,from_):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,s = img.shape
    img1 = img
    lines = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), from_,cp.deepcopy(F))
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        f = cp.deepcopy(F)
        print('dddd',pt1, pt2)
        r = getLine(f,pt2)
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        print(x0,y0)
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
    cv.imwrite('im2.jpg',img1)
    return img1


def getMatchedPoints(im1, im2):
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    #Ransac run to get 8 matched points
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    pts1, pts2 = [], []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    

    return np.int32(pts1), np.int32(pts2)

def pointFromLine(r,c):
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
    return [(x0,y0),(x1,y1)]


def drawLines2(img1, img2, pts1,pts2,F):
    r1,c1,s1 = img1.shape
    r2,c2,s2 = img2.shape
    for pt1,pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        line2 = getLine(cp.deepcopy(F), pt1)
        line2 = pointFromLine(line2,c2)
        
        
        
        
        line1 = getLine(np.transpose(cp.deepcopy(F)), pt2)
        line1 = pointFromLine(line1,c1)
        
        
        img1 = cv.line(img1, line1[0], line1[1], color,2)
        img1 = cv.circle(img1,tuple((pt1[0],pt1[1])),6,color,-1)
        
        img2 = cv.line(img2, line2[0], line2[1], color,2)
        img2 = cv.circle(img2,tuple((pt2[0],pt2[1])),6,color,-1)
        
        
    return img1, img2
        
def algError1(F,pts1,pts2):
    print("Calculating algebric error:")
    total = 0
    for pt1,pt2 in zip(pts1,pts2):
        x = np.array([pt1[0],pt1[1],1])
        x_ = np.array([pt2[0],pt2[1],1])
        # print(x,F)
        res = F.dot(x)
        res = abs(x_.dot(res))
        total += res
        print("point error: ",res)
    print("=================================")
    print("Avg error: ",total/len(pt1))
    print("=================================")
            
 
def getD(x,x_,l):
    #a^2 + b^2
    part2 = math.sqrt(l[0]**2 + l[1]**2)
    return (x_.dot(l))/part2
    
def algError2(F,pts1,pts2):
    print("Calculating epipolar error:")
    total = 0
    F_ = np.transpose(F)
    for pt1,pt2 in zip(pts1,pts2):
        x = np.array(pt1)
        x_ = np.array(pt2)
        
        l = F.dot(x)
        
        d1 = getD(x,x_,F.dot(x)) ** 2
        d2 = getD(x_,x,F_.dot(x_)) ** 2
        total += (d1+d2)
        print("point error: ",d1+d2)
    print("=================================")
    print("Avg error: ",total/len(pt1))
    print("=================================")




arg = (sys.argv)     
img1Path = (arg[1]) 
img2Path = (arg[2])
if arg[3] == '7':
    alg = cv.FM_7POINT
else:
    alg = cv.FM_8POINT

img1 = cv.imread(img1Path)
img2 = cv.imread(img2Path)

pts1, pts2 = None, None
if 'im_family' in img1Path:
    pts1 = [(412, 150, 1), (441, 320, 1), (121, 48, 1),  (229, 292, 1),  (560, 303, 1),(624,174, 1),(87,67, 1),(291,239, 1)]
    pts2 = [(288, 130, 1), (291, 343, 1), (575, 32, 1),  (227, 290, 1), (545, 339, 1),(700,153, 1),(555,54, 1),(533,236, 1)]
else:
    pts1 = [ (280, 129, 1),    (282, 229, 1),  (206,256, 1),    (313,358, 1),  (296,443, 1),  (527,531, 1),   (726,63, 1),    (5, 351, 1)]
    pts2 = [ (314, 90, 1),     (318, 174, 1),  (330,190, 1),    (393,266, 1),  (462,296, 1),  (637,304, 1),    (654,36, 1),    (269,255,1)]


pts1  = np.asarray(pts1)
pts2 = np.asarray(pts2)


F, mask = cv.findFundamentalMat(pts1,pts2,alg  )

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

pts1 = [np.array(p) for p in pts1]
pts2 = [np.array(p) for p in pts2]

im1,im2 = drawLines2(img1, img2, pts1,pts2,cp.deepcopy(F))
algError1(F,pts1,pts2)
print("\n\n")
algError2(F,pts1,pts2)

f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(im1)
cv.imwrite('im1.png',im1)
f.add_subplot(1,2, 2)
plt.imshow(im2)
cv.imwrite('im2.png',im2)
plt.show(block=True)


