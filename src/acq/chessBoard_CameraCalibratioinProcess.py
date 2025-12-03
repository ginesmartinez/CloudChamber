import os
import numpy as np
import cv2 as cv
import glob

# my_logger
from cloudChamberCommonCode import my_logger

# IO
from cloudChamberCommonCode import IO
from cloudChamberCommonCode import rawDataDirectory
from cloudChamberCommonCode import rawDataFileName

# ChessBoard Processing Parameters
from cloudChamberCommonCode import interestArea_x1
from cloudChamberCommonCode import interestArea_y1
from cloudChamberCommonCode import interestArea_x2
from cloudChamberCommonCode import interestArea_y2
from cloudChamberCommonCode import iImageIIntegral
from cloudChamberCommonCode import iImageFIntegral
from cloudChamberCommonCode import fname
from cloudChamberCommonCode import nx
from cloudChamberCommonCode import ny

# Filtering Processing Parameters
from cloudChamberCommonCode import timePeriod

#termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
#print(objp)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image place

images = glob.glob('*.jpeg') #get the images path
#print(images)

if (fname != "noCorrection") : 
    my_logger.info("Reference for chessboard correction %s" %(fname))
    img = cv.imread(fname) #load the image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #convert an image from BGR (Blue Green Red) 2 (to) GRAY
    invert_gray = 255 - gray #invert the black&white image (help with detection of chessboard corner because background must be white)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(invert_gray, (nx,ny), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp) #save the corner locations
    
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) #Refines the corner locations
        imgpoints.append(corners2) #save the refined corners locations
        #print(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (nx,ny), corners2, ret) #drawing the chessboard corners
        newfname = fname.replace('.jpeg', '_CBC.jpeg') #new file name CBC for ChessBoardCorners
        cv.imwrite(newfname, img) #saving the image'''
        my_logger.info("Corners for chessboard correction %s" %(newfname))

        #find the camera matrix (mtx), the distortions coefficients (dist), the rotation vectors (rvecs) and translation verctors (tvecs)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  
        #print("input Shape :", gray.shape[::-1])
        #print("output ret", ret)
        #print("Output mtx :", mtx)
        #print("Output dist :", dist)
        #print("Output rvecs :", rvecs)
        #print("Output tvecs :", tvecs )


        h,  w = img.shape[:2]
        #print("optimal:", img.shape[:2])
        #newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        #print("Output newcameramtx:", newcameramtx)
        #print("Output roi : ", roi) 

        dst = cv.undistort(gray, mtx, dist, None, mtx)
        #x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]
        newfname = fname.replace('.jpeg', '_Corrected.jpeg') #new file name Corrected for ChessBoardCorners
        cv.imwrite(newfname, dst)
        newfname2 = fname.replace('.jpeg', '_CorrectedSelected.jpeg') #new file name Corrected for ChessBoardCorners
        my_logger.info("Corrected chessboard image save in %s" %(fname))
        cv.imwrite(newfname2, dst[interestArea_y1:interestArea_y2,interestArea_x1:interestArea_x2])

my_logger.info("Data %s" %(rawDataDirectory))
my_logger.info("Files are %s" %(rawDataFileName))

if ((fname != "noCorrection") and ret ) :
    my_logger.info("Correcting images from %4d to %4d" %(iImageIIntegral, iImageFIntegral))
else :
    my_logger.info("NO-correction images from %4d to %4d" %(iImageIIntegral, iImageFIntegral))
io = IO(rawDataDirectory, rawDataFileName+"{}.jpeg")
 
# Iterating over images
iImage = iImageIIntegral
while (iImage < iImageFIntegral ) :
    # Loading raw image
    img = io.read(iImage)
    if (iImage%timePeriod ==0 ) : 
        if ((fname != "noCorrection") and ret ) :
            my_logger.info("Creating aberration corrected image  %4d" %(iImage))
        else :
            my_logger.info("Creating no-aberration corrected image  %4d" %(iImage))
    
    if ((fname != "noCorrection") and ret ) :
        dst = cv.undistort(img, mtx, dist, None, mtx)
    else :
        dst = img
    correctedFileName = io.dir + "aber_"+ io.fileTemplate.format(iImage)
    cv.imwrite(correctedFileName, dst[interestArea_y1:interestArea_y2,interestArea_x1:interestArea_x2])
    iImage=iImage+1

# image rate
firstFileName = io.dir + io.fileTemplate.format(iImageIIntegral)
lastFileName = io.dir + io.fileTemplate.format(iImageFIntegral)
imageRate = float(iImageFIntegral - iImageIIntegral)/(os.path.getmtime(lastFileName) - os.path.getmtime(firstFileName))
my_logger.info("First image is  %s" %(firstFileName))
my_logger.info("Last image is  %s" %(lastFileName))
my_logger.info("Image rate is %5.2f" %(imageRate))

    
