# Building
import sys
import logging
import os
import cv2
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# caution: path[0] is reserved for script path (or '' in REPL)
# path[1] in the directory where cloudChamberCommonCode can be
sys.path.insert(1, "../")

# my_logger
from cloudChamberCommonCode import my_logger

# Filtering Processing Parameters
from cloudChamberCommonCode import seuil
from cloudChamberCommonCode import iImageI
from cloudChamberCommonCode import iImageF
from cloudChamberCommonCode import imagesPerSecond
from cloudChamberCommonCode import integrationTime
from cloudChamberCommonCode import deltaTimeStep
from cloudChamberCommonCode import filteringOption
from cloudChamberCommonCode import occupancyFittingOption

# Background Estimation Parameters
from cloudChamberCommonCode import seuilDiff
from cloudChamberCommonCode import timeStep
from cloudChamberCommonCode import timePeriod

# IO
from cloudChamberCommonCode import IO
from cloudChamberCommonCode import rawDataDirectory
from cloudChamberCommonCode import rawDataFileName

# Settings of the logger
my_logger.info("Filtering of the Cloud Chamber pictures")

def exponentialconstant(X, Nzero, T0, Ncons ) :
    functionValue = Nzero * np.exp(-X/T0) + Ncons
    return (functionValue)

def constantFunction(X, Ncons ) :
    functionValue = Ncons + X- X
    return (functionValue)            

def background(io, iImage0) :
  # Initialization
  jImage =  iImage0+timeStep 
  img1 = io.read(iImage0)
  img2 = io.read(jImage)
  imgDiff = np.absolute(img2.astype(np.float16)- img1.astype(np.float16))
  imgBack = np.where (( imgDiff < seuilDiff ) , img1 , 0.)  
  counters = np.where (( imgDiff < seuilDiff ) , 1. , 0.)  

  # Comparing pairs of images and keeping zones where they are similar
  iImage=iImage0+timeStep
  img1 = io.read(iImage)
  while ( ((io.end) == False) and (iImage <(iImage0+timePeriod) )  ) :
    jImage = iImage + timeStep
    img2 = io.read(jImage)
    while ( ((io.end) == False) and jImage <(iImage+timePeriod) ) :
      jImage=jImage+timeStep
      img2 = io.read(jImage)
      imgDiff = np.absolute(img2.astype(np.float16)- img1.astype(np.float16))
      imgBack = np.where (( imgDiff < seuilDiff ), imgBack+img1 , imgBack)  
      counters = np.where (( imgDiff < seuilDiff ) , counters+1. , counters)  
    iImage = iImage + timeStep
    img1 = io.read(iImage)

  # Renormalization on pixel by pixel basis to generate the background image
  imgBack = imgBack/counters
  
  return imgBack

def filtering(imgBina) :
  idx = np.where(imgBina>0)
  imgFilt = imgBina
  if len(idx[0]) : 
    for iPix in np.nditer(idx):
      i = iPix[0]
      j = iPix[1]
      if ((i>2) and (j>2)) :
        imgArea = imgBina[i-3:i+3,j-3:j+3]
        imgFidu = imgBina[i-2:i+2,j-2:j+2]
        if (np.sum(imgArea)- np.sum(imgFidu) ==0) :
          #print(i, j, imgFilt[i,j])
          imgFilt[i-3:i+3,j-3:j+3]=0.
          #print(i, j, imgFilt[i,j])
  return imgFilt

def main() :
  if (filteringOption) :
    my_logger.info("Data %s" %(rawDataDirectory))
    my_logger.info("Files are %s" %(rawDataFileName))
    my_logger.info("Filtering images from %4d to %4d" %(iImageI, iImageF))
    io = IO(rawDataDirectory,"aber_"+ rawDataFileName + "{}.jpeg")
    
    # Iterating over images
    iImage = iImageI
    while (iImage < iImageF) :
      # Loading raw image
      img = io.read(iImage)
      if (iImage%timePeriod ==0) :
        my_logger.info("Creating filtering image  %4d" %(iImage))

      # Generation of background image every minutes (timePeriod images)
      if ((iImage%timePeriod == 0 ) or (iImage == iImageI) ) :
        imgBack = background(io, iImage)
        backFileName = io.dir + "bck_"+ io.fileTemplate.format(iImage)
        my_logger.info("Generating background image of the  cloud chamber for image %4d for %4d images " %(iImage, timePeriod))
        cv2.imwrite(backFileName, imgBack.astype(np.uint8))  
 
      # Generation of image difference with respect to to background image  
      imgDiff = np.absolute(img.astype(np.float16)- imgBack.astype(np.float16))
      diffFileName = io.dir + "diff_"+ io.fileTemplate.format(iImage)
      cv2.imwrite(diffFileName, imgDiff.astype(np.uint8))

      # Generation of binary image, over threshold
      imgBina = np.where( imgDiff.astype(np.uint8)>seuil, 255,0)
      BinaFileName = io.dir + "bina_"+ io.fileTemplate.format(iImage)
      cv2.imwrite(BinaFileName,imgBina.astype(np.uint8) )

      # Generation of filtered image to remove small potential cluster
      imgFilt = filtering (imgBina)
      FiltFileName = io.dir + "filt_"+ io.fileTemplate.format(iImage)
      cv2.imwrite(FiltFileName,imgFilt.astype(np.uint8) )

      iImage=iImage+1
    
  my_logger.info("Data %s" %(rawDataDirectory))
  my_logger.info("Files are %s" %(rawDataFileName))
  my_logger.info("Reading filtered images from %4d to %4d" %(iImageI, iImageF))
  io2 = IO(rawDataDirectory, "filt_aber_"+ rawDataFileName + "{}.jpeg")
  
  # Pixel occupancy distribution
  numberOfpoints = int((iImageF-iImageI)/integrationTime)
  timeStamp = np.zeros(numberOfpoints,dtype=float)
  timeStampError = np.ones(numberOfpoints,dtype=float)
  pixelOccupancyDistribution = np.zeros(numberOfpoints,dtype=float)
  pixelOccupancySquareDistribution = np.zeros(numberOfpoints,dtype=float)
  normalisation = 1.
  
  # Iterating over images
  iImage = iImageI
  while (iImage < iImageF) :
    
    imgFilt = io2.read(iImage)
    #print(imgFilt.shape[0], imgFilt.shape[1])
    #print(iImage)
    my_logger.info("Reading filtered image  %4d" %(iImage))
    
    #Normalisation
    if (iImage == iImageI) :
      normalisation =  float(len( np.where(imgFilt<300)[0]))
      print(normalisation) 

    #Occupancy calculation
    position = int ( (iImage -iImageI)/(integrationTime)) 
    relativeOccupancy = float(len( np.where(imgFilt>200)[0])) # binarized image 0 or 255
    relativeOccupancy = relativeOccupancy/normalisation*1000.
    pixelOccupancyDistribution[position] = pixelOccupancyDistribution[position] + relativeOccupancy
    pixelOccupancySquareDistribution[position] = pixelOccupancySquareDistribution[position] + (relativeOccupancy*relativeOccupancy)
    timeStamp[position] = timeStamp[position]+ float(iImage -iImageI) / imagesPerSecond / float(integrationTime/deltaTimeStep)
    iImage=iImage+int(deltaTimeStep)
  
  pixelOccupancyDistribution = pixelOccupancyDistribution/float(integrationTime/deltaTimeStep)
  pixelOccupanceErrorDistribution = np.sqrt(pixelOccupancySquareDistribution - pixelOccupancyDistribution*pixelOccupancyDistribution)/(integrationTime/deltaTimeStep)
  
#  for t in np.nditer(timeStamp) :
#    print(t)  
  # Main Plot Page
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
 
  ax.set_yscale("linear")
  ax.set_ylim(.1, 1.20*max(pixelOccupancyDistribution))
  ax.set_xlim(0.,(iImageF-iImageI)/imagesPerSecond )
  ax.set_xlabel(r"$\rm{Time \; (s)}$")
  ax.set_ylabel(r"$\rm{PerthousandOfPixelPerImage}$")

  # make data:
  ax.errorbar(timeStamp, pixelOccupancyDistribution, pixelOccupanceErrorDistribution, timeStampError, 'bo')

  # Fitting
  if (occupancyFittingOption == "exp") : 
    p1 = [1.,55., 1.]
    bounds1 = ([0.001, 6., 0.001],[1000., 500., 1000.])
    popt, pcov = curve_fit(f=exponentialconstant, xdata=timeStamp, ydata=pixelOccupancyDistribution, p0=p1,  bounds=bounds1)    
    #print(popt)
    #print(pcov)
    N0 = popt[0]
    N0Error = math.sqrt(pcov[0,0])
    my_logger.info("N0 is  %4.2f +- %4.3f pixel" %(N0, N0Error))
    halfTime = popt[1]*math.log(2)
    haltTimeError = math.sqrt(pcov[1,1])*math.log(2)
    my_logger.info("Half life time is  %4.2f +- %4.3f seconds" %(halfTime, haltTimeError))
    Nconst = popt[2]
    NconstError = math.sqrt(pcov[2,2])
    my_logger.info("Nconst is  %4.2f +- %4.3f pixel" %(Nconst, NconstError))
    ax.plot(timeStamp, exponentialconstant(timeStamp, *popt), color='red', linewidth=2.5, label=r'Fitted function')
    chi2 = np.sum( (exponentialconstant(timeStamp, *popt)-pixelOccupancyDistribution)*(exponentialconstant(timeStamp, *popt)-pixelOccupancyDistribution)/pixelOccupanceErrorDistribution/pixelOccupanceErrorDistribution)
    my_logger.info("Number of degrees of freedom is   %4d " %(len(timeStamp)))
    my_logger.info("chi2 is  %4.2f " %(chi2))
  
  else :
    p1 = [1.]
    bounds1 = ([0.0001],[1000.])
    popt, pcov = curve_fit(f=constantFunction, xdata=timeStamp, ydata=pixelOccupancyDistribution, p0=p1,  bounds=bounds1)    
    #print(popt)
    #print(pcov)
   
    Nconst = popt[0]
    NconstError = math.sqrt(pcov[0,0])
    my_logger.info("Nconst is  %4.2f +- %4.3f pixel" %(Nconst, NconstError))
    ax.plot(timeStamp, constantFunction(timeStamp, *popt), color='red', linewidth=2.5, label=r'Fitted function')
    chi2 = np.sum( (constantFunction(timeStamp, *popt)-pixelOccupancyDistribution)*(constantFunction(timeStamp, *popt)-pixelOccupancyDistribution)/pixelOccupanceErrorDistribution/pixelOccupanceErrorDistribution)
    my_logger.info("Number of degrees of freedom is   %4d " %(len(timeStamp)))
    my_logger.info("chi2 is  %4.2f " %(chi2))
    my_logger.info("chi2/n is  %4.2f " %(chi2/numberOfpoints))
   
  plt.savefig(rawDataDirectory+"filtering_ControlPlots.pdf")
  plt.show()

if __name__ == "__main__" :
  rc = main()
  sys.exit(rc)