# Building

import sys
import logging
import os
import cv2
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import cv2
import pickle

# my_logger
from cloudChamberCommonCode import my_logger

# webcam calibration factor
from cloudChamberCommonCode import calibrationFactor

# IO
from cloudChamberCommonCode import IO
from cloudChamberCommonCode import rawDataDirectory
from cloudChamberCommonCode import rawDataFileName
from cloudChamberCommonCode import timePeriod

#Merging Parameters
from cloudChamberCommonCode import maxLinePointDistance 
from cloudChamberCommonCode import maxRelativeAngle  
from cloudChamberCommonCode import maxRelativeDistance 

# Good cluster analysis parameters 
#from cloudChamberCommonCode import goodClusterMinClusterTransverseSigma 
#from cloudChamberCommonCode import goodClusterMaxClusterTransverseSigma 
#from cloudChamberCommonCode import goodClusterMinClusterLongitudinalSigma

# Good Cluster Selection
from cloudChamberCommonCode import goodCluster

# Settings of the logger
my_logger.info("Data %s" %(rawDataDirectory))
my_logger.info("Files are %s" %(rawDataFileName))
my_logger.info("Merging fragmented cluster")


def distanceLinePoint(point, line) :
  distance= math.fabs( -math.sin(line[2]*math.pi/180.)*(point[0]-line[0]) + math.cos(line[2]*math.pi/180.)*(point[1]-line[1]))
  return distance

def main() :

  io = IO(rawDataDirectory, "clus_filt_"+ rawDataFileName + "{}.jpeg")

  # Reading Raw cluster data from filteringProcess.py
  clusterDict = {}
  rawClusteringData = io.dir + "rawClusteringData.dat"
  clusterFile = open(rawClusteringData, "rb")
  clusterDict= pickle.load(clusterFile)
  clusterFile.close()

  # Merging fragmented clusters
  clusterDictMerged= {}
  mergingFragmentedClusterData = rawDataDirectory + "mergingFragmentedClusterData.dat"
  clusterMergedFile = open(mergingFragmentedClusterData, "wb")

  relativeDistanceDistributionSameEvent = np.empty(0,dtype=float)
  relativeDistanceDistributionSameEvent2 = np.empty(0,dtype=float)
  relativeAngleDistributionSameEvent = np.empty(0,dtype=float)
  relativeDistanceClusterLineDistributionSameImage = np.empty(0,dtype=float)
  
  for iImage,clusterList in clusterDict.items() :
    clusterMergedList = []
    mergedClusterNumberList = []
    if (iImage%timePeriod ==0) :
      my_logger.info("--- Merging fragmented cluster for image %d" %(iImage) )
    for cluster in clusterList :
      mergingStatus = 0
      if (goodCluster(cluster) and not(cluster[1] in mergedClusterNumberList))  :
        line = (cluster[2], cluster[3], cluster[4])
        point = (cluster[2], cluster[3])
        for cluster2 in clusterList :
          if (cluster2[1] != cluster[1] and not(cluster2[1] in mergedClusterNumberList) )  :
            # Calculation of the distance from cluster2 mean to the cluster line
            line2 = (cluster2[2], cluster2[3], cluster2[4])
            point2 = (cluster2[2], cluster2[3])
            relativeDistanceClusterLineSameImage = min(distanceLinePoint(point2, line),distanceLinePoint(point,line2))
            relativeDistanceSameImage = math.sqrt ( math.pow(cluster[2]-cluster2[2], 2)  +
                                           math.pow((cluster[3]-cluster2[3]),2) ) 
            relativeAngleSameImage = math.fabs(cluster[4]-cluster2[4])
            
            relativeDistanceDistributionSameEvent  = np.append(relativeDistanceDistributionSameEvent, relativeDistanceSameImage)
            relativeAngleDistributionSameEvent   = np.append(relativeAngleDistributionSameEvent, relativeAngleSameImage)
            relativeDistanceClusterLineDistributionSameImage = np.append(relativeDistanceClusterLineDistributionSameImage, relativeDistanceClusterLineSameImage)            
            
            if (relativeDistanceClusterLineSameImage<maxLinePointDistance/calibrationFactor and relativeAngleSameImage<maxRelativeAngle) :
              relativeDistanceDistributionSameEvent2  = np.append(relativeDistanceDistributionSameEvent2, relativeDistanceSameImage)

            # Merging condition
            if (relativeDistanceClusterLineSameImage<maxLinePointDistance/calibrationFactor and relativeAngleSameImage < maxRelativeAngle and relativeDistanceSameImage <maxRelativeDistance/calibrationFactor) :    
              cluster[13].extend(cluster2[13])
              mergingStatus = True
              mergedClusterNumberList.append(cluster2[1])               

        # Not optimal since this is the same code as in filteringProcess python program
        np_cluster = np.array( cluster[13] ).T
        clSize = np_cluster.shape[1]
        mean = np.mean( np_cluster, axis=1)
        cov = np.cov( np_cluster )
        ev, vp = np.linalg.eig( cov ) 
              
        if (ev[0]>ev[1]) : 
          theta = math.atan(vp[1,0]/vp[0,0])*180./math.pi
          sigmaLong = math.sqrt(12.*ev[0])/2.
          sigmaShort = math.sqrt(12.*ev[1])/2.
          extremeHighA = mean + vp[:,0]* math.sqrt(12.*ev[0])/2. 
          extremeHighB = mean - vp[:,0]* math.sqrt(12.*ev[0])/2. 
          extremeLowA = mean + vp[:,1]* math.sqrt(12.*ev[1])/2. 
          extremeLowB = mean - vp[:,1]* math.sqrt(12.*ev[1])/2. 
        else :
          theta = math.atan(vp[1,1]/vp[0,1])*180./math.pi
          sigmaLong = math.sqrt(12.*ev[1])/2.
          sigmaShort = math.sqrt(12.*ev[0])/2.
          extremeHighA = mean + vp[:,1]* math.sqrt(12.*ev[1])/2. 
          extremeHighB = mean - vp[:,1]* math.sqrt(12.*ev[1])/2. 
          extremeLowA = mean + vp[:,0]* math.sqrt(12.*ev[0])/2. 
          extremeLowB = mean - vp[:,0]* math.sqrt(12.*ev[0])/2. 
          theta= theta
        clusterMergedList.append((iImage, cluster[1],  mean[0], mean[1], theta, sigmaLong, sigmaShort, clSize, extremeHighA, extremeHighB, extremeLowA, extremeLowB, mergingStatus, cluster))
        
    clusterDictMerged[iImage]= clusterMergedList  
    #print(mergedClusterNumberList)  
    
  # Saving cluster in dictionary using iImage as key and the list of Merged clusters as value 
  my_logger.info("Saving Merged Clusters  in %s" %(clusterMergedFile.name))
  pickle.dump(clusterDictMerged, clusterMergedFile)
  clusterMergedFile.close()  

  # Representing merged cluster in the clus_filt jpeg images
  filtClusImg = (np.zeros(0), np.zeros(0), np.zeros(0))
  for cle,valeur in clusterDictMerged.items() :
    fileName = rawDataDirectory + "clus_filt_aber_" + rawDataFileName + str(cle) +  ".jpeg"  
    #print("yoyo", fileName)
    outputFileName = rawDataDirectory + "merg_clus_filt_aber_" + rawDataFileName + str(cle)+ ".jpeg"
    isHere = os.path.isfile(fileName)
    if isHere :
      #print("yoyo", fileName)
      filtClusImg = cv2.imread(fileName)
      for cluster in valeur :
        ellipseCenter = ( int(cluster[2]), int(cluster[3]))
        ellipseRadius =  ( int(1.25*cluster[5]), int(3.*cluster[6]))
        ellipseAngle = cluster[4]
        cv2.ellipse(filtClusImg, (ellipseCenter[0], ellipseCenter[1]), (ellipseRadius[0],ellipseRadius[1]), ellipseAngle, 0., 360.,(0,255,0))
      cv2.imwrite(outputFileName, filtClusImg.astype(np.uint8))
      #print("tutu", outputFileName)

  my_logger.info("Merging Pair cluster condition")
  my_logger.info("--- Relative line to point distance %5.1f pixels" %(maxLinePointDistance/calibrationFactor) )  
  my_logger.info("--- Relative angle %5.1f degres" %(maxRelativeAngle) ) 
  my_logger.info("--- Relative cluster to cluster distance %5.1f pixels" %(maxRelativeDistance/calibrationFactor) )  
  

  my_logger.info("Plots for cluster Merging" )    
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

  # Relative Distance Distribution in the same event
  ax[0,0].set_yscale("log")
  ax[0,0].set_ylim(.1, 1000)
  ax[0,0].set_xlim(0.,600.)
  ax[0,0].set_xlabel(r"$\rm{Relative \; Distance  \; SE (pixel)}$")
  ax[0,0].set_ylabel(r"$\rm{dN/dl \; (pixel)}$")
  #Fixing bin width
  binWidth =2.0
  #bin center calculation
  lengthValues=np.arange(min(relativeDistanceDistributionSameEvent), max(relativeDistanceDistributionSameEvent) + binWidth, binWidth)
  #bin center calculation
  histoValuesOut, lengthValues, patches =  ax[0,0].hist(relativeDistanceDistributionSameEvent, bins=lengthValues, log=False )
  lengthValuesCenter = np.array([0.5 * (lengthValues[i] + lengthValues[i+1]) for i in range(len(lengthValues)-1)])
  histoValuesOut[histoValuesOut<0.] = 0.
  histoErrorsOut = np.sqrt(histoValuesOut)
  histoErrorsOut[histoErrorsOut==0] = 1.
  #ax[0,0].errorbar(lengthValuesCenter, histoValuesOut, yerr=histoErrorsOut, fmt='o')
  
  # Relative Distance Distribution in the same event
  ax[0,1].set_yscale("log")
  ax[0,1].set_ylim(.1, 1000)
  ax[0,1].set_xlim(0.,600.)
  ax[0,1].set_xlabel(r"$\rm{Relative \; Distance \; Cut \; SE (pixel)}$")
  ax[0,1].set_ylabel(r"$\rm{dN/dl \; (pixel)}$")
  #Fixing bin width
  binWidth =2.0
  #bin center calculation
  lengthValues=np.arange(min(relativeDistanceDistributionSameEvent2), max(relativeDistanceDistributionSameEvent2) + binWidth, binWidth)
  #bin center calculation
  histoValuesOut, lengthValues, patches =  ax[0,1].hist(relativeDistanceDistributionSameEvent2, bins=lengthValues, log=False )
  lengthValuesCenter = np.array([0.5 * (lengthValues[i] + lengthValues[i+1]) for i in range(len(lengthValues)-1)])
  histoValuesOut[histoValuesOut<0.] = 0.
  histoErrorsOut = np.sqrt(histoValuesOut)
  histoErrorsOut[histoErrorsOut==0] = 1.
  #ax[0,0].errorbar(lengthValuesCenter, histoValuesOut, yerr=histoErrorsOut, fmt='o')

  # Relative Angle Distribution in the same event
  ax[1,0].set_yscale("log")
  ax[1,0].set_ylim(.1, 2500)
  ax[1,0].set_xlim(-200.,200.)
  ax[1,0].set_xlabel(r"$\rm{Relative \; Angle  \; SE (pixel)}$")
  ax[1,0].set_ylabel(r"$\rm{dN/dangle \; (degres)}$")
  #Fixing bin width
  binWidth =1.0
  #bin center calculation
  lengthValues=np.arange(min(relativeAngleDistributionSameEvent), max(relativeAngleDistributionSameEvent) + binWidth, binWidth)
  #bin center calculation
  histoValuesOut, lengthValues, patches =  ax[1,0].hist(relativeAngleDistributionSameEvent, bins=lengthValues, log=False )
  lengthValuesCenter = np.array([0.5 * (lengthValues[i] + lengthValues[i+1]) for i in range(len(lengthValues)-1)])
  histoValuesOut[histoValuesOut<0.] = 0.
  histoErrorsOut = np.sqrt(histoValuesOut)
  histoErrorsOut[histoErrorsOut==0] = 1.
  #ax[0,0].errorbar(lengthValuesCenter, histoValuesOut, yerr=histoErrorsOut, fmt='o')

  # Relative Distance Distribution mean cluster 2 to line cluster in the same event
  ax[1,1].set_yscale("log")
  ax[1,1].set_ylim(.1, 600)
  ax[1,1].set_xlim(0.,100.)
  ax[1,1].set_xlabel(r"$\rm{Relative \; point-line Distance  \; SE (pixel)}$")
  ax[1,1].set_ylabel(r"$\rm{dN/dl \; (pixel)}$")
  #Fixing bin width
  binWidth =0.25
  #bin center calculation
  lengthValues=np.arange(min(relativeDistanceClusterLineDistributionSameImage), max(relativeDistanceClusterLineDistributionSameImage) + binWidth, binWidth)
  #bin center calculation
  histoValuesOut, lengthValues, patches =  ax[1,1].hist(relativeDistanceClusterLineDistributionSameImage, bins=lengthValues, log=False )
  lengthValuesCenter = np.array([0.5 * (lengthValues[i] + lengthValues[i+1]) for i in range(len(lengthValues)-1)])
  histoValuesOut[histoValuesOut<0.] = 0.
  histoErrorsOut = np.sqrt(histoValuesOut)
  histoErrorsOut[histoErrorsOut==0] = 1.
  #ax[0,0].errorbar(lengthValuesCenter, histoValuesOut, yerr=histoErrorsOut, fmt='o')
  
  plt.savefig(io.dir + "mergingFragmentedCluster_ControlPlots2.pdf")
  plt.show()
 

# Generating control plots
  lengthDistribution = np.empty(0,dtype=float)
  lengthDistribution2 = np.empty(0,dtype=float)
  transverseDistribution = np.empty(0,dtype=float)
  transverseDistribution2 = np.empty(0,dtype=float)
  meanXDistribution = np.empty(0,dtype=float)
  meanYDistribution = np.empty(0,dtype=float)
  angleDistribution = np.empty(0,dtype=float)
  sizeDistribution = np.empty(0,dtype=float)
  mergedStatusCounter = 0
  clusterCounter =0
  for cle, valeur in clusterDictMerged.items() :
    for cluster in valeur :
      lengthDistribution     = np.append(lengthDistribution,calibrationFactor * 2.0 * cluster[5])
      transverseDistribution = np.append(transverseDistribution, calibrationFactor * 2.0* cluster[6])
      meanXDistribution      = np.append(meanXDistribution, calibrationFactor * cluster[2])
      meanYDistribution      = np.append(meanYDistribution, calibrationFactor * cluster[3])
      angleDistribution      = np.append(angleDistribution, cluster[4])
      sizeDistribution       = np.append(sizeDistribution, cluster[7])
      clusterCounter = clusterCounter+1
      if (cluster[12]==1) :
        mergedStatusCounter = mergedStatusCounter + 1 
      if (cluster[12]==0) :
        lengthDistribution2     = np.append(lengthDistribution2,calibrationFactor * 2.0 * cluster[5])
        transverseDistribution2 = np.append(transverseDistribution2, calibrationFactor * 2.0* cluster[6])
  my_logger.info("Merging cluster statistics" )
  my_logger.info("--- Number of cluster %d" %(clusterCounter) )
  my_logger.info("--- Number of merged cluster %d" %(mergedStatusCounter) )
      
  # Main Plot Page
  fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
 
  ax[0,0].set_yscale("log")
  ax[0,0].set_ylim(.1, 5000)
  ax[0,0].set_xlim(0.,100.)
  ax[0,0].set_xlabel(r"$\rm{Track \; length \; l (mm)}$")
  ax[0,0].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
  #Fixing bin width
  binWidth =1.0
  #bin center calculation
  lengthValues=np.arange(min(lengthDistribution), max(lengthDistribution) + binWidth, binWidth)
  #bin center calculation
  histoValues, lengthValues, patches =  ax[0,0].hist(lengthDistribution, bins=lengthValues, log=True )

  ax[1,0].set_yscale("log")
  ax[1,0].set_ylim(.1, 5000)
  ax[1,0].set_xlim(0.,100.)
  ax[1,0].set_xlabel(r"$\rm{Track \; length \; noMerged \; l (mm)}$")
  ax[1,0].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
  #Fixing bin width
  binWidth =1.0
  #bin center calculation
  lengthValues2=np.arange(min(lengthDistribution2), max(lengthDistribution2) + binWidth, binWidth)
  #bin center calculation
  histoValues, lengthValues2, patches =  ax[1,0].hist(lengthDistribution2, bins=lengthValues2, log=True )

  ax[0,1].set_yscale('log')
  ax[0,1].set_ylim(.1, 10000)
  ax[0,1].set_xlim(0.,30.)
  ax[0,1].set_xlabel(r"$\rm{Track \; transverse length \; l (mm)}$")
  ax[0,1].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
  #Fixing bin width
  binWidth =0.5
  transverseValues=np.arange(min(transverseDistribution), max(transverseDistribution) + binWidth, binWidth)
  histoValues, transverseValues, patches =  ax[0,1].hist(transverseDistribution, bins=transverseValues, log=True )

  ax[1,1].set_yscale('log')
  ax[1,1].set_ylim(.1, 10000)
  ax[1,1].set_xlim(0.,30.)
  ax[1,1].set_xlabel(r"$\rm{Track \; transverse length \; NoMerged \; l (mm)}$")
  ax[1,1].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
  #Fixing bin width
  binWidth =0.5
  transverseValues2=np.arange(min(transverseDistribution2), max(transverseDistribution2) + binWidth, binWidth)
  histoValues, transverseValues2, patches =  ax[1,1].hist(transverseDistribution2, bins=transverseValues2, log=True )


  ax[0,2].set_yscale('log')
  ax[0,2].set_ylim(.1, 5000)
  ax[0,2].set_xlim(0.,300.)
  ax[0,2].set_xlabel(r"$\rm{x (mm)}$")
  ax[0,2].set_ylabel(r"$\rm{N/dx \; (mm)}$")
  #Fixing bin width
  binWidth =5.0
  meanXvalues=np.arange(min(meanXDistribution), max(meanXDistribution) + binWidth, binWidth)
  histoValues, meanXvalues, patches =  ax[0,2].hist(meanXDistribution, bins=meanXvalues, log=True )

  ax[1,2].set_yscale('log')
  ax[1,2].set_ylim(.1, 5000)
  ax[1,2].set_xlim(0.,500.)
  ax[1,2].set_xlabel(r"$\rm{y (mm)}$")
  ax[1,2].set_ylabel(r"$\rm{N/dy \; (mm)}$")
  #Fixing bin width
  binWidth =5.0
  meanYvalues=np.arange(min(meanYDistribution), max(meanYDistribution) + binWidth, binWidth)
  histoValues, meanYvalues, patches =  ax[1,2].hist(meanYDistribution, bins=meanYvalues, log=True )

  ax[0,3].set_yscale('log')
  ax[0,3].set_ylim(.1, 5000)
  ax[0,3].set_xlim(-200.,200.)
  ax[0,3].set_xlabel(r"$\rm{angle (deg)}$")
  ax[0,3].set_ylabel(r"$\rm{N/dtheta}$")
  #Fixing bin width
  binWidth =1.0
  angleValues=np.arange(min(angleDistribution), max(angleDistribution) + binWidth, binWidth)
  histoValues, angleValues, patches =  ax[0,3].hist(angleDistribution, bins=angleValues, log=True )

  ax[1,3].set_yscale('log')
  ax[1,3].set_ylim(.1, 5000)
  ax[1,3].set_xlim(0.,20000.)
  ax[1,3].set_xlabel(r"$\rm{size}$")
  ax[1,3].set_ylabel(r"$\rm{N/dsize}$")
  #Fixing bin width
  binWidth =100
  sizeValues=np.arange(min(sizeDistribution), max(sizeDistribution) + binWidth, binWidth)
  histoValues, sizeValues, patches =  ax[1,3].hist(sizeDistribution, bins=sizeValues, log=True )
  plt.savefig(io.dir+"mergingFragmentedCluster_ControlPlots.pdf")
  plt.show()

if __name__ == "__main__" :
  rc = main()
  sys.exit(rc)