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

# my_logger
from cloudChamberCommonCode import my_logger

# web camera calibration
from cloudChamberCommonCode import calibrationFactor

# IO
from cloudChamberCommonCode import IO
from cloudChamberCommonCode import rawDataDirectory
from cloudChamberCommonCode import rawDataFileName

#Merging Parameters
#from cloudChamberCommonCode import maxLinePointDistance 
#from cloudChamberCommonCode import maxRelativeAngle  
#from cloudChamberCommonCode import maxRelativeDistance 

# Good cluster analysis parameters 
#from cloudChamberCommonCode import goodClusterMinClusterTransverseSigma 
#from cloudChamberCommonCode import goodClusterMaxClusterTransverseSigma 
#from cloudChamberCommonCode import goodClusterMinClusterLongitudinalSigma

# Good Cluster Selection
from cloudChamberCommonCode import goodCluster

# Settings of the logger
my_logger.info("Removing no-good and Correlated cluster in two images of the Cloud Chamber")

from cloudChamberCommonCode import timePeriod
# Removing process parameters
# Maximum relative distance between two correlated clusters in two different images
from cloudChamberCommonCode import maxCorrelatedRelativeDistance 
# Maximum relative angle between two correlated clusters in two different images
from cloudChamberCommonCode import maxCorrelatedRelativeAngle 
# Best choice of the correlated cluster between j=0 ad j=1
from cloudChamberCommonCode import qualitySigmaShort 

def main() :

  # Reading Merged cluster data from merging process.py
  clusterMergedDict = {}
  clusterMergedFile = open(rawDataDirectory + "mergingFragmentedClusterData.dat", "rb")
  clusterMergedDict= pickle.load(clusterMergedFile)
  clusterMergedFile.close()
  
  # Single distributions of selected clusters
  lengthDistribution     = np.empty(0,dtype=float)
  transverseDistribution = np.empty(0,dtype=float)
  meanXDistribution      = np.empty(0,dtype=float)
  meanYDistribution      = np.empty(0,dtype=float)
  angleDistribution      = np.empty(0,dtype=float)
  sizeDistribution       = np.empty(0,dtype=float)
  goodClusterCounter = 0

  # Relative comparison of clusters from consecutive images
  relativeLengthDistribution = np.empty(0,dtype=float)
  relativeLengthDistribution2 = np.empty(0,dtype=float)
  relativeAngleDistribution = np.empty(0,dtype=float)
  relativeAngleDistribution2 = np.empty(0,dtype=float)

  # Initializing list of cluster to be removed from file
  removeClusterListDict = {}
  for iImage, clusterList in clusterMergedDict.items() :
    removeClusterListDict[iImage]=[]

  my_logger.info("Data %s" %(rawDataDirectory))
  my_logger.info("Files are %s" %(rawDataFileName))
  my_logger.info("Identifying correlated clusters to be removed" )
  for iImage, clusterList in clusterMergedDict.items() :
    if(iImage%timePeriod==0) :
      my_logger.info("--- analyzing cluster list from image %d" %(iImage) )
    for cluster in clusterList :  
      if ( goodCluster(cluster) and not(cluster[1] in removeClusterListDict[iImage])) :  
        secondClusterBetter = False  
        for j in [1,2,3,4,5,6,7] :
          jImage = cluster[0]+j

          # Sometimes the cluster in iImage+1 is better than the initial cluster. 
          # if this happens the secondClusterBetter becomes true and 
          # all the comparison for j>Image+1 are skipped
          if (not (secondClusterBetter)) :
            if (jImage in clusterMergedDict) :
              #my_logger.info("--- Comparing with cluster list from image %d" %(jImage) )
              clusterList2 = clusterMergedDict[jImage]
              for cluster2 in clusterList2 :
                relativeDistance = math.sqrt ( math.pow(cluster[2]-cluster2[2], 2)  +
                                           math.pow((cluster[3]-cluster2[3]),2) ) 
                relativeAngle = math.fabs(cluster[4]-cluster2[4])
              
                if (relativeDistance < maxCorrelatedRelativeDistance/calibrationFactor and relativeAngle < maxCorrelatedRelativeAngle) :
                  if (j==-1) :
                    if ( math.fabs(cluster[6]-qualitySigmaShort) < math.fabs(cluster2[6]-qualitySigmaShort) ) : 
                      removeClusterListDict[jImage].append(cluster2[1])
                    else :                
                      removeClusterListDict[iImage].append(cluster[1])
                      secondClusterBetter = True
                  else :  
                    removeClusterListDict[jImage].append(cluster2[1])
                else : 
                  relativeLengthDistribution2 = np.append(relativeLengthDistribution2, relativeDistance)
                  relativeAngleDistribution2  = np.append(relativeAngleDistribution2, relativeAngle)
                relativeAngleDistribution   = np.append(relativeAngleDistribution, relativeAngle)
                relativeLengthDistribution  = np.append(relativeLengthDistribution, relativeDistance)
        
  my_logger.info("Removing no-good correlated clusters in image %d" %(iImage))  
  clusterDictRemoved= {}
  for iImage, clusterList in clusterMergedDict.items() :
    clusterListRemoved=[]
    for cluster in clusterList :
      if(goodCluster(cluster) and not(cluster[1] in removeClusterListDict[iImage])) :
        lengthDistribution     = np.append(lengthDistribution,2.0*calibrationFactor*cluster[5])
        transverseDistribution = np.append(transverseDistribution, 2.0*calibrationFactor*cluster[6])
        meanXDistribution      = np.append(meanXDistribution, calibrationFactor* cluster[2])
        meanYDistribution      = np.append(meanYDistribution, calibrationFactor* cluster[3])
        angleDistribution      = np.append(angleDistribution, cluster[4])
        sizeDistribution       = np.append(sizeDistribution, cluster[7])
        clusterListRemoved.append(cluster)
        if (2.0*calibrationFactor*cluster[5] > 55.) :
          my_logger.info("- Long cluster in image %d and length %f5.1 " %(iImage, 2.0*calibrationFactor*cluster[5]))
    clusterDictRemoved[iImage]=clusterListRemoved

  # Saving Selected cluster in dictionary using iImage as key and the list of clusters as value 
  clusterFile = open(rawDataDirectory + "MergedNoncorrelatedClusterData.dat", "wb")
  my_logger.info("Saving selected clusters in %s" %(clusterFile.name))
  pickle.dump(clusterDictRemoved, clusterFile)
  clusterFile.close()
          
  my_logger.info("Plots for cluster analysis" )    
  fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

  # Main Plot with the length track distribution
  ax[0,0].set_yscale("log")
  ax[0,0].set_ylim(.1, 1000)
  ax[0,0].set_xlim(0.,100.)
  ax[0,0].set_xlabel(r"$\rm{Track \; length \; l (mm)}$")
  ax[0,0].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
  #Fixing bin width
  binWidth =1.0
  #bin center calculation
  lengthValues=np.arange(min(lengthDistribution), max(lengthDistribution) + binWidth, binWidth)
  #bin center calculation
  histoValuesOut, lengthValues, patches =  ax[0,0].hist(lengthDistribution, bins=lengthValues, log=False )
  lengthValuesCenter = np.array([0.5 * (lengthValues[i] + lengthValues[i+1]) for i in range(len(lengthValues)-1)])
  histoValuesOut[histoValuesOut<0.] = 0.
  histoErrorsOut = np.sqrt(histoValuesOut)
  histoErrorsOut[histoErrorsOut==0] = 1.
  #ax[0,0].errorbar(lengthValuesCenter, histoValuesOut, yerr=histoErrorsOut, fmt='o')
  print(len(lengthDistribution[(lengthDistribution>15.) & (lengthDistribution<30.)]))

  ax[1,0].set_yscale('linear')
  ax[1,0].set_ylim(.1, 5000)
  ax[1,0].set_xlim(0.,30.)
  ax[1,0].set_xlabel(r"$\rm{Track \; transverse length \; l (mm)}$")
  ax[1,0].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
  #Fixing bin width
  binWidth =0.25
  transverseValues=np.arange(min(transverseDistribution), max(transverseDistribution) + binWidth, binWidth)
  histoValues, transverseValues, patches =  ax[1,0].hist(transverseDistribution, bins=transverseValues, log=True )

  ax[0,1].set_yscale('linear')
  ax[0,1].set_ylim(.1, 500)
  ax[0,1].set_xlim(0.,300)
  ax[0,1].set_xlabel(r"$\rm{x (mm)}$")
  ax[0,1].set_ylabel(r"$\rm{N/dx \; (mm)}$")
  #Fixing bin width
  binWidth =5.0
  meanXvalues=np.arange(min(meanXDistribution), max(meanXDistribution) + binWidth, binWidth)
  histoValues, meanXvalues, patches =  ax[0,1].hist(meanXDistribution, bins=meanXvalues, log=True )

  ax[1,1].set_yscale('linear')
  ax[1,1].set_ylim(.1, 700)
  ax[1,1].set_xlim(0.,500.)
  ax[1,1].set_xlabel(r"$\rm{y (mm)}$")
  ax[1,1].set_ylabel(r"$\rm{N/dy \; (mm)}$")
  #Fixing bin width
  binWidth =5.0
  meanYvalues=np.arange(min(meanYDistribution), max(meanYDistribution) + binWidth, binWidth)
  histoValues, meanYvalues, patches =  ax[1,1].hist(meanYDistribution, bins=meanYvalues, log=True )

  ax[2,0].set_yscale('linear')
  ax[2,0].set_ylim(.1, 200)
  ax[2,0].set_xlim(-180.,180.)
  ax[2,0].set_xlabel(r"$\rm{angle (deg)}$")
  ax[2,0].set_ylabel(r"$\rm{N/dtheta}$")
  #Fixing bin width
  binWidth =1.0
  angleValues=np.arange(min(angleDistribution), max(angleDistribution) + binWidth, binWidth)
  histoValues, angleValues, patches =  ax[2,0].hist(angleDistribution, bins=angleValues, log=True )

  #ax[2,1].set_yscale('log')
  #ax[2,1].set_ylim(.1, 5000)
  #ax[2,1].set_xlim(0.,5000.)
  #ax[2,1].set_xlabel(r"$\rm{size}$")
  #ax[2,1].set_ylabel(r"$\rm{N/dsize}$")
  #Fixing bin width
  #binWidth =20.
  #sizeValues=np.arange(min(sizeDistribution), max(sizeDistribution) + binWidth, binWidth)
  #histoValues, sizeValues, patches =  ax[2,1].hist(sizeDistribution, bins=sizeValues, log=True )

  ax[2,1].set_yscale("linear")
  ax[2,1].set_ylim(.1, 10000)
  ax[2,1].set_xlim(0.,500.)
  ax[2,1].set_xlabel(r"$\rm{Relative \; length \; l (pixel)}$")
  ax[2,1].set_ylabel(r"$\rm{dN/dlr \; (pixel)}$")
  #Fixing bin width
  binWidth =0.25
  #bin center calculation
  lengthValues=np.arange(min(relativeLengthDistribution), max(relativeLengthDistribution) + binWidth, binWidth)
  #bin center calculation
  histoValues, lengthValues, patches =  ax[2,1].hist(relativeLengthDistribution, bins=lengthValues, log=True )
  
  ax[0,2].set_yscale("linear")
  ax[0,2].set_ylim(.1, 10000)
  ax[0,2].set_xlim(0.,500.)
  ax[0,2].set_xlabel(r"$\rm{Relative \; length \; non correlated \; l (pixel)}$")
  ax[0,2].set_ylabel(r"$\rm{dN/dlr \; (pixel)}$")
  #Fixing bin width
  binWidth =0.25
  #bin center calculation
  lengthValues=np.arange(min(relativeLengthDistribution2), max(relativeLengthDistribution2) + binWidth, binWidth)
  #bin center calculation
  histoValues, lengthValues, patches =  ax[0,2].hist(relativeLengthDistribution2, bins=lengthValues, log=True )
  histoValues, lengthValues, patches =  ax[2,1].hist(relativeLengthDistribution2, bins=lengthValues, log=True )


  ax[1,2].set_yscale('linear')
  ax[1,2].set_ylim(.1, 10000)
  ax[1,2].set_xlim(0.,180.)
  ax[1,2].set_xlabel(r"$\rm{Rel angle  (non correlated)}$")
  ax[1,2].set_ylabel(r"$\rm{N/dtheta}$")
  #Fixing bin width
  binWidth =0.5
  angleValues=np.arange(min(relativeAngleDistribution), max(relativeAngleDistribution) + binWidth, binWidth)
  histoValues, angleValues, patches =  ax[1,2].hist(relativeAngleDistribution, bins=angleValues, log=True )


  ax[2,2].set_yscale('linear')
  ax[2,2].set_ylim(.1, 10000)
  ax[2,2].set_xlim(0.,180.)
  ax[2,2].set_xlabel(r"$\rm{Rel angle (deg)}$")
  ax[2,2].set_ylabel(r"$\rm{N/dtheta}$")
  #Fixing bin width
  binWidth =0.5
  angleValues=np.arange(min(relativeAngleDistribution2), max(relativeAngleDistribution2) + binWidth, binWidth)
  histoValues, angleValues, patches =  ax[2,2].hist(relativeAngleDistribution2, bins=angleValues, log=True )
  histoValues, angleValues, patches =  ax[1,2].hist(relativeAngleDistribution2, bins=angleValues, log=True )

  plt.savefig(rawDataDirectory + "removingAnalysis_ControlPlots.pdf")
  plt.show()
 

  # Representing selected cluster in the clus_filt jpeg images
  filtClusImg = (np.zeros(0), np.zeros(0), np.zeros(0))
  for iImage, clusterList in clusterDictRemoved.items() :
    fileName = rawDataDirectory + "filt_aber_"+rawDataFileName+str(iImage)+".jpeg"
    outputFileName = rawDataDirectory + "corr_filt_aber_"+rawDataFileName+str(iImage)+".jpeg"
    isHere = os.path.isfile(fileName)
    if isHere:
      filtClusImg = cv2.imread(fileName)
      for cluster in clusterList :
        ellipseCenter = ( int(cluster[2]), int(cluster[3]))
        ellipseRadius =  ( int(1.25*cluster[5]), int(3.*cluster[6]))
        ellipseAngle = cluster[4]
        cv2.ellipse(filtClusImg, (ellipseCenter[0], ellipseCenter[1]), (ellipseRadius[0],ellipseRadius[1]), ellipseAngle, 0., 360.,(0,255,0))
      cv2.imwrite(outputFileName, filtClusImg.astype(np.uint8))
  
if __name__ == "__main__" :
  rc = main()
  sys.exit(rc)