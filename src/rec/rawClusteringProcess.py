# Building

import sys
sys.path.insert(1, "../")
import logging
import os
import cv2
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt


# my_logger
from cloudChamberCommonCode import my_logger

# webcam calibration factor
from cloudChamberCommonCode import calibrationFactor

# Filtering Processing Parameters
from cloudChamberCommonCode import iImageI
from cloudChamberCommonCode import iImageF
from cloudChamberCommonCode import timePeriod

# clustering size threshold
from cloudChamberCommonCode import clusterSizeThreshold 

# IO
from cloudChamberCommonCode import IO
from cloudChamberCommonCode import rawDataDirectory
from cloudChamberCommonCode import rawDataFileName


# Settings of the logger
my_logger.info("Raw clustering of the filtered Cloud Chamber pictures")

def clusterizing(binaImg, iImage):
   
    clusterList = []
    ni = binaImg.shape[0]
    nj = binaImg.shape[1]
    dType = binaImg.dtype
    clusImg = np.zeros( (ni, nj, 3),dtype=dType)
    clusImg[:,:,0] = binaImg[:,:]
    clusImg[:,:,1] = binaImg[:,:]
    clusImg[:,:,2] = binaImg[:,:]
    
    done = np.ones( (ni, nj),dtype=dType)
    idx = np.where( binaImg > 0 )
    done[idx] = 0
    grp = 256
    if (iImage%timePeriod ==0) :
      my_logger.info("Clustering filtered binary image %4d" %(iImage))

    while (np.sum(done) != ni*nj):
      # new group/cluster
      idx = np.where(done == 0)
      i = idx[0][0]
      j = idx[1][0]
      neigh = [(i,j)]
      cluster = []
      clusterForImg = []
      while( len(neigh) > 0 ):
        (i, j) = neigh.pop()
        done[i,j] = 1
        clusterForImg.append((i,j))  
        # Permute i, j because of how display imshow
        cluster.append((j,i))
        if  ((i-1) >= 0) and (done[i-1, j] == 0):
          neigh.append( (i-1,j))
        if ((i+1) < ni) and ( done[i+1, j] == 0 ):
          neigh.append( (i+1,j))
        if ((j-1) >= 0) and ( done[i, j-1] == 0 ):
          neigh.append( (i,j-1))
        if ((j+1) < nj) and  ( done[i, j+1] == 0 ):
          neigh.append( (i,j+1))
      # End of a cluster
      np_cluster = np.array( cluster ).T
      clSize = np_cluster.shape[1]
      
      # Generating of cluster parameters
      if (len(cluster) > clusterSizeThreshold ):
        grp -= 1
        iCluster =256-grp
        # compute cluster features
        # mean_x = np.mean( np_cluster, axis=0)
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
        mergingStatus= False
        #if (theta>90.) :
        #  theta = theta-180.
        clusterList.append((iImage, iCluster,  mean[0], mean[1], theta, sigmaLong, sigmaShort, clSize, extremeHighA, extremeHighB, extremeLowA, extremeLowB, mergingStatus, cluster))
        my_logger.debug("iImage   : %4d " %(iImage))
        my_logger.debug("- iCluster : %4d " %(iCluster))
        my_logger.debug("- Position   : %6.3f, %6.3f" %(mean[0], mean[1]))
        my_logger.debug("- Inclination: %6.3f" %(theta))
        my_logger.debug("- Sigmas : %6.3f and %6.3f" %(sigmaLong, sigmaShort))
        my_logger.debug("- Extreme High A   : %6.3f, %6.3f" %(extremeHighA[0], extremeHighA[1]))
        my_logger.debug("- Extreme High B   : %6.3f, %6.3f" %(extremeHighB[0], extremeHighB[1]))
        my_logger.debug("- Extreme Low A   : %6.3f, %6.3f" %(extremeLowA[0], extremeLowA[1]))
        my_logger.debug("- Extreme Low B   : %6.3f, %6.3f" %(extremeLowB[0], extremeLowB[1])) 
        my_logger.debug("- Size : %4d " %(clSize))
       
        # Cluster principal lines
        ip0 = extremeHighA.astype(np.uint16)
        ip1= extremeHighB.astype(np.uint16)
        cv2.line(clusImg,(ip0[0],ip0[1]),(ip1[0],ip1[1]),(0,255,0),1)
        ip0 = extremeLowA.astype(np.uint16)
        ip1= extremeLowB.astype(np.uint16)
        #cv2.line(clusImg,(ip0[0],ip0[1]),(ip1[0],ip1[1]),(0,255,0),1)    
        ellipseCenter = ( int(mean[0]), int(mean[1]))
        ellipseRadius =  ( int(1.25*sigmaLong), int(3.*sigmaShort))
        ellipseAngle = theta
        cv2.ellipse(clusImg, (ellipseCenter[0], ellipseCenter[1]), (ellipseRadius[0],ellipseRadius[1]), ellipseAngle, 0., 360.,(0,0,255))
     
    return clusImg, clusterList


def main() :

  io = IO(rawDataDirectory, "filt_aber_"+rawDataFileName+"{}.jpeg")

  # Clustering Processing
  clusterDict = {}
  rawClusteringData = io.dir + "rawClusteringData.dat" 
  clusterFile = open(rawClusteringData, "wb")
  
  my_logger.info("Data %s" %(rawDataDirectory))
  my_logger.info("Files are %s" %(rawDataFileName))
  my_logger.info("Clusterizing filtered binary images from %4d to %4d" %(iImageI, iImageF))
  
  iImage = iImageI
  lengthDistribution = np.empty(0,dtype=float)
  transverseDistribution = np.empty(0,dtype=float)
  meanXDistribution = np.empty(0,dtype=float)
  meanYDistribution = np.empty(0,dtype=float)
  angleDistribution = np.empty(0,dtype=float)
  sizeDistribution = np.empty(0,dtype=float)
  
  while (iImage < iImageF) :
    # Loading filtered image
    BinaImg = io.read(iImage)   
    imgClus, clusterList = clusterizing( BinaImg, iImage)
  
    ClusFileName = io.dir + "clus_"+ io.fileTemplate.format(iImage)
    #ClusFileNameBin = io.dir + "clus_"+ io.fileTemplate.format(iImage)[0:-4]+"bindata"
    cv2.imwrite(ClusFileName,imgClus.astype(np.uint8) )
    
    clusterDict[iImage]= clusterList    
    iImage=iImage+1

  # Saving cluster in dictionary using iImage as key and the list of clusters as value 
  my_logger.info("Saving clusters from %4d to %4d images in %s" %(iImageI, iImageF, clusterFile.name))
  pickle.dump(clusterDict, clusterFile)
  clusterFile.close()

  # Generating control plots
  for cle, valeur in clusterDict.items() :
    for cluster in valeur :
      lengthDistribution     = np.append(lengthDistribution,calibrationFactor * 2.0 * cluster[5])
      transverseDistribution = np.append(transverseDistribution, calibrationFactor * 2.0* cluster[6])
      meanXDistribution      = np.append(meanXDistribution, calibrationFactor * cluster[2])
      meanYDistribution      = np.append(meanYDistribution, calibrationFactor * cluster[3])
      angleDistribution      = np.append(angleDistribution, cluster[4])
      sizeDistribution       = np.append(sizeDistribution, cluster[7])
      
  # Main Plot Page
  fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
 
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

  ax[1,0].set_yscale('log')
  ax[1,0].set_ylim(.1, 10000)
  ax[1,0].set_xlim(0.,30.)
  ax[1,0].set_xlabel(r"$\rm{Track \; transverse length \; l (mm)}$")
  ax[1,0].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
  #Fixing bin width
  binWidth =0.25
  transverseValues=np.arange(min(transverseDistribution), max(transverseDistribution) + binWidth, binWidth)
  histoValues, transverseValues, patches =  ax[1,0].hist(transverseDistribution, bins=transverseValues, log=True )

  ax[0,1].set_yscale('log')
  ax[0,1].set_ylim(.1, 5000)
  ax[0,1].set_xlim(0.,300.)
  ax[0,1].set_xlabel(r"$\rm{x (mm)}$")
  ax[0,1].set_ylabel(r"$\rm{N/dx \; (mm)}$")
  #Fixing bin width
  binWidth =5.0
  meanXvalues=np.arange(min(meanXDistribution), max(meanXDistribution) + binWidth, binWidth)
  histoValues, meanXvalues, patches =  ax[0,1].hist(meanXDistribution, bins=meanXvalues, log=True )

  ax[1,1].set_yscale('log')
  ax[1,1].set_ylim(.1, 5000)
  ax[1,1].set_xlim(0.,500.)
  ax[1,1].set_xlabel(r"$\rm{y (mm)}$")
  ax[1,1].set_ylabel(r"$\rm{N/dy \; (mm)}$")
  #Fixing bin width
  binWidth =5.0
  meanYvalues=np.arange(min(meanYDistribution), max(meanYDistribution) + binWidth, binWidth)
  histoValues, meanYvalues, patches =  ax[1,1].hist(meanYDistribution, bins=meanYvalues, log=True )

  ax[2,0].set_yscale('log')
  ax[2,0].set_ylim(.1, 5000)
  ax[2,0].set_xlim(-200.,200.)
  ax[2,0].set_xlabel(r"$\rm{angle (deg)}$")
  ax[2,0].set_ylabel(r"$\rm{N/dtheta}$")
  #Fixing bin width
  binWidth =1.0
  angleValues=np.arange(min(angleDistribution), max(angleDistribution) + binWidth, binWidth)
  histoValues, angleValues, patches =  ax[2,0].hist(angleDistribution, bins=angleValues, log=True )

  ax[2,1].set_yscale('log')
  ax[2,1].set_ylim(.1, 5000)
  ax[2,1].set_xlim(0.,20000.)
  ax[2,1].set_xlabel(r"$\rm{size}$")
  ax[2,1].set_ylabel(r"$\rm{N/dsize}$")
  #Fixing bin width
  binWidth =100
  sizeValues=np.arange(min(sizeDistribution), max(sizeDistribution) + binWidth, binWidth)
  histoValues, sizeValues, patches =  ax[2,1].hist(sizeDistribution, bins=sizeValues, log=True )
  plt.savefig(io.dir+"rawClustering_ControlPlots.pdf")
  plt.show()
 
if __name__ == "__main__" :
  rc = main()
  sys.exit(rc) 
