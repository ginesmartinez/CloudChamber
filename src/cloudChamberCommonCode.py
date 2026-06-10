import sys
import logging
import os
import cv2
import numpy as np
import yaml

# Settings of the logger
MY_FORMAT = "%(asctime)-24s %(levelname)-6s %(message)s"
logging.basicConfig(format=MY_FORMAT, level=logging.INFO)
my_logger=logging.getLogger() 

# Data analysis directory and files
#rawDataDirectory = "../DACQ_1807_two_camera_poire/"
#rawDataFileName = "img_C2_"  # C2 for DACQ_1807_two_camera_poire/
#rawDataDirectory = "../DACQ_220124/"
#rawDataFileName = "img_"  
#rawDataDirectory = "../DACQ_1807/"
#rawDataFileName = "img_C1_"  

#Load Yaml
if len(sys.argv) > 1:
    config_path = os.path.abspath(sys.argv[1])
    my_logger.info("Loading config from: %s" % config_path)
else:
    my_logger.error("No YAML config file provided!")
    my_logger.error("Usage: python script.py /path/to/experiment.yaml")
    sys.exit(1)

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# 1. Image Data Taking
rawDataDirectory = cfg["data"]["rawDataDirectory"]
rawDataFileName = cfg["data"]["rawDataFileName"]  

# Web camera calibration from aberration corrections and chessboard image July 2024
calibrationFactor = cfg["camera"]["calibrationFactor"] # mm per pixel

# 2. Chessboard processing Parameters 
#damierFileName = "noCorrection"
damierFileName   = cfg["chessboard"]["damierFileName"]
#dimension of the chessboard
nx = cfg["chessboard"]["nx"] #number of chessboard corner in x 
ny = cfg["chessboard"]["ny"] #number of chessboard corner in y

# Interest area in the chessboard processing code 
# Standard fiducial area of the image
interestArea_x1  = cfg["chessboard"]["interestArea_x1"] # Parameter zone of interest in the image x1 in pixels
interestArea_y1  = cfg["chessboard"]["interestArea_y1"] # Parameter zone of interest in the image y1
interestArea_x2  = cfg["chessboard"]["interestArea_x2"] # Parameter zone of interest in the image x2 : Warning x2 and y2 are lx and ly
interestArea_y2  = cfg["chessboard"]["interestArea_y2"] # Parameter zone of interest in the image y2
# DACQ juillet 2024
#interestArea_x1 = 650 # Parameter zone of interest in the image x1 in pixels
#interestArea_y1 = 60 # Parameter zone of interest in the image y1
#interestArea_x2 = 1260 # Parameter zone of interest in the image x2 : Warning x2 and y2 are lx and ly
#interestArea_y2 = 1030 # Parameter zone of interest in the image y2
# DACQ janvier 2024
#interestArea_x1 = 105 # Parameter zone of interest in the image x1 in pixels
#interestArea_y1 = 135 # Parameter zone of interest in the image y1
#interestArea_x2 = 530 # Parameter zone of interest in the image x2 : Warning x2 and y2 are lx and ly
#interestArea_y2 = 365 # Parameter zone of interest in the image y2

# Files to be processed
#iImageIIntegral = 0 
#iImageFIntegral = 2460 # DACQ_220124
#iImageFIntegral = 6005 # DACQ_1807
#iImageFIntegral = 3419 # DACQ_1807_two_camera_poire
iImageIIntegral = cfg["images"]["iImageIIntegral"]    # testdata
iImageFIntegral = cfg["images"]["iImageFIntegral"] # testdata


# 3. Filtering and clustering Processing Parameters
iImageI = cfg["images"]["iImageI"] # Parameter first image  
# timePeriod must therefore be defined before iImageF ( The final Image for filtering)

# Background Estimation Parameters
seuilDiff = cfg["background"]["seuilDiff"] # Parameter : pixel intensity difference threshold
timeStep = cfg["background"]["timeStep"] # Parameter : time step in unit of image number, to avoid correlation between sequential images
timePeriod = cfg["background"]["timePeriod"] # Parameter : time period considered to build the background image
#iImageF is not read from YAML but auto-detected at runtime.It is computed as the number of available aber_ images minus timePeriod, to ensure the background function always has enough images to look ahead.
iImageF = 0
while os.path.isfile(os.path.join(rawDataDirectory, "aber_" + rawDataFileName + str(iImageF) + ".jpeg")):
    iImageF += 1
iImageF = iImageF - 2 * timePeriod
my_logger.info("Auto-detected iImageF: %d" % iImageF) # Parameter last image

# Binarization
seuil = cfg["binarization"]["seuil"] # Parameter Threshold for binarization

# Calculation of the occupancy
#imagesPerSecond = 1.94 # January 18 2024, see output of chessboard correction processing
#imagesPerSecond = 1.84 # July 18 2024, see output of chessboard correction processing
#imagesPerSecond = 5.60  # Two camera poire July 18 2024
#imagesPerSecond = 1.83 # As calculated during the previous correction step fot gitlab test data
imagesPerSecond = cfg["occupancy"]["imagesPerSecond"] 
filteringOption = cfg["occupancy"]["filteringOption"] # !=0 default, 0 for doing only control plots
deltaTimeStep = cfg["occupancy"]["deltaTimeStep"] # in images, integer
integrationTime = cfg["occupancy"]["integrationTime"] * deltaTimeStep # in images, integer
occupancyFittingOption = cfg["occupancy"]["occupancyFittingOption"]

# 3. Raw Clustering Processing Parameters
# cluster size threshold
clusterSizeThreshold = cfg["clustering"]["clusterSizeThreshold"] # Parameter : minimum size of the cluster to be analyzed
#clusterSizeThreshold = 25 # DACQ_220124

# 4. Merging Processing Parameters
maxLinePointDistance = cfg["merging"]["maxLinePointDistance"] #(in mm)
#maxLinePointDistance = 3.0 #(in mm) for DACQ_220124
maxRelativeAngle = cfg["merging"]["maxRelativeAngle"] #(in degrés)
maxRelativeDistance = cfg["merging"]["maxRelativeDistance"] #(in mm)
#maxRelativeDistance = 50. #(in mm) for DACQ_220124


# Good cluster parameters  (in mm)
goodClusterMinClusterTransverseSigma = cfg["goodCluster"]["minTransverseSigma"]
goodClusterMaxClusterTransverseSigma = cfg["goodCluster"]["maxTransverseSigma"]
#goodClusterMaxClusterTransverseSigma = 3.2 # DACQ_220124
goodClusterMinClusterLongitudinalSigma = cfg["goodCluster"]["minLongitudinalSigma"]

# Good Cluster Selection
def goodCluster(cluster) :
  goodClusterStatus = False
  if(cluster[6]>goodClusterMinClusterTransverseSigma/calibrationFactor and cluster[6]<goodClusterMaxClusterTransverseSigma/calibrationFactor and cluster[5]>goodClusterMinClusterLongitudinalSigma/calibrationFactor) :
    goodClusterStatus = True
  return goodClusterStatus

# 5. Removing Correlated Cluster Processing Parameters
# Maximum relative distance between two correlated clusters in two different images in pixels
maxCorrelatedRelativeDistance = cfg["correlatedClusters"]["maxCorrelatedRelativeDistance"] # in mm
#maxCorrelatedRelativeDistance = 20. # DACQ 220124

# Maximum relative angle between two correlated clusters in two different images in pixels
maxCorrelatedRelativeAngle = cfg["correlatedClusters"]["maxCorrelatedRelativeAngle"] #
#maxCorrelatedRelativeAngle = 10. # DAQC_220124

# Best choice of the correlated cluster between j=0 ad j=1 (not used now)
#qualitySigmaShort = (goodClusterMinClusterTransverseSigma + goodClusterMaxClusterTransverseSigma)/2. 
qualitySigmaShort = cfg["correlatedClusters"]["qualitySigmaShort"] #
# 6. Final Analysis Distribution Process
# corona Volume in mm
coronaSize = cfg["finalAnalysis"]["coronaSize"] # 
# minimal length in mm
minLength = cfg["finalAnalysis"]["minLength"]
maxLength = cfg["finalAnalysis"]["maxLength"]

# Reading Image Interface Class
class IO:
  def __init__( self, dataDir="JPEG", fileTemplate= "img_{}.jpeg", pixx=0, pixy=0, lx=9999, ly=9999):
    self.dir = dataDir
    self.fileTemplate = fileTemplate
    # End of the file sequence
    self.end = False
    self.fileName = "empty"
    self.imgRead = 0
    self.pixx = pixx
    self.pixy = pixy
    self.lx = lx
    self.ly = ly
    my_logger.info("Creating IO object to read raw images of the Cloud Chamber")
    if (   (self.pixx > 0)          or
           (self.lx < 9999)         or
           (self.pixy > 0)          or
           (self.ly < 9999)              ) :
        my_logger.info("Selection image area from (%4d,%4d) to (%4d,%4d) pixel points" %(self.pixy, self.pixx, self.pixy+self.ly, self.pixx+self.lx))
    else :
        self.pixx = 0
        self.pixy = 0
        my_logger.info("Full image area")
      
  def nextRead(self):
    img = np.zeros(0)
    img2 = np.zeros(0)
    self.fileName = "/".join( (self.dir, self.fileTemplate.format(self.imgRead)) )
    my_logger.debug(self.fileName)
    isHere = os.path.isfile(self.fileName)
    if isHere:
      img = cv2.imread(self.fileName, cv2.IMREAD_GRAYSCALE)
      if ( (img.shape[0] > (self.pixx)          ) or
           (img.shape[0] > (self.pixx + self.lx)) or
           (img.shape[1] > (self.pixy)          ) or
           (img.shape[1] > (self.pixy + self.ly))   ) :
        img2 = img[self.pixx:self.pixx+self.lx, self.pixy:self.pixy+self.ly]
      else :
        img2 = img
      self.imgRead += 1
    else:
      self.end = True
    #
    my_logger.debug("Reading cloud chamber image %4d with IO object" %(self.imgRead))
    return img2

  def read(self, k):
    img = np.zeros(0)
    img2 = np.zeros(0)
    self.fileName = "/".join( (self.dir, self.fileTemplate.format(k)) )
    isHere = os.path.isfile(self.fileName)
    my_logger.debug(self.fileName)
    if isHere:
      img = cv2.imread(self.fileName, cv2.IMREAD_GRAYSCALE)
      if ( (img.shape[0] > (self.pixx)          ) or
           (img.shape[0] > (self.pixx + self.lx)) or
           (img.shape[1] > (self.pixy)          ) or
           (img.shape[1] > (self.pixy + self.ly))   ) :
        img2 = img[self.pixx:self.pixx+self.lx, self.pixy:self.pixy+self.ly]
      else:
        img2 = img
    else:
      self.end = True
    
    my_logger.debug("Reading cloud chamber image %4d with IO object" %(k))
    return img2      