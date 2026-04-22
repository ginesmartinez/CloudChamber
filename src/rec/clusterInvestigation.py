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

# Settings of the logger
MY_FORMAT = "%(asctime)-24s %(levelname)-6s %(message)s"
logging.basicConfig(format=MY_FORMAT, level=logging.INFO)
my_logger=logging.getLogger()
my_logger.info("Cluster Investigation")

def main() :

  # Reading Raw cluster data from filteringProcess.py
  clusterDict = {}
  clusterFile = open("RawClusterData.dat", "rb")  # or merged-cluster or non-correlated clusters
  clusterDict= pickle.load(clusterFile)
  clusterFile.close()

  for iImage,clusterList in clusterDict.items() :
    my_logger.info("--- Investigatin cluster for image %d" %(iImage) )
    for cluster in clusterList :
      if (cluster[6]<0.6 and cluster[6]>0.02) :
        print(cluster)

if __name__ == "__main__" :
  rc = main()
  sys.exit(rc)
