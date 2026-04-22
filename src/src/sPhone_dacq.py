import requests
import urllib.request
import cv2
import numpy as np
#import imutils
import time

url = "http://192.168.1.10:8080/shot.jpg"
url = "http://10.146.96.178:8080/shot.jpg"

# videoCaptureObject = cv2.VideoCapture(0)
nextImage = True
sumAcquisition=0.
sumComputing=0.
sumStore=0.
sumTotal=0.

n=0
while(nextImage):
  # Image acquisition
  sAcquisition = time.perf_counter_ns()
  print("Start image ", n)
  fileName = "DACQ/img_{}.png".format(n)
  # ret,frame = videoCaptureObject.read()
  img_resp = requests.get(url)
  # stream = urllib.request.urlopen(url).read()
  # img_arr = np.array(bytearray(urllib.request.urlopen(url).read()),dtype=np.uint8)
  img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
  img = cv2.imdecode(img_arr, -1)
  eAcquisition = time.perf_counter_ns()

  # Processing & Show image
  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
  # img = imutils.resize(img, width=1000, height=1800)
  cv2.imshow('Capturing Video',img)
  # cv2.imshow('Capturing Video',gray)
  eProcessing = time.perf_counter_ns()
  
  # Store
  cv2.imwrite(fileName, img)
  eStore = time.perf_counter_ns()
  
  # Continue or not
  # Wait a key stroke in ms
  key = cv2.waitKey(1)
  if( key & 0xFF == ord('q')):
    nextImage = False
  
  # Display in ms
  dtAcquisition = (eAcquisition - sAcquisition)* 1.e-6
  dtComputing = (eProcessing - eAcquisition)* 1.e-6
  dtStore = (eStore - eProcessing)* 1.e-6
  dtTotal = (eStore - sAcquisition)* 1.e-6
  print("  Total time={:.2f}, dacq={:.2f}, processing={:.2f}, storage={:.2f} in ms".format(dtTotal, dtAcquisition, dtComputing, dtStore) )
  print("  Rate={:.1f} fps".format( 1000./ (dtTotal) ))
  sumAcquisition += dtAcquisition
  sumComputing += dtComputing
  sumStore += dtStore
  sumTotal += dtTotal
  n += 1
  
print("-----------------------------------------------------------")  
print("Averages on {} images:".format(n))
print("  Total time={:.2f}, dacq={:.2f}, processing={:.2f}, storage={:.2f} in ms".format(
       n, sumTotal/n, sumAcquisition/n, sumComputing/n, sumStore/n) )
print("  Rate={} fps".format( (1000.0*n)/ sumTotal ))

# Terminate
videoCaptureObject.release()
cv2.destroyAllWindows()

