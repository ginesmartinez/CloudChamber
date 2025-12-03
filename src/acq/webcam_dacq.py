import cv2
import time

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# path[1] in the directory where cloudChamberCommonCode can be
sys.path.insert(1, "../")

# my_logger
from cloudChamberCommonCode import my_logger

# Current data directory and files
from cloudChamberCommonCode import rawDataDirectory
from cloudChamberCommonCode import rawDataFileName
 
storeImage = True
waitTime = .01  # Sleep in sec.

#camera test to look for the goodi id
#for i in range (0, 32):
#  camera_test = cv2.VideoCapture(i)
#  camera_test.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#  camera_test.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#  camera_test.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#  if camera_test.get(cv2.CAP_PROP_FRAME_WIDTH) != 0:
#    print('numero camera :',i)
#    print('camera test frame height : ', camera_test.get(cv2.CAP_PROP_FRAME_HEIGHT), ' px')
#    print('camera test frame width : ', camera_test.get(cv2.CAP_PROP_FRAME_WIDTH), ' px')
#    print('-----------------------------')

# Settings of the logger
my_logger.info("Taking pictures from webcam")
#camera 1 setup
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
camera.set(cv2.CAP_PROP_EXPOSURE, 10)

#camera 2 setup
#camera_2 = cv2.VideoCapture(9)
#camera_2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#camera_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#camera_2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

print('camera 1 frame height : ', camera.get(cv2.CAP_PROP_FRAME_HEIGHT), ' px')
print('camera 1 frame width : ', camera.get(cv2.CAP_PROP_FRAME_WIDTH), ' px')
print('camera 1 frame per second : ', camera.get(cv2.CAP_PROP_FPS), ' fps')
print('camera 1 frame per second : ', camera.get(cv2.CAP_PROP_EXPOSURE), ' au')
#print('camera 2 frame height : ', camera_2.get(cv2.CAP_PROP_FRAME_HEIGHT), ' px')
#print('camera 2 frame width : ', camera_2.get(cv2.CAP_PROP_FRAME_WIDTH), ' px')

nextImage = True
dtAcquisition = 0.
dtShow = 0.
dtComputing =0.
dtStore =0.
dtMonitoring=0.
dtTotal = 0.
sumAcquisition=0.
sumShow=0.
sumComputing=0.
sumStore=0.
sumMonitoring=0.
sumTotal=0.

n=0
nabs = 0
m=0
cycle = int (camera.get(cv2.CAP_PROP_FPS)/2. )
while(nextImage):
  # Image acquisition
  sAcquisition = time.perf_counter()
  ret,frame = camera.read()
  #ret,frame_2 = camera_2.read()
  eAcquisition = time.perf_counter()
    
  # Show image
  cv2.imshow('Capturing Video',frame)
  #cv2.imshow('Capturing Video',frame_2)
  eShow = time.perf_counter()

  # Computing 1 over cycle
  eComputing = time.perf_counter()
  eStore = eComputing 
  if (m == cycle) :
    m = 0
    n += 1
    my_logger.info("Saving image %d over %d" %(nabs, n))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    #gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY) 
    eComputing = time.perf_counter()
    
    fileName = rawDataDirectory+rawDataFileName+"{}.jpeg".format(nabs)
    #fileName_2 = "../../data/test/img_C2_{}.jpeg".format(nabs)
    cv2.imwrite(fileName, gray)
    my_logger.info("Picture %s" %(fileName))

    #cv2.imwrite(fileName_2, gray_2)
    eStore = time.perf_counter()
    nabs +=1
  m += 1

  #time.sleep(waitTime)


  # Monitoring in ms
  # Continue or not
  # Wait a key stroke in ms
  key = cv2.waitKey(1)
  if( key & 0xFF == ord('q')):
    nextImage = False
  eMonitoring = time.perf_counter()

  dtAcquisition = (eAcquisition - sAcquisition) *1.e3
  dtShow = (eShow - eAcquisition) *1.e3
  dtComputing = (eComputing - eShow) *1.e3
  dtStore = (eStore - eComputing) *1.e3
  dtMonitoring = (eMonitoring - eStore) *1.e3
  dtTotal = (eMonitoring - sAcquisition) *1.e3
  #print("  Total time={:.2f}, dacq={:.2f}, show={:.2f}, computing={:.2f}, storing={:.2f}, monitoring={:.2f} in ms".format(dtTotal, dtAcquisition, dtShow, dtComputing, dtStore, dtMonitoring) )
  #print("  Rate={:.1f} fps".format( 1.e3 / (dtTotal) ))
  sumAcquisition += dtAcquisition
  sumShow += dtShow
  sumComputing += dtComputing
  sumStore += dtStore
  sumMonitoring += dtMonitoring
  sumTotal += dtTotal
  n += 1
  
print("-----------------------------------------------------------")  
print("Full time on {} read (over {} stored) images:".format(n,nabs))
print("  Total time={:.2f}, dacq={:.2f}, show={:.2f}, computing (stored)={:.2f}, storage (stored)={:.2f}, monitoring={:.2f} in s".format(
       1.e-3*sumTotal, 1.e-3*sumAcquisition, 1.e-3*sumShow, 1.e-3*sumComputing, 1.e-3*sumStore, 1.e-3*sumMonitoring) )
print("Averages on {} read ({} stored) images:".format(n,nabs))
print("  Total time={:.2f}, dacq={:.2f}, show={:.2f}, computing (stored)={:.2f}, storage (stored)={:.2f} in ms".format(
       sumTotal/n, sumAcquisition/n, sumShow/n, sumStore/n, sumMonitoring/n, sumComputing/nabs, sumStore/nabs) )
print("  Acquisition and show  Rate={} fps".format( (1.e3*n)/ sumTotal  ))
print("  Computing and storing rate={} fps".format( (1.e3*nabs)/ sumTotal  ))

# Terminate
camera.release()
#camera_2.release()

cv2.destroyAllWindows()

