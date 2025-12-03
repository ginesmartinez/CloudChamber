# CloudChamber
Python code to take images of a cloud chamber and to analyse them: optical correction matrix, optical correction and fiducial volume, generation of the background image, raw clustering, merging of fragmented clusters. removing correlated cluster, final results. There are two applications : thoron lifetime and Radon in air measurement.   

## Configuration data
The code cloudChamberCommonCode.py defined all data that configures a given data taking period and analysis

## Data Taking
In acq/ directory the python program webcam_dacq.py to take a series of images of the webcam FULL HD WEBCAM using a linux computer. See specifications of the webcam in the data directory.




