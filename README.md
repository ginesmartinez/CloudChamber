# CloudChamber
Python code to take images of a cloud chamber and to analyse them: optical correction matrix, optical correction and fiducial volume, generation of the background image, raw clustering, merging of fragmented clusters. removing correlated cluster, final results. There are two applications : thoron lifetime and Radon in air measurement.   

## Configuration data
The code cloudChamberCommonCode.py defined all data that configures a given data taking period and analysis

## Image Data Taking
In src/acq/ directory the python3 script webcam_dacq.py to take a series of images of the webcam (to be used at the end of the day with FULL HD WEBCAM using a linux computer). See specifications of the webcam FULL HD in the doc directory.

## Optical Aberration correction and Image Fiducial Area
IN src/acq/ direction the python3 script chessBoard_CameraCalibrationProcess.py performs the optical aberration correction (using the chess board image in data/ImageDamier_FullResolution/ ) and selects the fiducial area of the image in which the radioactive tracks wil be reconstructed ans analyzed.


