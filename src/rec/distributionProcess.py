# Building

import sys
sys.path.insert(1, "../")
import logging
import os
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.optimize import curve_fit

# my_logger
from cloudChamberCommonCode import my_logger

from cloudChamberCommonCode import iImageI
from cloudChamberCommonCode import iImageF
from cloudChamberCommonCode import imagesPerSecond

# webcam calibration factor
from cloudChamberCommonCode import calibrationFactor

# IO
from cloudChamberCommonCode import rawDataDirectory

# Good Cluster Selection
from cloudChamberCommonCode import goodCluster

# Response function of the cloud Chamber
# Exponential with a gaussian end point and a gaussian function
def exponentialEndpoint(X, Aexp, L0, Agauss, mu, relativeSigma ) :
    sigma = mu * relativeSigma
    functionValue1 = (Aexp * np.exp(-X/L0))
    functionValue2 = [1 if x<mu else np.exp(-1.0 * (x-mu)**2 / (2.*sigma**2)) for x in X]
    return (functionValue1*functionValue2)

def gaussian(X, Aexp, L0, Agauss, mu, relativeSigma) :
    sigma = mu * relativeSigma
    functionValue3 = Agauss * np.exp( -(X-mu)**2 / (2.*sigma**2) )
    return (functionValue3)

def responseFunction(X, Aexp, L0, Agauss, mu, relativeSigma) :
    functionValue1 = exponentialEndpoint(X, Aexp, L0, Agauss, mu, relativeSigma)
    functionValue2 = gaussian(X, Aexp, L0, Agauss, mu, relativeSigma)
    return (functionValue1 + functionValue2)

def twoEnergiesResponseFunction(X, A1exp, L01, A1gauss, mu1, A2exp, L02, A2gauss, mu2, relativeSigma) : 
    functionValue1 = responseFunction(X, A1exp, L01, A1gauss, mu1, relativeSigma) 
    functionValue2 = responseFunction(X, A2exp, L02, A2gauss, mu2, relativeSigma) 
    return (functionValue1+functionValue2)

def threeEnergiesResponseFunction(X, A1exp, L01, A1gauss, mu1, A2exp, L02, A2gauss, mu2, A3exp, L03, A3gauss, mu3, relativeSigma) : 
    functionValue1 = responseFunction(X, A1exp, L01, A1gauss, mu1, relativeSigma) 
    functionValue2 = responseFunction(X, A2exp, L02, A2gauss, mu2, relativeSigma) 
    functionValue3 = responseFunction(X, A3exp, L03, A3gauss, mu3, relativeSigma) 
    return (functionValue1+functionValue2+functionValue3)

def main() :

    from cloudChamberCommonCode import interestArea_x1
    from cloudChamberCommonCode import interestArea_y1
    from cloudChamberCommonCode import interestArea_x2
    from cloudChamberCommonCode import interestArea_y2
    
    from cloudChamberCommonCode import coronaSize
    from cloudChamberCommonCode import minLength
    from cloudChamberCommonCode import maxLength
    from cloudChamberCommonCode import calibrationFactor 


    # Reading final good, non-merged and non-correlated cluster
    clusterDict = {}
    clusterFile = open(rawDataDirectory + "MergedNoncorrelatedClusterData.dat", "rb")
    clusterDict= pickle.load(clusterFile)
    clusterFile.close()
  
    # Single distributions of selected clusters
    lengthDistribution     = np.empty(0,dtype=float)
    transverseDistribution = np.empty(0,dtype=float)
    meanXDistribution      = np.empty(0,dtype=float)
    meanYDistribution      = np.empty(0,dtype=float)
    angleDistribution      = np.empty(0,dtype=float)
    sizeDistribution       = np.empty(0,dtype=float)
  
 

    mergedRatio = 0.
    noMergedCounter =0 
    totalCounter= 0.
    # Fiducial volume cuts
    fiduX1 = coronaSize /  calibrationFactor
    fiduX2 = ((interestArea_x2 - interestArea_x1) - coronaSize) /  calibrationFactor
    fiduY1 = coronaSize / calibrationFactor
    fiduY2 = ((interestArea_y2-interestArea_y1) - coronaSize) /  calibrationFactor
    print (fiduX1, fiduX2, fiduY1, fiduY2)
    for iImage, clusterList in clusterDict.items() :
        for cluster in clusterList :
            if (goodCluster(cluster) and  2.0*calibrationFactor*cluster[5] > minLength and 2.0*calibrationFactor*cluster[5] < maxLength) :
                print("Cluster X:", cluster[2], "Y:", cluster[3])
                print("FiduX:", fiduX1, fiduX2)
                print("FiduY:", fiduY1, fiduY2)
                 # Fiducial volume selection
                if (cluster[2]>fiduX1 and cluster[2]<fiduX2 and cluster[3]>fiduY1 and cluster[3]<fiduY2) : 
                    print (" Cluster 3")
                    lengthDistribution     = np.append(lengthDistribution, 2.0*calibrationFactor*cluster[5])
                    transverseDistribution = np.append(transverseDistribution, 2.0*calibrationFactor*cluster[6])
                    meanXDistribution      = np.append(meanXDistribution, calibrationFactor*cluster[2])
                    meanYDistribution      = np.append(meanYDistribution, calibrationFactor*cluster[3])
                    angleDistribution      = np.append(angleDistribution, cluster[4])
                    sizeDistribution       = np.append(sizeDistribution, cluster[7])
                    totalCounter = totalCounter+1
                    if (cluster[12]==0) :
                        noMergedCounter = noMergedCounter + 1 

    my_logger.info("Data %s" %(rawDataDirectory))
    my_logger.info("Image range from %d until %d" %(iImageI, iImageF))
    my_logger.info("Number of images is %d" %(iImageF-iImageI))
    my_logger.info("Image rate is %5.2f per second" %(imagesPerSecond))
    my_logger.info("Total number of good cluster length larger than %5.2f mm is %5d" %(minLength, totalCounter))
    my_logger.info("with a number of no-merged good cluster of %5d" %(noMergedCounter))
    mergedRatio = float((totalCounter-noMergedCounter))/float(totalCounter)
    my_logger.info("Merged good cluster ratio is %5.2f" %(mergedRatio))
    clusterRate = float(totalCounter)/float(iImageF-iImageI)*imagesPerSecond
    clusterRateError = math.sqrt(float(totalCounter))/float(iImageF-iImageI)*imagesPerSecond
    my_logger.info("Cluster rate is %5.4f +- %5.4f per second" %(clusterRate, clusterRateError))    
    my_logger.info("Fiducial X interval (%5.2f, %5.2f) in mm" %(fiduX1*calibrationFactor, fiduX2*calibrationFactor))
    my_logger.info("Fiducial Y interval (%5.2f, %5.2f) in mm" %(fiduY1*calibrationFactor, fiduY2*calibrationFactor))
    fiducialArea = (fiduY2-fiduY1)*(fiduX2-fiduX1)*calibrationFactor*calibrationFactor/1.e6
    my_logger.info("Fiducial area %7.5f m2" %(fiducialArea))
    decayRateDensity = clusterRate / fiducialArea / 0.15
    decayRateDensityError = clusterRateError / fiducialArea / 0.15
    my_logger.info("Decay Rate %7.2f+-%7.2f per second and per m3" %(decayRateDensity, decayRateDensityError ))
    my_logger.info("Plots for cluster analysis" )    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 7))

    # Main Plot with the length track distribution
    ax[0,0].set_yscale("linear")
    ax[0,0].set_ylim(.1, 1000.)
    ax[0,0].set_xlim(0.,100.)
    ax[0,0].set_xlabel(r"$\rm{Track \; length \; l (mm)}$")
    ax[0,0].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
    #Fixing bin width
    binWidth =1.0
    #bin center calculation
    lengthValues=np.arange(min(lengthDistribution), max(lengthDistribution) + binWidth, binWidth)
    #lengthValues=np.arange(8., max(lengthDistribution) + binWidth, binWidth)
    #bin center calculation
    histoValuesOut, lengthValues, patches =  ax[0,0].hist(lengthDistribution, bins=lengthValues, log=True )
    lengthValuesCenter = np.array([0.5 * (lengthValues[i] + lengthValues[i+1]) for i in range(len(lengthValues)-1)])
    histoValuesOut[histoValuesOut<0.] = 0.
    histoErrorsOut = np.sqrt(histoValuesOut)
    histoErrorsOut[histoErrorsOut==0] = 1.
    #ax[0,0].errorbar(lengthValuesCenter, histoValuesOut, yerr=histoErrorsOut, fmt='o')
   

    fittingOption = -1    
    if (fittingOption==0) :
        #Fitting the curve to one energy
        p1 = [1000.,10., 200., 41.7, 0.037]
        bounds1 = ([10., 6., 10., 20., 0.01],[10000., 15., 500., 90., 0.300])
        popt, pcov = curve_fit(f=responseFunction, xdata=lengthValuesCenter, ydata=histoValuesOut, p0=p1, sigma=histoErrorsOut, bounds=bounds1)    
        print(popt)

        ax[0,0].plot(lengthValuesCenter, responseFunction(lengthValuesCenter, *popt), color='blue', linewidth=2.5, label=r'Fitted function')
        ax[0,0].plot(lengthValuesCenter, exponentialEndpoint(lengthValuesCenter, *popt), color='red', linewidth=2.5, label=r'Fitted function')
        ax[0,0].plot(lengthValuesCenter, gaussian(lengthValuesCenter, *popt), color='orange', linewidth=2.5, label=r'Fitted function')
        gaussianValues = gaussian(lengthValuesCenter, *popt)
        print(sum(gaussianValues) )
    if (fittingOption==1) :
        # Fitting to two energies 
        p2 = [1000., 10., 100., 37., 1000., 10., 100., 45., 0.037]
        bounds2 = ([10., 5., 10., 20., 10., 5., 10., 20., 0.01],[10000., 15., 1000., 90., 10000., 15., 1000., 90., 0.300])
        popt, pcov = curve_fit(f=twoEnergiesResponseFunction, xdata=lengthValuesCenter, ydata=histoValuesOut, p0=p2, sigma=histoErrorsOut, bounds=bounds2)    
        print(popt)
        popt_rp = [popt[0], popt[1], popt[2], popt[3], popt[8]]
        ax[0,0].plot(lengthValuesCenter, responseFunction(lengthValuesCenter, *popt_rp), color='orange', linewidth=2.5, label=r'Fitted function')
        popt_rp = [popt[4], popt[5], popt[6], popt[7], popt[8]]
        ax[0,0].plot(lengthValuesCenter, responseFunction(lengthValuesCenter, *popt_rp), color='red', linewidth=2.5, label=r'Fitted function')
        ax[0,0].plot(lengthValuesCenter, twoEnergiesResponseFunction(lengthValuesCenter, *popt), color='blue', linewidth=2.5, label=r'Fitted function')

    ax[1,0].set_yscale('linear')
    ax[1,0].set_ylim(.1, 1000)
    ax[1,0].set_xlim(0.,20.)
    ax[1,0].set_xlabel(r"$\rm{Track \; transverse length \; l (mm)}$")
    ax[1,0].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
    #Fixing bin width
    binWidth =0.25
    transverseValues=np.arange(min(transverseDistribution), max(transverseDistribution) + binWidth, binWidth)
    histoValues, transverseValues, patches =  ax[1,0].hist(transverseDistribution, bins=transverseValues, log=True )

    ax[0,1].set_yscale('linear')
    ax[0,1].set_ylim(.1, 600)
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
    binWidth =3.0
    angleValues=np.arange(min(angleDistribution), max(angleDistribution) + binWidth, binWidth)
    histoValues, angleValues, patches =  ax[2,0].hist(angleDistribution, bins=angleValues, log=True )

    ax[2,1].set_yscale('log')
    ax[2,1].set_ylim(.1, 5000)
    ax[2,1].set_xlim(0.,5000.)
    ax[2,1].set_xlabel(r"$\rm{size}$")
    ax[2,1].set_ylabel(r"$\rm{N/dsize}$")
    #Fixing bin width
    binWidth =20.0
    sizeValues=np.arange(min(sizeDistribution), max(sizeDistribution) + binWidth, binWidth)
    histoValues, sizeValues, patches =  ax[2,1].hist(sizeDistribution, bins=sizeValues, log=True )

    plt.savefig(rawDataDirectory+"finalClusterAnalysis_ControlPlots.pdf")    
    plt.show()
    

  
if __name__ == "__main__" :
    rc = main()
    sys.exit(rc)
