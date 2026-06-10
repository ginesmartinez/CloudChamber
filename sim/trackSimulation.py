#!/usr/bin/env python3
# Python script for simulation of alpha tracks in the cloud chamber
# v1 Gines Martinez  Mars 2023 (gines.martinez@subatech.in2p3.fr )
# v2 Gines Martinez Janvier 2025

# Reference system is the center of the cloud chamber, on the bottom of the active volume
# z value increase from 0 (bottom part, close to the alcohol to the higher part until z=100 mm)

import sys
import sys
import os
import logging
sys.path.insert(1, "../src/")

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.optimize import curve_fit

# Reminder : export PYTHONPATH='/Users/martinez/Work/GitLocalRepository/CloudChamber/src/'
# Area in pixel from Common Code
from cloudChamberCommonCode import interestArea_x1
from cloudChamberCommonCode import interestArea_y1
from cloudChamberCommonCode import interestArea_x2
from cloudChamberCommonCode import interestArea_y2
# Calibration factor in mm/pixel
from cloudChamberCommonCode import calibrationFactor
# Size of the non fiducial volume
from cloudChamberCommonCode import coronaSize
# minimal length of the projected alpha track
from cloudChamberCommonCode import minLength
# maximal length of the projected alpha track
from cloudChamberCommonCode import maxLength

# Calculation of the volume
xLength = (interestArea_x2 - interestArea_x1) * calibrationFactor  # in mm
yLength = (interestArea_y2 - interestArea_y1) * calibrationFactor  # in mm
zLength = 150 # in mm

# Settings of the logger
MY_FORMAT = "%(asctime)-24s %(levelname)-6s %(message)s"
logging.basicConfig(format=MY_FORMAT, level=logging.INFO)
my_logger=logging.getLogger()
my_logger.info("Simulations of alpha tracks in the cloud chamber")
my_logger.info("Volume Lx Ly Lz :  %4.8f x  %4.8f x  %4.8f mm3" %(xLength, yLength, zLength))

# Track number to be simulated in the full volume 
trackNumber = 10000
my_logger.info("Number of simulated track in the volume is %d" %(trackNumber))
# Track number counter
trackNumberCounter =0 
# Track seen in the cloud chamber fully include in the fiducial volume
trackNumberSeen = 0
trackNumberSeenIn = 0
trackNumberSeenOut = 0

# It is not clear what is the thickness of the active volume. 
# The estimation given in appendix C is 2-3 mm
# A additional question could be the uniformity of the active volume 
# in the horizontal plane We assume perfect uniformity of the active volume 
# thickness
activeVolumeHeight = 150. # in mm, see page 4 of appendix C
my_logger.info("Height of the active volume :  %4.8f  mm" %(activeVolumeHeight))

# Parameter of the Bethe-Bloch formula in W. R. Leo page 24
# One has to pay attention that this formula is valid for
# beta x gamma range 0.1 < beta x gamma < 1000 (see pdg), but alpha particles
# below 10 MeV, beta x gamma are below 0.1 Therefore this formula 
# is a bad approximation for alpha particles from radioactive decays
betheBlochNormalisation = 0.1535 # MeV cm^2/g 
electronMass = 511000 # eV/c^2
alphaCharge = 2 # electron charge units
alphaRelativeStragglingRange = 0.1 #0.037 # See SRIM Calculation form Vincent METIVIER in Git repository
alphaMass = 3727 # MeV/c^2
airZoverA = 0.5
airDensity  = 0.00120479*293.0/298.0 # g/cm^3, see Geant4 Material Database (at 25 degres)
#airDensity  = 0.00120479*293./258. # g/cm^3, extrapolation at -15 degres
airExcitationPotential = 85.7 # eV, see Geant4 Material Database

def betheBloch(alphaEnergy) : 
    if (alphaEnergy>0.) :
        alphaTotalEnergy = alphaEnergy + alphaMass
        alphaMomentum = math.sqrt(math.pow(alphaTotalEnergy,2)-math.pow(alphaMass,2))
        alphaBeta = alphaMomentum/alphaTotalEnergy
        alphaGammaLorentz = 1./math.sqrt(1. - alphaBeta*alphaBeta) 
        if (alphaBeta*alphaGammaLorentz <0.1 or alphaBeta*alphaGammaLorentz>1000) :
            my_logger.warning("Alpha Beta out of rage of validity 0.1<beta*gamma<1000")
            my_logger.warning("Alpha Beta*Gamma is %4.8f," %(alphaBeta*alphaGammaLorentz))
       
        my_logger.debug("Alpha Beta is %4.8f," %(alphaBeta))
        my_logger.debug("Alpha Gamma is %4.8f," %(alphaGammaLorentz))
        alphaWmax = 2. *electronMass * alphaBeta * alphaGammaLorentz
        my_logger.debug("Alpha Wmax is %4.8f," %(alphaWmax))
        dEdx = -betheBlochNormalisation * airZoverA  * airDensity * alphaCharge**2 / alphaBeta**2 * (math.log(2.*electronMass*alphaBeta**2 * alphaGammaLorentz**2 * alphaWmax / airExcitationPotential**2)-2*alphaBeta**2) 
    else :
        dEdx =0
    return dEdx

# Integration of the dEdx form Bethe-Bloch formula
def alphaLength(alphaEnergy) : # Alpha energy in MeV
    # Choice of the step as the length to loose 0.1% of the initial energy   
    step = -0.001 *  alphaEnergy/betheBloch(alphaEnergy)
    # Runge-Kutta method (RK4) to estimate the length of the track, when alpha energy is null
    alphaLength=0
    while (alphaEnergy>0) :
        k1 = betheBloch(alphaEnergy)
        k2 = betheBloch(alphaEnergy + k1*step/2.)
        k3 = betheBloch(alphaEnergy + k2*step/2.)
        k4 = betheBloch(alphaEnergy + k3*step)
#        alphaEnergy = alphaEnergy + k2*step
        alphaEnergy = alphaEnergy+(k1+2*k2+2*k3+k4)*step/6.
        alphaLength = alphaLength+step
#        my_logger.debug("Alpha Energy, energy loss and position is %4.4f, %4.4f,%4.4f," %(alphaEnergy, k1, alphaLength)) 
    return 10*alphaLength # in mm

# Projected Range for protons/alpha in dry air from
# https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
# https://physics.nist.gov/PhysRefData/Star/Text/ASTAR.html
# these values are a good estimation for alpha particles in the range 1-10 MeV
def rangeNIST(alphaEnergy) :
    f = open("AlphaRange_DryAir.dat", "r")
    content = False
    energy =[]
    range=[]
    for line in f :
        values = line.split()
        if (values[0] == "1.000E-03") :
            content = True
        if (content) :   
           energy.append(float(values[0]))
           range.append(float(values[1])/airDensity)
    n=1
    rangeNIST =0
    while (alphaEnergy>=energy[n-1]) :
       if (energy[n]>alphaEnergy and energy[n-1]<=alphaEnergy) :
           rangeNIST = range[n-1] + (range[n]-range[n-1])/(energy[n]-energy[n-1])*(alphaEnergy-energy[n-1]) 
       n=n+1
    my_logger.debug("Range is %4.4f " %(rangeNIST))
    f.close()
    return  rangeNIST*10 # in mm

# Alpha emission point is defined as a python list (x,y,z)
# Alpha emission point are assumed to be uniformly random in the
# considered volume 
def alphaEmmission() :
    vertex = [random.uniform(-xLength/2., xLength/2.), random.uniform(-yLength/2., yLength/2.), random.uniform(0., zLength)]
    return vertex

# Alpha direction (u, v, w) is also defined as a list
# The direction of the alpha particle is assumed to be isotropic
def alphaDirection() :
    phi = random.uniform(0, 2 * math.pi)
    cos_theta = random.uniform(-1., 1.)
    u = [math.cos(phi) * math.sqrt(1-cos_theta*cos_theta), math.sin(phi) * math.sqrt(1-cos_theta*cos_theta), cos_theta]
    return u

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


#Algorithme
def main() :
    # From https://www.nndc.bnl.gov/ensdf/ :
    # alphaEnergy Radon    222 5.48948 MeV (99.92% 3.82 days)
    # alphaEnergy Polonium 218 6.00255  MeV (99.999% 3.1 minutes)
    # AlphaEnergy Polonium 214 7.68682 MeV (99.995 164 mu-s)
    # but there are two beta decays before 214Po, 214Pb(26,8 minutes) and 214Bi (19.9 minutes)
    alphaEnergy222Rn = 5.48948  # MeV
    alphaEnergy218Po = 6.00255  # MeV
    alphaEnergy214Po = 7.68682  # MeV
    
    alphaLength = [rangeNIST(alphaEnergy222Rn),  rangeNIST(alphaEnergy218Po), rangeNIST(alphaEnergy214Po)]
    my_logger.info("Alpha path length 222Rn is %4.2f" %(alphaLength[0]))
    my_logger.info("Alpha path length 218Po is %4.2f" %(alphaLength[1]))
    my_logger.info("Alpha path length 214Po is %4.2f" %(alphaLength[2]))

    trackNumberCounter=0
    trackNumberSeen=0
    trackNumberSeenIn=0
    trackNumberSeenOut=0
    trackNumberVertexIn=0
    trackNumberBothVertexIn=0

    # test for histogramming
    lengthDistribution = np.zeros(trackNumber,dtype=float)
    lengthDistributionIn = np.zeros(trackNumber,dtype=float)
    lengthDistributionOut = np.zeros(trackNumber,dtype=float)
    
    # Relative mixing of radionuclides
    relativeRadionuclideRatio = [0.0, 0.80, 0.998]

    while trackNumberCounter < trackNumber-1 :
        trackNumberCounter=trackNumberCounter+1

        # Vertex Generation of the alpha 
        vertex = alphaEmmission()
        my_logger.debug("Alpha emmission is %4.2f,%4.2f,%4.2f" %(vertex[0],vertex[1],vertex[2]))

        # Modelling of the density correction to the track length
        # An approximate effect on the air density as a function of z vertex of the alpha emission
        # It is assumed T=25 degrees , so air dry density : see airDensity variable value
        # vertex[2] between 0 and zLength of the cloud chamber. At z=0 25 degrees, at z=zLength -25 degrees 
        alphaLengthCorrection = (zLength - vertex[2])/zLength * 248.0/298.0

        # Verification if the alpha decay happens inside the active volume
        if (vertex[2]>0. and vertex[2]<activeVolumeHeight) :
            trackNumberVertexIn = trackNumberVertexIn + 1

        # Generation isotopically of the alpha direction
        u = alphaDirection()
        my_logger.debug("Alpha direction is %4.4f, %4.4f, %4.4f" %(u[0], u[1], u[2]))
    
        # Choice of the radionuclide and Smearing of the alpha Length
        choice = np.random.uniform(0.,1.)
        choiceRadionuclide = 0
     
        if (choice>relativeRadionuclideRatio[1]) :
            choiceRadionuclide = 1
            if (choice>relativeRadionuclideRatio[2]) :
                #print(choice)               
                choiceRadionuclide = 2
                #print(alphaLength[2])
        
        # Calculation of the alpha track length in dry air taking into account the struggling and temperature evolution on the vertical axis
        alphaLengthEByE = alphaLength[choiceRadionuclide] * np.random.normal(1.0, alphaRelativeStragglingRange) #* alphaLengthCorrection * 1.2

        # straight line \vec{v}(t) = \vec{vertex} + t \vec{u}
        alphaProjection = 0. 
        activeVolumeHeightEByE = activeVolumeHeight #* np.random.normal(1.0, activeVolumeHeightDispersion) 
        if (u[2]==0.) :  # protection for track direction in the xy plane
            if (vertex[2]>0. and vertex[2]<activeVolumeHeightEByE) :
                t1=0.
                t2 = alphaLengthEByE
            else :
                t1 = -9999.
                t2 = -9999.
        else :
            t2 = (activeVolumeHeightEByE-vertex[2])/u[2] # upper activeVolume
            t1 = -vertex[2]/u[2] #lower activeVolume
  
        if (vertex[2]>0. and vertex[2]<activeVolumeHeight) :
            if ( ( (vertex[2] + alphaLengthEByE*u[2]) >0.) and ((vertex[2] + alphaLengthEByE*u[2]) <activeVolumeHeight)  ) : 
                trackNumberBothVertexIn = trackNumberBothVertexIn + 1 

        
        my_logger.debug("Initial t values is %4.4f, %4.4f " %(t1, t2))
    
        if (t1<0.) :
            t1=0.
        if (t2<0.) :
            t2=0.
        if (t2>alphaLengthEByE) :
            t2 = alphaLengthEByE
        if (t1>alphaLengthEByE) : 
            t1 = alphaLengthEByE
            
        my_logger.debug("Final t values is %4.4f, %4.4f " %(t1, t2))

        x1 = vertex[0] + t1*u[0]
        y1 = vertex[1] + t1*u[1]
        x2 = vertex[0] + t2*u[0]
        y2 = vertex[1] + t2*u[1]
        alphaProjection = math.sqrt( (x2-x1)**2 + (y2-y1)**2)

        if (alphaProjection<0.) :
            my_logger.info("Alpha emission is %4.2f,%4.2f,%4.2f" %(vertex[0],vertex[1],vertex[2]))
            my_logger.info("Alpha direction is %4.4f, %4.4f, %4.4f" %(u[0], u[1], u[2]))
            my_logger.info("Alpha projection length is %4.4f" %(alphaProjection))
            my_logger.info("t1 and t2 are %4.4f %4.4f" %(t1,t2))

        # Gravity center of the track
        x_center = vertex[0] + (t1+t2)/2.*u[0]
        y_center = vertex[1] + (t1+t2)/2.*u[1]
        # Calculer average t as the center of the alphatrack to selection on a fiducial volume with respect to the center, as it is done with real data
        if ( (alphaProjection>minLength) and (x_center>-(xLength/2.-coronaSize) and x_center<(xLength-coronaSize)/2.) and (y_center>-(yLength/2.-coronaSize) and y_center<(yLength/2.-coronaSize)) ) : 
            # Track seen
            trackNumberSeen+=1
            lengthDistribution[trackNumberSeen]=alphaProjection
            if ((t1 == 0. and t2 == alphaLengthEByE) or (t2 == 0. and t1 == alphaLengthEByE) ) :
                # Track seen is fully in the active volume
                trackNumberSeenIn=trackNumberSeenIn+1
                lengthDistributionIn[trackNumberSeenIn]=alphaProjection
            else :
                # Track seen is partially in the active volume
                trackNumberSeenOut = trackNumberSeenOut+1
                lengthDistributionOut[trackNumberSeenOut]=alphaProjection
    # Final Plot
    my_logger.info("Track number is %6d " %(trackNumber))
    my_logger.info("Track number counter is %6d " %(trackNumberCounter))
    my_logger.info("Track number seen is %6d " %(trackNumberSeen))
    my_logger.info("Track number seen IN is %6d " %(trackNumberSeenIn))
    my_logger.info("Track number seen OUT is %6d " %(trackNumberSeenOut))
    my_logger.info("Track number Vertex IN is %6d " %(trackNumberVertexIn))
    my_logger.info("Track number Both Vertex IN is %6d " %(trackNumberBothVertexIn))

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))

    # Main Plot with the length track distribution
    ax[0].set_yscale('log')
    ax[0].set_ylim(0.1, 10000)
    ax[0].set_xlim(0.,100.)
    ax[0].set_xlabel(r"$\rm{Track \; length \; l (mm)}$")
    ax[0].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
    #Fixing bin width
    binWidth =1.
    lengthValues=np.arange(min(lengthDistribution), max(lengthDistribution) + binWidth, binWidth)
    # bin center calculation
    lengthValuesCenter = np.array([0.5 * (lengthValues[i] + lengthValues[i+1]) for i in range(len(lengthValues)-1)])

    histoValues, lengthValues, patches =  ax[0].hist(lengthDistribution[(lengthDistribution>0) & (lengthDistribution<100)], bins=lengthValues, log=True )

    # Removing negative values for error calculation
    histoValues[histoValues<0.] = 0.
    histoErrors = np.sqrt(histoValues)
    histoErrors[histoErrors==0] = 1.
    ax[0].errorbar(lengthValuesCenter, histoValues, yerr=histoErrors, fmt='o')

    # Fitting Interval
    Lmin = 15.
    Lmax = 99.
    jIni = np.intc(Lmin/binWidth+1)
    jFinal= np.intc(Lmax/binWidth)
    lengthValuesInterval  = lengthValuesCenter[jIni:jFinal]
    histoValuesInterval = histoValues[jIni:jFinal]
    histoErrors = histoErrors[jIni:jFinal]
    my_logger.info("Fitting between %4.2f and %4.2f mm" %(lengthValuesInterval[0],lengthValuesInterval[len(lengthValuesInterval)-1]))
    
    # alpha tracks with one extreme out of the active volume
    ax[1].set_yscale('log')
    ax[1].set_ylim(0.1, 10000)
    ax[1].set_xlim(0.,100.)
    ax[1].set_xlabel(r"$\rm{Track \; length \; Out \; l (mm)}$")
    ax[1].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
    histoValuesOut, lengthValuesOut, patches =  ax[1].hist(lengthDistributionOut[(lengthDistributionOut>0) & (lengthDistributionOut<100)], bins=lengthValues, log=True )
    histoValuesOut[histoValuesOut<0.] = 0.
    histoErrorsOut = np.sqrt(histoValuesOut)
    histoErrorsOut[histoErrorsOut==0] = 1.
    ax[1].errorbar(lengthValuesCenter, histoValuesOut, yerr=histoErrorsOut, fmt='o')

    # alpha tracks with both extremes in the active volume
    ax[2].set_yscale('log')
    ax[2].set_ylim(0.1, 10000)
    ax[2].set_xlim(0.,100.)
    ax[2].set_xlabel(r"$\rm{Track \; length \; In \; l (mm)}$")
    ax[2].set_ylabel(r"$\rm{dN/dl \; (mm)}$")
    histoValuesIn, lengthValuesIn, patches =  ax[2].hist(lengthDistributionIn[(lengthDistributionIn>0) & (lengthDistributionIn<100)], bins=lengthValues, log=True )
    histoValuesIn[histoValuesIn<0.] = 0.
    histoErrorsIn = np.sqrt(histoValuesIn)
    histoErrorsIn[histoErrorsIn==0] = 1.
    ax[2].errorbar(lengthValuesCenter, histoValuesIn, yerr=histoErrorsIn, fmt='o')

    fittingOption = 3    
    if (fittingOption==0) :
        # Fitting the curve tow one energy
        p1 = [1000.,10., 200., 41.7, 0.037]
        bounds1 = ([100., 3., 100., 20., 0.02],[10000., 20., 2000., 90., 0.60])
        popt, pcov = curve_fit(f=responseFunction, xdata=lengthValuesInterval, ydata=histoValuesInterval, p0=p1, sigma=histoErrors, bounds=bounds1)    
        print(popt)

        ax[0].plot(lengthValuesInterval, responseFunction(lengthValuesInterval, *popt), color='blue', linewidth=2.5, label=r'Fitted function')
        ax[0].plot(lengthValuesInterval, exponentialEndpoint(lengthValuesInterval, *popt), color='red', linewidth=2.5, label=r'Fitted function')
        ax[0].plot(lengthValuesInterval, gaussian(lengthValuesInterval, *popt), color='orange', linewidth=2.5, label=r'Fitted function')
        gaussianValues = gaussian(lengthValuesInterval, *popt)
        gaussianIntegral = sum(gaussianValues) 
        rawTrackIntegral = len(lengthDistribution[(lengthDistribution>30) & (lengthDistribution<50)])
        my_logger.info("Histo In integral %6d " %(len(lengthDistributionIn[(lengthDistributionIn>0) & (lengthDistributionIn<57)]))) 
        my_logger.info("Gaussian integral %6d " %(gaussianIntegral)) 
        my_logger.info("Total integral between 20-50 is %6d " %(rawTrackIntegral))

        ax[1].plot(lengthValuesInterval, exponentialEndpoint(lengthValuesInterval, *popt), color='red', linewidth=2.5, label=r'Fitted function')
        ax[2].plot(lengthValuesInterval, gaussian(lengthValuesInterval, *popt), color='blue', linewidth=2.5, label=r'Fitted function')

        my_logger.info("Total number of tracks simulated in the cloud chamber %6d " %(trackNumber))
        my_logger.info("Total volume of the cloud chambers %8.5f m3" %(fullVolumeHeight*fullVolumeWidth**2/1000./1000./1000.) )
        my_logger.info("Total number of tracks in the in peak %6d" %(gaussianIntegral))
        my_logger.info("Factor to get the Bq/m3 from the decay rate in the active volume of the chamber after fitting the peak %8.2f " %(trackNumber/(fullVolumeHeight*fullVolumeWidth**2/1000./1000./1000.)/gaussianIntegral)) 
        my_logger.info("If the decay rate in the peak in the active volume of the chamber is 1/60 Hz, then the Bq/m3 should be %8.2f Bq/m3" %(trackNumber/60./(fullVolumeHeight*fullVolumeWidth**2/1000./1000./1000.)/gaussianIntegral))

    if (fittingOption==1) :
        # Fitting to two energies 
        p2 = [1000., 10., 200., 42., 1000., 10., 200., 75., 0.037]
        bounds2 = ([100., 5., 10., 30., 100., 5., 10., 30., 0.015],[10000., 15., 1000., 90., 10000., 15., 1000., 90., 0.050])
        popt, pcov = curve_fit(f=twoEnergiesResponseFunction, xdata=lengthValuesInterval, ydata=histoValuesInterval, p0=p2, sigma=histoErrors, bounds=bounds2)    
        print(popt)
        popt_rp = [popt[0], popt[1], popt[2], popt[3], popt[8]]
        ax[0].plot(lengthValuesInterval, responseFunction(lengthValuesInterval, *popt_rp), color='orange', linewidth=2.5, label=r'Fitted function')
        popt_rp = [popt[4], popt[5], popt[6], popt[7], popt[8]]
        ax[0].plot(lengthValuesInterval, responseFunction(lengthValuesInterval, *popt_rp), color='red', linewidth=2.5, label=r'Fitted function')
        ax[0].plot(lengthValuesInterval, twoEnergiesResponseFunction(lengthValuesInterval, *popt), color='blue', linewidth=2.5, label=r'Fitted function')
       
    if (fittingOption==2) :
        p3 = [1000., 10., 200., 42., 700., 10., 200., 47., 100., 10., 200., 75., 0.037]
        bounds3 = ([100., 5., 10., 30., 100., 5., 10., 30., 100., 5., 10., 30., 0.015],[10000., 15., 1000., 90., 10000., 15., 1000., 90., 10000., 15., 1000., 90., 0.050])
        popt, pcov = curve_fit(f=threeEnergiesResponseFunction, xdata=lengthValuesInterval, ydata=histoValuesInterval, p0=p3, sigma=histoErrors, bounds=bounds3)    
        print(popt)
        popt_rp = [popt[0], popt[1], popt[2], popt[3], popt[12]]
        ax[0].plot(lengthValuesInterval, responseFunction(lengthValuesInterval, *popt_rp), color='orange', linewidth=2.5, label=r'Fitted function')
        popt_rp = [popt[4], popt[5], popt[6], popt[7], popt[12]]
        ax[0].plot(lengthValuesInterval, responseFunction(lengthValuesInterval, *popt_rp), color='red', linewidth=2.5, label=r'Fitted function')
        popt_rp = [popt[8], popt[9], popt[10], popt[11], popt[12]]
        ax[0].plot(lengthValuesInterval, responseFunction(lengthValuesInterval, *popt_rp), color='pink', linewidth=2.5, label=r'Fitted function')
        
        ax[0].plot(lengthValuesInterval, threeEnergiesResponseFunction(lengthValuesInterval, *popt), color='blue', linewidth=2.5, label=r'Fitted function')
       

        #responseFunction(X, Agauss, mu, relativeSigma, relativeGaus2Exp, L0) :
        #popt_responsefunction = [popt[0], popt[2]]
        #ax[0].plot(lengthValuesInterval, responseFunction(lengthValuesInterval, *popt), color='blue', linewidth=2.5, label=r'Fitted function')

    
    #ax[1].plot(lengthValuesInterval, exponentialEndpoint(lengthValuesInterval, *popt), color='red', linewidth=2.5, label=r'Fitted function')

   
    #ax[2].plot(lengthValuesInterval, gaussian(lengthValuesInterval, *popt), color='blue', linewidth=2.5, label=r'Fitted function')


    plt.savefig("TrackSimulationResults.pdf")
    plt.show()

if __name__ == "__main__" :
  rc = main()
  sys.exit(rc)