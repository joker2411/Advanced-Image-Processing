
import cv2
import numpy as np
import math
import scipy
from scipy import misc
from matplotlib import pyplot as plt

def createLogGaborFilters(k,N, nscale, norient, minWaveLength, mult,sigmaOnf, dThetaOnSigma):
    
    #filterbank_spatial=[]
    filterbank_freq=[]

    
    # Pre-allocate cell array
    '''
    filterbank_spatial=[]
    for i in range(0,nscale):
        filterbank_spatial.append([])
        for j in range(0,norient):
            filterbank_spatial[i].append(0)
    '''
    
            
    filterbank_freq=[]
    for i in range(0,nscale):
        filterbank_freq.append([])
        for j in range(0,norient):
            filterbank_freq[i].append(0)
   

    BP=[]
    for i in range(0,nscale):
        BP.append([])
        for j in range (0,1):
             BP[i].append(0)
     
    # Pre-compute some stuff to speed up filter construction
    # Set up X and Y matrices with ranges normalised to +/- 0.5
    # The following code adjusts things appropriately for odd and even values
    # of rows and columns.
    
    
    if N%2==1:
        xRange = np.arange(-(N-1)/2,((N-1)/2)+1, 1)
        xRange=xRange/(N-1)
        
    else:
        xRange = np.arange(-N/2,N/2, 1)
        xRange=xRange/N
        
   
    if k%2==1:
        yRange = np.arange(-(k-1)/2,((k-1)/2)+1, 1)
        yRange=yRange/(k-1)
    else:
        yRange = np.arange(-k/2,k/2, 1)
        yRange=yRange/k
    

    x,y = np.meshgrid(xRange, yRange, sparse=False)
    
    
    radius = np.sqrt(x**2 + y**2)       # Matrix values contain *normalised* radius from centre.
    
    theta = np.arctan2(y,x)             # Matrix values contain polar angle.
    
    radius=np.fft.fftshift(radius)  # Quadrant shift radius and theta so that filters
    theta=np.fft.fftshift(theta)
    
    radius[0][0]=1                  # Get rid of the 0 radius value at the 0
                                    # Get rid of the 0 radius value at the 0
                                      # frequency point (now at top-left corner)
                                      # so that taking the log of the radius will 
                                      # not cause trouble.
    
    
    
    
    
    sintheta = np.sin(theta)
    costheta=np.cos(theta)
    
    del x; del y; del theta;    # save a little memory
    
    # Filters are constructed in terms of two components.
    # 1) The radial component, which controls the frequency band that the filter
    #    responds to
    # 2) The angular component, which controls the orientation that the filter
    #    responds to.
    # The two components are multiplied together to construct the overall filter.
    
    # Construct the radial filter components...
    # First construct a low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries.  All log Gabor filters are multiplied by
    # this to ensure no extra frequencies at the 'corners' of the FFT are
    # incorporated. This keeps the overall norm of each filter not too dissimilar.

    logGabor=[]
  
    for i in range(0,nscale):
        logGabor.append([]) 
    
    for s in range (0,nscale):
        wavelength = minWaveLength*pow(mult,(s))
        
        fo = 1.0/wavelength                  # Centre frequency of filter.
        logGabor[s] = np.exp((-(np.log(radius/fo))**2) / (2 * (np.log(sigmaOnf))**2))
       
        logGabor[s][0][0] = 0;              # Set the value at the 0
                                            # frequency point of the filter 
                                            # back to zero (undo the radius fudge).
        # Compute bandpass image for each scale
        #logGabor[s] = logGabor[s]
        #BP[s] = np.fft.ifft2(np.fft.ifftshift(imagefft * logGabor[s]))
        #BP[s] = np.fft.ifft2(imagefft * logGabor[s])
        #BP[s]=BP[s].real

        
    cnt=1
    # The main loop...
    for o in range (0,norient):# For each orientation.

        angl = ((o+1)-1)*np.pi/(norient)    # Calculate filter angle.
        wavelength = minWaveLength          # Initialize filter wavelength.
        

        # Pre-compute filter data specific to this orientation
        # For each point in the filter matrix calculate the angular distance from the
        # specified filter orientation.  To overcome the angular wrap-around problem
        # sine difference and cosine difference values are first computed and then
        # the atan2 function is used to determine angular distance.
        
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)  # Difference in sine.
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)  # Difference in cosine.
        dtheta = abs(np.arctan2(ds,dc))         # Absolute angular distance.

        # Calculate the standard deviation of the angular Gaussian
        # function used to construct filters in the freq. plane.
        thetaSigma = np.pi/norient/dThetaOnSigma
        spread = np.exp((-dtheta**2) / (2 * thetaSigma**2))
        
        for s in range  (0,nscale):     # For each scale.
            filter1 = logGabor[s] * spread      # Multiply by the angular spread to get the filter
            
            # save each filter in filterbank_freq[][]
            filterbank_freq[s][o] =  filter1
            
            cnt+=1
            wavelength = wavelength * mult;         # Finally calculate Wavelength of next filter
        #end .. #For each scale                     # ... and process the next scale
    # end .. # For each orientation
    return filterbank_freq
    
    
