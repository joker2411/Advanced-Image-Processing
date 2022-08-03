import numpy as np
def logGaborConvolve(img, filterbank_freq,nscale,norient):
    imagefft = np.fft.fft2(img) # Fourier transform of image

    EO=[]
    for i in range(0,nscale):
        EO.append([])
        for j in range(0,norient):
            EO[i].append(0)
    for s in range(0,nscale):
        for o in range(0,norient):
            EO[s][o] = np.fft.ifft2(np.fft.ifftshift(imagefft * filterbank_freq[s][o]))

            
    return EO    
