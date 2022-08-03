import cv2


import LogGaborConvolve as lgc
import numpy as np
import scipy.misc
import scipy.ndimage



def LogXorMedian(img,RG,filterbank_freq,nscale,norient):
    #Obtaining 24 filtered images using log Gabor transform at n=4 (Scales)  and  m=6 (Orientations)
    n=nscale
    m=norient
   

    filteredImages=lgc.logGaborConvolve(img, filterbank_freq,nscale,norient)

    reI=[] 
    ImI=[]
    F_Mag=[]     #Magnitude of log Gabor transformed Image
    F_Phase=[]   #Phase of log Gabor transformed Image

    for i in range (0,n):
        
        reI.append([])
        ImI.append([])
        F_Mag.append([])
        F_Phase.append([])

        for j in range (0,m):
            
            reI[i].append(filteredImages[i][j].real)
            ImI[i].append(filteredImages[i][j].imag)
            F_Mag[i].append(np.sqrt((reI[i][j]**2)+(ImI[i][j]**2)))
            F_Phase[i].append(np.arctan(ImI[i][j]/reI[i][j]))



    del reI
    del ImI
    '''
    from matplotlib import pyplot as plt

    fig=plt.figure(figsize=(12, 7))
    columns = 6
    rows = 4
    count=1
    plt.title("Magnitude Pattern of log Gabor Transformed Images")
    plt.axis("off")
    for i in range (0,n):
        for j in range (0,m):
             fig.add_subplot(rows, columns, count)
             
             plt.axis("off")
             plt.imshow(F_Mag[i][j], cmap = 'gray')
             count=count+1
    plt.show()


    fig=plt.figure(figsize=(12, 7))
    columns = 6
    rows = 4
    count=1
    plt.title("Phase Pattern of log Gabor Transformed Images")
    plt.axis("off")

    for i in range (0,n):
        for j in range (0,m):
             fig.add_subplot(rows, columns, count)
             plt.axis("off")
             plt.imshow(F_Phase[i][j], cmap = 'gray')
             count=count+1
    plt.show()
    '''
    
    # Next step fuses and salts phase and magnitude patterns of log Gabor transformed with the help of random grid [ XOR operation]

    tempXS = [] 
    XS=[]

    '''
    fig=plt.figure(figsize=(12, 7))
    columns = 6
    rows = 4
    counter=1
    plt.title("XOR")
    plt.axis("off")
    '''
    for i in range (0,n):
        tempXS.append([])
        XS.append([])
        for j in range (0,m):
            
            tempXS[i].append(cv2.bitwise_xor(F_Mag[i][j].astype(np.uint8),F_Phase[i][j].astype(np.uint8)))
            XS[i].append(cv2.bitwise_xor(tempXS[i][j].astype(np.uint8),RG.astype(np.uint8)))

            '''
            fig.add_subplot(rows, columns, counter)
            plt.axis("off")
            plt.imshow(XS[i][j], cmap = 'gray')
            counter+=1

    plt.show()
    '''
    #print("XS")
    #print(XS[0][0][0][0])

    # To make salted Fuse patterns non-invertible , 2-D Median Filtering is applied

    MedianFilteredImg=[]

    '''
    fig=plt.figure(figsize=(12, 7))
    columns = 6
    rows = 4
    counter=1
    plt.title("Median Filtered Images")
    plt.axis("off")
    '''
    
    for i in range (0,n):
        MedianFilteredImg.append([])
        for j in range (0,m):
            Q=scipy.ndimage.filters.median_filter(XS[i][j],size=(5,5),footprint=None, output=None, mode='reflect',cval=0.0,origin=0)
            MedianFilteredImg[i].append(Q)
            '''
            fig.add_subplot(rows, columns, counter)
            plt.axis("off")
            plt.imshow(MedianFilteredImg[i][j], cmap = 'gray')
            counter+=1
    plt.show()
    '''        
    

    #print(MedianFilteredImg[1][1][1][1])


    for r in range(0,n):
        for c in range(0,m):
            dst = MedianFilteredImg[r][c].reshape(100*100,1)
            dst=np.array(dst)
            if r==0 and c==0:
                dataMatrixPart=dst
            else:
                dataMatrixPart=np.append(dataMatrixPart,dst,axis=0)
                
                
 
    
    return dataMatrixPart
    

    
            




        










