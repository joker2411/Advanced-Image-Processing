import numpy as np
#import scipy.misc
import scipy.ndimage
def F_Key(Key,HogFeature):#,lenHOG,p):
    
    
    HogFeature=np.array(HogFeature)
    fvsParts=np.split(HogFeature,2)
    X1=fvsParts[0]
    Y1=fvsParts[1]

    
    Key=np.array(Key)
    KeyParts=np.split(Key,2)
    X2=KeyParts[0]
    Y2=KeyParts[1]

    
    X2=np.array(X2)
    X1=np.array(X1)
    Y1=np.array(Y1)
    Y2=np.array(Y2)
    
    Tf=np.sqrt((X2-X1)**2+(Y2-Y1)**2)

    TfM=scipy.ndimage.filters.median_filter(Tf,size=3,footprint=None, output=None, mode='reflect',cval=0.0,origin=0)
  
    return TfM
