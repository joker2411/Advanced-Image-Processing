# -*- coding: utf-8 -*-
"""

@author: home
"""

import FusionWithKey as FK

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import time
from scipy import interp

import csv

def hog(img):
    bin_n = 32 # Number of bins

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


# To have all images in dataset of same size
k=112
N=92

#......Locating Database.......
main_folder_path="E:\AIP_Project\DATASET\ORL" #Th_Vis_fusion\crop_thermal"#"D:\FaceDataset\Database\IRIS_crop\\visible"   #"D:\XOR based\database" #"D:\Database\palmprint"#
main_folder_content=(os.listdir(main_folder_path))
NumberOfSub=len(main_folder_content)
S_Path=[]
for subfolderNo in range(0,NumberOfSub):
    S_Path.append(main_folder_path+'\\'+main_folder_content[subfolderNo])
sub_folder_content=[]
for subfolderNo in range(0,NumberOfSub):
    sub_folder_content.append(os.listdir(S_Path[subfolderNo]))
    
R=len(sub_folder_content) #No. of rows sub_folder_content[][]...40
C=len(sub_folder_content[0]) #No. of cols sub_folder_content[][]...10






#



#DataMatrix
ID=[]
for x in range (0,R):  #R: number of subjects
    print('Subject: ',x+1) #C : number of samples per subject
    for y in range(0,C):
        ID.append(x)
        
        ImgPath=S_Path[x]+'\\'+sub_folder_content[x][y]
        Img=cv2.imread(ImgPath,0)
        
        #plt.subplot(121),plt.imshow(Img, cmap = 'gray')
        #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        
        Img = np.float32(Img) / 255.0
        #Img=np.asarray(Img,dtype=np.float64)
        
        #print(Img.shape)
        #print(Img.shape)
        Img = cv2.resize(Img, (k,N), interpolation=cv2.INTER_CUBIC)
        
        #plt.subplot(122),plt.imshow(Img,cmap = 'gray')
        #plt.title('Resized Image'), plt.xticks([]), plt.yticks([])
        #plt.show()
        #print(Img.shape)

        '''
        # Calculate gradient 
        gx = cv2.Sobel(Img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(Img, cv2.CV_32F, 0, 1, ksize=1)

        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)


        plt.subplot(131),plt.imshow(gx, cmap = 'gray')
        plt.title('gx'), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(gy,cmap = 'gray')
        plt.title('gy'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(mag,cmap = 'gray')
        plt.title('mag'), plt.xticks([]), plt.yticks([])
        plt.show()
        

        
        
        Img=(Img.reshape(k*N,1))
        transformedFeatureVector=Img
        '''
        
        HogFeature=hog(Img)
        lenHOG=len(HogFeature)

        HogFeature=HogFeature.reshape(lenHOG,1)
#
#        #print(HogFeature.shape)
#        Key=KeyMat[1]
#        
#        Key=Key[0:lenHOG].reshape(lenHOG,1)
#        #print(Key.shape)
#        p=3
        #transformedFeatureVector=FK.F_Key(Key,HogFeature)#,lenHOG,p)
        
        #print((transformedFeatureVector.shape))
        transformedFeatureVector=HogFeature
        if x==0 and y==0:
            dataMatrixRG=np.column_stack((transformedFeatureVector)).T
        else:
            dataMatrixRG=np.column_stack((dataMatrixRG,transformedFeatureVector))
        


print(dataMatrixRG.shape)     
print('\n .....Feature Extraction completed.....')







data=dataMatrixRG.T #Transpose of dataMatrix

Avg_far=[]
Avg_frr=[]
Avg_rec_rates=[]
Nfolds=C

for K in range(0,Nfolds):
    #print('\n\n------------------------------------------------------')
    #print('Fold ',K+1)
    test_data=[]
    train_data=[]
    
    ids_test=[]
    ids_train=[]
    
    cnt=0
    for i in range(0,R):
        for j in range(0,C):
            if j==K:
                test_data.append(data[cnt].astype(np.float64))
                ids_test.append(ID[cnt])
            else:
                train_data.append(data[cnt].astype(np.float64))
                ids_train.append(ID[cnt])
            cnt+=1
    
    test_data=((np.array(test_data)).T)
    train_data=((np.array(train_data)).T)
    


    X_train = train_data.T 
    X_test = test_data.T
    y_train = ids_train
    y_test = ids_test
    


    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=0))       
    # Fitting Naive Bayes to the Training set
    #classifier = MultinomialNB()
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    #print('y_score')
    #print(y_score)
    y_pred = classifier.predict(X_test)
    #print(y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    #print('Confusion Matrix')
    #print(cm)
    cr = classification_report(y_test, y_pred)
    #print('Classification Report')
    #print(cr)

    accuracy = accuracy_score(y_test, y_pred)
    #print('Accuracy Score',accuracy)

    # Binarize the output
    from sklearn.preprocessing import label_binarize

    classes=np.unique(ID)
    y_test = label_binarize(y_test, classes)
    n_classes = y_test.shape[1]

    #Begin : FRR and FAR ROC
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    #print('fpr')
    #print(fpr)
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    #print('fpr')
    #print(fpr)
    #print('tpr')
    #print(tpr)
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(0,40):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Fold '+ str(K+1) + ' Accuracy Score '+ str(accuracy*100)+'%' )

    
    time.sleep(1.0)
