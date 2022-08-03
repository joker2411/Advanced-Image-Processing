# -*- coding: utf-8 -*-
"""

@author: arpita
"""

print('Loading images from a database and computing Gabor magnitude features.')
print('During this step we also need to  construct the \n Gabor filter bank and extract the Gabor magnitude features.')
print('This may take a while.')

import os
import cv2
import numpy as np

import LGandXORandMedian as lg
import CreateLogGaborFilter as createlgf
import csv
from Classification import Results


#...........Standard Image Size [k x N]..........
k=100 
N=100

#.....Creating log Gabor Filter Bank........

nscale=4
norient=6
minWaveLength=3
mult=3
sigmaOnf=0.55
dThetaOnSigma=1.5

filterbank_freq=createlgf.createLogGaborFilters(k,N, nscale, norient, minWaveLength, mult,sigmaOnf, dThetaOnSigma)


#......Locating Database.......
main_folder_path="E:\AIP_Project\DATASET\ORL"#"D:\XOR based\database"
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


#Read of Random Grid Matrix(RG) [240000x90]

RG=[]
with open('RandomMatrix.csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        for i in range(0,len(row)):
            row[i]=int(row[i])
        RG.append(row)

print('Random Grid Matrix Loaded')
RG=np.array(RG)
RG=RG.astype(np.uint8)




''''
In this scheme size of median filter is fixed...
#Let size of median filter be p=3
#P=3

p=[]
with open('MedianFilterSize(p).csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        for i in range(0,len(row)):
            row[i]=int(row[i])
        p.append(row)

print('MedianFilterSize(p) Loaded') 
p=np.array(p)
'''




#DataMatrix
ID=[]
for x in range (0,R):  #R: number of subjects
    print('Subject: ',x+1) #C : number of samples per subject
    for y in range(0,C):
        ID.append(x)
        
        ImgPath=S_Path[x]+'\\'+sub_folder_content[x][y]
        Img=cv2.imread(ImgPath,0)
        Img=np.asarray(Img,dtype=np.float64)
        Img = cv2.resize(Img, (k,N), interpolation=cv2.INTER_CUBIC)
        #Img=(Img.reshape(k*N,1))
        Rg=RG[0:k*N,x] #Image Size :[k x N]
        Rg=Rg.reshape(k,N)
        Rg=Rg.astype(np.uint8)
        # For worst case scenario (same RG for all users) set x=1 or x=2 as required...
        
        #transformedFeatureVector=Img
        transformedFeatureVector=lg.LogXorMedian(Img,Rg,filterbank_freq,nscale,norient) #(24x100x100,1)
        #print((transformedFeatureVector[0]))
        if x==0 and y==0:
            dataMatrixRG=np.column_stack((transformedFeatureVector)).T
        else:
            dataMatrixRG=np.column_stack((dataMatrixRG,transformedFeatureVector))
        

        #print(len(transformedFeatureVector))
        #print(featureVector[800][0])
        
#print(len(dataMatrixRG))        
#print(len(dataMatrixRG[200]))
print('\n .....Feature Extraction completed.....')

data=dataMatrixRG.T #Transpose of dataMatrix


Avg_far=[]
Avg_frr=[]
Avg_rec_rates=[]

Nfolds=C
for K in range(0,Nfolds):
    print('\n\n------------------------------------------------------')

    print('Fold ',K+1)
    test_data=[]
    train_data=[]
    
    ids_test=[]
    ids_train=[]
    
    cnt=0
    for i in range(0,R):
        for j in range(0,C):
            if j==0 :
                test_data.append(data[cnt].astype(np.float64))
                ids_test.append(ID[cnt])
            else:
                train_data.append(data[cnt].astype(np.float64))
                ids_train.append(ID[cnt])
            cnt+=1
    
    test_data=((np.array(test_data)).T)
    train_data=((np.array(train_data)).T)
        
     
    results,test_features,classifyResults,output=Results.compute_results(train_data,ids_train,test_data,ids_test)
    Avg_far.append(output.DET_far_rate)
    Avg_frr.append(output.DET_frr_rate)
    Avg_rec_rates.append(output.CMC_rec_rates)
    
FAR,FRR, Rec_Rates=Results.average_output(Avg_far,Avg_frr,Avg_rec_rates,Nfolds)
print('Average ROC')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(FAR*100, FRR*100,label=' ROC curve ',color='deeppink', linestyle=':', linewidth=4)

plt.xlim([0.0, 100])
plt.ylim([0.0, 100])

plt.xlabel('False Accept Rate')
plt.ylabel('False Reject Rate')
plt.show()


#Calculate Equal Error Rate
EER=Results.compute_avg_EER(FAR,FRR)
print('======================================================================')
print('SOME PERFORMANCE METRICS:')
print('Identification experiments:')
print('Log Gabor - XOR- Median')
print('The equal error rate on the evaluation set equals ',EER,'%')
print('The rank one recognition rate of the experiments equals',Rec_Rates[0]*100, '%')






















