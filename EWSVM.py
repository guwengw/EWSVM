# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:03:06 2021

@author: guwen
"""


import numpy as np
from sklearn.svm import SVC
import utils 

from sklearn.metrics import f1_score

C = 10

svc = SVC(C, kernel='rbf', gamma='auto', probability=True, random_state=2018)


trainRate = 0.7

algorithm = "EWSVM"

data_list = ['mushroom']

priors = [0.3,0.5,0.7]

lambda_dir = {'mushroom': 0.1}

for dataset in data_list: 
    
    n_unl = 800
    
    n_pos = 400

    lambd = lambda_dir[dataset]
    
    x,t = utils.load_dataset(dataset)   
    
    x = utils.dataPreProcess(x)
    
    xtrain,xtest,ttrain,ttest = utils.trainTestSpilt(x,t,trainRate)
    
    xtest,ttest = utils.multiClassTest(xtest,ttest)
    
    result = []
    
    for prior in priors:
        
        print(prior)
        
        f1 = []
        
        for i in range(20):
            
            print(dataset + str(prior) + str(i))
    
            P,U = utils.positiveUnlabeledSpilt(xtrain,ttrain,n_unl,prior,n_pos)
            
            M = utils.computeCostMatrix(P,U)
            
            Gs = utils.sinkhornTransport(n_unl,n_pos,lambd,M)
            
            entro = utils.computeEntropy(Gs)
            
            X_train = np.r_[P, U]
            y_train = np.r_[np.ones(len(P)), np.ones(len(U))*(-1)]
            
            weight = np.r_[np.ones(P.shape[0]),entro]
            svc.fit(X_train, y_train, sample_weight=weight)
            y_pred = svc.predict(xtest)
            
            errorE = len(np.where((y_pred == ttest ) == False)[0]) 
            errorRateE = errorE / ttest.shape[0]
            
            f1EWSVM = f1_score(ttest,y_pred,average = 'micro')
            f1.append(f1EWSVM)
            
        result.append(f1)
    

