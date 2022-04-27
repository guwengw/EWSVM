# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:01:53 2021

@author: guwen
"""


import numpy as np
import scipy as sp
import ot
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


def load_dataset(dataset):
    if dataset == "mushroom":
        x, t = load_svmlight_file("mushrooms.txt")
        x = x.toarray()
        x = np.delete(x, 77, 1)  
        t[t == 1] = 1
        t[t == 2] = -1   
        
    t = t.astype('int')
    return x,t
        
    
def dataPreProcess(x):
    div = np.max(x, axis=0) - np.min(x, axis=0)
    div[div == 0] = 1
    x = (x - np.min(x, axis=0)) / div
    return x

def trainTestSpilt(x,t,trainRate):
    trainSize = int(t.shape[0] * trainRate)
    xtrain,xtest,ttrain,ttest = train_test_split(x, t, train_size=trainSize)
    return xtrain,xtest,ttrain,ttest

def multiClassTrain(xtrain,ttrain):
    xtrain_p = xtrain[ttrain == 1]
    ttrain_p = ttrain[ttrain == 1] 
    
    xtrain_n = xtrain[ttrain == -1]
    ttrain_n = ttrain[ttrain == -1] 

    xtrain_pn = np.r_[xtrain_p, xtrain_n]
    ttrain_pn = np.r_[ttrain_p, ttrain_n]
    
    return xtrain_pn,ttrain_pn

def multiClassTest(xtest,ttest):
    xtest_p = xtest[ttest == 1]
    ttest_p = ttest[ttest == 1] 
    
    xtest_n = xtest[ttest == -1]
    ttest_n = ttest[ttest == -1] 

    xtest_pn = np.r_[xtest_p, xtest_n]
    ttest_pn = np.r_[ttest_p, ttest_n]
    
    return xtest_pn,ttest_pn
    
    
def positiveUnlabeledSpilt(x,t,n_unl,prior,n_pos):
    size_u_p = int(prior * n_unl)
    size_u_n = n_unl - size_u_p

    xp_t = x[t == 1]
    tp_t = t[t == 1]    
    
    xp, xp_other, _, tp_o = train_test_split(xp_t, tp_t, train_size=n_pos)

    xup, _, _, _ = train_test_split(xp_other, tp_o, train_size=size_u_p)


    xn_t = x[t == -1]
    tn_t = t[t == -1]
    xun, _, _, _ = train_test_split(xn_t, tn_t, train_size=size_u_n)
    xu = np.concatenate([xup, xun], axis=0)

    P = xp
    U = xu

    return P,U

def computeCostMatrix(P,U):
    M = sp.spatial.distance.cdist(U, P)
    return M

def sinkhornTransport(n_unl,n_pos,lambd,M):
    p = ot.unif(n_unl)
    q = ot.unif(n_pos)
    #Gs = ot.sinkhorn(p, q, M, lambd, numItermax=2000,verbose=True)
    Gs = ot.sinkhorn(p, q, M, lambd, numItermax=2000)
    return Gs

def computeEntropy(Gs):
    #Gs = Gs.T
    entro = np.zeros(Gs.shape[0])
    for i in range(Gs.shape[0]):
        entro[i] = sp.stats.entropy(Gs[i])
        
    entro_min = min(entro)
    entro_max = max(entro)
    
    entro = (entro-entro_min)/(entro_max-entro_min)
    return entro

