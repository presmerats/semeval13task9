from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score

from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import numpy as np
import json
import pickle
import pprint
import copy
from time import gmtime, strftime





def loadData(filename):
    
    with open(filename,'r') as f:
        return json.load(f)
    
    return None


def saveModel( clf, filename):
    
    joblib.dump(clf, filename) 
    return

   
def loadModel(filename):
    return joblib.load(filename) 

# def transformToDataDict(data,n=None,ertype="ER"):
    
#     #print("transformToDataDict ertype="+ertype)
#     if ertype=="ER":
#         return transformToDataDictER(data,n)
#     elif ertype=="NER":
#         return transformToDataDictNER(data,n)
#     else:
#         return transformToDataDictER(data,n)
    

# def transformToDataDictER(data,n=None):
#     """
#         target is just BIO tag
        
#         Transform data.json form into a suitable dictionary
#         Accepts test sets without the target/class variable
#         (named 'biotag' in the usually generated dictionary)
#     """
#     if n is None:
#         n=len(data)

#     targets = [ e['biotag'] 
#                 if 'biotag' in e.keys() else ''
#                 for e in data[:n]  ]
        
#     features = []
#     for wdict in data[:n]:
#         aux_dict = copy.deepcopy(wdict)
#         #aux_dict = wdict

#         if 'biotag' in aux_dict.keys():
#             aux_dict.pop('biotag',None)
#         if 'drugtype' in aux_dict.keys():
#             aux_dict.pop('drugtype',None)
#         if 'offsetend' in aux_dict.keys():
#             aux_dict.pop('offsetend',None)
#         if 'offsetstart' in aux_dict.keys():
#             aux_dict.pop('offsetstart',None)
#         if 'sentenceid' in aux_dict.keys():
#             aux_dict.pop('sentenceid',None)
        

#         # we don't need to sort the keys right?
#         features.append(aux_dict)

#     return features, targets


# def transformToDataDictNER(data,n=None):
#     """
#         Target is now: BIO tag and drug group name
        
#         Transform data.json form into a suitable dictionary
#         Accepts test sets without the target/class variable
#         (named 'biotag' in the usually generated dictionary)
#     """
#     if n is None:
#         n=len(data)

#     targets = [ e['biotag'] + "-" + e['drugtype'] 
#                 if 'biotag' in e.keys() and 'drugtype' in e.keys() 
#                 else ''
#                 for e in data[:n]  ]

#     features = []
#     for wdict in data[:n]:
#         aux_dict = copy.deepcopy(wdict)
#         #aux_dict = wdict

#         if 'biotag' in aux_dict.keys():
#             aux_dict.pop('biotag',None)
#         if 'drugtype' in aux_dict.keys():
#             aux_dict.pop('drugtype',None)
#         if 'offsetend' in aux_dict.keys():
#             aux_dict.pop('offsetend',None)
#         if 'offsetstart' in aux_dict.keys():
#             aux_dict.pop('offsetstart',None)
#         if 'sentenceid' in aux_dict.keys():
#             aux_dict.pop('sentenceid',None)
                
#         # we don't need to sort the keys right?
#         features.append(aux_dict)

    

    
#     return features, targets


def newPipeline_fromFile( datafile='../data/models/data.json', n=None):

    # loading data
    data = loadData(datafile)
    if n is None:
        n=len(data)

    return newPipelineSVM(data,n)


def newPipeline( X, Y,n=None, ertype="ER", algotype="CRF", weights="", cv=False):
    
    """
        ertype= "ER" means Entity Recognition
        ertype= "NER" means Named entity recognitioin
    """

    if n is None:
        n=len(X)

    # separate features and class
    #X, Y = transformToDataDict(data,n, ertype)

    #for test in Y:
    #    if test != "O-":
    #        print(test, sep=" ")

    #for test in X[:3]:
    #    pprint.pprint(test)


    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    # pprint.pprint(X[29][1:3])
    # print(len(X))
    # print(Y[29])
    # print(len(Y))
    
    if not cv:
        crf.fit(X, Y)
        return crf
    else:
        
        kf = KFold(n_splits=2, shuffle=True, random_state=1)
        scores = cross_val_score(crf, X, Y, cv=kf, n_jobs=2)
        scores.append(scores.mean())
        print(scores)
        return crf, scores


def savePipeline(pipeline):
    
    # save pipeline
    datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
    modelfilename = '../data/models/svm-pipeline_'+ datesuffix +'.pkl'
    joblib.dump(pipeline, modelfilename) 
    
    return modelfilename

   
def loadPipeline(filename):
    return joblib.load(filename) 


def predict_fromfile(pipeline, datafile):
    
    # load from file
    testdata = loadData(datafile)
    return predict(pipeline, testdata)


    
def predict(pipeline, X, ertype="ER", algotype=None):
    """
        ertype="ER"  drugtype will be part of X (the features)
        ertype="NER" drugtype will be part of Y with BIO tags (the targets)
    """
    #print("predict ertype="+ertype)
    # convert features

    #for test in testdata:
    #    if "drugtype" in test.keys() and test["drugtype"]!="":
    #        pprint.pprint(test)
    #        break

    #X, Y = transformToDataDict(testdata, ertype=ertype)


    
    #for test in Y:
    #    if test != 'O-':
    #       pprint.pprint(test)

    # predict
    return pipeline.predict(X)

