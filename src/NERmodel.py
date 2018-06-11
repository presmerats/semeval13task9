from FeatureExtraction import FeatureExtraction
from CustomFeatures import MyFeatures
import SVMtrainer
import CRFtrainer
import traceback, sys
import json
import subprocess 
import yaml
import os
import pprint
from joblib import Parallel, delayed
import joblib
import time
from time import gmtime, strftime
import copy
import multiprocessing
from multiprocessing import Process, Value, Lock, Pool
from functools import partial
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
from sklearn.model_selection import cross_val_score



mylock = Lock()

class NERmodel():
    
    def __init__(self, algorithm="lsvm", featureset=None,name="", modeltype="ER"):
        
        self.name = name
        
        self.modeltype = modeltype
        
        self.featureset = MyFeatures()
        if not featureset is None:
            self.featureset = featureset
        
        self.algotype=algorithm
        self.algo = SVMtrainer
        if algorithm.lower() in ["svm","lsvm","lwsvm"]:
            self.algo = SVMtrainer

        elif algorithm.lower() == "crf":
            self.algo = CRFtrainer
        #elif algorithm.lower() == "randomforest":
        #    self.algo = RFtrainer

        
        self.pipeline = None
        self.pipelinefile = None
        
        self.traindata = None
        self.traindatafile = None
        
        self.testdata = None
        self.testdatafile = None
        
        self.predictionResults = None
        self.predictionResultFile = None

        self.scores =[]
        
    def __enter__(self):
        # nothing to set up besides __init__
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __del__(self):
        print("freeing NERmodel instance...")
        del self.traindata
        self.traindata  = None
        del self.testdata
        self.testdata = None
        self.featureset = None
        self.pipeline = None
        
    def setName(self, name):
        self.name = name
        
    def trainFeatureExtraction(self, datafile, limit=None):
        self.traindata = FeatureExtraction(
            datafile, 
            self.featureset, 
            algoformat=self.algotype,
            targetformat=self.modeltype
            )
        self.traindata.load()
        self.traindata.extractFeatures(limit)

        if self.algotype == "lwsvm":
            self.traindata.weightsVector()
        
    def setTraindataFile(self, filepath):
        self.traindatafile = filepath
       
    def saveTrainingFeatures(self, filepath=None):
        if not filepath is None:
            self.traindatafile = filepath
        self.traindata.save(self.traindatafile)
    
    def loadTrainingFeatures(self, filepath=None):
        if not filepath is None:
            self.traindatafile = filepath
        
        if self.traindata is None:
            self.traindata = FeatureExtraction('', self.featureset)
            
        with open(self.traindatafile,'r') as f:
            self.traindata.datadict = json.load(f)
            self.traindata.transformToXY(self.traindata.datadict)
      
    def newModelPipeline(self,ertype="NER",algotype="lwsvm",cv=False):
        if algotype=="lwsvm":
            self.pipeline, self.vectorizer = self.algo.newPipeline(
                self.traindata.X,
                self.traindata.Y, 
                ertype=ertype,
                algotype=algotype,
                weights=self.traindata.weights)
        else:
            if cv:
                self.pipeline, self.scores = self.algo.newPipeline(
                    self.traindata.X,
                    self.traindata.Y, 
                    ertype=ertype,
                    algotype=algotype,
                    weights=self.traindata.weights,
                    cv=cv)
            else:
                self.pipeline= self.algo.newPipeline(
                    self.traindata.X,
                    self.traindata.Y, 
                    ertype=ertype,
                    algotype=algotype,
                    weights=self.traindata.weights)
        
    def setPipelinefile(self,filepath):
        self.pipelinefile=filepath
        
    def saveModelPipeline(self):
        self.pipelinefile = self.algo.savePipeline(self.pipeline)
        return self.pipelinefile
    
    def loadModelPipeline(self, filepath=None):
        if not filepath is None:
            self.pipelinefile = filepath
        self.pipeline = self.algo.loadPipeline(self.pipelinefile)
        
    
    def testFeatureExtraction(self, folderpath, limit=None):
        self.testdata = FeatureExtraction(
            folderpath, 
            self.featureset,
            algoformat=self.algotype,
            targetformat=self.modeltype
            )
        self.testdata.load()
        self.testdata.extractFeatures(limit)
        
    def setTestdataFile(self, filepath):
        self.testdatafile = filepath
       
    def saveTestFeatures(self, filepath=None):
        if not filepath is None:
            self.testdatafile = filepath
        
        self.testdata.save(self.testdatafile)
    
    def loadTestFeatures(self, filepath=None): 
        if not filepath is None:
            self.testdatafile = filepath
            
        if self.testdata is None:
            self.testdata = FeatureExtraction('', self.featureset)
            
        with open(self.testdatafile,'r') as f:
            
            self.testdata.datadict = json.load(f)
            self.testdata.transformToXY(self.traindata.datadict)
    
    def predict(self,mode=None,algotype="lsvm"):
        """
            mode="ER" sets drugtype as a feature
            mode="NER" sets drugtype as part of the target and hence not in the features
        """
        #print(self.modeltype)
        if not mode and self.modeltype:
            mode = self.modeltype
        elif not mode:
            mode = "ER"
            
        if algotype=="lwsvm":
            #print("the model is " + mode)
            self.predictionResults = self.algo.predict(
                self.pipeline, 
                self.testdata.X,
                ertype=mode,
                algotype=self.algotype,
                vectorizer=self.vectorizer)

        else:
            #print("the model is " + mode)
            self.predictionResults = self.algo.predict(
                self.pipeline, 
                self.testdata.X,
                ertype=mode,
                algotype=self.algotype)
        
   
        
        
    def setPredictionResultsFile(self, filepath):
        self.predictionResultFile = filepath
        
        
    def parseSingleTagOutput(self, tag, wordtuple):
        
        sentenceid = wordtuple['sentenceid']
        offset1 = wordtuple['offsetstart']
        offset2 = wordtuple['offsetend']
        word = wordtuple['word']


        tag = tag
        drugtype = "null"
        if self.modeltype=="NER":
            tag2 = tag.split('-')
            tag = tag2[0]
            drugtype = tag2[1]
        else:
            # this should be extracted from the prediction result
            #drugtype = wordtuple['drugtype']
            drugtype= "null"
            
        return [offset1,offset2,word, drugtype,tag,sentenceid]
        
    def formatPrediction(self,matchdict,debug=False):
        
        finallist = []
        for k,v in matchdict.items():
            drugword = []
            #create new element until B is found again
            for name in v:
                offset1 = int(name[0])
                offset2 = int(name[1])-1
                word = name[2]
                drugtype = name[3]
                #print("drugtype")
                #print(drugtype)
                tag = name[4]
                # traverse updating offset2, 
                # until B is found, where new line is created
                if debug:
                    print("drugword: ",k,offset1,offset2,word,drugtype)
                if tag == 'B':
                    if len(drugword)>0:
                        if debug:
                            print("appending",drugword)
                        finallist.append(drugword)
                    # restart the drugword list
                    drugword = []
                elif tag == 'I':
                    if debug:
                        print("tag I",k,word)
                    
                if len(drugword)==0:
                    drugword = [k,offset1,offset2,word,drugtype]
                #elif drugword[-1][0] == k and \
                #     drugword[-1][2] <= offset2 and \
                else:
                    
                    if debug:
                        print("-->before drugword",drugword[1],drugword[2],drugword[3])
                    drugword[2]=offset2
                    drugword[3]=drugword[3]+" "+word
                    if debug:
                        print("-->updated drugword",drugword[1],drugword[2],drugword[3])
                
            if len(drugword)>0:
                if debug:
                    print("appending",drugword)
                finallist.append(drugword)
        
        return finallist
    
    
    def parsePredictionOutput(self, filepath=None, debug=False):
        if not filepath is None:
            self.predictionResultFile = filepath

        #for tag in self.predictionResults:
        #    print(tag,sep=" ")
         
        #print(" non O tags:")
        #print(sum([ 1 for tag in self.predictionResults if not tag.startswith('O')])) 
        


        # match tag to word , grouping by sentenceid
        matchdict={}
        if self.algo.__name__ == "SVMtrainer":
            for i in range(len(self.predictionResults)):
                tag = self.predictionResults[i]
                
                if not tag.startswith('O'):
                    predlist = self.parseSingleTagOutput(tag,self.testdata.datadict[i])
                    sentenceid = predlist[-1]
                    if not sentenceid in matchdict.keys():
                        matchdict[sentenceid]=[]
                    matchdict[sentenceid].append(predlist[:-1])   
        else:
            # CRF version
            # list of sentences with list of features for each word
            #pprint.pprint(self.predictionResults[:2])

            # print metrics by sentence
            #print(metrics.flat_f1_score(self.testdata.Y, self.predictionResults,
            #          average='weighted', labels=labels))
        
            # adapt to the format of evaluation
            for i in range(len(self.predictionResults)):
                sentfeatures = self.testdata.datadict[i]
                senttags = self.predictionResults[i]
                for j in range(len(sentfeatures)):

                    tag = senttags[j]
                    wordfeatures = sentfeatures[j]

                    if not tag.startswith('O'):
                        predlist = self.parseSingleTagOutput(tag,wordfeatures)
                        sentenceid = predlist[-1]
                        if not sentenceid in matchdict.keys():
                            matchdict[sentenceid]=[]
                        matchdict[sentenceid].append(predlist[:-1])   

        if debug:
            print("matchdict")
            print(len(matchdict.keys()))

            if len(matchdict.keys())>0:
                firstkey = list(matchdict.keys())[0]
                print(firstkey,matchdict[firstkey])
        
        # generate the lines grouping by BI..I
        
        if debug:
            print()
            print()
            print("--prediction-formatting----------------------------")
        finallist = self.formatPrediction(matchdict,debug)
        
        if debug:
            print("predictions length",len(finallist)) 
            if len(finallist)>0:
                print(finallist[:2])
        self.writePredictionOutput(finallist, debug)
        

    def parseTestSetOutput(self, filepath=None, debug=False):
        """
            Parses the solutions of the biotags for each word of the test set
            The biotag is extracted from the dataset (it's not the prediction, but the actual correct value)
        """
        
        tempPredictionResultsFile = self.predictionResultFile
        if not filepath is None:
            self.predictionResultFile = filepath
         
        matchdict={}
        
        if self.algo.__name__ == "SVMtrainer":
            for i in range(len(self.predictionResults)):
                
                tag = self.testdata.datadict[i]["biotag"]
                if self.modeltype=="NER":
                    tag = self.testdata.datadict[i]["biotag"] +"-" + self.testdata.datadict[i]["drugtype"] 
                
                if not tag.startswith('O'):
                    predlist = self.parseSingleTagOutput(tag,self.testdata.datadict[i])
                    sentenceid = predlist[-1]
                    if not sentenceid in matchdict.keys():
                        matchdict[sentenceid]=[]
                    matchdict[sentenceid].append(predlist[:-1])  
        else:
            # CRF version
            # list of sentences with list of features for each word
            #pprint.pprint(self.predictionResults[:2])

            # print metrics by sentence
            #print(metrics.flat_f1_score(self.testdata.Y, self.predictionResults,
            #          average='weighted', labels=labels))
        
            # adapt to the format of evaluation
            for i in range(len(self.testdata.datadict)):
                sentfeatures = self.testdata.datadict[i]
                senttags = self.testdata.Y[i]
                for j in range(len(sentfeatures)):

                    tag = senttags[j]
                    wordfeatures = sentfeatures[j]

                    if not tag.startswith('O'):
                        predlist = self.parseSingleTagOutput(tag,wordfeatures)
                        sentenceid = predlist[-1]
                        if not sentenceid in matchdict.keys():
                            matchdict[sentenceid]=[]
                        matchdict[sentenceid].append(predlist[:-1])  

            
        # generate the lines grouping by BI..I
        if debug:
            if len(matchdict.keys())>0:
                firstkey = list(matchdict.keys())[0]
                print(matchdict[firstkey])
            print()
            print()
            print("--Solution-formatting----------------------------")
        finallist = self.formatPrediction(matchdict,debug)
        
        if debug:
            print("solutions length",len(finallist)) 
            if len(finallist)>0:
                print(finallist[:2])
        self.writePredictionOutput(finallist, debug)
        
        # restore original value
        self.predictionResultFile = tempPredictionResultsFile




        
    def writePredictionOutput(self, finallist, debug=False):
        
        with open(self.predictionResultFile, 'w') as f:
            for line in finallist:
                print(
                    line[0],
                    str(line[1]) + "-" + str(line[2]),
                    line[3],line[4],
                    sep='|',file=f
                )
                
        if debug:
            for line in finallist:
                print(
                    line[0],
                    str(line[1]) + "-" + str(line[2]),
                    line[3],line[4],
                    sep='|'
                )
            print("...")
        
        
    def manualEvaluation(self, solutionsFile, debug=False):
        
        # open resultsFile
        # open solutions file
        total = 0
        cor = 0
        mis = 0
        par = 0
        accuracy = 0
        
        with open(self.predictionResultFile,'r') as predfile, open(solutionsFile,'r') as solfile:
         
            solf = solfile.readlines()
            predf = predfile.readlines()
            
            # how many lines of the solf are in the prediction file
            for line in solf:
                try:
                    elems = line.split("|")
                    sentid = elems[0]
                    offset = elems[1]
                    offsets = offset.split('-')
                    o1 =int( offsets[0])
                    o2 = int(offsets[1])
                    biotag = elems[2]
                    total+=1

                    #print("looking for ",sentid)
                    found= False
                    sentid_partial_matches = []
                    for pred in predf:
                        elems2 = pred.split("|")
                        sentid2 = elems2[0]
                        offset2 = elems2[1]
                        offsets2 = offset2.split('-')
                        o1b = int(offsets2[0])
                        o2b = int(offsets2[1])
                        biotag2 = elems2[2]
                        #print("      ->",sentid2)
                        if sentid2 == sentid and offset2 == offset and biotag2 == biotag:
                            if debug:
                                print("match!:",line,pred,"")
                            cor+=1
                            found=True
                            break
                        elif sentid2 == sentid and \
                             ((o1<=o1b and o1b<=o2 and o2<=o2b) or \
                              (o1b<=o1 and o1<=o2b and o2b<=o2) or \
                              (o1<=o1b and o2b<=o2) or \
                              (o1b<=o1 and o2<=o2b)):
                            # look at all the sentid2 options

                            if debug:
                                print("partial:",line,pred,o1,o2,o1b,o2b)
                            found=True
                            par+=1
                            break

                    if not found:

                        mis+=1
                except:
                    pass
            
            if total>0:
                accuracy = cor/total
            else:
                accuracy = 0        
        
        print("accuracy",accuracy,"total",total,"cor",cor,"par",par,"mis",mis,"")
        
        # extract the accuracy value
        if not accuracy is None:
            with open(self.predictionResultFile+'_result2','w+') as r:
                # write feature list and 
                r.write("accuracy: "+str(accuracy)+"\n")
                r.write("features: ")
                r.write(self.featureset.printfActiveFeatureFunctions())

            return self.predictionResultFile+'_result2', accuracy
        
        return None, accuracy
    
               
    def autoEvaluation(self, testFolder):
        
        scores = {}

        process = subprocess.Popen([
            'java',
            '-jar', 
            '../evaluation/evaluateNER.jar', 
            testFolder, 
            self.predictionResultFile
        ], stdout=subprocess.PIPE)
        out, err = process.communicate()
        with open(self.predictionResultFile+'_eval','w') as f:
            f.write(str(out))
        #print(out
        
        # extract the accuracy value
        accuracy, prec, rec, f1 = (0, 0,0,0)
        try:
            with open(self.predictionResultFile+'_eval','r') as f:
                scores = self.parseEvaluation(out.decode("utf-8") )
                print(
                    "accuracy", float(scores["exact"]["cor"])/float(scores["exact"]["total"]),
                    "precision",scores["exact"]["prec"],
                    "recall",scores["exact"]["recall"],
                    "F1",scores["exact"]["F1"])
        except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60) 
                print("some error occurred while parsin scores files")

            

        return scores
    
    def parseEvaluationLine(self, line):
        newdict={}
        try:
            scores = line
            scores = scores.split()
            if len(scores)> 4:
                newdict["cor"] = scores[0]
                newdict["inc"] = scores[1]
                newdict["par"] = scores[2]
                newdict["mis"] = scores[3]
                newdict["spu"]= scores[4]
                newdict["total"] = scores[5]
                newdict["prec"] = scores[6]
                newdict["recall"] = scores[7]
                newdict["F1"] = scores[8]
            else:
                newdict["P"] = scores[0]
                newdict["R"] = scores[1]
                newdict["F1"] = scores[2]
        except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60) 
                print(line)
                print(newdict)

        return newdict
    
    def parseEvaluation(self, resultstring):
        """
            construct a dict will the results of the scores file
        """

        result = {}
        
        resultType = 'Exact matching'
        indexResult = 0
        if self.modeltype == "NER":
            resultType = 'type matching'
            indexResult = 0
        
        resultstring = resultstring.split("\n")
        for i in range(len(resultstring)): 
            line = resultstring[i]
            if line.startswith("Strict matching (") and i < len(resultstring) - 2:
                result["strict"]=self.parseEvaluationLine(resultstring[i+2])
    
            elif line.startswith("Partial") and i < len(resultstring) - 2:
                result["partial"]=self.parseEvaluationLine(resultstring[i+2])
            elif line.startswith("type matching") and i < len(resultstring) - 2:
                result["type"]=self.parseEvaluationLine(resultstring[i+2])
            elif line.startswith("Exact matching on drug_n") and i < len(resultstring) - 2:
                result["drug_n"]=self.parseEvaluationLine(resultstring[i+2])
            elif line.startswith("Exact matching on drug") and i < len(resultstring) - 2:
                result["drug"]=self.parseEvaluationLine(resultstring[i+2])
            elif line.startswith("Exact matching on brand") and i < len(resultstring) - 2:
                result["brand"]=self.parseEvaluationLine(resultstring[i+2])
            elif line.startswith("Exact matching on group") and i < len(resultstring) - 2:
                result["group"]=self.parseEvaluationLine(resultstring[i+2])
            
            elif line.startswith("MACRO-AVERAGE") and i < len(resultstring) - 2:
                result["macro"]=self.parseEvaluationLine(resultstring[i+2])

            elif line.startswith("Exact ") and i < len(resultstring) - 2:
                result["exact"]=self.parseEvaluationLine(resultstring[i+2])
            else:
                print("unrecognized line: ",line)
            
        
        return result
    
    
    # static method to test models in batch
    def initConf(filepath=None):
        """
            every plugin uses its own config file
        """
        if not filepath:
            conf = yaml.load(open(os.getcwd()+'/models.conf','r'))
            conf["filepath"] = filepath
            return conf
        else:
            print(filepath)
            conf = yaml.load(open(filepath,'r'))
            conf["filepath"] = filepath
            return conf

    def writeConfBack(conf, filepath=None):
        if not filepath:
            filepath = conf["filepath"]
        with open(filepath,'w') as f:
            f.write(yaml.dump(conf))


    def writeConf(conf,modelname,key, msg,session="default", todisk=False ):

        if modelname=="":
            if todisk:
                # load from disk
                diskconf = NERmodel.initConf(conf["filepath"])
                #  overwrite indicated model with all
                diskconf[key] = msg
                # write to disk
                NERmodel.writeConfBack(diskconf)
            else:
                conf[key]=msg


        for modeln, params in conf["models"].items():
            try:
                if modelname == modeln:
                    if "results" not in params.keys():
                        params["results"]={}
                    if params["results"] is None:
                        params["results"]={}
                    if session not in params["results"].keys():
                        params["results"][session]={}
                    params["results"][session][key]=msg
                    if todisk:
                        # load from disk
                        diskconf = NERmodel.initConf(conf["filepath"])

                        # traverse models
                        for modeld, paramsd in diskconf["models"].items():

                            if modeld == modelname:
                                #  overwrite indicated model with all
                                diskconf["models"][modeld] = params
                                # write to disk
                                NERmodel.writeConfBack(diskconf)

                                break
                    break
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60) 

    def trainTestModel(mf, training_set_name, session, mylock=mylock, cv=True):

        print("NERmodel.trainTestModel cv=",cv)

        fset=mf[0]
        modelname=mf[1]
        modeltype=mf[2]
        jobconf = mf[4]

        start = time.time()
        NERmodel.writeConf(
            jobconf,modelname,"start", start, session=session )

        NERmodel.writeConf(
            jobconf,modelname,"status", "doing", session=session )
        
        algorithm="lsvm"
        if "algorithm" in mf[3].keys():
            algorithm = mf[3]["algorithm"]
        print(mf[3].keys())
        print(algorithm)

        limitTrain=None
        if "limitTraining" in mf[3].keys():
            limitTrain = mf[3]["limitTraining"]
        limitTest=None
        if "limitTest" in mf[3].keys():
            limitTest = mf[3]["limitTest"]
            
        try:

            # fsetObject = joblib.load(fset) 
            # model = NERmodel(featureset=fsetObject, name=training_set_name+modelname, modeltype=modeltype)
            
            with NERmodel(
                featureset=fset, 
                name=training_set_name+modelname, 
                modeltype=modeltype, 
                algorithm=algorithm,
                ) as model:

                # model = NERmodel(
                #     featureset=fset, 
                #     name=training_set_name+modelname, 
                #     modeltype=modeltype, 
                #     algorithm=algorithm)
                
                model.trainFeatureExtraction(
                    jobconf["trainingFolder"], 
                    limit=limitTrain)

                # model.saveTrainingFeatures(jobconf["savingFolder"] +"/" +jobconf["session"]+"-"+model.name+".json")
                
                model.newModelPipeline(ertype=model.modeltype, algotype=model.algotype, cv=cv)

                if cv:
                    print(model.scores)
                    NERmodel.writeConf(
                        jobconf,modelname,"cv", model.scores, session=session)
                

                modelfile= model.saveModelPipeline()
                NERmodel.writeConf(
                    jobconf,modelname,"modelfile", modelfile, session=session)
                
                model.testFeatureExtraction(
                    jobconf["testFolder"], 
                    limit=limitTest)

                model.predict()

                # manual accuracy computation
                solutionsFile = jobconf["resultsFolder"]+'/task9.1_'+training_set_name+model.name+'_sol.txt'
                model.parseTestSetOutput(solutionsFile,debug=False)
                
                predictionsFile = jobconf["resultsFolder"]+'/task9.1_'+training_set_name+model.name+'.txt'
                model.parsePredictionOutput(predictionsFile,debug=False)
                
                #results_files[i] = jobconf["savingFolder"]+ "/" + jobconf["session"]+"-"+model.name+'-output.csv'

                print("model "+model.name+":")
                scores = model.autoEvaluation(jobconf["testFolder"])
                _dummy, accuracy2 = model.manualEvaluation(
                    solutionsFile,debug=False)
                
                NERmodel.writeConf(
                    jobconf,modelname,"scores", scores, session=session)
                NERmodel.writeConf(
                    jobconf,modelname,
                    "accuracy_manual", 
                    accuracy2, 
                    session=session)
                NERmodel.writeConf(
                    jobconf,
                    modelname,
                    "algorithm", 
                    model.algo.__name__, session=session)
                

                NERmodel.writeConf(
                        jobconf,
                        modelname,
                        "resultsFile",
                        model.predictionResultFile+'_result', session=session )
                
                NERmodel.writeConf(
                        jobconf,
                        modelname,
                        "scoresFile",
                        model.predictionResultFile[:-4]+'_scores.log', session=session )
                
                NERmodel.writeConf(jobconf,modelname,"status", "done", session=session)

                end = time.time()
                NERmodel.writeConf(jobconf,modelname,"end", end, session=session )
                total_time = end - start
                NERmodel.writeConf(jobconf,modelname,"total_time", total_time, session=session)
                print("job time:",total_time)
                print()

                with mylock:
                    NERmodel.writeConf(jobconf,modelname,"status", "done", session=session, todisk=True )
                

        except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                print("model "+modelname+" has failed... ")
                
                NERmodel.writeConf(jobconf,modelname,"status", "error", session=session )

        #trying to free memory
        gc.collect()
        
        #return jobconf,modelname


    def TestModel( mf, training_set_name, session, modelfile, mylock=mylock):


        print("from inside TestModel")

        fset=mf[0]
        modelname=mf[1]
        modeltype=mf[2]
        jobconf = mf[4]

        start = time.time()
        NERmodel.writeConf(jobconf,modelname,"start", start, session=session )

        NERmodel.writeConf(jobconf,modelname,"status", "doing", session=session )

        algorithm="lsvm"
        if "algorithm" in mf[3].keys():
            algorithm = mf[3]["algorithm"]

        limitTrain=None
        if "limitTraining" in mf[3].keys():
            limitTrain = mf[3]["limitTraining"]
        limitTest=None
        if "limitTest" in mf[3].keys():
            limitTest = mf[3]["limitTest"]
            
        try:

            
            model = NERmodel(
                featureset=fset, 
                name=training_set_name+modelname, 
                modeltype=modeltype, 
                algorithm=algorithm,
                )

            model.loadModelPipeline(filepath=modelfile)
            NERmodel.writeConf(jobconf,modelname,"modelfile", modelfile, session=session)
            

            model.testFeatureExtraction(
                jobconf["testFolder"], 
                limit=limitTest)

            print("testdata",len(model.testdata.X))

            model.predict()


            # manual accuracy computation
            solutionsFile = jobconf["resultsFolder"]+'/task9.1_'+training_set_name+model.name+'_sol.txt'
            model.parseTestSetOutput(solutionsFile,debug=False)
            
            predictionsFile = jobconf["resultsFolder"]+'/task9.1_'+training_set_name+model.name+'.txt'
            model.parsePredictionOutput(predictionsFile,debug=False)
            
            #results_files[i] = jobconf["savingFolder"]+ "/" + jobconf["session"]+"-"+model.name+'-output.csv'

            print("model "+model.name+":")
            scores = model.autoEvaluation(jobconf["testFolder"])
            _dummy, accuracy2 = model.manualEvaluation(solutionsFile,debug=False)
            
            NERmodel.writeConf(jobconf,modelname,"scores", scores, session=session)
            NERmodel.writeConf(jobconf,modelname,"accuracy_manual", accuracy2, session=session)
            NERmodel.writeConf(jobconf,modelname,"algorithm", model.algo.__name__, session=session)
            

            NERmodel.writeConf(
                    jobconf,
                    modelname,
                    "resultsFile",
                    model.predictionResultFile+'_result', session=session )
            
            NERmodel.writeConf(
                    jobconf,
                    modelname,
                    "scoresFile",
                    model.predictionResultFile[:-4]+'_scores.log', session=session )
            
            NERmodel.writeConf(jobconf,modelname,"status", "done", session=session)

            end = time.time()
            NERmodel.writeConf(jobconf,modelname,"end", end, session=session )
            total_time = end - start
            NERmodel.writeConf(jobconf,modelname,"total_time", total_time, session=session)
            print("job time:",total_time)
            print()


            with mylock:
                NERmodel.writeConf(jobconf,modelname,"status", "done", session=session, todisk=True )

        except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                print("model "+modelname+" has failed... ")
                
                NERmodel.writeConf(jobconf,modelname,"status", "error", session=session )

        #trying to free memory
        model.free()
        model=None
        gc.collect()

        #return jobconf,modelname



    def parallelbatchTraining(configfile="models.conf",status=True):

        conf = NERmodel.initConf(configfile)
        total_start = time.time()

        # name the batch training
        datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
        session = conf["session"] + datesuffix
        training_set_name =  conf["session"]
        print(training_set_name)

        # load the different MyFeature instances
        fs = []
        for modelname,params in conf["models"].items():
            try:
                alreadydone = False
                #print("evaluating",modelname)
                if "results" in params.keys():
                    for rname, result in params["results"].items():

                        if status and result["status"] == "done":
                            print("skipping model",modelname)
                            alreadydone= True
                            break
                    
                if not alreadydone:
                    #print("alreadydone",alreadydone)
                    mf = MyFeatures(params=params)

                    #datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
                    #featuresetFile = conf["savingFolder"] +'featureset_'+ datesuffix +'.pkl'
                    #joblib.dump(mf, featuresetFile)

                    fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                NERmodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])       

        # Parallelization
        numcores=7
        if "numcores" in conf.keys() and conf["numcores"] is not None:
            print("setting numcores to ",conf["numcores"])
            numcores = conf["numcores"]

        

        

        # using Process
        # mylock = Lock()
        # tot = len(fs)
        # ktot = int(len(fs)/numcores)
        # rest = tot - ktot*numcores
        # for k in range(ktot):
        #     procs = [Process(target=NERmodel.trainTestModel,args=(mf,training_set_name, session, mylock)) for mf in fs[k*numcores:k*numcores+numcores]]
        #     for p in procs: p.start()
        #     for p in procs: p.join()
        # procs = [Process(target=NERmodel.trainTestModel,args=(mf,training_set_name, session, mylock)) for mf in fs[ktot*numcores:ktot*numcores+rest]]
        # for p in procs: p.start()
        # for p in procs: p.join()
        

        # using pool.apply
        pool = Pool(processes=numcores, maxtasksperchild=1)
        procs = [pool.apply_async(NERmodel.trainTestModel,args=(mf,training_set_name, session)) for mf in fs]
        #results = [p.get() for p in procs]        
        pool.close()
        
        pool.join()

        
        # sequential version
        #jobresults = [NERmodel.trainTestModel(mf,training_set_name, session) for mf in fs]



        total_end = time.time()
        total_time =total_end - total_start
        print("total batch time", total_time)
        NERmodel.writeConf(conf,"","total_time", total_time ,todisk=True)


    def parallelbatchCVTraining(configfile="models.conf",status=True):

        conf = NERmodel.initConf(configfile)
        total_start = time.time()

        # name the batch training
        datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
        session = conf["session"] + datesuffix
        training_set_name =  conf["session"]
        print(training_set_name)

        # load the different MyFeature instances
        fs = []
        for modelname,params in conf["models"].items():
            try:
                alreadydone = False
                #print("evaluating",modelname)
                if "results" in params.keys():
                    for rname, result in params["results"].items():

                        if status and result["status"] == "done":
                            print("skipping model",modelname)
                            alreadydone= True
                            break
                    
                if not alreadydone:
                    #print("alreadydone",alreadydone)
                    mf = MyFeatures(params=params)

                    #datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
                    #featuresetFile = conf["savingFolder"] +'featureset_'+ datesuffix +'.pkl'
                    #joblib.dump(mf, featuresetFile)

                    fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                NERmodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])       

        # Parallelization
        numcores=7
        if "numcores" in conf.keys() and conf["numcores"] is not None:
            print("setting numcores to ",conf["numcores"])
            numcores = conf["numcores"]


        # using pool.apply
        pool = Pool(processes=numcores, maxtasksperchild=1)
        procs = [pool.apply_async(NERmodel.trainTestModel,args=(mf,training_set_name, session, True)) for mf in fs]
        #results = [p.get() for p in procs]        
        pool.close()
        
        pool.join()

        
        # sequential version
        #jobresults = [NERmodel.trainTestModel(mf,training_set_name, session) for mf in fs]



        total_end = time.time()
        total_time =total_end - total_start
        print("total batch time", total_time)
        NERmodel.writeConf(conf,"","total_time", total_time ,todisk=True)
       
     
    def parallelbatchTesting(configfile="models.conf",status=True,modelfile=None):

        conf = NERmodel.initConf(configfile)
        total_start = time.time()

        # name the batch training
        datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
        session = conf["session"] + datesuffix
        training_set_name =  conf["session"]
        print(training_set_name)

        # load the different MyFeature instances
        fs = []
        for modelname,params in conf["models"].items():
            try:
                alreadydone = False
                print("evaluating",modelname)
                if "results" in params.keys():
                    for rname, result in params["results"].items():

                        if status and result["status"] == "done":
                            print("skipping model",modelname)
                            alreadydone= True
                            break
                    
                if not alreadydone:
                    print("alreadydone",alreadydone)
                    mf = MyFeatures(params=params)

                    #datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
                    #featuresetFile = conf["savingFolder"] +'featureset_'+ datesuffix +'.pkl'
                    #joblib.dump(mf, featuresetFile)



                    fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))


                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                NERmodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])   


        # Parallelization
        numcores=4
        if numcores in conf.keys() and conf["numcores"] is not None:
            numcores = conf["numcores"]
        # mylock = Lock()
        # procs = [Process(target=NERmodel.TestModel,args=(mf,training_set_name, session,modelfile, mylock)) for mf in fs]

        pool = Pool(processes=numcores, maxtasksperchild=1)
        procs = [pool.apply_async(NERmodel.TestModel,args=(mf,training_set_name, session,modelfile)) for mf in fs]
        #results = [p.get() for p in procs]        
        pool.close()
        pool.join()


        

        print("after pool")
 

        total_end = time.time()
        total_time =total_end - total_start
        print("total batch time", total_time)
        NERmodel.writeConf(conf,"","total_time", total_time ,todisk=True)
       
     

    def batchTraining(configfile="models.conf",status=True):

        conf = NERmodel.initConf(configfile)
        total_start = time.time()

        # name the batch training
        datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
        session = conf["session"] + datesuffix
        training_set_name =  conf["session"]
        print(training_set_name)

        # load the different MyFeature instances
        fs = []
        for modelname,params in conf["models"].items():
            try:
                if status and params["status"] == "done":
                    continue
                    
                mf = MyFeatures(params=params)

                #datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
                #featuresetFile = conf["savingFolder"] +'featureset_'+ datesuffix +'.pkl'
                #joblib.dump(mf, featuresetFile)

                fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                NERmodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])   

        # sequential version
        jobresults = [NERmodel.trainTestModel(mf,training_set_name, session) for mf in fs]

        
        for jobr in jobresults:
            jobconf = jobr[0]
            modelname = jobr[1]
            NERmodel.writeConf(jobconf,modelname,"status", "done", session=session, todisk=True )

        total_end = time.time()
        total_time =total_end - total_start
        print("total batch time", total_time)
        NERmodel.writeConf(conf,"","total_time", total_time ,todisk=True)



    def batchTestManualAccuracy(configfile="models.conf", modelfilepath='../data/models/svm-pipeline_20180518151600.pkl'):

        conf = NERmodel.initConf(configfile)

        # name the batch training
        datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
        session = conf["session"] + datesuffix
        training_set_name =  conf["session"]
        print(training_set_name)

        # load the different MyFeature instances
        fs = []
        #for i in range(len(conf["models"])):
        for modelname,params in conf["models"].items():
            try:
                mf = MyFeatures(params=params)
                fs.append((mf,modelname,params["modeltype"],params))
                NERmodel.writeConf(conf,modelname,"status", "doing", session=session )
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                NERmodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])   

        # initialize
        models=[]
        models_pickles =[None]*len(fs)
        results_files = [None]*len(fs)
        evaluation_files = [None]*len(fs)
        for i in range(len(fs)):
            mf = fs[i]
            try:
                fset=mf[0]
                modelname=mf[1]
                modeltype=mf[2]
                limitTrain=None
                if "limitTraining" in mf[3].keys():
                    limitTrain = mf[3]["limitTraining"]
                limitTest=None
                if "limitTest" in mf[3].keys():
                    limitTest = mf[3]["limitTest"]
                    
                model = NERmodel(featureset=fset, name=modelname, modeltype=modeltype)
                model.setName(training_set_name)
                models.append(model)
                
                if False:
                    model.trainFeatureExtraction(
                        conf["trainingFolder"], 
                        limit=limitTrain)
                    model.saveTrainingFeatures(conf["savingFolder"] +"/" +conf["session"]+"-"+model.name+".json")
                    model.newModelPipeline(ertype=model.modeltype)
                    models_pickles[i] = model.saveModelPipeline()
                    modelfile=models_pickles[i]
                    NERmodel.writeConf(conf,modelname,"modelfile", modelfile, session=session)
                
                model.loadModelPipeline(modelfilepath)
                model.testFeatureExtraction(
                    conf["testFolder"], 
                    limit=limitTest)
                
                
                model.predict()

                print("model "+model.name+":")
                
                # task accuracy and other evaluation measures
                predictionsFile = conf["resultsFolder"]+'/task9.1_'+training_set_name+model.name+'_'+str(i)+'.txt'
                model.parsePredictionOutput(predictionsFile,debug=False)
                #results_files[i] = conf["savingFolder"]+ "/" + conf["session"]+"-"+model.name+'-output.csv'
                evaluation_files[i], accuracy, precision, recall, f1 = model.autoEvaluation(conf["testFolder"])
                NERmodel.writeConf(conf,modelname,"accuracy", accuracy, session=session)
                NERmodel.writeConf(conf,modelname,"precision", precision, session=session)
                NERmodel.writeConf(conf,modelname,"recall", recall, session=session)
                NERmodel.writeConf(conf,modelname,"f1", f1, session=session)
                
                # manual accuracy computation
                solutionsFile = conf["resultsFolder"]+'/task9.1_'+training_set_name+model.name+'_'+str(i)+'_sol.txt'
                model.parseTestSetOutput(solutionsFile,debug=False)
               
                evaluation_files[i], accuracy2 = model.manualEvaluation(solutionsFile,debug=False)
                NERmodel.writeConf(conf,modelname,"accuracy_manual", accuracy2, session=session)
                
                
                # computing accuracy from goldNer
                goldNERFile = conf["resultsFolder"]+'/../../src/goldNER.txt'
                evaluation_files[i], accuracy3 = model.manualEvaluation(goldNERFile,debug=False)
                NERmodel.writeConf(conf,modelname,"accuracy_goldNER", accuracy3, session=session)
                
                print()
                print()
                NERmodel.writeConf(
                        conf,
                        modelname,
                        "resultsFile",
                        model.predictionResultFile+'_result', session=session )
                
                NERmodel.writeConf(
                        conf,
                        modelname,
                        "scoresFile",
                        model.predictionResultFile[:-4]+'_scores.log', session=session )
                
                NERmodel.writeConf(conf,modelname,"status", "done", session=session)


            except Exception:
                print("Exception in user code:",i)
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                print("model "+str(i)+" has failed... ")
                models_pickles[i] = None
                NERmodel.writeConf(conf,modelname,"status", "error", session=session )


