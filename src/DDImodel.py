from FeatureExtractionDDI import FeatureExtractionDDI
from CustomFeaturesDDI import MyFeaturesDDI
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



mylock = Lock()

class DDImodel():
    
    def __init__(self, algorithm="lsvm", featureset=None,name="", modeltype="ER"):
        
        self.name = name
        
        self.modeltype = modeltype

        
        print("init DDI")

        self.featureset = MyFeaturesDDI()
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

        self.scores = []

        print("after all init")
        
    def __enter__(self):
        # nothing to set up besides __init__
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __del__(self):
        print("freeing DDImodel instance...")
        del self.traindata
        self.traindata  = None
        del self.testdata
        self.testdata = None
        self.featureset = None
        self.pipeline = None
        
    def setName(self, name):
        self.name = name
        
    def trainFeatureExtraction(self, datafile, limit=None,
        ):

        self.traindata = FeatureExtractionDDI(
            datafile, 
            self.featureset, 
            algoformat=self.algotype,
            targetformat=self.modeltype
            )
        self.traindata.load()
        self.traindata.extractFeaturesNG(limit)

        
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
            self.traindata = FeatureExtractionDDI('', self.featureset)
            
        with open(self.traindatafile,'r') as f:
            self.traindata.datadict = json.load(f)
            self.traindata.transformToXY(self.traindata.datadict)

    def loadDDIFeatures(self,
        filepath, 
        limit=None, 
        topcount=None, 
        topfeatures=[],
        sentencefeatures=[],
        wordfeatures=[],
        windowfeatures=[],
        window=5):
        if filepath is not None:
            self.traindata.loadFilterDDIFeatures(
                        filepath=filepath,
                        limit=limit,
                        topcount=topcount,
                        topfeatures=topfeatures,
                        sentencefeatures=sentencefeatures,
                        wordfeatures=wordfeatures,
                        windowfeatures=windowfeatures,
                        window=window)

      
    def newModelPipeline(self,ertype="DDI",algotype="lsvm",cv=False):
        
        if cv:
            self.pipeline, self.scores = self.algo.newPipeline(
                self.traindata.X,
                self.traindata.Y, 
                ertype=ertype,
                algotype=algotype,
                    cv=cv)
        else:
            self.pipeline = self.algo.newPipeline(
                self.traindata.X,
                self.traindata.Y, 
                ertype=ertype,
                algotype=algotype)
        
    def setPipelinefile(self,filepath):
        self.pipelinefile=filepath
        
    def saveModelPipeline(self):
        self.pipelinefile = self.algo.savePipeline(self.pipeline)
        return self.pipelinefile
    
    def loadModelPipeline(self, filepath=None):
        if not filepath is None:
            self.pipelinefile = filepath
        self.pipeline = self.algo.loadPipeline(self.pipelinefile)
        
    
    def testFeatureExtraction(self, folderpath, 
        limit=None):
        self.testdata = FeatureExtractionDDI(
            folderpath, 
            self.featureset,
            algoformat=self.algotype,
            targetformat=self.modeltype
            )
        self.testdata.load()
        self.testdata.extractFeaturesNG(limit)
        
        
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
            self.testdata = FeatureExtractionDDI('', self.featureset)
            
        with open(self.testdatafile,'r') as f:
            
            self.testdata.datadict = json.load(f)
            self.testdata.transformToXY(self.traindata.datadict)

    def loadDDITestFeatures(self,
        filepath, 
        limit=None, 
        topcount=None, 
        topfeatures=[],
        sentencefeatures=[],
        wordfeatures=[],
        windowfeatures=[],
        window=5):
        if filepath is not None:
            self.testdata.loadFilterDDIFeatures(
                        filepath=filepath,
                        limit=limit,
                        topcount=topcount,
                        topfeatures=topfeatures,
                        sentencefeatures=sentencefeatures,
                        wordfeatures=wordfeatures,
                        windowfeatures=windowfeatures,
                        window=window)
    
    def predict(self,mode=None):
        """
            mode="ER" sets drugtype as a feature
            mode="NER" sets drugtype as part of the target and hence not in the features
        """
        #print(self.modeltype)
        if not mode and self.modeltype:
            mode = self.modeltype
        elif not mode:
            mode = "ER"
            
        #print("the model is " + mode)
        self.predictionResults = self.algo.predict(
            self.pipeline, 
            self.testdata.X,
            ertype=mode)
        
   
        
        
    def setPredictionResultsFile(self, filepath):
        self.predictionResultFile = filepath
        
        
    def parseSingleTagOutput(self, tag, pairdict):
        """
            Goal: idSentence|IdDrug1|IdDrug2| prediction|type
        """
        
        sentenceid = pairdict['sentenceid']
        e1id = pairdict['e1id']
        e2id = pairdict['e2id']
        #ddi = pairdict['ddi']
        dditype = pairdict['type']


        tag = tag
        tagtype = "null"
        if self.modeltype=="DDI":
            # print("parseSingleTagOutput",tag,tag.split('-'))
            tag2 = tag.split('-')
            tag = tag2[0]
            try:
                dditype = tag2[1]
            except:
                dditype = "null"
        else:
            # this should be extracted from the prediction result
            #drugtype = wordtuple['drugtype']
            dditype= "null"
            
        if str(tag)=="true":
            tag = 1
        else:
            tag = 0

        return [sentenceid,e1id,e2id,tag,dditype]
        

    
    
    def parsePredictionOutput(self, filepath=None, debug=False):
        if not filepath is None:
            self.predictionResultFile = filepath

        #for tag in self.predictionResults:
        #    print(tag,sep=" ")
         
        #print(" non O tags:")
        #print(sum([ 1 for tag in self.predictionResults if not tag.startswith('O')])) 
        


        # match tag to word , grouping by sentenceid
        finallist=[]
        if self.algo.__name__ == "SVMtrainer":
            for i in range(len(self.predictionResults)):
                tag = self.predictionResults[i]
                
                if not tag.startswith('False'):
                    predlist = self.parseSingleTagOutput(tag,self.testdata.datadict[i])
                    sentenceid = predlist[0]
                    finallist.append(predlist)   
        else:
            
            # CRF version
            # list of sentences with list of features for each word
            #pprint.pprint(self.predictionResults[:2])

            # print metrics by sentence
            #print(metrics.flat_f1_score(self.testdata.Y, self.predictionResults,
            #          average='weighted', labels=labels))
        
            # adapt to the format of evaluation
            # for i in range(len(self.predictionResults)):
            #     sentfeatures = self.testdata.datadict[i]
            #     senttags = self.predictionResults[i]
            #     for j in range(len(sentfeatures)):

            #         tag = senttags[j]
            #         wordfeatures = sentfeatures[j]

            #         if not tag.startswith('O'):
            #             predlist = self.parseSingleTagOutput(tag,wordfeatures)
            #             sentenceid = predlist[-1]
            #             if not sentenceid in matchdict.keys():
            #                 matchdict[sentenceid]=[]
            #             matchdict[sentenceid].append(predlist[:-1])   
            pass

        if debug:
            print("finallist")
            print(len(finallist))

            
                
        if debug:
            print("predictions length",len(finallist)) 
            if len(finallist)>0:
                print(finallist[:2])
        self.writePredictionOutput(finallist, debug)
        

    def parseTestSetOutput(self, filepath=None, debug=False):
        """
            Parses the solutions of the biotags for each word of the test set
            The biotag is extracted from the dataset (it's not the prediction
            , but the actual correct value)
        """
        
        tempPredictionResultsFile = self.predictionResultFile
        if not filepath is None:
            self.predictionResultFile = filepath
         
        finallist=[]
        
        if self.algo.__name__ == "SVMtrainer":
            for i in range(len(self.predictionResults)):
                
                tag = str(self.testdata.datadict[i]["ddi"])
                if self.modeltype=="DDI":
                    tag = str(self.testdata.datadict[i]["ddi"]) +"-" + self.testdata.datadict[i]["type"] 
                
                if not tag.startswith('False'):
                    predlist = self.parseSingleTagOutput(tag,self.testdata.datadict[i])
                    sentenceid = predlist[0]
                    finallist.append(predlist)  
        else:
            # CRF version
            # list of sentences with list of features for each word
            #pprint.pprint(self.predictionResults[:2])

            # print metrics by sentence
            #print(metrics.flat_f1_score(self.testdata.Y, self.predictionResults,
            #          average='weighted', labels=labels))
        
            # adapt to the format of evaluation
            # for i in range(len(self.testdata.datadict)):
            #     sentfeatures = self.testdata.datadict[i]
            #     senttags = self.testdata.Y[i]
            #     for j in range(len(sentfeatures)):

            #         tag = senttags[j]
            #         wordfeatures = sentfeatures[j]

            #         if not tag.startswith('O'):
            #             predlist = self.parseSingleTagOutput(tag,wordfeatures)
            #             sentenceid = predlist[-1]
            #             if not sentenceid in matchdict.keys():
            #                 matchdict[sentenceid]=[]
            #             matchdict[sentenceid].append(predlist[:-1])  
            pass
            
        
       
        if debug:
            print("solutions length",len(finallist)) 
            if len(finallist)>0:
                print(finallist[:2])
        self.writePredictionOutput(finallist, debug)
        
        # restore original value
        self.predictionResultFile = tempPredictionResultsFile




        
    def writePredictionOutput(self, finallist, debug=False):
        """
            
            from this [sentenceid,e1id,e2id,tag,dditype]

            to 

            idSentence|IdDrug1|IdDrug2| prediction|type

        """
        
        with open(self.predictionResultFile, 'w') as f:
            for line in finallist:
                print(
                    line[0],
                    line[1],line[2],
                    str(line[3]),
                    line[4],
                    sep='|',file=f
                )
                
        if debug:
            for line in finallist:
                print(
                    line[0],
                    line[1],line[2],
                    str(line[3]),
                    line[4],
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
                    e1 = elems[1]
                    e2 = elems[2]
                    ddi = elems[3]
                    dditype = elems[4]
                    total+=1

                    #print("looking for ",sentid)
                    found= False
                    sentid_partial_matches = []
                    for pred in predf:
                        elems2 = pred.split("|")
                        sentid2 = elems2[0]
                        e12 = elems2[1]
                        e22 = elems2[2]
                        ddi2 = elems2[3]
                        dditype2 = elems2[4]

                        if sentid2 == sentid and e12 == e1 and e22 == e2 and ddi2==ddi and dditype2 == dditype:
                            if debug:
                                print("exact!:",line,pred,"")
                            cor+=1
                            found=True
                            break
                        elif sentid2 == sentid and e12 == e1 and e22 == e2 and ddi2==ddi:
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
        
        return None, accuracy
    
               
    def autoEvaluation(self, testFolder):
        """
            java -jar evaluateDDI.jar <goldDir> <submissionFile>
        """
        scores = {}

        process = subprocess.Popen([
            'java',
            '-jar', 
            '../evaluation/evaluateDDI.jar', 
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
        except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60) 
                print("some error occurred while parsin scores files")

        try:
                print(
                    "accuracy", float(scores["partial"]["tp"])/float(scores["partial"]["total"]),
                    "exact accuracy", float(scores["exact"]["tp"])/float(scores["exact"]["total"]),
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
                newdict["tp"] = scores[0]
                newdict["fp"] = scores[1]
                newdict["fn"] = scores[2]
                newdict["total"] = scores[3]
                newdict["prec"] = scores[4]
                newdict["recall"] = scores[5]
                newdict["F1"] = scores[6]
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
        print("parseEvaluation",resultstring,"")

        result = {}
        
        resultType = 'partial'  # for BDDI
        indexResult = 0
        if self.modeltype == "DDI":
            resultType = 'exact'
            indexResult = 0
        
        resultstring = resultstring.split("\n")
        for i in range(len(resultstring)): 
            line = resultstring[i]
            if line.startswith("Partial Evaluation") and i < len(resultstring) - 2:
                result["partial"]=self.parseEvaluationLine(resultstring[i+2])
    
            elif line.startswith("Detection and") and i < len(resultstring) - 2:
                result["exact"]=self.parseEvaluationLine(resultstring[i+2])
            elif line.startswith("Scores for ddi with type mechanism") and i < len(resultstring) - 2:
                result["mechanism"]=self.parseEvaluationLine(resultstring[i+2])
            elif line.startswith("Scores for ddi with type effect") and i < len(resultstring) - 2:
                result["effect"]=self.parseEvaluationLine(resultstring[i+2])
            elif line.startswith("Scores for ddi with type advise") and i < len(resultstring) - 2:
                result["advise"]=self.parseEvaluationLine(resultstring[i+2])
            
            elif line.startswith("Scores for ddi with type int") and i < len(resultstring) - 2:
                result["int"]=self.parseEvaluationLine(resultstring[i+2])
            

            elif line.startswith("MACRO-AVERAGE") and i < len(resultstring) - 2:
                result["macro"]=self.parseEvaluationLine(resultstring[i+2])

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
                diskconf = DDImodel.initConf(conf["filepath"])
                #  overwrite indicated model with all
                diskconf[key] = msg
                # write to disk
                DDImodel.writeConfBack(diskconf)
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
                        diskconf = DDImodel.initConf(conf["filepath"])

                        # traverse models
                        for modeld, paramsd in diskconf["models"].items():

                            if modeld == modelname:
                                #  overwrite indicated model with all
                                diskconf["models"][modeld] = params
                                # write to disk
                                DDImodel.writeConfBack(diskconf)

                                break
                    break
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60) 

    def trainTestModel(mf, training_set_name, session, mylock=mylock, cv=True):

        print("DDImodel.trainTestModel cv=",cv)
        fset=mf[0]
        modelname=mf[1]
        modeltype=mf[2]
        jobconf = mf[4]

        start = time.time()
        DDImodel.writeConf(
            jobconf,modelname,"start", start, session=session )

        DDImodel.writeConf(
            jobconf,modelname,"status", "doing", session=session )
        
        algorithm="svm"
        if "algorithm" in mf[3].keys():
            algorithm = mf[3]["algorithm"]

        limitTrain=None
        if "limitTraining" in mf[3].keys():
            limitTrain = mf[3]["limitTraining"]
        limitTest=None
        if "limitTest" in mf[3].keys():
            limitTest = mf[3]["limitTest"]
            
        try:

            # fsetObject = joblib.load(fset) 
            # model = DDImodel(featureset=fsetObject, name=training_set_name+modelname, modeltype=modeltype)
            
            with DDImodel(
                featureset=fset, 
                name=training_set_name+modelname, 
                modeltype=modeltype, 
                algorithm=algorithm) as model:

                # model = DDImodel(
                #     featureset=fset, 
                #     name=training_set_name+modelname, 
                #     modeltype=modeltype, 
                #     algorithm=algorithm)
                
                model.trainFeatureExtraction(
                    jobconf["trainingFolder"], 
                    limit=limitTrain)

                

                model.saveTrainingFeatures(jobconf["savingFolder"] +"/" +jobconf["session"]+"-"+model.name+".json")
                

                model.newModelPipeline(ertype=model.modeltype, algotype=model.algotype, cv=cv)

                if cv:
                    print(model.scores)
                    NERmodel.writeConf(
                        jobconf,modelname,"cv", model.scores, session=session)

                modelfile= model.saveModelPipeline()
                DDImodel.writeConf(
                    jobconf,modelname,"modelfile", modelfile, session=session)
                
                model.testFeatureExtraction(
                    jobconf["testFolder"], 
                    limit=limitTest)

                model.predict()

                # manual accuracy computation
                solutionsFile = jobconf["resultsFolder"]+'/task9.1_'+training_set_name+model.name+'_sol.txt'
                model.parseTestSetOutput(solutionsFile,debug=False)
                
                predictionsFile = jobconf["resultsFolder"]+'/task9.1_'+training_set_name+model.name+'.txt'
                model.parsePredictionOutput(predictionsFile,debug=True)
                
                #results_files[i] = jobconf["savingFolder"]+ "/" + jobconf["session"]+"-"+model.name+'-output.csv'

                print("model "+model.name+":")
                scores = model.autoEvaluation(jobconf["testFolder"])
                _dummy, accuracy2 = model.manualEvaluation(
                    solutionsFile,debug=False)
                
                DDImodel.writeConf(
                    jobconf,modelname,"scores", scores, session=session)
                DDImodel.writeConf(
                    jobconf,modelname,
                    "accuracy_manual", 
                    accuracy2, 
                    session=session)
                DDImodel.writeConf(
                    jobconf,
                    modelname,
                    "algorithm", 
                    model.algotype, session=session)
                

                DDImodel.writeConf(
                        jobconf,
                        modelname,
                        "resultsFile",
                        model.predictionResultFile+'_result', session=session )
                
                DDImodel.writeConf(
                        jobconf,
                        modelname,
                        "scoresFile",
                        model.predictionResultFile[:-4]+'_scores.log', session=session )
                
                DDImodel.writeConf(jobconf,modelname,"status", "done", session=session)

                end = time.time()
                DDImodel.writeConf(jobconf,modelname,"end", end, session=session )
                total_time = end - start
                DDImodel.writeConf(jobconf,modelname,"total_time", total_time, session=session)
                print("job time:",total_time)
                print()

                with mylock:
                    DDImodel.writeConf(jobconf,modelname,"status", "done", session=session, todisk=True )
                

        except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                print("model "+modelname+" has failed... ")
                
                DDImodel.writeConf(jobconf,modelname,"status", "error", session=session )

        #trying to free memory
        gc.collect()
        
        #return jobconf,modelname





    def parallelbatchTraining(configfile="models.conf",status=True):

        conf = DDImodel.initConf(configfile)
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
                    mf = MyFeaturesDDI(params=params)

                    #datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
                    #featuresetFile = conf["savingFolder"] +'featureset_'+ datesuffix +'.pkl'
                    #joblib.dump(mf, featuresetFile)

                    fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                DDImodel.writeConf(conf,modelname,"status", "error", session=session )

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
        #     procs = [Process(target=DDImodel.trainTestModel,args=(mf,training_set_name, session, mylock)) for mf in fs[k*numcores:k*numcores+numcores]]
        #     for p in procs: p.start()
        #     for p in procs: p.join()
        # procs = [Process(target=DDImodel.trainTestModel,args=(mf,training_set_name, session, mylock)) for mf in fs[ktot*numcores:ktot*numcores+rest]]
        # for p in procs: p.start()
        # for p in procs: p.join()
        

        # using pool.apply
        pool = Pool(processes=numcores, maxtasksperchild=1)
        procs = [pool.apply_async(DDImodel.trainTestModel,args=(mf,training_set_name, session)) for mf in fs]
        #results = [p.get() for p in procs]        
        pool.close()
        
        pool.join()

        
        # sequential version
        #jobresults = [DDImodel.trainTestModel(mf,training_set_name, session) for mf in fs]



        total_end = time.time()
        total_time =total_end - total_start
        print("total batch time", total_time)
        DDImodel.writeConf(conf,"","total_time", total_time ,todisk=True)
       
    def parallelbatchCVTraining(configfile="models.conf",status=True):

        conf = DDImodel.initConf(configfile)
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
                    mf = MyFeaturesDDI(params=params)

                    #datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
                    #featuresetFile = conf["savingFolder"] +'featureset_'+ datesuffix +'.pkl'
                    #joblib.dump(mf, featuresetFile)

                    fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                DDImodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])       

        # Parallelization
        numcores=7
        if "numcores" in conf.keys() and conf["numcores"] is not None:
            print("setting numcores to ",conf["numcores"])
            numcores = conf["numcores"]

        

        # using pool.apply
        pool = Pool(processes=numcores, maxtasksperchild=1)
        procs = [pool.apply_async(DDImodel.trainTestModel,args=(mf,training_set_name, session, True)) for mf in fs]
        #results = [p.get() for p in procs]        
        pool.close()
        
        pool.join()

        
        # sequential version
        #jobresults = [DDImodel.trainTestModel(mf,training_set_name, session) for mf in fs]



        total_end = time.time()
        total_time =total_end - total_start
        print("total batch time", total_time)
        DDImodel.writeConf(conf,"","total_time", total_time ,todisk=True)
       

    def featureExtractionDDI(configfile="models.conf",status=True):

        """
            assume only one data set is processed
        """

        conf = DDImodel.initConf(configfile)
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
                    mf = MyFeaturesDDI(params=params)

                    #datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
                    #featuresetFile = conf["savingFolder"] +'featureset_'+ datesuffix +'.pkl'
                    #joblib.dump(mf, featuresetFile)

                    fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                DDImodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])    

        for mf in fs:
            fset=mf[0]
            modelname=mf[1]
            modeltype=mf[2]
            jobconf = mf[4]

            start = time.time()
            DDImodel.writeConf(
                jobconf,modelname,"start", start, session=session )

            DDImodel.writeConf(
                jobconf,modelname,"status", "doing", session=session )
            
            algorithm="svm"
            if "algorithm" in mf[3].keys():
                algorithm = mf[3]["algorithm"]

            limitTrain=None
            if "limitTraining" in mf[3].keys():
                limitTrain = mf[3]["limitTraining"]
            limitTest=None
            if "limitTest" in mf[3].keys():
                limitTest = mf[3]["limitTest"]
                
            try:

                with DDImodel(
                    featureset=fset, 
                    name=training_set_name+modelname, 
                    modeltype=modeltype, 
                    algorithm=algorithm) as model:

                    
                    model.trainFeatureExtraction(
                        jobconf["trainingFolder"], 
                        limit=limitTrain)

                    model.saveTrainingFeatures(jobconf["savingFolder"] +"/" +jobconf["session"]+"-"+model.name+".json")
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                


    def featureExtractionDDITest(configfile="models.conf",status=True):

        """
            assume only one data set is processed
        """

        conf = DDImodel.initConf(configfile)
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
                    mf = MyFeaturesDDI(params=params)

                    #datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
                    #featuresetFile = conf["savingFolder"] +'featureset_'+ datesuffix +'.pkl'
                    #joblib.dump(mf, featuresetFile)

                    fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                DDImodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])    

        for mf in fs:
            fset=mf[0]
            modelname=mf[1]
            modeltype=mf[2]
            jobconf = mf[4]

            start = time.time()
            DDImodel.writeConf(
                jobconf,modelname,"start", start, session=session )

            DDImodel.writeConf(
                jobconf,modelname,"status", "doing", session=session )
            
            algorithm="svm"
            if "algorithm" in mf[3].keys():
                algorithm = mf[3]["algorithm"]

            limitTrain=None
            if "limitTraining" in mf[3].keys():
                limitTrain = mf[3]["limitTraining"]
            limitTest=None
            if "limitTest" in mf[3].keys():
                limitTest = mf[3]["limitTest"]
                
            try:

                with DDImodel(
                    featureset=fset, 
                    name=training_set_name+modelname, 
                    modeltype=modeltype, 
                    algorithm=algorithm) as model:

                    
                    model.testFeatureExtraction(
                        jobconf["testFolder"], 
                        limit=limitTrain)
                
                    model.saveTestFeatures(jobconf["savingFolder"] +"/" +jobconf["session"]+"-test.json")
                    
                    
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                


    def featureExtractionDDIstep2(configfile="models.conf",status=True):

        """
            assume only one data set is processed
        """

        conf = DDImodel.initConf(configfile)
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
                    mf = MyFeaturesDDI(params=params)

                    #datesuffix = strftime("%Y%m%d%H%M%S", gmtime())
                    #featuresetFile = conf["savingFolder"] +'featureset_'+ datesuffix +'.pkl'
                    #joblib.dump(mf, featuresetFile)

                    fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                DDImodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])    

        for mf in fs:
            fset=mf[0]
            modelname=mf[1]
            modeltype=mf[2]
            jobconf = mf[4]
            modelparams = mf[3]

            start = time.time()
            DDImodel.writeConf(
                jobconf,modelname,"start", start, session=session )

            DDImodel.writeConf(
                jobconf,modelname,"status", "doing", session=session )
            
            algorithm="svm"
            if "algorithm" in modelparams.keys():
                algorithm = modelparams["algorithm"]

            limitTrain=None
            if "limitTraining" in modelparams.keys():
                limitTrain = modelparams["limitTraining"]
            limitTest=None
            if "limitTest" in modelparams.keys():
                limitTest = modelparams["limitTest"]

            topcount = 50
            if "topcount" in modelparams.keys():
                topcount = modelparams["topcount"]


                
            try:

                with DDImodel(
                    featureset=fset, 
                    name=training_set_name+modelname, 
                    modeltype=modeltype, 
                    algorithm=algorithm) as model:

                    model.traindata = FeatureExtractionDDI('../data/results',
                        algoformat=algorithm,
                        targetformat=modeltype)

                    model.traindata.finalizeFeatures(
                        jobconf["trainingFolder"],
                        limit=limitTrain,
                        topcount=topcount,
                        topfeatures=modelparams["topcountfeatures"],
                        sentencefeatures=modelparams["sentencefeatures"],
                        wordfeatures=modelparams["wordfeatures"],
                        windowfeatures=modelparams["windowfeatures"]
                        )

                    
                    model.saveTrainingFeatures(jobconf["savingFolder"] +"/" +jobconf["session"]+".json")
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)


    def trainTestDDIModel(mf, training_set_name, session, mylock=mylock):


        fset=mf[0]
        modelname=mf[1]
        modeltype=mf[2]
        jobconf = mf[4]
        modelparams = mf[3]

        start = time.time()
        DDImodel.writeConf(
            jobconf,modelname,"start", start, session=session )

        DDImodel.writeConf(
            jobconf,modelname,"status", "doing", session=session )
        
        algorithm="svm"
        if "algorithm" in mf[3].keys():
            algorithm = mf[3]["algorithm"]

        limitTrain=None
        if "limitTraining" in mf[3].keys():
            limitTrain = mf[3]["limitTraining"]
        limitTest=None
        if "limitTest" in mf[3].keys():
            limitTest = mf[3]["limitTest"]

        topcount = 50
        if "topcount" in mf[3].keys():
            topcount = mf[3]["topcount"]

        if modeltype == "DDI":
            jobconf["trainingFolder"] = jobconf["trainingDDIfile"]
            jobconf["testFeatures"] = jobconf["testDDIfile"]
        elif modeltype == "BDDI":
            jobconf["trainingFolder"] = jobconf["trainingBDDIfile"]
            jobconf["testFeatures"] = jobconf["testBDDIfile"]

        if "window" not in modelparams.keys():
            modelparams["window"]=3

        print("DDI model",modelname)
        try:
            
            with DDImodel(
                featureset=fset, 
                name=training_set_name+modelname, 
                modeltype=modeltype, 
                algorithm=algorithm) as model:

                # json training features jobconf["trainingFolder"]
                model.traindata = FeatureExtractionDDI('../data/results')
                model.loadDDIFeatures(
                    filepath=jobconf["trainingFolder"],
                    limit=limitTrain,
                    topcount=topcount,
                    topfeatures=modelparams["topcountfeatures"],
                    sentencefeatures=modelparams["sentencefeatures"],
                    wordfeatures=modelparams["wordfeatures"],
                    windowfeatures=modelparams["windowfeatures"],
                    window=modelparams["window"])



                model.newModelPipeline(ertype=model.modeltype, algotype=model.algotype)

                modelfile= model.saveModelPipeline()
                DDImodel.writeConf(
                    jobconf,modelname,"modelfile", modelfile, session=session)
                
                # json training features jobconf["testFolder"]
                model.testdata = FeatureExtractionDDI('../data/results')
                model.loadDDITestFeatures(
                    filepath=jobconf["testFeatures"],
                    limit=limitTest,
                    topcount=topcount,
                    topfeatures=modelparams["topcountfeatures"],
                    sentencefeatures=modelparams["sentencefeatures"],
                    wordfeatures=modelparams["wordfeatures"],
                    windowfeatures=modelparams["windowfeatures"],
                    window=modelparams["window"])

                print("inside model, loaded testing",jobconf["testFeatures"])


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
                
                DDImodel.writeConf(
                    jobconf,modelname,"scores", scores, session=session)
                DDImodel.writeConf(
                    jobconf,modelname,
                    "accuracy_manual", 
                    accuracy2, 
                    session=session)
                DDImodel.writeConf(
                    jobconf,
                    modelname,
                    "algorithm", 
                    model.algotype, session=session)
                

                DDImodel.writeConf(
                        jobconf,
                        modelname,
                        "resultsFile",
                        model.predictionResultFile+'_result', session=session )
                
                DDImodel.writeConf(
                        jobconf,
                        modelname,
                        "scoresFile",
                        model.predictionResultFile[:-4]+'_scores.log', session=session )
                
                DDImodel.writeConf(jobconf,modelname,"status", "done", session=session)

                end = time.time()
                DDImodel.writeConf(jobconf,modelname,"end", end, session=session )
                total_time = end - start
                DDImodel.writeConf(jobconf,modelname,"total_time", total_time, session=session)
                print("job time:",total_time)
                print()

                with mylock:
                    DDImodel.writeConf(jobconf,modelname,"status", "done", session=session, todisk=True )
                
            

        except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                print("model "+modelname+" has failed... ")
                
                DDImodel.writeConf(jobconf,modelname,"status", "error", session=session )

        #trying to free memory
        gc.collect()
        
        #return jobconf,modelname
        print("after DDI model")


    def parallelDDIbatchTraining(configfile="models.conf",status=True):

        conf = DDImodel.initConf(configfile)
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
                    mf = MyFeaturesDDI(params=params)

                    fs.append((mf,modelname,params["modeltype"],params,copy.deepcopy(conf)))
                
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                DDImodel.writeConf(conf,modelname,"status", "error", session=session )

        print([e[1] for e in fs])       

        # Parallelization
        numcores=5
        if "numcores" in conf.keys() and conf["numcores"] is not None:
            print("setting numcores to ",conf["numcores"])
            numcores = conf["numcores"]
        

        # using pool.apply
        pool = Pool(processes=numcores, maxtasksperchild=1)
        procs = [pool.apply_async(DDImodel.trainTestDDIModel,args=(mf,training_set_name, session)) for mf in fs]
        #results = [p.get() for p in procs]        
        pool.close()
        pool.join()



        total_end = time.time()
        total_time =total_end - total_start
        print("total batch time", total_time)
        DDImodel.writeConf(conf,"","total_time", total_time ,todisk=True)
       

