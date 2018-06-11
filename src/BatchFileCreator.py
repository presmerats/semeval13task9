import os, sys
import yaml
import pprint
import traceback
import jinja2
from jinja2 import Template
import subprocess
import shutil
from pathlib import Path
import copy
import itertools
import random



class BatchFileCreator():

    def __init__(self,filepath, backupfolder):
        self.backupfolder = backupfolder
        self.filepath = filepath

        self.featurelist = [
            "isTitleCase",
            "isUpperCase",
            "isLowerCase",
            "hasDigits",
            "hasStrangeChars",
            "moreThan10chars",
            "prefix",
            "prefix3",
            "prefix4",
            "prefix5",
            "suffix",
            "suffix3",
            "suffix4",
            "suffix5",
            "lenprefix",
            "lensuffix",
            "lenword",
            "wordStructure",
            "wordStructure2",
            "wordStructureLong",
            "wordStructureLong2",
            "lemma",
            "pos",
            "w2v",
            "word",
            "chunk",
            "chunkGroup",
            "lookup"
            ]

        self.ortholist = [
            "isTitleCase",
            "isUpperCase",
            "isLowerCase",
            "hasDigits",
            "hasStrangeChars",
            "moreThan10chars",
            "prefix",
            "prefix3",
            "prefix4",
            "prefix5",
            "suffix",
            "suffix3",
            "suffix4",
            "suffix5",
            "lenprefix",
            "lensuffix",
            "lenword",
            "wordStructure",
            #"wordStructure2",
            #"wordStructureLong",
            #"wordStructureLong2",
        ]




    def backup(self, filepath=None, backupfolder=None):
        if backupfolder is None:
            backupfolder = self.backupfolder

        if filepath is None:
            filepath = self.filepath

        # get filename
        filepath2 = os.path.basename(filepath)
        # create backup folder if it doesnt exist
        mybackupfolder = Path(backupfolder)
        if not mybackupfolder.exists():
            os.makedirs(backupfolder)

        shutil.copy(filepath,os.path.join(backupfolder,filepath2+'.back' ))

    def load(self):
        return yaml.load(open(self.filepath,'r'))

    def save(self, dictionary):
        if not Path(os.path.dirname(self.filepath)).exists():
            os.makedirs(os.path.dirname(self.filepath))

        with open(self.filepath,'w') as f:
            f.write(yaml.dump(dictionary))


    def removeResults(self):
        batch = self.load()

        if "models" not in batch.keys():
            return False

        for name, model in batch["models"].items():

            if "results" not in model.keys():
                continue

            model["results"] = {}   

        self.save(batch)


    def removeResultsOld(self):
        batch = self.load()

        if "models" not in batch.keys():
            return False

        for name, model in batch["models"].items():

            if "resultsold" not in model.keys():
                continue

            del model["resultsold"]

        self.save(batch)

    def resetTotalTime(self):
        batch = self.load()
        if "total_time" not in batch.keys():
            return False
        batch["total_time"] = 0
        self.save(batch)

    def resetStatus(self):
        batch = self.load()

        if "models" not in batch.keys():
            return False

        for name, model in batch["models"].items():

            if "results" not in model.keys():
                continue
            for rname, result in model["results"].items():
                if "status" in result.keys():
                    result["status"] = "pending"

        self.save(batch)

    def removeAccuracies(self):
        batch = self.load()

        if "models" not in batch.keys():
            return False

        for name, model in batch["models"].items():

            if "accuracy" in model.keys():
                del model["accuracy"]
            if "accuracy_manual" in model.keys():
                del model["accuracy_manual"]
            if "goalaccuracy" in model.keys():
                del model["goalaccuracy"]
            if "precision" in model.keys():
                del model["precision"]
            if "recall" in model.keys():
                del model["recall"]
            if "f1" in model.keys():
                del model["f1"]


            if "results" not in model.keys():
                continue
            for rname, result in model["results"].items():
                if "accuracy" in result.keys():
                    del result["accuracy"]
                if "goalaccuracy" in result.keys():
                    del result["goalaccuracy"]
                if "accuracy_manual" in result.keys():
                    del result["accuracy_manual"]
                if "precision" in result.keys():
                    del result["precision"]
                if "recall" in result.keys():
                    del result["recall"]
                if "f1" in result.keys():
                    del result["f1"]
                if "algorithm" in result.keys():
                    del result["algorithm"]

        self.save(batch)     


    def removeModelByName(self, thename):
        batch = self.load()

        if "models" not in batch.keys():
            return False

        for name, model in batch["models"].items():
            if name == thename:
                del batch["models"][name]
                break
        self.save(batch)

    def getBeforeWindow(self, model):
        count_before = 0

        if "fbefore5" in model.keys() and len(model["fbefore5"])>0:
            count_before = 5
        elif "fbefore4" in model.keys() and len(model["fbefore4"])>0:
            count_before = 4
        elif "fbefore3" in model.keys() and len(model["fbefore3"])>0:
            count_before = 3
        elif "fbefore2" in model.keys() and len(model["fbefore2"])>0:
            count_before = 2
        elif "fbefore1" in model.keys() and len(model["fbefore1"])>0:
            count_before = 1

        return count_before

    def getAfterWindow(self, model):
        count_after = 0
        
        if "fafter5" in model.keys() and len(model["fafter5"])>0:
            count_after = 5
        elif "fafter4" in model.keys() and len(model["fafter4"])>0:
            count_after = 4
        elif "fafter3" in model.keys() and len(model["fafter3"])>0:
            count_after = 3
        elif "fafter2" in model.keys() and len(model["fafter2"])>0:
            count_after = 2
        elif "fafter1" in model.keys() and len(model["fafter1"])>0:
            count_after = 1

        return count_after

    def getWindow(self,model):
        count_after = self.getAfterWindow(model)
        count_before = self.getBeforeWindow(model)

        return max(count_after, count_before)

    def isW1(self,model):
        return self.getWindow(model)==1

    def isW2(self,model):
        return self.getWindow(model)==2

    def isW3(self,model):
        return self.getWindow(model)==3

    def isW4(self,model):
        return self.getWindow(model)==4

    def isW5(self,model):
        return self.getWindow(model)==5


    def featureInList(self, model, listname, featurename):
        return listname in model.keys() and featurename in model[listname]

    def hasFeature(self, model, featurename):
        return self.featureInList(model, "fafter5", featurename) \
            or self.featureInList(model, "fafter4", featurename) \
            or self.featureInList(model, "fafter3", featurename) \
            or self.featureInList(model, "fafter2", featurename) \
            or self.featureInList(model, "fafter1", featurename) \
            or self.featureInList(model, "fcurrent", featurename) \
            or self.featureInList(model, "fbefore1", featurename) \
            or self.featureInList(model, "fbefore2", featurename) \
            or self.featureInList(model, "fbefore3", featurename) \
            or self.featureInList(model, "fbefore4", featurename) \
            or self.featureInList(model, "fbefore5", featurename)


    def removeModelByPattern(self, patternname):
        batch = self.load()

        if "models" not in batch.keys():
            return False

        copydict = copy.deepcopy(batch["models"])
        for name, model in batch["models"].items():
            if patternname == "w1" and self.isW1(model):
                del copydict[name]
            elif patternname == "w2" and self.isW2(model):
                del copydict[name]
            elif patternname == "w3" and self.isW3(model):
                del copydict[name]
            elif patternname == "w4" and self.isW4(model):
                del copydict[name]
            elif patternname == "w5" and self.isW5(model):
                del copydict[name]
            elif patternname in self.featurelist and self.hasFeature(model, patternname):
                del copydict[name]

        batch["models"] = copydict

                
        self.save(batch)

    def replicateWindow(self, numt, model):
        count_after = self.getAfterWindow(model)
        count_before = self.getBeforeWindow(model)

        if count_after < numt:
            for i in range(count_after+1,numt+1):
                #print("filling ","fafter"+str(i)," with ","fafter"+str(count_after))
                model["fafter"+str(i)] =model["fafter"+str(count_after)][:]

        if count_before < numt:
            for i in range(count_before+1,numt+1):
                #print("filling ","fbefore"+str(i)," with ","fbefore"+str(count_before))

                model["fbefore"+str(i)] =model["fbefore"+str(count_before)][:]

    def reduceWindow(self, numt, model):
        count_after = self.getAfterWindow(model)
        count_before = self.getBeforeWindow(model)

        if count_after > numt:
            for i in range(numt+1,count_after+1,):
                #print("filling ","fafter"+str(i)," with ","fafter"+str(count_after))
                model["fafter"+str(i)] =[]

        if count_before > numt:
            for i in range(numt+1,count_before+1):
                #print("filling ","fbefore"+str(i)," with ","fbefore"+str(count_before))

                model["fbefore"+str(i)] =[]


    def replicateBatch(self, target, newname):
        batch = self.load()

        # rename session and filpeath inside the batch file
        batch["session"]=os.path.basename(newname.replace(".yaml",""))
        batch["filepath"]=newname
        self.filepath = newname

        if target.lower() == "medline".lower():
            batch["testFolder"] = '../data/LaboCase/Test/Test for DrugNER task/MedLine'
        elif target.lower() == "drugbank".lower():
            batch["testFolder"] = '../data/LaboCase/Test/Test for DrugNER task/DrugBank'
        elif target.lower() == "mixed".lower():
            batch["testFolder"] = '../data/LaboCase/Test/Test for DrugNER task/mixed'
        elif target.lower() == "crf".lower():
            if "models" not in batch.keys():
                return False
            for name, model in batch["models"].items():
                model["algorithm"] = target.lower()
            
        elif target.lower() == "svm".lower():
            if "models" not in batch.keys():
                return False
            for name, model in batch["models"].items():
                model["algorithm"] = target.lower()
        elif target.lower() == "ER".lower():
            if "models" not in batch.keys():
                return False
            for name, model in batch["models"].items():
                model["modeltype"] = target.upper()
        elif target.lower() == "NER".lower():
            if "models" not in batch.keys():
                return False
            for name, model in batch["models"].items():
                model["modeltype"] = target.upper()
        elif target.lower() == "DDI".lower():
            if "models" not in batch.keys():
                return False
            for name, model in batch["models"].items():
                model["modeltype"] = target.upper()
        elif target.lower() in ["w1","w2","w3","w4","w5"]:
            if "models" not in batch.keys():
                return False
            for name, model in batch["models"].items():
                
                window = self.getWindow(model)
                numt = int(target[-1])
                if window < numt:
                    #print(name)
                    self.replicateWindow(numt,model)
                elif window > numt:
                    self.reduceWindow(numt,model)
        self.save(batch) 


    def forceBatch(self, target, newname):
        """
            window to be exactly target
        """
        batch = self.load()

        # rename session and filpeath inside the batch file
        batch["session"]=os.path.basename(newname.replace(".yaml",""))
        batch["filepath"]=newname
        self.filepath = newname

        if target.lower() in ["w1","w2","w3","w4","w5"]:
            if "models" not in batch.keys():
                return False
            for name, model in batch["models"].items():
                
                window = self.getWindow(model)
                numt = int(target[-1])
                self.replicateWindow(numt,model)
                self.reduceWindow(numt,model)
                
        self.save(batch) 


    def mergeBatch(self, filepath2, newname):
        """
            merges models from self.filepath and filepath2
            into filepath(attributes of this filepath) but saving it with newname
        """
        batch = self.load()

        # rename session and filpeath inside the batch file
        batch["session"]=os.path.basename(newname.replace(".yaml",""))
        batch["filepath"]=newname
        self.filepath = newname

        # second batch
        bfc2 = BatchFileCreator(
        filepath=filepath2,
        backupfolder=self.backupfolder)
        batch2 = bfc2.load()



        if "models" not in batch.keys() or \
           "models" not in batch2.keys():
            return False

        # quick approach
        # renaming if needed!
        auxdict = copy.deepcopy(batch2["models"])
        for name, model in batch2["models"].items():
            if name in batch["models"].keys():
                auxdict[name+"b"]= model
                del auxdict[name]
        batch2["models"]=auxdict

        batch["models"].update(batch2["models"])

        #for name, model in batch2["models"].items():
       
        self.save(batch) 


    def generateSingleModel(self, window="3", typewindow="symmetric",
        wordtypes=["lemma","pos","chunkGroup","lookup","wordStructure"],
        windowtypes=["pos","lookup","wordStructure"],
        
        modeltype="NER",
        algorithm="CRF",
        ):


        basicModel ={
            "algorithm": algorithm,
            "fafter1": [],
            "fafter2": [],
            "fafter3": [],
            "fafter4": [],
            "fafter5": [],
            "fbefore1": [],
            "fbefore2": [],
            "fbefore3": [],
            "fbefore4": [],
            "fbefore5": [],
            "fcurrent": wordtypes[:],
            "limitTest": None,
            "limitTraining": None,
            "modelfile": None,
            "modeltype": modeltype,
            "resultsFile": None,
            "scoresFile": None,
            "status": "pending"
        }

        if typewindow=="symmetric":
            for i in range(int(window)):
                basicModel["fafter"+str(i+1)]=windowtypes[:]
                basicModel["fbefore"+str(i+1)]=windowtypes[:]
        else:
            # expected "-1:+2"
            beforelength = int(typewindow[1])
            afterlength = int(typewindow[4])
            for i in range(beforelength):
                basicModel["fbefore"+str(i+1)]=windowtypes[:]
            for i in range(afterlength):
                basicModel["fafter"+str(i+1)]=windowtypes[:]

                

        return basicModel


    def generateSingleBatchFile(self,
        filepath="../data/batches/b0520/example.yaml",
        testdata="MedLine",
        session="tMDmNERaCRF",
        window="3", 
        typewindow="symmetric",
        wordtypes=["lemma","pos","chunkGroup","lookup","wordStructure"],
        windowtypes=["pos","lookup","wordStructure"],
        modeltype="NER",
        algorithm="CRF",
        combinations=None,
        numcores=4):
        
        basicBatchDict = {
            "filepath": filepath,
            'numcores': numcores,
            "resultsFolder": "../data/results/",
            "savingFolder": "../data/models/",
            "session": session,
            "testFolder": "../data/LaboCase/Test/Test for DrugNER task/"+testdata,
            "total_time": 0,
            "trainingFolder": "../data/LaboCase/Train",
            "models": {}
        }

        # filtering types list

        # expand types list
        # orthographic are summarized
        # but lemma, pos, chunk, chunkGroup, lookup
        # are directly written..
        wordfeatures = []
        for ftype in wordtypes:
            if ftype in self.featurelist:
                wordfeatures.append(ftype)
            elif ftype=="ortho":
                wordfeatures.extend(self.ortholist)

        windowfeatures = []
        for ftype in windowtypes:
            if ftype in self.featurelist:
                windowfeatures.append(ftype)
            elif ftype=="ortho":
                windowfeatures.extend(self.ortholist)

        # randomization / combinations
        if combinations is  None:
            basicBatchDict["models"]["m000"]=self.generateSingleModel(
                window,
                typewindow,
                wordfeatures,
                windowfeatures,
                modeltype,
                algorithm)
            self.filepath = filepath
            self.save(basicBatchDict)
        else:
            # generate different combinations
            if combinations.startswith("all"):
                numcomb = int(combinations[3:])
                # all combinations of numcomb
                windowfeatures2 = list(itertools.combinations(windowfeatures,numcomb))
                wordfeatures2 = list(itertools.combinations(wordfeatures,numcomb))
                combinations = list(itertools.product(windowfeatures2,wordfeatures2))

                #pprint.pprint(combinations)
            else:
                numrand = int(combinations[6:])
                # shuffle numrand times
                combinations = []
                for i in range(numrand):
                    random.shuffle(windowfeatures)
                    random.shuffle(wordfeatures)
                    combinations.append(
                        (
                           copy.deepcopy(windowfeatures[:-1]),
                           copy.deepcopy(wordfeatures[:-1])
                        )
                        )



            # generate respective models
            newmodels = {}
            for i in range(len(combinations)):
                combi = combinations[i]
                #print(combi)
                windowfeatures = list(combi[0])
                wordfeatures = list(combi[1])
                
                newmodels["m"+str(i)] = self.generateSingleModel(

                window,
                typewindow,
                wordfeatures,
                windowfeatures,
                modeltype,
                algorithm)


            # save to basicBatchDict

            basicBatchDict["models"]=newmodels
            self.filepath = filepath
            self.save(basicBatchDict)




    def generateIncrementalBatchFile(self,
        filepath="../data/batches/b0520/example.yaml",
        testdata="MedLine",
        session="tMDmNERaCRF",
        window="3", 
        typewindow="symmetric",
        pairs=[],
        modeltype="NER",
        algorithm="CRF",
        combinations=None,
        basicBatchDict=None,
        numcores=4):
        
        if basicBatchDict is None:
            basicBatchDict = {
                "filepath": filepath,
                'numcores': numcores,
                "resultsFolder": "../data/results/",
                "savingFolder": "../data/models/",
                "session": session,
                "testFolder": "../data/LaboCase/Test/Test for DrugNER task/"+testdata,
                "total_time": 0,
                "trainingFolder": "../data/LaboCase/Train",
                "models": {}
            }


        themodels={}
        pnum = 0
        for pair in pairs:
            pnum+=1

            wordtypes=pair[0]
            windowtypes=pair[1]


            # filtering types list

            # expand types list
            # orthographic are summarized
            # but lemma, pos, chunk, chunkGroup, lookup
            # are directly written..
            wordfeatures = []
            for ftype in wordtypes:
                if ftype in self.featurelist:
                    wordfeatures.append(ftype)
                elif ftype=="ortho":
                    wordfeatures.extend(self.ortholist)

            windowfeatures = []
            for ftype in windowtypes:
                if ftype in self.featurelist:
                    windowfeatures.append(ftype)
                elif ftype=="ortho":
                    windowfeatures.extend(self.ortholist)

            # randomization / combinations
            if combinations is  None:
                themodels["m"+str(pnum)]=self.generateSingleModel(
                    window,
                    typewindow,
                    wordfeatures,
                    windowfeatures,
                    modeltype,
                    algorithm)

            else:
                # generate different combinations
                if combinations.startswith("all"):
                    numcomb = int(combinations[3:])
                    # all combinations of numcomb
                    windowfeatures2 = list(itertools.combinations(windowfeatures,numcomb))
                    wordfeatures2 = list(itertools.combinations(wordfeatures,numcomb))
                    combinations = list(itertools.product(windowfeatures2,wordfeatures2))

                    #pprint.pprint(combinations)
                else:
                    numrand = int(combinations[6:])
                    # shuffle numrand times
                    combinations = []
                    for i in range(numrand):
                        random.shuffle(windowfeatures)
                        random.shuffle(wordfeatures)
                        combinations.append(
                            (
                               copy.deepcopy(windowfeatures[:-1]),
                               copy.deepcopy(wordfeatures[:-1])
                            )
                            )



                # generate respective models
                newmodels = {}
                for i in range(len(combinations)):
                    combi = combinations[i]
                    #print(combi)
                    windowfeatures = list(combi[0])
                    wordfeatures = list(combi[1])
                    
                    newmodels["m"+str(i)] = self.generateSingleModel(

                    window,
                    typewindow,
                    wordfeatures,
                    windowfeatures,
                    modeltype,
                    algorithm)


                # save to basicBatchDict

                themodels.update(newmodels)

        basicBatchDict["models"]=themodels
        self.filepath = filepath
        self.save(basicBatchDict)


    def generate(self, root="d21",
        folder="../data/batches/b0521/",
        typewindow="symmetric",
        window="3",
        wordtypes=["lemma","pos","chunkGroup","lookup","wordStructure"],
        windowtypes=["pos","lookup","wordStructure"],
        combinations=None):

        """ 
            In: 
                folder where batchfiles will be saved
                typewindow, window  
                wordtypes list of features/types for word
                windowtypes list of features/types for window
                combinations alli, randomi, None -> generation mechanism
        
            out: writes different batch files
                 
                 generation mechanism:
                    - first the combinations and the wordtypes, windotypes will be combined to create a list of models
                    - this list of models will be saved in different batchfiles, one for each combination of
                        - Medline, DrugBank or Mixed testfolder
                        - ER, NER; DDI model type
                        - SVm, CRF algorithm


        for algo in ["svm", "crf"]:
            for testype in ["MedLine", "DrugBank", "Mixed"]:
                for modeltype in ["ER", "NER", "DDI"]:
                    for windowlength in ["1", "2", "3", "4", "5"]:


        """

        for algo in ["crf"]:
            for testype in ["MedLine", "DrugBank", "Mixed"]:
                for modeltype in ["NER"]:
                    for windowlength in ["1", "2", "3", "5"]:

                        numcors=6
                        if windowlength=="5":
                            numcors=4

                        testtype2 ={
                            "MedLine":"MD", 
                            "DrugBank": "DB", 
                            "Mixed": "MX"
                        }
                        session = root \
                            + "t" + testtype2[testype].upper() \
                            + "m" + modeltype.upper() \
                            + "a" + algo.upper() \
                            + "w" + windowlength

                        filename = os.path.join(
                            folder,
                            session + ".yaml")

                        self.generateSingleBatchFile(
                            filepath=filename,
                            wordtypes=wordtypes,
                            windowtypes=windowtypes,
                            typewindow=typewindow,
                            window=windowlength,
                            combinations=combinations,
                            testdata=testype,
                            session=session,
                            modeltype=modeltype,
                            algorithm=algo,
                            numcores=numcors)


    def generateMultiple(self, root="d21",
        folder="../data/batches/b0521/",
        typewindow="symmetric",
        window="3",
        pairs=[],
        combinations=None):

        """ 
            same as generate() but with pairs of wordfeatures and windowfeatures

        """

        for algo in ["crf"]:
            for testype in ["MedLine", "DrugBank", "Mixed"]:
                for modeltype in ["NER"]:
                    for windowlength in ["1", "2", "3", "5"]:

                        numcors=6
                        if windowlength=="5":
                            numcors=4

                        testtype2 ={
                            "MedLine":"MD", 
                            "DrugBank": "DB", 
                            "Mixed": "MX"
                        }
                        session = root \
                            + "t" + testtype2[testype].upper() \
                            + "m" + modeltype.upper() \
                            + "a" + algo.upper() \
                            + "w" + windowlength

                        filename = os.path.join(
                            folder,
                            session + ".yaml")

                        self.generateIncrementalBatchFile(
                            filepath=filename,
                            pairs=pairs,
                            typewindow=typewindow,
                            window=windowlength,
                            combinations=combinations,
                            testdata=testype,
                            session=session,
                            modeltype=modeltype,
                            algorithm=algo,
                            numcores=numcors)





    def exampleW5clean():
        bfc = BatchFileCreator(
        filepath='../data/batches/b0517/debug_examplew5.yaml',
        backupfolder='../data/batches/b0517/backup')
        bfc.backup()

        # w5 clean
        bfc.removeModelByPattern("w1")
        bfc.removeModelByPattern("w2")
        bfc.removeModelByPattern("w3")
        bfc.removeModelByPattern("w4")
        # replicate to crf,svm,md,db,mx,er,ner

        # tMDmERaCRF
        bfc.replicateBatch("crf",'../data/batches/b0517/debug_examplew5_aCRF.yaml')
        bfc = BatchFileCreator(
        filepath='../data/batches/b0517/debug_examplew5_aCRF.yaml',
        backupfolder='../data/batches/b0517/backup')
        
        bfc.replicateBatch("MedLine",'../data/batches/b0517/debug_examplew5_tMDaCRF.yaml')
        bfc = BatchFileCreator(
        filepath='../data/batches/b0517/debug_examplew5_tMDaCRF.yaml',
        backupfolder='../data/batches/b0517/backup')
        
        bfc.replicateBatch("ER",'../data/batches/b0517/debug_examplew5_tMDmERaCRF.yaml')
        tMDmERaCRF = BatchFileCreator(
        filepath='../data/batches/b0517/debug_examplew5_tMDmERaCRF.yaml',
        backupfolder='../data/batches/b0517/backup')

        # tDBmERaCRF
        tMDmERaCRF.replicateBatch("DrugBank",'../data/batches/b0517/debug_examplew5_tDBmERaCRF.yaml')

        # tMXmERaCRF
        tMDmERaCRF.replicateBatch("mixed",'../data/batches/b0517/debug_examplew5_tMXmERaCRF.yaml')


        # tMDmNERaCRF
        tMDmERaCRF.replicateBatch("NER",'../data/batches/b0517/debug_examplew5_tMXmNERaCRF.yaml')

        # tMDmERaSVM
        tMDmERaCRF.replicateBatch("svm",'../data/batches/b0517/debug_examplew5_tMXmERaSVM.yaml')

        

    def exampleW3clean():
        bfc = BatchFileCreator(
        filepath='../data/batches/b0517/debug_examplew5.yaml',
        backupfolder='../data/batches/b0517/backup')
        bfc.backup()

        # w3 clean
        bfc.removeModelByPattern("w1")
        bfc.removeModelByPattern("w2")
        bfc.removeModelByPattern("w4")
        bfc.removeModelByPattern("w5")
        # replicate to crf,svm,md,db,mx,er,ner

        # tMDmERaCRF
        bfc.replicateBatch("crf",'../data/batches/b0517/debug_examplew3_aCRF.yaml')

        bfc = BatchFileCreator(
        filepath='../data/batches/b0517/debug_examplew3_aCRF.yaml',
        backupfolder='../data/batches/b0517/backup')
        bfc.replicateBatch("MedLine",'../data/batches/b0517/debug_examplew3_aCRF.yaml')
        bfc = BatchFileCreator(
        filepath='../data/batches/b0517/debug_examplew3_aCRF.yaml',
        backupfolder='../data/batches/b0517/backup')
        bfc.replicateBatch("NER",'../data/batches/b0517/debug_examplew3_tMDmNERaCRFw3.yaml')

        tMDmNERaCRF = BatchFileCreator(
        filepath='../data/batches/b0517/debug_examplew5_tMDmNERaCRFw3.yaml',
        backupfolder='../data/batches/b0517/backup')

        # tMDmNERaCRFw5 
        # transform all windows to w5
        tMDmNERaCRF.replicateBatch("w5",'../data/batches/b0517/debug_examplew3_tMDmNERaCRFw5.yaml')

        tMDmNERaCRFw5 = BatchFileCreator(
        filepath='../data/batches/b0517/debug_examplew3_tMDmNERaCRFw5.yaml',
        backupfolder='../data/batches/b0517/backup')

        # replicate into crf, svm, md, db, er and ner

        tMDmNERaCRF.replicateBatch("ER",'../data/batches/b0517/debug_examplew3_tMDmERaCRFw5.yaml')

        tMDmNERaCRF.replicateBatch("SVM",'../data/batches/b0517/debug_examplew3_tMDmERaSVMw5.yaml')

        tMDmNERaCRF.replicateBatch("mixed",'../data/batches/b0517/debug_examplew3_tMXmERaCRFw5.yaml')

        tMDmNERaCRF.replicateBatch("DrugBank",'../data/batches/b0517/debug_examplew3_tDBmERaCRFw5.yaml')


    def exampleMerge():
        bfc = BatchFileCreator(
            filepath='../data/batches/b0517/debug_merge.yaml',
            backupfolder='../data/batches/b0517/backup')
        bfc.mergeBatch(
            filepath2='../data/batches/b0517/debug_merge2.yaml',
            newname='../data/batches/b0517/debug_merge_res.yaml')

    def batch20180522():

        # generating
        bfc = BatchFileCreator(
            filepath='',
            backupfolder='')
       
        bfc.generate(
            root="d21-01",
            folder="../data/batches/b0522/",
            wordtypes=[
                "lemma",
                "lookup",
                "chunkGroup",
                "wordStructure"],
            windowtypes=[
                "lemma",
                "lookup",
                "chunkGroup",
                "wordStructure"],
            combinations=None)

       
        bfc.generate(
            root="d21-02",
            folder="../data/batches/b0522/",
            wordtypes=[
                "lemma",
                "lookup",
                "chunkGroup",
                "wordStructure2"],
            windowtypes=[
                "lemma",
                "lookup",
                "chunkGroup",
                "wordStructure2"],
            combinations=None)

       
        bfc.generate(
            root="d21-03",
            folder="../data/batches/b0522/",
            wordtypes=[
                "lemma",
                "lookup",
                "chunkGroup",
                "ortho"],
            windowtypes=[
                "lemma",
                "lookup",
                "chunkGroup",
                "ortho"],
            combinations=None)

        bfc.generate(
            root="d21-04",
            folder="../data/batches/b0522/",
            wordtypes=[
                "pos",
                "lookup",
                "chunkGroup",
                "ortho"],
            windowtypes=[
                "pos",
                "lookup",
                "chunkGroup",
                "ortho"],
            combinations=None)

        bfc.generate(
            root="d21-05",
            folder="../data/batches/b0522/",
            wordtypes=[
                "pos",
                "lemma",
                "lookup",
                "chunkGroup",
                "ortho"],
            windowtypes=[
                "pos",
                "lookup",
                "chunkGroup",
                "ortho"],
            combinations=None)

        bfc.generate(
            root="d21-06",
            folder="../data/batches/b0522/",
            wordtypes=[
                "pos",
                "lookup",
                "chunk",
                "ortho"],
            windowtypes=[
                "pos",
                "lookup",
                "chunk",
                "ortho"],
            combinations=None)


        bfc.generate(
            root="d21-07",
            folder="../data/batches/b0522/",
            wordtypes=[
                "ortho"],
            windowtypes=[
                "ortho"],
            combinations=None)

        bfc.generate(
            root="d21-08",
            folder="../data/batches/b0522/",
            wordtypes=[
                "lemma","ortho"],
            windowtypes=[
                "lemma","ortho"],
            combinations=None)

        bfc.generate(
            root="d21-09",
            folder="../data/batches/b0522/",
            wordtypes=[
                "pos","ortho"],
            windowtypes=[
                "pos","ortho"],
            combinations=None)

        # bfc.generate(
        #     root="d21-10",
        #     folder="../data/batches/b0522/",
        #     wordtypes=[
        #         "pos","ortho", "lemma","lookup","chunkGroup"],
        #     windowtypes=[
        #         "pos","ortho", "lemma","lookup","chunkGroup"],
        #     combinations="all3")

        bfc.generate(
            root="d21-11",
            folder="../data/batches/b0522/",
            wordtypes=[
                "pos","wordStructure", "lemma","lookup","chunkGroup"],
            windowtypes=[
                "pos","wordStructure", "word","lookup","chunkGroup"],
            combinations=None)

        bfc.generate(
            root="d21-12",
            folder="../data/batches/b0522/",
            wordtypes=[
                "pos","wordStructure", "lemma","lookup","chunkGroup"],
            windowtypes=[
                "pos","wordStructure","lookup","chunkGroup"],
            combinations=None)

        bfc.generate(
            root="d21-13",
            folder="../data/batches/b0522/",
            wordtypes=[
                "pos","wordStructure", "lemma","lookup","chunkGroup","prefix3","prefix4","prefix5",
                "suffix3","suffix4","suffix5"],
            windowtypes=[
                "pos","wordStructure","prefix3","prefix4","prefix5",
                "suffix3","suffix4","suffix5"],
            combinations="all3")


    def batch20180522good():

        # generating
        bfc = BatchFileCreator(
            filepath='',
            backupfolder='')
       
        bfc.generateMultiple(
            root="d22",
            folder="../data/batches/b0522/",
            combinations=None,
            pairs=[
                (    
                    ["lemma",
                    "lookup",
                    "chunkGroup",
                    "wordStructure"],
                    [
                    "lemma",
                    "lookup",
                    "chunkGroup",
                    "wordStructure"]
                ),
                (
                    [
                    "lemma",
                    "lookup",
                    "chunkGroup",
                    "wordStructure2"],
                    [
                    "lemma",
                    "lookup",
                    "chunkGroup",
                    "wordStructure2"]
                ),
                (
                    [
                    "pos","wordStructure", "lemma","lookup","chunkGroup"],[
                    "pos","wordStructure","lookup","chunkGroup"]
                ),
                (
                    [
                    "lemma",
                    "lookup",
                    "chunkGroup",
                    "ortho"],
                    [
                    "lemma",
                    "lookup",
                    "chunkGroup",
                    "ortho"]
                ),
                (
                    [
                    "pos",
                    "lookup",
                    "chunkGroup",
                    "ortho"],
                    [
                    "pos",
                    "lookup",
                    "chunkGroup",
                    "ortho"]
                ),
                (
                    [
                    "pos",
                    "lemma",
                    "lookup",
                    "chunkGroup",
                    "ortho"],
                    [
                    "pos",
                    "lookup",
                    "chunkGroup",
                    "ortho"]
                ),
                (
                    [
                    "pos",
                    "lookup",
                    "chunk",
                    "ortho"],
                    [
                    "pos",
                    "lookup",
                    "chunk",
                    "ortho"]
                ),
                (
                    [
                    "ortho"],
                    [
                    "ortho"]
                ),
                ([
                    "lemma","ortho"],
                    [
                    "lemma","ortho"]),
                ([
                    "pos","ortho"],
                    [
                    "pos","ortho"]),
                ([
                    "pos","wordStructure", "lemma","lookup","chunkGroup"],
                    [
                    "pos","wordStructure", "word","lookup","chunkGroup"]),
                ]
            )


        bfc.generate(
            root="d22b",
            folder="../data/batches/b0522/",
            wordtypes=[
                "pos","wordStructure","lookup"],
            windowtypes=[
                "pos","wordStructure","prefix3","prefix4","prefix5",
                "suffix3","suffix4","suffix5"],
            combinations="all3")


if __name__ == '__main__':

    # bfc = BatchFileCreator(
    #     filepath='../data/batches/d20tMDmNERaCRFw5.yaml',
    #     backupfolder='../data/batches/backup')
    # bfc.backup()
    # bfc.removeResults()


    # for root, dirs, files in os.walk('../data/batches/b0520', topdown = False):
    #     for name in files:
    #         if os.path.basename(name).endswith(".yaml"):
                
    #             bfc = BatchFileCreator(
    #             filepath=os.path.join(root, name),
    #             backupfolder=os.path.join(root, 'backup')
    #             )

    #             bfc.backup()
                # bfc.removeResultsOld()
                # bfc.resetTotalTime()
                # bfc.resetStatus()
                # bfc.removeAccuracies()
    #             #bfc.removeResults()
                

    # bfc = BatchFileCreator(
    #     filepath='../data/batches/b0517/debug_batch7.yaml',
    #     backupfolder='../data/batches/b0517/backup')
    # bfc.backup()
    #bfc.removeResults()
    # bfc.removeResultsOld()
    # bfc.resetTotalTime()
    # bfc.resetStatus()
    # bfc.removeAccuracies()
    #bfc.removeModelByName('mod062')
    #bfc.removeModelByPattern("w3")
    #bfc.removeModelByPattern("chunkGroup")
    #bfc.removeModelByPattern("lookup")
    #bfc.removeModelByPattern("hasDigits")
    #bfc.removeModelByPattern("lemma")
    #bfc.replicateBatch("w5",'../data/batches/b0517/debug_batch7_crf.yaml')
    #bfc.replicateBatch("w2",'../data/batches/b0517/debug_batch7_crf.yaml')
    #bfc.forceBatch("w1",'../data/batches/b0517/debug_batch7_crf.yaml')

    # merging
    # bfc = BatchFileCreator(
    #     filepath='../data/batches/b0517/debug_merge.yaml',
    #     backupfolder='../data/batches/b0517/backup')
    # bfc.mergeBatch(
    #     filepath2='../data/batches/b0517/debug_merge2.yaml',
    #     newname = '../data/batches/b0517/debug_merge_res.yaml'
    #     )

    # generating
    # bfc = BatchFileCreator(
    #     filepath='',
    #     backupfolder='')
    # bfc.generateSingleBatchFile()

    # bfc.generateSingleBatchFile(
    #     filepath="../data/batches/b0520/example_wasymmetric.yaml",
    #     typewindow="-4:+2")

    # bfc.generateSingleBatchFile(
    #     filepath="../data/batches/b0520/example_types.yaml",
    #     wordtypes=["ortho","pos"],
    #     windowtypes=["wordStructure","lookup"],
    #     typewindow="symmetric", window="4")

    # bfc.generateSingleBatchFile(
    #     filepath="../data/batches/b0520/example_types_asymwin.yaml",
    #     wordtypes=["ortho","pos"],
    #     windowtypes=["ortho","lookup"],
    #     typewindow="-5:+2")

    # bfc.generateSingleBatchFile(
    #     filepath="../data/batches/b0520/example_types_random.yaml",
    #     wordtypes=["chunk","lemma","pos"],
    #     windowtypes=["chunk","lookup"],
    #     typewindow="symmetric",
    #     window="4",
    #     combinations="random5")


    # bfc.generateSingleBatchFile(
    #     filepath="../data/batches/b0520/example_types_combi.yaml",
    #     wordtypes=["lemma","pos"],
    #     windowtypes=["chunk","lookup","wordStructure"],
    #     typewindow="-5:+2",
    #     combinations="all2")
    

    # bfc.generate(
    #     root="d21",
    #     folder="../data/batches/b0521/",
    #     wordtypes=["lemma","pos","chunkGroup","lookup","wordStructure"],
    #     windowtypes=["pos","lookup","wordStructure"],
    #     combinations="all3")
    
    BatchFileCreator.batch20180522good()