import os, sys
import yaml
import pprint
import traceback
import jinja2
from jinja2 import Template
import subprocess


class Reporting():

    def __init__(self, basefolder):
        self.folder = basefolder
        self.models = []
        self.featuresLists = ["fafter1","fafter2","fafter3","fafter4","fafter5","fbefore1","fbefore2","fbefore3","fbefore4","fbefore5","fcurrent","topcountfeatures","windowfeatures","wordfeatures","sentencefeatures"]

    def gatherData(self):

        print("gathering data")

        for root, dirs, files in os.walk(self.folder, topdown = False):
            for name in files:
                if os.path.basename(name).endswith(".yaml"):
                    self.parseBatchFile(os.path.join(root, name))

    def parseBatchFile(self, filepath):

        batch = yaml.load(open(filepath,'r'))

        # for each model get name, type, score
        if "models" not in batch.keys():
            return

        for name,params in batch["models"].items():
            try:

                modeltype = "ER"
                if "modeltype" in params.keys():
                    modeltype = params["modeltype"]

                # scores
                bestf1 = 0.0
                prec = 0
                rec = 0
                bestmf1 = 0.0
                mprec = 0
                mrec = 0
                if "results" in params.keys() and params["results"] is not None:
                    
                    if modeltype in ["DDI","BDDI"]:
                        try:
                            print(params["results"]["scores"])
                        except:
                            pass

                    for runname,values in params["results"].items():
                        if "scores" in values.keys():
                            # if modeltype=="ER":
                            #     f1 = values["scores"]["exact"]["F1"]
                            #     if float(f1) > bestf1:
                            #         bestf1 = float(f1)
                            #         prec = values["scores"]["exact"]["prec"]
                            #         rec = values["scores"]["exact"]["recall"]
                            # elif modeltype=="NER":
                            #     f1 = values["scores"]["macro"]["F1"]
                            #     if float(f1) > bestf1:
                            #         bestf1 = float(f1)
                            #         prec = values["scores"]["macro"]["P"]
                            #         rec = values["scores"]["macro"]["R"]
                            if modeltype=="ER":
                                f1 = values["scores"]["exact"]["F1"]
                                if float(f1) > bestf1:
                                    bestf1 = float(f1)
                                    prec = values["scores"]["exact"]["prec"]
                                    rec = values["scores"]["exact"]["recall"]
                                    bestmf1 = float(values["scores"]["macro"]["F1"])
                                    mprec = values["scores"]["macro"]["P"]
                                    mrec = values["scores"]["macro"]["R"]
                            elif modeltype=="NER":
                                if "macro" in values["scores"].keys():
                                    mf1 = values["scores"]["macro"]["F1"]
                                    if float(mf1) > bestmf1:
                                        bestf1 = float(values["scores"]["exact"]["F1"])
                                        prec = values["scores"]["exact"]["prec"]
                                        rec = values["scores"]["exact"]["recall"]
                                        bestmf1 = float(mf1)
                                        mprec = values["scores"]["macro"]["P"]
                                        mrec = values["scores"]["macro"]["R"]

                                else:
                                    print(values["scores"].keys())

                            elif modeltype=="DDI" :
                                if "macro" in values["scores"].keys():
                                    mf1 = values["scores"]["macro"]["F1"].replace(",",".")
                                    if float(mf1) > bestmf1:
                                        bestf1 = float(values["scores"]["partial"]["F1"].replace(",","."))
                                        prec = values["scores"]["partial"]["prec"]
                                        rec = values["scores"]["partial"]["recall"]
                                        bestmf1 = float(mf1.replace(",","."))
                                        mprec = values["scores"]["macro"]["P"]
                                        mrec = values["scores"]["macro"]["R"]
                                
                                else:
                                    print(values["scores"].keys())
                            elif modeltype=="BDDI":
                                if "partial" in values["scores"].keys():
                                    f1 = values["scores"]["partial"]["F1"].replace(",",".")
                                    if float(f1) > bestf1:
                                        bestf1 = float(values["scores"]["partial"]["F1"].replace(",","."))
                                        prec = values["scores"]["partial"]["prec"]
                                        rec = values["scores"]["partial"]["recall"]
                                        bestmf1 = float(values["scores"]["macro"]["F1"].replace(",","."))
                                        mprec = values["scores"]["macro"]["P"]
                                        mrec = values["scores"]["macro"]["R"]
                                
                                else:
                                    print(values["scores"].keys())
                            

                session = ""
                if "session" in batch.keys():
                    session = batch["session"]


                algorithm="svm"
                if "algorithm" in params.keys() and \
                    params["algorithm"] != ""  and \
                    len(params["algorithm"])>0:
                    algorithm = params["algorithm"]

                numfeatures = self.countFeatures(params, modeltype)
                typefeatures = self.typeFeatures(params, modeltype)

                window = self.countWindow(params, modeltype)

                completename = session + name
                completename = completename.replace("_","\\_")

                newmodel = { 
                    "name": completename,
                    "type": modeltype,
                    "algorithm": algorithm.upper(),
                    "numfeatures" : numfeatures,
                    "typefeatures" : typefeatures,
                    "window": window,
                    "prec": prec,
                    "rec": rec,
                    "f1": bestf1,
                    "mprec": mprec,
                    "mrec": mrec,
                    "mf1": bestmf1,
                    }

                self.models.append(newmodel)
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60) 


    def countWindow(self, params, modeltype):
        wbefore = ""
        wafter = ""

        if modeltype in ["DDI", "BDDI"]:
            return "-3:+3"

        if "fbefore5" in params.keys() and len(params["fbefore5"])>0:
            wbefore = "-5"
        elif "fbefore4" in params.keys() and len(params["fbefore4"])>0:
            wbefore = "-4"
        elif "fbefore3" in params.keys() and len(params["fbefore3"])>0:
            wbefore = "-3"
        elif "fbefore2" in params.keys() and len(params["fbefore2"])>0:
            wbefore = "-2"
        elif "fbefore1" in params.keys() and len(params["fbefore1"])>0:
            wbefore = "-1"
        
        if "fafter5" in params.keys() and len(params["fafter5"])>0:
            wafter = "+5"
        elif "fafter4" in params.keys() and len(params["fafter4"])>0:
            wafter = "+4"
        elif "fafter3" in params.keys() and len(params["fafter3"])>0:
            wafter = "+3"
        elif "fafter2" in params.keys() and len(params["fafter2"])>0:
            wafter = "+2"
        elif "fafter1" in params.keys() and len(params["fafter1"])>0:
            wafter = "+1"


        return wbefore + ":" + wafter

    def countFeatures(self,params, modeltype):

        count = 0

        for lf in self.featuresLists:
            try:
                for elem in params[lf]:
                    if lf in ["wordfeatures","windowfeatures"]:
                        count+=7

                    count += 1
            except:
                pass
        return count

    def getTypeFeatures(self, featurelist, feature):

        if feature in [
            
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
            ]:
            if featurelist.find("ort")==-1:
                featurelist += "ort" + ","
        elif feature in ["lemma"]:
            if featurelist.find("lm")==-1:
                featurelist += "lm" + ","
        elif feature in ["pos"]:
            if featurelist.find("pos")==-1:
                featurelist += "pos" + ","
        elif feature in ["lookup"]:
            if featurelist.find("lo")==-1:
                featurelist += "lo" + ","
        elif feature in ["w2v"]:
            if featurelist.find("w2v")==-1:
                featurelist += "w2v" + ","
        elif feature in ["chunk"]:
            if featurelist.find("ch")==-1:
                featurelist += "ch" + ","
        elif feature in ["chunkGroup"]:
            if featurelist.find("chg")==-1:
                featurelist += "chg" + ","
        elif feature in ["vb_count", "cc_count", "md_count", "dt_count", "negationLemma"]:
            if featurelist.find("sent")==-1:
                featurelist += "sent" + ","
        elif feature in ["ttrigram"]:
            if featurelist.find("tri")==-1:
                featurelist += "tri" + ","
        elif feature in ["tword"]:
            if featurelist.find("tw")==-1:
                featurelist += "tw" + ","
        elif feature in ["tlemma"]:
            if featurelist.find("tl")==-1:
                featurelist += "tl" + ","
        elif feature in ["tpos"]:
            if featurelist.find("tp")==-1:
                featurelist += "tp" + ","


        return featurelist

    def typeFeatures(self,params, modeltype):
        
        typefeatures = ""
        wordfeatures = ""
        windowfeatures = ""
        topcountfeatures = ""
        sentencefeatures = ""

        for lf in self.featuresLists:
            try:
                if lf == "fcurrent" :
                    for feature in params[lf]:
                        wordfeatures = self.getTypeFeatures(wordfeatures, feature)

                elif lf == "wordfeatures":
                    for feature in params[lf]:
                        wordfeatures = self.getTypeFeatures(wordfeatures,+feature)

                elif lf == "topcountfeatures":
                    for feature in params[lf]:
                        topcountfeatures = self.getTypeFeatures(topcountfeatures, "t"+feature)
               
                elif lf == "sentencefeatures":
                    for feature in params[lf]:
                        sentencefeatures = self.getTypeFeatures(sentencefeatures, feature)
                else:
                    for feature in params[lf]:
                        windowfeatures = self.getTypeFeatures(windowfeatures, feature)

            except:
                pass
        return wordfeatures[:-1] + "+" + windowfeatures[:-1] + "+" + topcountfeatures[:-1] + "+" + sentencefeatures[:-1]       


    def sort(self):

        print("sorting")
        self.models = sorted(self.models, key=lambda x: x["f1"], reverse=True)

    def sortmf1(self):

        print("sorting")
        self.models = sorted(self.models, key=lambda x: x["mf1"], reverse=True)



    def reportCommandline(self):
        #pprint.pprint(self.models)
        pprint.pprint([model for model in self.models if model["f1"]>-10.0])

    def reportTable(self, ttype=None):

        print("report table")

        if ttype is None:
            data = [model for model in self.models if model["f1"]>-10.0]
        elif ttype=="ER":
            self.sort()
            data = [model for model in self.models if model["type"]==ttype and model["f1"]>-10.0]
        elif ttype=="NER":
            self.sortmf1()
            data = [model for model in self.models if model["type"]==ttype and model["mf1"]>-10.0]
        elif ttype=="DDI" or ttype=="BDDI":
            self.sortmf1()
            data = [model for model in self.models if model["type"]==ttype and model["mf1"]>-10.0]

        with open('templates/scores.j2','r') as templatefile:
            template = Template(templatefile.read())

            renderer_template = template.render(results=data)

            build_d='../data/reports/'
            if not os.path.exists(build_d):  # create the build directory if not existing
                os.makedirs(build_d)

            out_file = build_d +"_"+str(ttype)+"_"+ "scores"
            with open(out_file+".tex", "w") as f:  # saves tex_code to outpout file
                f.write(renderer_template)  

            process = subprocess.Popen([
                'pdflatex -output-directory {} {}'.format(
                    os.path.realpath(build_d), 
                    os.path.realpath(out_file + '.tex'))
                ], 
                shell=True,
                stdout=subprocess.PIPE)
            out, err = process.communicate()
            

    def reportTable2(self, ttype=None):

        print("report table")

        if ttype is None:
            data = [model for model in self.models if model["f1"]>-10.0]
        elif ttype=="ER":
            self.sort()
            data = [model for model in self.models if model["type"]==ttype and model["f1"]>-10.0]
        elif ttype=="NER":
            self.sortmf1()
            data = [model for model in self.models if model["type"]==ttype and model["mf1"]>-10.0]
        elif ttype=="DDI" or ttype=="BDDI":
            self.sortmf1()
            data = [model for model in self.models if model["type"]==ttype and model["mf1"]>-10.0]

        with open('templates/score2.j2','r') as templatefile:
            template = Template(templatefile.read())

            renderer_template = template.render(results=data)

            build_d='../data/reports/'
            if not os.path.exists(build_d):  # create the build directory if not existing
                os.makedirs(build_d)

            out_file = build_d +"_"+str(ttype)+"_"+ "scores2"
            with open(out_file+".tex", "w") as f:  # saves tex_code to outpout file
                f.write(renderer_template)  

            process = subprocess.Popen([
                'pdflatex -output-directory {} {}'.format(
                    os.path.realpath(build_d), 
                    os.path.realpath(out_file + '.tex'))
                ], 
                shell=True,
                stdout=subprocess.PIPE)
            out, err = process.communicate()


if __name__ == '__main__':
    rp = Reporting(basefolder='../data/batches/')
    rp.gatherData()
    rp.sort()
    #rp.reportCommandline()
    #rp.reportTable(ttype="ER")
    rp.reportTable(ttype="NER")
    rp.reportTable(ttype="DDI")
    #rp.reportTable(ttype="BDDI")
    rp.reportTable2(ttype="NER")
    rp.reportTable2(ttype="DDI")
    