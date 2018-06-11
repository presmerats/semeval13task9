import lxml
import os, sys
from xml.dom.minidom import parse, parseString
from bs4 import BeautifulSoup
import pprint
import nltk
import re
import copy
import pickle
import json
import numpy as np

from FreelingFeatures import FreelingFeatures
from BIOFeatures import BIOTagger, BIOTagger2
from CustomFeatures import MyFeatures
from ChunkFeatures import MyChunker
from DrugBankLookup import DrugBankLookup
from W2VFeatures import W2VFeatures


class FeatureExtraction():
    """
    datastructure:
    dict
    { sentenceid :
        {
            text
            entities: [entity{},..]
            pairs: [pair]
                features:
                tokens
                POS
                lemma
                parsing
                ...and other...
        }}

    
    """
    data = []
    files_path = ""
    
    
    def __init__(self, path, customfeatures = None, algoformat="SVM", targetformat="ER"):
        self.files_path = path
        self.data = []
        self.freeling = FreelingFeatures()
        self.biotagger = BIOTagger()
        self.biotagger2 = BIOTagger2()
        self.custom = MyFeatures()
        self.chunker = MyChunker()
        self.drugbank = DrugBankLookup("../data/Dictionaries/drugbank.pkl", pickled=True)
        self.algoformat = algoformat
        self.targetformat = targetformat
        #self.wordembedding = W2VFeatures()

        if not customfeatures is None:
            self.custom = customfeatures
        self.verification=False
        self.weights = None
        

    def __del__(self):
        print("freeing FeatureExtraction instance...")
        del self.chunker 
        del self.data 
        del self.drugbank 
        del self.custom 
        del self.biotagger2
        del self.biotagger
        del self.freeling 

        self.chunker = None
        self.data = None
        self.drugbank = None
        self.custom = None
        self.biotagger2 = None
        self.biotagger = None
        self.freeling = None



    def load(self):
        """
            walks recursively the file system beginning at self.files_path
            then parses each file tino the corresponding elem in the datastruct
        """
        #os.chdir(self.files_path)
        for root, dirs, files in os.walk(self.files_path, topdown = False):
            for name in files:
                if os.path.basename(name).endswith(".xml"):
                    self.parseXml(os.path.join(root, name))
                    

    def parseXml(self,filepath):
        """
            Parses one single xml file, saving data to the self.data datastruct
        """
        with open(filepath, 'r') as f:
            #parse
            dom = parse(f)

            # traverse and save each elem to self.data
            for node in dom.getElementsByTagName('sentence'):
                sentence = {}
                # get id
                if node.hasAttribute('id'):
                    sentence['id'] = node.getAttribute('id')
                # get text
                if node.hasAttribute('text'):
                    sentence['text'] = node.getAttribute('text')

                sentence['entities'] =[]
                sentence['pairs'] =[]
                    
                # get child entities
                #    get charOffset, id, text, type
                for child in node.getElementsByTagName('entity'):
                    entity ={}

                    if child.hasAttribute('id'):
                        entity['id'] = child.getAttribute('id')
                    if child.hasAttribute('charOffset'):
                        entity['charOffset'] = child.getAttribute('charOffset')
                    if child.hasAttribute('text'):
                        entity['text'] = child.getAttribute('text')
                    if child.hasAttribute('type'):
                        entity['type'] = child.getAttribute('type')                    

                    sentence['entities'].append(entity)
                        
                
                # get child pairs
                #    get ddi, e1, e2, id
                for child in node.getElementsByTagName('pair'):
                    pair ={}

                    if child.hasAttribute('ddi'):
                        pair['ddi'] = child.getAttribute('ddi')
                    if child.hasAttribute('e1'):
                        pair['e1'] = child.getAttribute('e1')
                    if child.hasAttribute('e2'):
                        pair['e2'] = child.getAttribute('e2')
                    if child.hasAttribute('id'):
                        pair['id'] = child.getAttribute('id')
                    if child.hasAttribute('type'):
                        pair['type'] = child.getAttribute('type')
                    
                    sentence['pairs'].append(pair)                
                
                self.data.append(sentence)
        
    def getMaxWordLength(self):

        wlength = 0
        for element in self.data:
            for word in element["text"].split():
                l = len(word)
                if l > wlength:
                    wlength=l

        return wlength


            
    def extractFeatures(self, limit=None):
        """
            This function calls different plugins to extract:
            -tokens
            -pos
            -lemmas
            -parsing...
        """
        
        if not limit is None:
            self.data = self.data[:limit]

        # normalization of lengths
        mwl = self.getMaxWordLength()

            
        for element in self.data:
            # clean
            self.biotagger2.prepareElementBIOOffsets(element)
            self.biotagger2.cleanElement2(element)
            
            # Tokenize & POS
            element["features"] = self.freeling.processTextOriginal(element['text'], element['id'])
                  
            # BIO tag
            if self.verification:
                self.biotagger2.verifyOffsets(element)
            self.biotagger2.bIOtag(element)
            if self.verification:
                self.biotagger2.verifyBIOtag(element)
            
            # Chunking (nltk)
            self.chunker.chunkElement(element)
            
            # DrugBank Lookup
            self.drugbank.addLookupFeature(element)
            
            # Word embedding - Word 2 Vector and it's cluster using Kmean
            #self.wordembedding.addW2VFeature(element)
            #self.wordembedding.addW2VClusterFeature(element)
            
            # Custom features & and final reorganization of features
            self.custom.addFeatures(element, maxwordlength=mwl)
            
        # Final global verifications
        if self.verification:
            self.numtokenerrors =self.biotagger2.verifyGlobalOffsets(self.data)
            self.numbioerrors = self.biotagger2.verifyGlobalBIOtags(self.data)
        self.prepareFeatures()



        
    def prepareFeatures(self):
        """
            Format features in a dict format for later processing correctly

            format svm
                a list of word features dict
            format crf
                a list of list of sentence word features dict
            
           
        """
        features = []
        if self.algoformat.lower() in ["svm","lwsvm","lsvm"]:
            for element in self.data:
                for featuresdict in element["features"]:
                    # create a dict of each tuple
                    # 0 is the word, 6 is the class, the others are features
                    #newdict = {}
                    #for fi in range(len(word)):
                    #    newdict[str(fi)] = word[fi]
                    features.append(featuresdict)

        elif self.algoformat.lower() == "crf":
            for element in self.data:
                sentencelist = []
                for featuresdict in element["features"]:
                    sentencelist.append(featuresdict)  

                features.append(sentencelist)      
                
        # datadict contains features and labels
        self.datadict = features
        #pprint.pprint(features[:1])

        # X contains only features
        # Y contains only labels
        
        self.X, self.Y = self.transformToXY(self.datadict)


    def transformToXY(self,data,n=None):
        """
            target is just BIO tag
            
            Transform data.json form into a suitable dictionary
            Accepts test sets without the target/class variable
            (named 'biotag' in the usually generated dictionary)

        """

        print("transformToXY",self.algoformat)
        if n is None:
            n=len(data)

        if self.algoformat.lower()=="lsvm" or \
           self.algoformat.lower()=="svm":
            targets = [self.filterTarget(e)
                       for e in data[:n]]

            features = []
            for wdict in data[:n]:
                aux_dict = self.filterERfeatures(copy.deepcopy(wdict))
                
                features.append(aux_dict)

            return features, targets

        elif self.algoformat.lower()=="lwsvm":
            targets = [self.filterTarget(e)
                       for e in data[:n]]

            features = []
            for wdict in data[:n]:
                aux_dict = self.filterERfeatures(copy.deepcopy(wdict))
                
                features.append(aux_dict)

            print("featureextraction","algoformat",self.algoformat,len(features),len(targets))
        
            return features, targets

        else:
            # CRF format has a list of features/target per sentence
            targets = [[self.filterTarget(word) 
                        for word in sent]
                       for sent in data[:n]]

            features = []
            for sent in data[:n]:
                wordflist = []
                for wdict in sent:
                    aux_dict = self.filterERfeatures(copy.deepcopy(wdict))
                
                    wordflist.append(aux_dict)

                features.append(wordflist)

            return features, targets

    def filterTarget(self, e):
        if self.targetformat=="ER":
            return e['biotag'] if 'biotag' in e.keys() else ''
        else:
            return e['biotag'] + "-" + e['drugtype'] if 'biotag' in e.keys() and 'drugtype' in e.keys() else ''

    def filterERfeatures(self, aux_dict):

        if 'biotag' in aux_dict.keys():
            aux_dict.pop('biotag',None)
        if 'drugtype' in aux_dict.keys():
            aux_dict.pop('drugtype',None)
        if 'offsetend' in aux_dict.keys():
            aux_dict.pop('offsetend',None)
        if 'offsetstart' in aux_dict.keys():
            aux_dict.pop('offsetstart',None)
        if 'sentenceid' in aux_dict.keys():
            aux_dict.pop('sentenceid',None)

        return aux_dict

    def weightsVector(self):
        # only for NER models and lwsvm algorithm

        # weights vector
        targets = self.Y
        weights=np.ones(len(targets))
        print("len targets",len(targets))
        print("len weights",len(weights))

        for i in range(len(targets)):
            target = targets[i]
            if not target.startswith("O"):
                # weigth is 7.0
                weights[i] *= 7.0


        self.weights = weights

    
    def save(self, filename='data.json'):
        """
            saves the self.data to disk for further processing
            options:
                - json, yaml, xml, csv...
                - 
        """
        with open(filename,'w') as f:            
            json.dump(self.datadict,f)
                
        
        
    def printdata(self, n=100):
        if n>=len(self.data):
            n=len(self.data)
        pprint.pprint(self.data[:n])
        

    def printerrors(self):
        print("biotag errors count:")
        print(self.numbioerrors)
        try:
            print("token offset errors count:")
            print(self.numtokenerrors)
            self.biotagger2.printErrorOffsets(self.data)
        except:
            pass
        
    
    def printerrorsBIO(self, n=None):
        if n is None or n>=len(self.data):
            n=len(self.data)
            
        pprint.pprint([elem for elem in self.data[:n] if not elem["BIOok"] ])

        
        
        
if __name__ == '__main__':
        
    #os.chdir(os.path.dirname(sys.argv[0]))
    print(os.getcwd())
    fe =FeatureExtraction("./data/LaboCase/Train")
    fe.load()
    fe.extractFeatures()
    print(fe.numerrors)
    #fe.printdata(n=300)
    fe.printerrors()
    fe.save()  