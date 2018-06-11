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
import traceback
from time import time

from FreelingFeatures import FreelingFeatures
from BIOFeatures import BIOTagger, BIOTagger2
from CustomFeaturesDDI import MyFeaturesDDI
from ChunkFeatures import MyChunker
from DrugBankLookup import DrugBankLookup
from W2VFeatures import W2VFeatures
from FeatureExtraction import FeatureExtraction




class FeatureExtractionDDI(FeatureExtraction):
    """

    """
    data = []
    files_path = ""
    
    
    def __init__(self, path, customfeatures = None, algoformat="SVM", targetformat="BDDI"):
        self.files_path = path
        self.data = []
        self.freeling = FreelingFeatures()
        #self.biotagger = BIOTagger()
        self.biotagger = None
        self.biotagger2 = BIOTagger2()
        self.custom = MyFeaturesDDI()
        self.chunker = MyChunker()
        self.biotagger = None
        #self.drugbank = DrugBankLookup("../data/Dictionaries/drugbank.pkl", pickled=True)
        self.drugbank = None
        self.algoformat = algoformat[:3]
        self.targetformat = targetformat
        #self.wordembedding = W2VFeatures()

        if not customfeatures is None:
            self.custom = customfeatures
        self.verification=False

        self.errorlog = []
        self.numerrors = 0

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
            "wordStructure2",
            "wordStructureLong",
            "wordStructureLong2",
            ]
        
        

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

    def registerError(self, message):
        self.errorlog.append(message)
        self.numerrors+=1

    # def extractFeatures(self, limit=None):
    #     """
    #         For each sentence
    #             For each pair
    #                 features: numwords, num V postags, list of postags, words?
    #                 target: ddi(1-0), dditype(effect,etc)
    #                 ->[sentid, ]


    #     """
        
    #     if not limit is None:
    #         self.data = self.data[:limit]

    #     # normalization of lengths
    #     mwl = self.getMaxWordLength()


    #     # parse tree
    #     parser = Parser()
    #     #dparser = MaltParser('../data/grammars/maltparser-1.8.1/', 'engmalt.linear-1.7.mco')
    
    #     #error log
    #     self.errorlog=[]
    #     self.numerrors=0

    #     for element in self.data:

    #         # prepare entity offsets 
    #         self.biotagger2.prepareElementBIOOffsets(element) # for later usage we keep this one
    #         self.biotagger2.prepareElementDDIOffsets(element) # this gives us the offsets of entities
            
    #         # clean (correct for \n\r)
    #         self.biotagger2.cleanElement2(element)

    #         # tokenize & POS
    #         element["features"] = self.freeling.processText(element['text'], element['id'])
            
    #         # BIO tag
    #         # if self.verification:
    #         #     self.biotagger2.verifyOffsets(element)
    #         # self.biotagger2.bIOtag(element)
    #         # if self.verification:
    #         #     self.biotagger2.verifyBIOtag(element)
            
    #         # Chunking (nltk)
    #         self.chunker.chunkElement(element)
            
    #         # DrugBank Lookup
    #         #self.drugbank.addLookupFeature(element)
            
    #         # Word embedding - Word 2 Vector and it's cluster using Kmean
    #         #self.wordembedding.addW2VFeature(element)
    #         #self.wordembedding.addW2VClusterFeature(element)
            
    #         # parse tree
    #         try:
    #             element["cfgtree"] = parser.parse(element["text"])
    #         except Exception:
    #             self.registerError(traceback.format_stack())
    #             element["cfgtree"] = []
    #         # dependency tree?
        

    #         # get pairs and their features
    #         element["ddifeatureset"] = []
    #         for pair in element['pairs']:
    #             pair_features={}

    #             pairid = pair['id']
    #             # get e1 id, e2 id  , ddi, type, 
    #             e1id = pair['e1']
    #             e2id = pair['e2']
    #             # pair eid's with offsets
    #             # get e1 offset, e2 offset (from entity list)
    #             pair['e1offset'] = [ (offsetgroup[0],offsetgroup[2]) 
    #                                  for offsetgroup in element['ddioffsets']
    #                                  if offsetgroup[4] == e1id ]
    #             pair['e2offset'] = [ (offsetgroup[0],offsetgroup[2]) 
    #                                  for offsetgroup in element['ddioffsets']
    #                                  if offsetgroup[4] == e2id ]

    #             # get target values
    #             targetddi = pair['ddi']
    #             targettype = "null"
    #             if "type" in pair.keys():
    #                 targettype = pair['type'] # paper talks about it , but training has not that

    #             # this are the metadata and the labels/classes/target values
    #             pair_features = {
    #                 'sentenceid': element['id'],
    #                 'pid': pairid,
    #                 'e1id': e1id,
    #                 'e2id': e2id,
    #                 'ddi': targetddi,
    #                 'type': targettype
    #             }

    #             # now we collect the features of this pair
    #             # get features -> CustomFeatures.py?
    #                 # remove e1 and e2 from the sentence (using the offsets)
    #                 # split into before, between, after?
    #                 # counts of words, lemmas, pos (vrb, determiners, NN) 
    #                 # sequences? sequence of pos-tags, of lemmas, of words
    #                 # 
            
    #             # shortest path from CFG parse tree
    #             e1 = self.getWordFromOffset(element,pair['e1offset'][0][0])
    #             e2 = self.getWordFromOffset(element,pair['e2offset'][0][0])
                
    #             pair_features["shortestpathCFG"] = self.shortestPath(
    #                 element["text"],
    #                 [ w['word'] for w in element["features"]],
    #                 element["cfgtree"],
    #                 e1,
    #                 e2
    #                 )
                
    #             # Custom features & and final reorganization of features
    #             self.custom.addFeatures(pair_features, element)

    #             element["ddifeatureset"].append(pair_features)
            
    #     # Final global verifications
    #     if self.verification:
    #         self.numtokenerrors = 0 # probably verify something like number of entity pairs?
    #         # self.numtokenerrors =self.biotagger2.verifyGlobalOffsets(self.data)
    #         # self.numbioerrors = self.biotagger2.verifyGlobalBIOtags(self.data)
    #     self.prepareFeatures()

    
    def extractFeaturesNG(self, limit=None):
        """
            For each sentence
                For each pair
                    features: numwords, num V postags, list of postags, words?
                    target: ddi(1-0), dditype(effect,etc)
                    ->[sentid, ]


        """
        
        if not limit is None:
            self.data = self.data[:limit]

        # normalization of lengths
        mwl = self.getMaxWordLength()

        # freeling dependency parsing
        # sentences = [element["text"] for element in self.data]
        # starttime = time()
        # deptrees = self.freeling.processSentences(sentences)
        # endtime = time()
        # print(endtime - starttime)
        # print(len(sentences))
        # print(len(deptrees))

            

        jj=-1
        for element in self.data:
            jj+=1

            # prepare entity offsets 
            self.biotagger2.prepareElementBIOOffsets(element) # for later usage we keep this one
            self.biotagger2.prepareElementDDIOffsets(element) # this gives us the offsets of entities
            
            # clean (correct for \n\r)
            self.biotagger2.cleanElement2(element)

            # tokenize & POS
            element["features"],deptrees = self.freeling.processText(element['text'], element['id'])

            element["deptree"]=[]
            # for handling deptree one by one (correct displacement due to missing )
            if len(deptrees)>0:
                element["deptree"]=deptrees[0]
            
            # # for handling deptree in block  
            # if jj < len(deptrees):
            #     element["deptree"] = deptrees[jj]   
            
            # BIO tag
            # if self.verification:
            #     self.biotagger2.verifyOffsets(element)
            # self.biotagger2.bIOtag(element)
            # if self.verification:
            #     self.biotagger2.verifyBIOtag(element)
            
            # Chunking (nltk)
            self.chunker.chunkElement(element)
            
            # DrugBank Lookup
            #self.drugbank.addLookupFeature(element)
            
            # Word embedding - Word 2 Vector and it's cluster using Kmean
            #self.wordembedding.addW2VFeature(element)
            #self.wordembedding.addW2VClusterFeature(element)
            


            # get pairs and their features
            element["ddifeatureset"] = []
            for pair in element['pairs']:
                pair_features={}

                pairid = pair['id']
                # get e1 id, e2 id  , ddi, type, 
                e1id = pair['e1']
                e2id = pair['e2']
                # pair eid's with offsets
                # get e1 offset, e2 offset (from entity list)
                pair['e1offset'] = [ (offsetgroup[0],offsetgroup[2]) 
                                     for offsetgroup in element['ddioffsets']
                                     if offsetgroup[4] == e1id ]
                pair['e2offset'] = [ (offsetgroup[0],offsetgroup[2]) 
                                     for offsetgroup in element['ddioffsets']
                                     if offsetgroup[4] == e2id ]

                # get target values
                targetddi = pair['ddi']
                targettype = "null"
                if "type" in pair.keys():
                    targettype = pair['type'] # paper talks about it , but training has not that

                # this are the metadata and the labels/classes/target values
                pair_features = {
                    'sentenceid': element['id'],
                    'pid': pairid,
                    'e1id': e1id,
                    'e2id': e2id,
                    'ddi': targetddi,
                    'type': targettype
                }

                # now we collect the features of this pair
                # get features -> CustomFeatures.py?
                    # remove e1 and e2 from the sentence (using the offsets)
                    # split into before, between, after?
                    # counts of words, lemmas, pos (vrb, determiners, NN) 
                    # sequences? sequence of pos-tags, of lemmas, of words
                    # 
            
                # shortest path from CFG parse tree
                e1 = self.getWordFromOffset(element,pair['e1offset'][0][0])
                e2 = self.getWordFromOffset(element,pair['e2offset'][0][0])
                
                # pair_features["shortestpathCFG"] = self.shortestPath(
                #     element["text"],
                #     [ w['word'] for w in element["features"]],
                #     element["cfgtree"],
                #     e1,
                #     e2
                #     )

                #print(element["deptree"])
                pair_features["shortestpathDep"] = self.shortestPathDep(
                    element["deptree"],
                    e1,
                    e2
                    )

                pair_features["trigrams"] = self.getTrigrams(pair_features["shortestpathDep"])

                #print(pair_features["shortestpathDep"])

                
                # Custom features & and final reorganization of features
                self.custom.addFeatures(pair_features, element)

                element["ddifeatureset"].append(pair_features)
            


        # Final global verifications
        if self.verification:
            self.numtokenerrors = 0 # probably verify something like number of entity pairs?
            # self.numtokenerrors =self.biotagger2.verifyGlobalOffsets(self.data)
            # self.numbioerrors = self.biotagger2.verifyGlobalBIOtags(self.data)
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
        for element in self.data:
            features.extend(element["ddifeatureset"])      
                
        # datadict contains features and labels
        self.datadict = features
        #pprint.pprint(features[:1])

        # X contains only features
        # Y contains only labels
        try:
            self.X, self.Y = self.transformToXY(self.datadict)
        except:
            self.X = None
            self.Y = None

    def transformToXY(self,data,n=None):
        """
            target is just BIO tag
            
            Transform data.json form into a suitable dictionary
            Accepts test sets without the target/class variable
            (named 'biotag' in the usually generated dictionary)

        """
        if n is None:
            n=len(data)

        targets = [self.filterTarget(e)
                   for e in data[:n]]

        features = []
        for wdict in data[:n]:
            aux_dict = self.filterERfeatures(copy.deepcopy(wdict))
            
            features.append(aux_dict)

        return features, targets

    def filterTarget(self, e):
        if self.targetformat=="BDDI":
            return e['ddi'] if 'ddi' in e.keys() else ''
        else:
            return e['ddi'] + "-" + e['type'] if 'ddi' in e.keys() and 'type' in e.keys() else ''

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
        if 'pid' in aux_dict.keys():
            aux_dict.pop('pid',None)
        if 'e1id' in aux_dict.keys():
            aux_dict.pop('e1id',None)
        if 'e2id' in aux_dict.keys():
            aux_dict.pop('e2id',None)
        if 'ddi' in aux_dict.keys():
            aux_dict.pop('ddi',None)
        if 'type' in aux_dict.keys():
            aux_dict.pop('type',None)

        if 'e1_offsetend' in aux_dict.keys():
            aux_dict.pop('e1_offsetend',None)
        if 'e1_offsetstart' in aux_dict.keys():
            aux_dict.pop('e1_offsetstart',None)
        if 'e1_sentenceid' in aux_dict.keys():
            aux_dict.pop('e1_sentenceid',None)
        if 'e2_offsetend' in aux_dict.keys():
            aux_dict.pop('e2_offsetend',None)
        if 'e2_offsetstart' in aux_dict.keys():
            aux_dict.pop('e2_offsetstart',None)
        if 'e2_sentenceid' in aux_dict.keys():
            aux_dict.pop('e2_sentenceid',None)
    
    

        if 'shortestpathDep' in aux_dict.keys():
            aux_dict.pop('shortestpathDep',None)
        if 'trigrams' in aux_dict.keys():
            aux_dict.pop('trigrams',None)
        if 'verbLemma' in aux_dict.keys():
            aux_dict.pop('verbLemma',None)
        if 'e2negationLemmaid' in aux_dict.keys():
            aux_dict.pop('negationLemma',None)
        if 'wordlist' in aux_dict.keys():
            aux_dict.pop('wordlist',None)
        if 'lemmalist' in aux_dict.keys():
            aux_dict.pop('lemmalist',None)
 
        return aux_dict


          
    def findword(self,tree, word):
        #print("tree", type(tree), tree==word)
        #print(dir(tree))
        if isinstance(tree,nltk.tree.Tree):
            result=[tree.label()]
            for stree in tree:
                subresult = self.findword(stree, word)
                #print("sub",subresult)
                if subresult is not None:
                    result.extend(subresult)
                    #print("result",result)
                    return result 
                    break
            return None
        elif isinstance(tree,str) and tree.lower()==word.lower():
            return []
        else:
            return None
        

    def shortestPath(self,text, words, tree, word1, word2):

        
        try:

            #print(tree)
            #print(type(tree))

            #print(tree)
            #print("calling findword",word1.lower())
            path1 = self.findword(tree,word1 )

            #print("","calling findword",word2.lower())
            path2 = self.findword(tree,word2 )
            #print()
            #print(path1)
            #print(path2)

            # compare both paths
            #   -> find first different element
            j = 0
            for i in range(1,min(len(path1),len(path2))):
                if path1[i] != path2[i]:
                    j = i - 1
                    break
                    
            # now join both list from the jth element
            # we need to take into account the "order" of appearance in the tree
            # left or right, which is left to the other one, cuz it's tree will be reversed
            #  S VP NP Mary
            #  S VP NP Bob <-> Bob Np VP S
            #  always the reversed list goes first and that's it?
            sublist1 = path1[j:]
            #print("sublist1",sublist1)
            if j< len(path2)-1:
                j=j+1
            sublist2 = path2[j:]
            #print("sublist2",sublist2)
            sublist2.reverse()
            #print("sublist2",sublist2)
            shortestpath = sublist2 + sublist1

            return shortestpath

        except Exception:
            
            self.registerError(
                [text+"\n"+str(words)+"\n"+str(tree)+"\n"+str(word1)+"\n"+str(word2)+"\n",
                ''.join(traceback.format_stack())
                ])
            print("Exception in user code:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            
            print(text)
            print(words)
            print(tree)
            print(word1,
                word2)
            print()
            return None


    def findwordDepNLTK(self,tree, word):
        #print("tree", type(tree), tree==word)
        #print(dir(tree))
        if isinstance(tree,str) \
            and tree.lower() == word.lower():
            return [tree]
        elif isinstance(tree,nltk.tree.Tree) \
            and tree.label().lower()==word.lower():
            return [tree.label()]
        elif isinstance(tree,nltk.tree.Tree):
            result=[tree.label()]
            for stree in tree:
                subresult = self.findwordDep(stree, word)
                #print("sub",subresult)
                if subresult is not None:
                    result.extend(subresult)
                    #print("result",result)
                    return result 
                    break
            return None
        
        else:
            return None


    def findwordDepFreeling(self,tree, word):
        #print("tree", type(tree), tree==word)
        #print(dir(tree))
        #print("findWordDepFreeling.-----------------")
        #print(tree, word)
        if isinstance(tree,str) \
            and tree.lower() == word.lower():
            return [tree]
        elif isinstance(tree,tuple) and \
             len(tree)>0 and \
             isinstance(tree[0],str) and \
             tree[0].lower() == word.lower():
            return tree[0]
        elif isinstance(tree,tuple): 
            result=[]
            if len(tree)>0:
                subresult = self.findwordDepFreeling(tree[0], word)
                #print("sub",subresult)
                if subresult is not None:
                    result.extend(list(subresult))
                else:
                    # strange case that should not happen?
                    result=[tree[0]]
            
            if len(tree)>1:
                for stree in tree[1:]:
                    subresult = self.findwordDepFreeling(stree, word)
                    #print("sub",subresult)
                    if subresult is not None and \
                       isinstance(subresult, list) or \
                       isinstance(subresult, tuple):
                        result.extend(list(subresult))
                        #print("result",result)
                        return result 
                        break
                    elif subresult is not None and \
                       isinstance(subresult, str):
                        result.append(subresult)
                        #print("result",result)
                        return result 
                        break
            return None
        
        else:
            return None
        

    def shortestPathDep(self, tree, word1, word2):


        
        try:
            #print(tree, "\n",word1, word2)
            #print(type(tree))

            #print(tree)
            #print("calling findword",word1.lower())
            path1 = self.findwordDepFreeling(tree,word1 )

            #print("","calling findword",word2.lower())
            path2 = self.findwordDepFreeling(tree,word2 )
            #print()
            #print(path1)
            #print(path2)

            

            # compare both paths
            #   -> find first different element
            
            # fill smallest path with empties
            # l1 = len(path1)
            # l2 = len(path2)
            # if l1 < l2 :
            #     path1.extend([' ']*(l2 - l1))
            # elif l1 > l2:
            #     path2.extend([' ']*(l1 - l2))

            # now they should be the same size
            j = -1
            for i in range(1,min(len(path1),len(path2))):
                #print(i,path1[i],path2[i])
                if path1[i] != path2[i]:
                    j = i - 1
                    break

            if j == -1:
                if min(len(path1),len(path2)) > 0 :
                    j = min(len(path1),len(path2)) -1 
                else:
                    j=0

            # now join both list from the jth element
            # we need to take into account the "order" of appearance in the tree
            # left or right, which is left to the other one, cuz it's tree will be reversed
            #  S VP NP Mary
            #  S VP NP Bob <-> Bob Np VP S
            #  always the reversed list goes first and that's it?
            sublist1 = path1[j:]
            #print("sublist1",sublist1)
            if j< len(path2)-1:
                j=j+1
            sublist2 = path2[j:]
            #print("sublist2",sublist2)
            sublist2.reverse()
            #print("sublist2",sublist2)
            shortestpath = sublist2 + sublist1
            

            return shortestpath

        except Exception:

            self.registerError(
                [str(tree)+"\n"+str(word1)+"\n"+str(word2)+"\n",
                traceback.format_stack()
                ])
            print("Exception in user code:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            
            print(tree)
            print(word1,
                word2)
            print()
            return None

    def getTrigrams(self, shortestpath):

        trigrams=[]
        try:
            if len(shortestpath)<3:
                return trigrams

            for i in range(2,len(shortestpath)):
                trigrams.append(
                    (shortestpath[i-2],
                     shortestpath[i-1],
                     shortestpath[i]))
        except:
            pass
            
        return trigrams

    def getWordFromOffset(self, element, offset):
        """
            to adapt to pystatparser
            if word offset is 0 -> apply lower()
        """
        try:
            previousOffset = -1
            followingOffset = -1
            minDistance = 100
            for elem in element["features"]:
                
                if abs(int(elem["offsetstart"])-int(offset)) < minDistance:
                    if int(elem["offsetstart"]) < int(offset):
                        previousOffset = int(elem["offsetstart"])
                    else:
                        followingOffset = int(elem["offsetstart"])

                if elem["offsetstart"]==offset:

                    return elem["word"] if elem["word"].isupper() else elem["word"].lower()
                elif int(offset)==int(elem["offsetstart"])-1 or \
                     int(offset)==int(elem["offsetstart"])+1:


                    return elem["word"]  if elem["word"].isupper() else elem["word"].lower()
                elif int(offset)==int(elem["offsetstart"])-2 or \
                     int(offset)==int(elem["offsetstart"])+2:


                    return elem["word"]  if elem["word"].isupper() else elem["word"].lower()
            

            offsetstartFound=[e["offsetstart"] for e in element["features"]][0]
            self.registerError(
                    " offset not found"+"\n"+
                    "text "+element["text"]+"\n"+
                    "offsets "+
                    str(offsetstartFound)+"\n"+
                    "query offset"+"\n"+str(offset)
                    ) 

            print(
                "text",element["text"],"\n"
                "offsets",[[e["offsetstart"] for e in element["features"]]],"\n"
                "query offset",str(offset))

            # extract the closes offset word (previous or following)
            fl = [elem["word"] for elem in element["features"] if elem["offsetstart"]==previousOffset]
            if len(fl)>0:
                return fl[0]


            # extract from original text
            #return element["text"][offset:followingOffset]

        except Exception:
            self.registerError(traceback.format_stack())
            

    
        return None
        
  
    def save(self, filename='data.json'):
        """
            saves the self.data to disk for further processing
            options:
                - json, yaml, xml, csv...
                - 
        """
        with open(filename,'w') as f:            
            json.dump(
                {
                 'data': self.data,
                 'allfeatures': self.datadict,
                 'errorlog': self.errorlog,
                 'numerrors': self.numerrors,
                 'X': self.X,
                 'Y': self.Y
                },f) 


    def wordsetLookup(self,  deptrees, wordset):
        """
            get all words of each tree
            to set
            wordset to set
            compare
        """
        print("wordsetLookup",len(deptrees))

        selectedTree = None
        ws = set(wordset)
        for t in deptrees:
            treewordset = self.getWordsFromTree(t)
            ws1 = set(treewordset)
            if ws.issubset(ws1):
                selectedTree = t
                break

        return selectedTree

    def getWordFromTree(self, tree):

        print(tree)


    def loadDDIFeaturesFile(self, filepath, limit=None):

        with open(filepath,'r') as f:
        
            loadeddata = json.load(f)
            self.data = loadeddata["data"]
            self.datadict = loadeddata["allfeatures"]
            self.X = loadeddata["X"]
            self.Y = loadeddata["Y"]

            if limit is not None:
                self.data = self.data[:limit]
                self.datadict = self.datadict[:limit]
                self.X = self.X[:limit]
                self.Y = self.Y[:limit]




    def filterWordFeatures(self, featuresdict, wordfeatures=[]):
        # expand word features orth -> all orthograhpic features
        if "ortho" in wordfeatures:
            wordfeatures.extend(self.ortholist)

        aux = copy.deepcopy(featuresdict)
        for k,v in aux.items():
            if k[3:] not in wordfeatures:
                del featuresdict[k]


    def fillDict(self, dictorig, dictdest, keyprefix):
        #aux=copy.deepcopy(dictorig)
        aux= dictorig
        for k,v in aux.items(): 
            dictdest[str(keyprefix)+k]=v

    def prepareEntityWordFeatures(self,
        wordfeatures=[],
        windowfeatures=[],
        window=3):
        # basic features of the entities
        # data["features"]
        # for each datadict["ddifeatures"]
        #  get pair entities id
        #  get entity info from data["pairs"]
        #  get entity faetures with offset, and pair id, and entity id

        # initialize features as empty
        entity1_features_template = {}
        aux=self.data[0]["features"][0]
        for k,v in aux.items(): 
            entity1_features_template["e1_"+k]=""
        entity2_features_template = {}
        for k,v in aux.items(): 
            entity2_features_template["e2_"+k]=""
            

        error_offset_list = []
        

        for jj in range(len(self.datadict)):
            e = self.datadict[jj]
            pid = e["pid"]
            sid = e["sentenceid"]
            e1id = e["e1id"]
            e2id = e["e2id"]
            
            e1o=-1
            e2o=-1
            e1f=copy.deepcopy(entity1_features_template)
            e2f=copy.deepcopy(entity2_features_template)
            found1=False
            found2=False
            e1word = ""
            e2word = ""
            e1offset_original = ""
            e2offset_original = ""
            for s in self.data:
                if sid == s["id"]:
                    for p in s["pairs"]:
                        if p["id"] == pid:
                            e1o = p["e1offset"][0][0]
                            e2o = p["e2offset"][0][0]
                            e1word = ' '.join(p["e1offset"][0][1])

                    for e in s["entities"]:
                        if e["id"]==e1id:
                            #e1word = e["text"]
                            e1offset_original = e["charOffset"]
                        if e["id"]==e2id:
                            e2word = e["text"]
                            e2offset_original = e["charOffset"]
                    
                    # window init
                    winit={}
                    self.fillDict(s["features"][0],winit,"___")
                    w1m1 = w1m2 = w1m3 = w1p1 = w1p2 = w1p3 = w2m1 = w2m2 = w2m3 = w2p1 = w2p2 = w2p3 = winit 

                    # entities and their window features gathering
                    for h in range(len(s["features"])):
                        w = s["features"][h]

                        if found1==False and \
                           (w["offsetstart"]==e1o or w["offsetstart"]==e1o-1):
                            # aux=copy.deepcopy(w)
                            # for k,v in aux.items(): 
                            #     e1f["e1_"+k]=v
                            self.fillDict(w, e1f, "e1_")
                            found1=True
                        if found2==False and \
                           ( w["offsetstart"]==e2o or w["offsetstart"]==e2o-1):
                            # aux=copy.deepcopy(w)
                            # for k,v in aux.items(): 
                            #     e2f["e2_"+k]=v
                            self.fillDict(w, e2f, "e2_")
                            found2=True
                        
                        # window features
                        if not found1:
                            if h>2:
                                w = s["features"][h-3]
                                self.fillDict(w, w1m3, "w13")
                            if h>1:
                                w = s["features"][h-2]
                                self.fillDict(w, w1m2, "w12")
                            if h>0:
                                w = s["features"][h-1]
                                self.fillDict(w, w1m1, "w11")
                            if h<len(s["features"])-1:
                                w = s["features"][h+1]
                                self.fillDict(w, w1p1, "w01")
                            if h<len(s["features"])-2:
                                w = s["features"][h+2]
                                self.fillDict(w, w1p2, "w02")
                            if h<len(s["features"])-3:
                                w = s["features"][h+3]
                                self.fillDict(w, w1p3, "w03")

                        if not found2:
                            if h>2:
                                w = s["features"][h-3]
                                self.fillDict(w, w2m3, "w23")
                            if h>1:
                                w = s["features"][h-2]
                                self.fillDict(w, w2m2, "w22")
                            if h>0:
                                w = s["features"][h-1]
                                self.fillDict(w, w2m1, "w21")
                            if h<len(s["features"])-1:
                                w = s["features"][h+1]
                                self.fillDict(w, w2p1, "w31")
                            if h<len(s["features"])-2:
                                w = s["features"][h+2] 
                                self.fillDict(w, w2p2, "w32")
                            if h<len(s["features"])-3: 
                                w = s["features"][h+3]
                                self.fillDict(w, w2p3, "w33")

                    if found1==False or found2==False:
                        info = {
                            'e1' : e1word,
                            'e1id': e1id,
                            'e1o':e1o,
                            'e1o_origin': e1offset_original,
                            'e1w': e1word,
                            'e2' : e2word,
                            'e22id': e2id,
                            'e2o':e2o,
                            'e2o_origin': e2offset_original,
                            'e2w': e2word, 
                            'words': [(w["offsetstart"],w["word"]) for w in s["features"]]
                        }
                        error_offset_list.append(info)

                    self.filterWordFeatures(e1f, wordfeatures)
                    self.filterWordFeatures(e2f,
                        wordfeatures)
                    self.filterWordFeatures(w1m1,
                        windowfeatures)
                    self.filterWordFeatures(w1m2,
                        windowfeatures)
                    self.filterWordFeatures(w1m3,
                        windowfeatures)
                    self.filterWordFeatures(w1p1,
                        windowfeatures)
                    self.filterWordFeatures(w1p2,
                        windowfeatures)
                    self.filterWordFeatures(w1p3,
                        windowfeatures)
                    self.filterWordFeatures(w2m1,
                        windowfeatures)
                    self.filterWordFeatures(w2m2,
                        windowfeatures)
                    self.filterWordFeatures(w2m3,
                        windowfeatures)
                    self.filterWordFeatures(w2p1,
                        windowfeatures)
                    self.filterWordFeatures(w2p2,
                        windowfeatures)
                    self.filterWordFeatures(w2p3,
                        windowfeatures)

                    self.datadict[jj].update(e1f)
                    self.datadict[jj].update(e2f)
                    if window>=1:    
                        self.datadict[jj].update(w1m1)
                    if window>=2:    
                        self.datadict[jj].update(w1m2)
                    if window>=3:    
                        self.datadict[jj].update(w1m3)
                    if window>=1:    
                        self.datadict[jj].update(w1p1)
                    if window>=2:    
                        self.datadict[jj].update(w1p2)
                    if window>=3:    
                        self.datadict[jj].update(w1p3)
                    if window>=1:
                        self.datadict[jj].update(w2m1)
                    if window>=2:
                        self.datadict[jj].update(w1m2)
                    if window>=3:
                        self.datadict[jj].update(w1m3)
                    if window>=1:
                        self.datadict[jj].update(w1p1)
                    if window>=2:
                        self.datadict[jj].update(w1p2)
                    if window>=3:
                        self.datadict[jj].update(w1p3)

                    break

        json.dump(error_offset_list,open('debug_offsets.json','w+'))

    def prepareSentenceFeatures(self, 
        sentencefeatures=[]):

        if len(sentencefeatures)==0:
            return

        # filtering out:
        # vb_count, md_count, dt_count, cc_count

        for e in self.datadict:

            #pprint.pprint(e)

            if "vb_count" in sentencefeatures:
                e["sefe_vb_count"] = copy.deepcopy(e["vb_count"])
            del e["vb_count"]
            if "md_count" in sentencefeatures:
                e["sefe_md_count"] = copy.deepcopy(e["md_count"])
            del e["md_count"]
            if "dt_count" in sentencefeatures:
                e["sefe_dt_count"] = copy.deepcopy(e["dt_count"])
            del e["dt_count"]
            if "cc_count" in sentencefeatures:
                e["sefe_cc_count"] = copy.deepcopy(e["cc_count"])
            del e["cc_count"]
            if "negationLemma" not in sentencefeatures:
                e["sefe_negationLemma"] = copy.deepcopy(e["negationLemma"])
            del e["negationLemma"]









    def prepareCountFeatures(self, 
        topcount=None, 
        topfeatures=[],
        sentencefeatures=[]):

        if len(topfeatures)==0:
            return

        # correct trigrams replace drug by "e"
        for e in self.datadict:
            try:
                e["trigrams"][0][0]="e"       
            except:
                pass

            try:
                e["trigrams"][-1][-1]="e" 
            except:
                pass
        
        # count trigram, words and lemmas features
        # get all trigrams
        trigrams = [ (t[0].lower(), t[1].lower(), t[2].lower())
                    for e in self.datadict 
                    for t in e["trigrams"] 
                    if len(e["trigrams"])>1 and  e["ddi"] == "true"]
    
        threshold = 100
        if topcount is not None:
            threshold = topcount

        if "trigram" in topfeatures:
            # frequency of each trigram in trigrams vs freq in trigrams2
            fd = nltk.FreqDist(trigrams)
            topTrigrams = [f[0] for f in fd.most_common(threshold)]
            countdictT = {}
            for i in range(len(topTrigrams)):
                countdictT["tofe_trigram_"+str(i)+"_"+str(topTrigrams[i])] = 0
        
        # words and lemmas

        if "word" in topfeatures:
            words = [ w.lower()
                        for e in self.datadict 
                        for w in e["wordlist"]
                        if  e["ddi"] == "true"]
            fd = nltk.FreqDist(words)
            topWords = [f[0] for f in fd.most_common(threshold)]
            countdictW = {}
            for i in range(len(topWords)):
                countdictW["tofe_word_"+str(i)+"_"+str(topWords[i])] = 0
       
        if "lemma" in topfeatures:
            lemmas = [ l
                        for e in self.datadict 
                        for l in e["lemmalist"]
                        if  e["ddi"] == "true"]
            fd = nltk.FreqDist(lemmas)
            topLemmas = [f[0] for f in fd.most_common(threshold)]
            countdictL = {}
            for i in range(len(topLemmas)):
                countdictL["tofe_lemma_"+str(i)+"_"+str(topLemmas[i])] = 0
            
        if "pos" in topfeatures:
            wordsbypos = [l
                          for e in self.datadict 
                          for l in e["verbLemma"].split()
                          if e["ddi"] == "true"]
            wordsbypos2 = [l
                           for e in self.datadict 
                           for l in e["negationLemma"].split()
                           if e["ddi"] == "true"]
            fd = nltk.FreqDist(wordsbypos + wordsbypos2)
            topPos = [f[0] for f in fd.most_common(threshold)]
            countdictP = {}
            for i in range(len(topPos)):
                countdictP["tofe_pos_"+str(i)+"_"+str(topPos[i])] = 0


        # now count appearance of each top value in each pair inside 
        for e in self.datadict:
            # counts of words
            if "trigram" in topfeatures:
                cT = copy.deepcopy(countdictT)
                for t in e["trigrams"]:
                    if str(t) in cT.keys():
                        cT[str(t)] += 1 
                e.update(cT)

            if "word" in topfeatures:
                cW = copy.deepcopy(countdictW)
                for t in e["wordlist"]:
                    if str(t) in cW.keys():
                        cW[str(t)] += 1  
                e.update(cW)

            if "lemma" in topfeatures:
                cL = copy.deepcopy(countdictL)
                for t in e["lemmalist"]:
                    if str(t) in cL.keys():
                        cL[str(t)] += 1  
                e.update(cL)


            if "pos" in topfeatures:
                cP = copy.deepcopy(countdictP)
                for t in e["verbLemma"]:
                    if str(t) in cP.keys():
                        cP[str(t)] += 1  
                e.update(cP)




    def finalizeFeatures(self, 
        filepath, 
        limit=None, 
        topcount=None, 
        topfeatures=[],
        sentencefeatures=[],
        wordfeatures=[],
        windowfeatures=[],
        window=5):

        with open(filepath,'r') as f:
        
            loadeddata = json.load(f)
            self.data = loadeddata["data"]
            self.datadict = loadeddata["allfeatures"]

            if limit is not None:
                self.data = self.data[:limit]
                self.datadict = self.datadict[:limit]

            self.prepareEntityWordFeatures(
                wordfeatures=wordfeatures,
                windowfeatures=windowfeatures,
                window=window)

            self.prepareCountFeatures(
                topcount=topcount,
                topfeatures=topfeatures
                )

            self.prepareSentenceFeatures(
                sentencefeatures=sentencefeatures
                )

        self.prepareDDIFeatures()




    def loadFilterDDIFeatures(self, 
        filepath, 
        limit=None, 
        topcount=None, 
        topfeatures=[],
        sentencefeatures=[],
        wordfeatures=[],
        windowfeatures=[],
        window=5):

        with open(filepath,'r') as f:
        
            loadeddata = json.load(f)
            self.data = loadeddata["data"]
            self.datadict = loadeddata["allfeatures"]
            self.X = loadeddata["X"]
            self.Y = loadeddata["Y"]

            if limit is not None:
                self.datadict = self.datadict[:limit]

            # # change to filerEntityWordFeatures

            # word features
            if "ortho" in wordfeatures:
                wordfeatures.extend(self.ortholist)
            # window features
            if "ortho" in windowfeatures:
                windowfeatures.extend(self.ortholist)

            for jj in range(len(self.X)):
                e = self.X[jj]
                auxdict = copy.deepcopy(e)

                # for each dict, filter out 
                # the features that do not 
                # belong th the word or window

                for k,v in auxdict.items():

                    try:
                        if (k.startswith("e1_") or \
                           k.startswith("e2_")) and \
                           k[3:] not in wordfeatures or \
                           k.startswith('___'):
                            del e[k]

                        # for each dict, filter out the 
                        # features that do not belong to
                        # the word or window


                        # now filter by feature prefix = length of windo
                
                        if k.startswith("w0") or \
                           k.startswith("w1") or \
                           k.startswith("w2") or \
                           k.startswith("w3"):

                            if int(k[2:3]) > window or \
                               k[3:] not in windowfeatures:
                                del e[k]
                                #print("removed",k,k[2:3],k[3:],k[3:] not in windowfeatures,type(window), window, int(k[2:3]) > window,  )


                        if k.startswith("sefe_") and \
                           k[5:] not in sentencefeatures:
                            del e[k]

                        if k.startswith("tofe_"):
                            # find if any has prefix 
                            ks = k.split("_")
                            if len(ks)>1 and \
                               ks[1] not in topfeatures or \
                               len(ks)>2 and \
                               int(ks[2]) > int(topcount):
                                del e[k]
                    except Exception:
                        print("Exception in user code:")
                        print("-"*60)
                        traceback.print_exc(file=sys.stdout)
                        print("-"*60)

        #pprint.pprint(self.X[10:11])


    def prepareDDIFeatures(self):     
        try:
            self.X, self.Y = self.transformToXY(self.datadict)
        except Exception:
            print("Exception in user code:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)

            self.X = None
            self.Y = None


    def wordFeaturesExtension(self, 
        filepath=None,
        resultpath=None,
        filepath2=None,
        resultpath2=None):


        if filepath is None:
            filepath= '../data/features/d25tDBmDDIaSVM-Dependency-d25tDBmDDIaSVMm0.json'

        if resultpath is None:
            resultpath = '../data/features/finalDDI.json'

        self = FeatureExtractionDDI('')
        with open(filepath,'r') as f:
            
            loadeddata = json.load(f)
            print(loadeddata.keys())
            self.data = loadeddata["data"]
            self.datadict = loadeddata["allfeatures"]
            wordfeatures = [
                #"lookup",
                "getWord",
                "lemma",
                "pos",
                "chunk",
                "chunkGroup",
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
                ]

            custom = MyFeaturesDDI(params={"wordfeatures":wordfeatures})

            for e in self.data:
                custom.addFeaturesExtended(e)


            #pprint.pprint(self.data[1]["features"])

            # write again to disk
            loadeddata["data"] = self.data

            with open(resultpath,'w+') as f:            
                json.dump(loadeddata,f) 


        if filepath2 is None:
            filepath2= '../data/features/d25tDBmDDIaSVM-test-Dependency-d25tDBmDDIaSVMm0-test.json'

        if resultpath2 is None:
            resultpath2 = '../data/features/finalDDI_test.json'

        self = FeatureExtractionDDI('')
        with open(filepath2,'r') as f:
            
            loadeddata = json.load(f)
            print(loadeddata.keys())
            self.data = loadeddata["data"]
            self.datadict = loadeddata["allfeatures"]
            wordfeatures = [
                #"lookup",
                "getWord",
                "lemma",
                "pos",
                "chunk",
                "chunkGroup",
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
                ]

            custom = MyFeaturesDDI(params={"wordfeatures":wordfeatures})

            for e in self.data:
                custom.addFeaturesExtended(e)

            # write again to disk
            loadeddata["data"] = self.data

            #pprint.pprint(self.data[1]["features"])

            with open(resultpath2,'w+') as f:            
                json.dump(loadeddata,f) 
        

    def correctFeatureFiles(self):
        self.loadDDIFeaturesFile('../data/features/featuresDDItrain.json')
        aux = copy.deepcopy(self.X)
        for i in range(len(self.X)):
            x = self.X[i]
            for k,v in x.items():
                if k.startswith("___"):
                    del aux[i][k]
        self.save('../data/features/featuresDDItrain_DrugBank.json')


        self.loadDDIFeaturesFile('../data/features/featuresBDDItrain.json')
        aux = copy.deepcopy(self.X)
        for i in range(len(self.X)):
            x = self.X[i]
            for k,v in x.items():
                if k.startswith("___"):
                    del aux[i][k]
        self.save('../data/features/featuresBDDItrain_DrugBank.json')

        self.loadDDIFeaturesFile('../data/features/featuresDDItest.json')
        aux = copy.deepcopy(self.X)
        for i in range(len(self.X)):
            x = self.X[i]
            for k,v in x.items():
                if k.startswith("___"):
                    del aux[i][k]
        self.save('../data/features/featuresDDItest_DrugBank.json')

        self.loadDDIFeaturesFile('../data/features/featuresBDDItest.json')
        aux = copy.deepcopy(self.X)
        for i in range(len(self.X)):
            x = self.X[i]
            for k,v in x.items():
                if k.startswith("___"):
                    del aux[i][k]
        self.save('../data/features/featuresBDDItest_DrugBank.json')

if __name__ == '__main__':
        
    #os.chdir(os.path.dirname(sys.argv[0]))
    # print(os.getcwd())
    # fe =FeatureExtraction("./data/LaboCase/Train")
    # fe.load()
    # fe.extractFeatures()
    # print(fe.numerrors)
    # #fe.printdata(n=300)
    # fe.printerrors()
    # fe.save()  

    # freeling = FreelingFeatures()
    # sentences = [
    #     "Today I've missed a very important class about Multi Layer Perceptron and Neural Networkds.",
    #     "I wish I had time to read a good book about it.",
    #     "Maybe in another life I can enroll on a MIT course and actually learn something."]
    # starttime = time()
    # deptrees = freeling.processSentences(sentences)
    # endtime = time()
    # print(endtime - starttime)
    # print(deptrees)
    

    self = FeatureExtractionDDI('')
    #self.wordFeaturesExtension()
    self.correctFeatureFiles()






