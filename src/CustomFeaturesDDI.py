from CustomFeatures import MyFeatures
import re
import copy
import nltk
import FreelingFeatures
import pprint
import random
import math

class MyFeaturesDDI(MyFeatures):

    def __init__(self, params=None):
        
        self.maxwordlength = 30
        self.features_list = [
            self.inSameChunk,
            self.verbLemma,
            self.negationLemma,
            self.posCounts,
            self.getWords,
            self.getLemmas,
            self.pos,
            self.lemma,
            self.lookup,
            self.chunk,
            self.chunkGroup,
            # self.w2v,
            # derived ones
            self.isTitleCase,
            self.isUpperCase,
            self.isLowerCase,
            self.hasDigits,
            self.hasStrangeChars,
            self.moreThan10chars,
            self.prefix,
            self.prefix3,
            self.prefix4,
            self.prefix5,
            self.suffix,
            self.suffix3,
            self.suffix4,
            self.suffix5,
            self.lenprefix,
            self.lensuffix,
            self.lenword,
            self.wordStructure,
            self.wordStructure2,
            self.wordStructureLong,
            self.wordStructureLong2,
        ]
        self.ddifeatures = []
        self.morpho_features = []
        # self.window_1_before = []
        # self.window_1_after = []
        # self.window_2_before = []
        # self.window_2_after = []
        # self.window_3_before = []
        # self.window_3_after = []
        # self.window_4_before = []
        # self.window_4_after = []
        # self.window_5_before = []
        # self.window_5_after = []
        #initialize functions
        if params is not None: 
            self.initialize_feature_list("features", params,self.ddifeatures)
            self.initialize_feature_list("wordfeatures", params,self.morpho_features)
            # self.initialize_feature_list("fbefore1", params,self.window_1_before)
            # self.initialize_feature_list("fafter1", params,self.window_1_after)
            # self.initialize_feature_list("fbefore2", params,self.window_2_before)
            # self.initialize_feature_list("fafter2", params,self.window_2_after)
            # self.initialize_feature_list("fbefore3", params,self.window_3_before)
            # self.initialize_feature_list("fafter3", params,self.window_3_after)
            # self.initialize_feature_list("fbefore4", params,self.window_4_before)
            # self.initialize_feature_list("fafter4", params,self.window_4_after)
            # self.initialize_feature_list("fbefore5", params,self.window_5_before)
            # self.initialize_feature_list("fafter5", params,self.window_5_after)
        else:
            self.initialize_basic_feature_list(self.ddifeatures)
            # self.initialize_basic_feature_list(self.morpho_features)
            # self.initialize_basic_feature_list(self.window_1_before)
            # self.initialize_basic_feature_list(self.window_1_after)
            # self.initialize_basic_feature_list(self.window_2_before)
            # self.initialize_basic_feature_list(self.window_2_after)
            # self.initialize_basic_feature_list(self.window_3_before)
            # self.initialize_basic_feature_list(self.window_3_after)
            # self.initialize_basic_feature_list(self.window_4_before)
            # self.initialize_basic_feature_list(self.window_4_after)
            # self.initialize_basic_feature_list(self.window_5_before)
            # self.initialize_basic_feature_list(self.window_5_after)
            
            
            
    def initialize_feature_list(self, nameslist,params, funclist):
        if nameslist in params.keys():
            nameslist = params[nameslist]
            for f in self.features_list:
                if f.__name__ in nameslist:
                    funclist.append(f)

    def initialize_basic_feature_list(self, funclist):
        for f in self.features_list:
            funclist.append(f)
       
    def isFeatureMethod(self, methodname):
        if methodname.startswith('__') or \
           methodname.startswith('window_') or \
           methodname in [
               'f',
               'isFeatureMethod',
               'addFeatures',
               'shuffleFeatureList',
               'deriveNewFeatureSet',
               'printActiveFeatureFunctions'
           ]:
            return False
        else:
            return True
        
    def shuffleFeatureList(self, listobject, leaveout=0):

        newlist = listobject[:]
        random.shuffle(newlist)
        lastindex = len(newlist) - leaveout
        if lastindex <= 0:
            lastindex= len(listobject)
        newlist2 = newlist[:lastindex]
        newlist2.sort(key= lambda x: x.__name__)
        return newlist2
        
    def deriveNewFeatureSet(self, degree=0):
        self.morpho_features = self.shuffleFeatureList(self.morpho_features,degree)
        self.window_1_before = self.shuffleFeatureList(self.window_1_before,degree)
        self.window_1_after = self.shuffleFeatureList(self.window_1_after,degree)
        self.window_2_before = self.shuffleFeatureList(self.window_2_before,degree)
        self.window_2_after = self.shuffleFeatureList(self.window_2_after,degree)
        self.window_3_before = self.shuffleFeatureList(self.window_3_before,degree)
        self.window_3_after = self.shuffleFeatureList(self.window_3_after,degree)
        self.window_2_before = self.shuffleFeatureList(self.window_4_before,degree)
        self.window_2_after = self.shuffleFeatureList(self.window_4_after,degree)
        self.window_3_before = self.shuffleFeatureList(self.window_5_before,degree)
        self.window_3_after = self.shuffleFeatureList(self.window_5_after,degree)
        
    def printActiveFeatureFunc(self, listobject):
       
        for e in listobject:
            print(e.__name__)
        print()
            
    def printActiveFeatureFunctions(self):
        self.printActiveFeatureFunc(self.morpho_features)
        self.printActiveFeatureFunc(self.window_1_before)
        self.printActiveFeatureFunc(self.window_1_after)
        self.printActiveFeatureFunc(self.window_2_before)
        self.printActiveFeatureFunc(self.window_2_after)
        self.printActiveFeatureFunc(self.window_3_before)
        self.printActiveFeatureFunc(self.window_3_after)
        self.printActiveFeatureFunc(self.window_4_before)
        self.printActiveFeatureFunc(self.window_4_after)
        self.printActiveFeatureFunc(self.window_5_before)
        self.printActiveFeatureFunc(self.window_5_after)

    def printfActiveFeatureFunc(self, listobject):
        result = "[ "
        for e in listobject:
            result = result + str(e.__name__) + ", "
        
        result = result + "]\n"
        return result
 
    def printfActiveFeatureFunctions(self):
        result = ""
        result = result + " current: \n"
        result = result + self.printfActiveFeatureFunc(self.morpho_features)
        result = result + " before1: \n"
        result = result + self.printfActiveFeatureFunc(self.window_1_before)
        result = result + " after1: \n"
        result = result + self.printfActiveFeatureFunc(self.window_1_after)
        result = result + " before2: \n"
        result = result + self.printfActiveFeatureFunc(self.window_2_before)
        result = result + " after2: \n"
        result = result + self.printfActiveFeatureFunc(self.window_2_after)
        result = result + " before3: \n"
        result = result + self.printfActiveFeatureFunc(self.window_3_before)
        result = result + " after3: \n"
        result = result + self.printfActiveFeatureFunc(self.window_3_after)
        result = result + " before4: \n"
        result = result + self.printfActiveFeatureFunc(self.window_4_before)
        result = result + " after4: \n"
        result = result + self.printfActiveFeatureFunc(self.window_4_after)
        result = result + " before5: \n"
        result = result + self.printfActiveFeatureFunc(self.window_5_before)
        result = result + " after5: \n"
        result = result + self.printfActiveFeatureFunc(self.window_5_after)
        return result


    def addFeature(self, resultFeatures, element, i, function_list, word_window_position):
        pos = word_window_position
        sign = "-"+str(math.fabs(pos)) if pos<0 else "+"+str(math.fabs(pos))

        if pos<0 and i>-1*pos-1 or \
           pos>0 and i<len(element["features"])-pos:
            windowword = element["features"][i+pos]
            for func in function_list:
                if func.__name__=="w2v":
                    resultFeatures.update(func(windowword, sign))
                else:
                    resultFeatures[func.__name__+":"+sign]=func(windowword)
        else:
            #for j in range(len(self.window_1_before)):
            for func in function_list:  
                if func.__name__=="w2v":
                    resultFeatures.update(func(element["features"][i], sign, empty=True))
                else:
                    resultFeatures[func.__name__+":"+sign]= ''

    def addFeatures(self, pair_features, element, maxwordlength=30):
        """
            Each feature called from the list
            will add features to the pair_features dictionary
        """
        self.maxwordlength = maxwordlength
        
        for func in self.ddifeatures:
            func(pair_features, element)
        
        
    def addFeaturesExtended(self, element, maxwordlength=30):
        """
            General Entry point (public function)
            this will call all other functionalities
        """
        self.maxwordlength = maxwordlength

        totalResult = []
        for i in range(len(element["features"])):
            
            wordTuple = element["features"][i]
            resultFeatures = {}

            # basic features(word and target!)
            resultFeatures['word']=wordTuple['word']
            resultFeatures['sentenceid']=wordTuple['sentenceid']
            resultFeatures['offsetstart']=wordTuple['offsetstart']
            resultFeatures['offsetend']=wordTuple['offsetend']
            #resultFeatures['biotag']=wordTuple['biotag']
            #resultFeatures['drugtype']=wordTuple['drugtype']
            
            # single features
            for func in self.morpho_features:
                if func.__name__=="w2v":
                    resultFeatures.update(func(wordTuple, ""))
                else:
                    #print(func.__name__)
                    resultFeatures[func.__name__]=func(wordTuple)
            
            totalResult.append(resultFeatures)
        
        element["features"] = totalResult

        
 
    # DDI features

    def getWords(self, pair_features, element):
        thelist = [word["word"] for word in  element["features"]]
        pair_features["wordlist"] = thelist
        


    def getLemmas(self, pair_features, element):
        thelist = [word["lemma"] for word in  element["features"]]
        pair_features["lemmalist"] = thelist


    def inSameChunk(self, pair_features, element):
        """
            perform chunking (or get info from chunks)
            and determine if both entities happend inside the same chunk or not
        """

        # get entity 1 chunk group name
        # get entity 2 chunk group name
        # return true if they are the same, None other wise?
        # return true if hey happend within the same chunk

        # get both e1 and e2 chunkgroup
        # if they are different -> return false
        # if they are the same
        #     traverse the chunktree looking for both see if they are in the same chunk or not
        pass

    def verbLemma(self, pair_features, element):
        """
            retrieve the lemma of word that has a Verb related pos tag
        """
        verb_words=[]
        for elem in element["features"]:        
            if elem["pos"] in ["VB","VBD","VBG","VBN","VBP","VBZ","MD"]:
                verb_words.append(elem["word"])

        pair_features["verbLemma"] =" ".join(verb_words)

            



    def negationLemma(self, pair_features, element):
        """
            return the words around verb pos tag
            https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
        """
        verb_words=[]
        for elem in element["features"]:        
            if elem["pos"] in ["DT","MD"]:
                verb_words.append(elem["word"])

        pair_features["negationLemma"] =" ".join(verb_words)

         

    def posCounts(self, pair_features, element):
        """
            count the number of Verb words in between both entities
            https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk

            https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        """


        # get sentence pos tags (element["features"])
        verbs_count = 0
        md_count = 0
        dt_count = 0
        cc_count = 0
        pdt_count = 0
        pos_count = 0
        to_count = 0
        for elem in element["features"]:        
            if elem["pos"] in ["VB","VBD","VBG","VBN","VBP","VBZ",]:
                verbs_count+=1

            elif elem["pos"] in ["MD",]:
                md_count+=1

            elif elem["pos"] in ["DT",]:
                dt_count+=1

            elif elem["pos"] in ["CC",]:
                cc_count+=1

            elif elem["pos"] in ["PDT",]:
                pdt_count+=1
            elif elem["pos"] in ["POS",]:
                pos_count+=1
            elif elem["pos"] in ["TO",]:
                to_count+=1

        pair_features["vb_count"]=verbs_count
        pair_features["md_count"]=md_count
        pair_features["dt_count"]=dt_count
        pair_features["cc_count"]=cc_count