import re
import copy
import nltk
import FreelingFeatures
import pprint
import random
import math

class MyFeatures():

    def __init__(self, params=None):
        
        self.maxwordlength = 30
        self.features_list = [
            self.getWord,
            self.lemma,
            self.pos,
            self.lookup,
            self.chunk,
            self.chunkGroup,
            self.w2v,
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
        self.morpho_features = []
        self.window_1_before = []
        self.window_1_after = []
        self.window_2_before = []
        self.window_2_after = []
        self.window_3_before = []
        self.window_3_after = []
        self.window_4_before = []
        self.window_4_after = []
        self.window_5_before = []
        self.window_5_after = []
        #initialize functions
        if params: 
            self.initialize_feature_list("fcurrent", params,self.morpho_features)
            self.initialize_feature_list("fbefore1", params,self.window_1_before)
            self.initialize_feature_list("fafter1", params,self.window_1_after)
            self.initialize_feature_list("fbefore2", params,self.window_2_before)
            self.initialize_feature_list("fafter2", params,self.window_2_after)
            self.initialize_feature_list("fbefore3", params,self.window_3_before)
            self.initialize_feature_list("fafter3", params,self.window_3_after)
            self.initialize_feature_list("fbefore4", params,self.window_4_before)
            self.initialize_feature_list("fafter4", params,self.window_4_after)
            self.initialize_feature_list("fbefore5", params,self.window_5_before)
            self.initialize_feature_list("fafter5", params,self.window_5_after)
        else:
            self.initialize_basic_feature_list(self.morpho_features)
            self.initialize_basic_feature_list(self.window_1_before)
            self.initialize_basic_feature_list(self.window_1_after)
            self.initialize_basic_feature_list(self.window_2_before)
            self.initialize_basic_feature_list(self.window_2_after)
            self.initialize_basic_feature_list(self.window_3_before)
            self.initialize_basic_feature_list(self.window_3_after)
            self.initialize_basic_feature_list(self.window_4_before)
            self.initialize_basic_feature_list(self.window_4_after)
            self.initialize_basic_feature_list(self.window_5_before)
            self.initialize_basic_feature_list(self.window_5_after)
            
            
            
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

        
    def addFeatures(self, element, maxwordlength="30"):
        """
            General Entry point (public function)
            this will call all other functionalities
        """
        self.maxwordlength = maxwordlength

        totalResult = []
        for i in range(len(element["features"])):
            
            wordTuple = element["features"][i]
            resultFeatures = {}
            """
            # remove non basic features
            # ok-    0-make sure feature names are not repeated so overwritten
            # ko-    1-remove anything that is not word, lemma, pos? but then window features won't work well 
            # -> better to remove non appearing features from the corresponding list
            # 2-add functions for those precreated features and list them
            # 3-add them to the definitions of the batches
            """
            # basic features(word and target!)
            resultFeatures['word']=wordTuple['word']
            resultFeatures['sentenceid']=wordTuple['sentenceid']
            resultFeatures['offsetstart']=wordTuple['offsetstart']
            resultFeatures['offsetend']=wordTuple['offsetend']
            resultFeatures['biotag']=wordTuple['biotag']
            resultFeatures['drugtype']=wordTuple['drugtype']
            
            # single features
            for func in self.morpho_features:
                if func.__name__=="w2v":
                    resultFeatures.update(func(wordTuple, ""))
                else:
                    resultFeatures[func.__name__+"_word"]=func(wordTuple)
            
            # context features
            # 1-word window
            self.addFeature(resultFeatures, element, i, self.window_1_before,-1)
            self.addFeature(resultFeatures, element, i, self.window_1_after,1)      
            # if i>0:
            #     beforeword = element["features"][i-1]
            #     for func in self.window_1_before:
            #         resultFeatures[func.__name__+":-1"]=func(beforeword)
            # else:
            #     #for j in range(len(self.window_1_before)):
            #     for func in self.window_1_before:  
            #         resultFeatures[func.__name__+":-1"]= ''

            # if i<len(element["features"])-1:
            #     afterword = element["features"][i+1]
            #     for func in self.window_1_after:
            #         resultFeatures[func.__name__+":+1"]=func(afterword)
            # else:
            #     #for j in range(len(self.window_1_after)):
            #     for func in self.window_1_after:  
            #         resultFeatures[func.__name__+":+1"]=''
                                                     
            # 2-word window
            self.addFeature(resultFeatures, element, i,  self.window_2_before,-2) 
            self.addFeature(resultFeatures, element,  i, self.window_2_after,2)  
            # if i>1:
            #     beforeword2 = element["features"][i-2]
            #     for func in self.window_2_before:
            #         resultFeatures[func.__name__+":-2"]=func( beforeword2)
            # else:
            #     #for j in range(len(self.window_2_before)):
            #     for func in self.window_2_before:
            #         resultFeatures[func.__name__+":-2"]=''
                                  
            # if i<len(element["features"])-2:
            #     afterword2 = element["features"][i+2]
            #     for func in self.window_2_after:
            #         resultFeatures[func.__name__+":+2"]=func( afterword2)
            # else:
            #     #for j in range(len(self.window_2_after)):
            #     for func in self.window_2_after:
            #         resultFeatures[func.__name__+":+2"]=''
                                 
            # 3-word window
            self.addFeature(resultFeatures, element,  i, self.window_3_before,-3)
            self.addFeature(resultFeatures, element,  i, self.window_3_after,3) 
            # if i>2:
            #     beforeword3 = element["features"][i-3]
            #     for func in self.window_3_before:
            #         resultFeatures[func.__name__+":-3"]=func( beforeword3)
            # else:
            #     #for j in range(len(self.window_3_before)):
            #     for func in self.window_3_before:
            #         resultFeatures[func.__name__+":-3"]=''
                                  
            # if i<len(element["features"])-3:
            #     afterword3 = element["features"][i+3]
            #     for func in self.window_3_after:
            #         resultFeatures[func.__name__+":+3"]=func( afterword3)
            # else:
            #     #for j in range(len(self.window_3_after)):
            #     for func in self.window_3_after:
            #         resultFeatures[func.__name__+":+3"]=''
                                            
            # 4-word window
            self.addFeature(resultFeatures, element,  i, self.window_4_before,-4)
            self.addFeature(resultFeatures, element,  i, self.window_4_after,4)
            # if i>3:
            #     beforeword4 = element["features"][i-4]
            #     for func in self.window_4_before:
            #         resultFeatures[func.__name__+":-4"]=func( beforeword4)
            # else:
                
            #     for func in self.window_4_before:
            #         resultFeatures[func.__name__+":-4"]=''
                                  
            # if i<len(element["features"])-4:
            #     afterword4 = element["features"][i+4]
            #     for func in self.window_4_after:
            #         resultFeatures[func.__name__+":+4"]=func( afterword4)
            # else:
                
            #     for func in self.window_4_after:
            #         resultFeatures[func.__name__+":+4"]=''

            # 5-word window
            self.addFeature(resultFeatures, element,  i, self.window_5_before,-5)
            self.addFeature(resultFeatures, element,  i, self.window_5_after,5)
            # if i>5:
            #     beforeword5 = element["features"][i-5]
            #     for func in self.window_5_before:
            #         resultFeatures[func.__name__+":-5"]=func( beforeword5)
            # else:
                
            #     for func in self.window_5_before:
            #         resultFeatures[func.__name__+":-5"]=''
                                  
            # if i<len(element["features"])-5:
            #     afterword5 = element["features"][i+5]
            #     for func in self.window_5_after:
            #         resultFeatures[func.__name__+":+5"]=func( afterword5)
            # else:
            #     for func in self.window_5_after:
            #         resultFeatures[func.__name__+":+5"]=''


            totalResult.append(resultFeatures)
        
        
        element["features"] = totalResult
        
      
    
    def template(self, tuplefeature ):
        """
            example
        """
        word = tuplefeature["word"]
        pos = tuplefeature["pos"]
        lemma = tuplefeature["lemma"]
        return len(word)/len(lemma)

    
    # already in place features
    
    def getWord(self, wordfeatures):    
        word = wordfeatures["word"].lower()
        return word
    
    def lemma(self, wordfeatures):
        return wordfeatures["lemma"]
    
    
    def pos(self, wordfeatures):
        return wordfeatures["pos"]
    
    
    def lookup(self, wordfeatures):
        return wordfeatures["lookup"]
    
    
    def chunk(self, wordfeatures):
        return wordfeatures["chunk"]
    
    def chunkGroup(self, tuplefeature):
        
        return tuplefeature["chunkGroup"]  
    
            
    # derive features 
    def isTitleCase(self, wordfeatures):
       
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        #return word.istitle()
        return word[0].istitle()
    
    def isUpperCase(self, wordfeatures):
       
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        return word.isupper()
    
    def isLowerCase(self, wordfeatures):
       
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        return word.islower()
        
    def hasDigits(self, wordfeatures):
       
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        #return word.isalnum() and not word.isalpha()
        return not re.match(r'[0-9]',word) is None
    
    def hasStrangeChars(self, wordfeatures):
   
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        return len(re.findall(r'[\W_]',word))>0
        
    def moreThan10chars(self, wordfeatures):
   
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        return len(word) > 10
    
    def prefix(self, wordfeatures):
   
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        i = word.lower().find(lemma)
        if i>-1:
            return word[0:i]
            
        return ""
    
    def prefix3(self, wordfeatures):
        word = wordfeatures["word"]
        return word[0:3]
    
    def prefix4(self, wordfeatures):
        word = wordfeatures["word"]
        return word[0:4]
    
    def prefix5(self, wordfeatures):
        word = wordfeatures["word"]
        return word[0:5]
            
    def suffix(self, wordfeatures):
   
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        i = word.lower().find(lemma)
        if i>-1:
            return word[i+len(lemma):]
            
        return ""

    def suffix3(self, wordfeatures):
        word = wordfeatures["word"]
        return word[-3:]

    def suffix4(self, wordfeatures):
        word = wordfeatures["word"]
        return word[-4:]

    def suffix5(self, wordfeatures):
        word = wordfeatures["word"]
        return word[-5:]
    
    def lenprefix(self, wordfeatures):
   
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        i = word.lower().find(lemma)
        if i>-1:
            return len(word[0:i])/self.maxwordlength
            
        return 0
    
    def lensuffix(self, wordfeatures):
   
        word = wordfeatures["word"]
        pos = wordfeatures["pos"]
        lemma = wordfeatures["lemma"]
        
        i = word.lower().find(lemma)
        if i>-1:
            return len(word[i+len(lemma):])/self.maxwordlength
            
        return 0
    
    def lenword(self, wordfeatures):
        
        word = wordfeatures["word"]
        
        return len(word)/self.maxwordlength


    def wordStructure2(self, tuplefeature ):
        """
            without non alphanumeric symbols
        
        """
        
        word = tuplefeature["word"]
        
        result = ""
        for c in word:
            #print("parsing",c)
            if str(c).isupper() and ( len(result)==0 or len(result)>0 and result[-1]!="X"):
                result = result + "X"
            elif str(c).islower() and ( len(result)==0 or len(result)>0 and result[-1]!="x"):
                result = result + "x"
            elif re.match(r'[0-9]',str(c)) and ( len(result)==0 or len(result)>0 and result[-1]!="0"):
                result = result + "0"
      
        return result
    
    def wordStructure(self, tuplefeature ):
        """
        
        """
        
        word = tuplefeature["word"]
        
        result = ""
        for c in word:
            #print("parsing",c)
            if str(c).isupper() and ( len(result)==0 or len(result)>0 and result[-1]!="X"):
                result = result + "X"
            elif str(c).islower() and ( len(result)==0 or len(result)>0 and result[-1]!="x"):
                result = result + "x"
            elif re.match(r'[0-9]',str(c)) and ( len(result)==0 or len(result)>0 and result[-1]!="0"):
                result = result + "0"
            elif re.match(r'[\W,_]',str(c)) and ( len(result)==0 or len(result)>0 and result[-1]!="0"):
                result = result + "_"
            
            
            
            
        return result
 
    def wordStructureLong(self, tuplefeature ):
        """
            With full length (only replaces char by it's common structure symbol 
            like
            Paracetamol2.0 -> Xxxxxxxxxxx0O0
        
        """
        
        word = tuplefeature["word"]
        
        result = ""
        for c in word:
            #print("parsing",c)
            if str(c).isupper() and ( len(result)==0 or len(result)>0):
                result = result + "X"
            elif str(c).islower() and ( len(result)==0 or len(result)>0 ):
                result = result + "x"
            elif re.match(r'[0-9]',str(c)) and ( len(result)==0 or len(result)>0 ):
                result = result + "0"
            elif re.match(r'[\W,_]',str(c)) and ( len(result)==0 or len(result)>0):
                result = result + "_"
            
        return result

    
    def wordStructureLong2(self, tuplefeature ):
        """
            without non alphanumeric symbols
        """
        
        word = tuplefeature["word"]
        
        result = ""
        for c in word:
            #print("parsing",c)
            if str(c).isupper() and ( len(result)==0 or len(result)>0):
                result = result + "X"
            elif str(c).islower() and ( len(result)==0 or len(result)>0 ):
                result = result + "x"
            elif re.match(r'[0-9]',str(c)) and ( len(result)==0 or len(result)>0 ):
                result = result + "0"
            #elif re.match(r'[\W,_]',str(c)) and ( len(result)==0 or len(result)>0):
            #    result = result + "_"
            
        return result
    

    def w2v(self, featuredict, windowpos, empty=False):
        """
            add all the features "w2vi" that start with "w2v"
            (all the dimension of the word2vec)
        """
        result_w2vfeatures_dict = {}
        featurekeys = list(featuredict.keys())
        for featurename in featurekeys:
            if featurename.startswith("w2v"):
                if not empty:
                    result_w2vfeatures_dict[featurename+windowpos]=featuredict[featurename]
                else:
                    # for when the wordwindow is not possible due to word near sentence boundary
                    result_w2vfeatures_dict[featurename+windowpos]=0

        return result_w2vfeatures_dict