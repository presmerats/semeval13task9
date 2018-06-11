import lxml
import os, sys
from xml.dom.minidom import parse, parseString
import xmltodict
from bs4 import BeautifulSoup
import pprint
import nltk
import re
import copy
import pickle
import json




class DrugBankLookup():
    """

    """
    data = []
    files_path = ""
    
    
    def __init__(self, path, pickled=False):
        self.files_path = path
        self.data = {}
            
        if pickled:
            self.loadPickle(path)

    def __del__(self):
        del self.data
        self.data = None
            
    def save(self,path="lookupdict"):
        with open(path + '.pkl', 'wb') as output:
            pickle.dump(self.data, output)
            
    def loadPickle(self,path):
        with open(path, 'rb') as output:
            self.data = pickle.load(output)
       
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
                    #print(name)
                    #self.parseXmlfast(os.path.join(root, name))
                    

    def parseXmlfast(self,filepath):
        """
            Parses one single xml file, saving data to the self.data datastruct
            as a dictionary image of the original xml
        """
        with open(filepath, 'r') as f:
            #parse
            doc = xmltodict.parse(f.read())
            self.data.update(doc)
        
    def parseXml(self,filepath):
        """
            Parses one single xml file, saving data to the self.data datastruct
            as a list of names and synonyms (nothing else is saved!)
        """
        with open(filepath, 'r') as f:
            #parse
            doc = xmltodict.parse(f.read())
            
            newlist = []

            d2 = doc['drugbank']
            d3 = d2['drug']
            for element in d3:
                newlist.append(element['name'].lower())
                try:
                    synonyms  = element['synonyms']
                    if synonyms:
                        for k,v in synonyms.items():
                            for elem in v:
                                newlist.append(elem['#text'].lower())
                                
                except:
                    pass

        self.data = newlist

            
    def lookup(self,word):
        """
            
        """
        return word in self.data
    
    def addLookupFeature(self,element):
       
        for feature in element["features"]:
            feature["lookup"] = self.lookup(feature["word"].lower())
       
        
        
        
if __name__ == '__main__':
    
    fe2 =DrugBankLookup("../data/Dictionaries/drugbank.pkl", pickled=True)
    print(len(fe2.data))
    print(fe2.lookup('paracetamol'))
    print(fe2.lookup('paracetaml'))

    
    #fe =DrugBankLookup("../data/Dictionaries/")
    #fe.load()
    #fe.lookup('paracetamol')
    #fe.lookup('overfit')
    #fe.save("../data/Dictionaries/drugbank")