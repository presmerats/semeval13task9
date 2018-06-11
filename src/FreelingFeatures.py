import pyfreeling as freeling
import pprint
from time import time

class FreelingFeatures():

    def __init__(self):

        freeling.util_init_locale("default")
        self.lang= "en"
        self.ipath="/usr/local"
        self.lpath=self.ipath + "/share/freeling/" + self.lang + "/"
        self.tk=freeling.tokenizer(self.lpath+"tokenizer.dat")
        self.sp=freeling.splitter(self.lpath+"splitter.dat")

        # create the analyzer with the required set of maco_options  
        self.morfo=freeling.maco(self.my_maco_options(self.lang,self.lpath));
        #  then, (de)activate required modules   
        self.morfo.set_active_options (False,  # UserMap 
                                  False,  # NumbersDetection,  
                                  True,  # PunctuationDetection,   
                                  False,  # DatesDetection,    
                                  True,  # DictionarySearch,  
                                  True,  # AffixAnalysis,  
                                  False, # CompoundAnalysis, 
                                  True,  # RetokContractions,
                                  False,  # MultiwordsDetection,  
                                  True,  # NERecognition,     
                                  False, # QuantitiesDetection,  
                                  True); # ProbabilityAssignment                 
        # create tagger
        self.tagger = freeling.hmm_tagger(self.lpath+"tagger.dat",True,2)


        # create sense annotator
        self.sen = freeling.senses(self.lpath+"senses.dat");
        # create sense disambiguator
        self.wsd = freeling.ukb(self.lpath+"ukb.dat");
        # create dependency parser
        self.parser = freeling.dep_treeler(self.lpath+"dep_treeler/dependences.dat");

        

    def my_maco_options(self, lang,lpath) :
        ## -----------------------------------------------
        ## Set desired options for morphological analyzer
        ## -----------------------------------------------

        # create options holder 
        opt = freeling.maco_options(lang);

        # Provide files for morphological submodules. Note that it is not 
        # necessary to set file for modules that will not be used.
        opt.UserMapFile = "";
        opt.LocutionsFile = lpath + "locucions.dat"; 
        opt.AffixFile = lpath + "afixos.dat";
        opt.ProbabilityFile = lpath + "probabilitats.dat"; 
        opt.DictionaryFile = lpath + "dicc.src";
        opt.NPdataFile = lpath + "np.dat";
        opt.PunctuationFile = lpath + "../common/punct.dat";
        return opt;

    def tokenize(self, text):
        self.lw = self.tk.tokenize(text)

        # all in one list
        return list((w.get_form())  for w in self.lw)
        

    def processWords(self):
        """
          A string must already be tokenized into self.lw
          (see tokenize(self, text))
        """
        self.ls = self.sp.split(self.lw)
        self.ls = self.morfo.analyze(self.ls)
        self.ls = self.tagger.analyze(self.ls)

        # separated by sentences
        #return list([(w.get_form(),w.get_lemma(),w.get_tag()) for w in s.get_words()] for s in ls)
        
        # all in one list
        return list((w.get_form(),w.get_lemma(),w.get_tag())  for s in self.ls for w in s.get_words())


    def processTextOriginal(self, text, sentenceid):
        """
         this generates all the features at once
         """

        self.lw = self.tk.tokenize(text)
        self.ls = self.sp.split(self.lw)
        self.ls = self.morfo.analyze(self.ls)
        self.ls = self.tagger.analyze(self.ls)
        # annotate and disambiguate senses     
        self.ls = self.sen.analyze(self.ls);
        self.ls = self.wsd.analyze(self.ls);
        # parse sentences
        #self.ls = self.parser.analyze(self.ls)
        

        # separated by sentences
        #return list([(w.get_form(),w.get_lemma(),w.get_tag()) for w in s.get_words()] for s in ls)
        
        #for s in self.ls:
        #    print(dir(s))
        #    for w in s.get_words():
        #        print(w.get_form())
        #        print(dir(w))
        #        try:
        #            print("AFFIXES")
        #            print(w.AFFIXES)
        #            print(dir(w.AFFIXES))
        #            print("NER")
        #            print(w.NER)
        #            print(dir(w.NER))
        #            print(w.get_n_selected())             
        #            print(w.get_n_unselected())
        #            print("ANALYSIS")
        #            print(w.get_analysis())
        #            print(dir(w.get_analysis()))
        #            print()
        #        except:
        #            pass
        # 
        
        # all in one list
        #return list((w.get_form(),w.get_lemma(),w.get_tag(), sentenceid, w.get_span_start(), w.get_span_finish())  for s in self.ls for w in s.get_words())
        #  dict version

        

        featurelist = list(
            {
                'word': w.get_form(),
                'lemma': w.get_lemma(),
                'pos': w.get_tag(), 
                'sentenceid': sentenceid, 
                'offsetstart': w.get_span_start(), 
                'offsetend': w.get_span_finish()
            }  for s in self.ls for w in s.get_words())

        return featurelist

    def processText(self, text, sentenceid):
        """
         this generates all the features at once
         """

        self.lw = self.tk.tokenize(text)
        self.ls = self.sp.split(self.lw)
        self.ls = self.morfo.analyze(self.ls)
        self.ls = self.tagger.analyze(self.ls)
        # annotate and disambiguate senses     
        self.ls = self.sen.analyze(self.ls);
        self.ls = self.wsd.analyze(self.ls);
        # parse sentences
        self.ls = self.parser.analyze(self.ls)
        

        # separated by sentences
        #return list([(w.get_form(),w.get_lemma(),w.get_tag()) for w in s.get_words()] for s in ls)
        
        #for s in self.ls:
        #    print(dir(s))
        #    for w in s.get_words():
        #        print(w.get_form())
        #        print(dir(w))
        #        try:
        #            print("AFFIXES")
        #            print(w.AFFIXES)
        #            print(dir(w.AFFIXES))
        #            print("NER")
        #            print(w.NER)
        #            print(dir(w.NER))
        #            print(w.get_n_selected())             
        #            print(w.get_n_unselected())
        #            print("ANALYSIS")
        #            print(w.get_analysis())
        #            print(dir(w.get_analysis()))
        #            print()
        #        except:
        #            pass
        # 
        
        # all in one list
        #return list((w.get_form(),w.get_lemma(),w.get_tag(), sentenceid, w.get_span_start(), w.get_span_finish())  for s in self.ls for w in s.get_words())
        #  dict version

        

        featurelist = list(
            {
                'word': w.get_form(),
                'lemma': w.get_lemma(),
                'pos': w.get_tag(), 
                'sentenceid': sentenceid, 
                'offsetstart': w.get_span_start(), 
                'offsetend': w.get_span_finish()
            }  for s in self.ls for w in s.get_words())

        return featurelist,self.processSentenceForDeptree(text)



    


    def processSentences(self, sentences):
        """
         this generates all the features at once
         """

        # starttime = time()
        # print("Freeling, analyzing sentences")
        # process input text
        text = "\n".join(sentences)

        # tokenize input line into a list of words
        lw = self.tk.tokenize(text)
        # split list of words in sentences, return list of sentences
        ls = self.sp.split(lw)
        
        ls = self.morfo.analyze(ls)
        ls = self.tagger.analyze(ls)
        # annotate and disambiguate senses     
        ls = self.sen.analyze(ls);
        ls = self.wsd.analyze(ls);
        # parse sentences
        ls = self.parser.analyze(ls)

        # endtime = time()
        # print(endtime - starttime)


        # starttime = time()
        self.ProcessSentences(ls)
        # endtime = time()
        # print(endtime - starttime)

        return self.deptrees

    def processSentenceForDeptree(self, sentence):
        """
         this generates all the features at once
         """

        # starttime = time()
        # print("Freeling, analyzing sentences")
        

        # tokenize input line into a list of words
        lw = self.tk.tokenize(sentence)
        # split list of words in sentences, return list of sentences
        ls = self.sp.split(lw)
        
        ls = self.morfo.analyze(ls)
        ls = self.tagger.analyze(ls)
        # annotate and disambiguate senses     
        ls = self.sen.analyze(ls);
        ls = self.wsd.analyze(ls);
        # parse sentences
        ls = self.parser.analyze(ls)

        # endtime = time()
        # print(endtime - starttime)


        #starttime = time()
        self.ProcessSentences(ls)
        #endtime = time()
        #print(endtime - starttime)

        return self.deptrees


    def ProcessSentences(self, ls):
        """
            get parse tree as a tree??
            compute shortestpath between 2 words?

        """
        #print("Process Sentences")

        # for each sentence in list
        self.deptrees = []
        for s in ls :
            # for each node in dependency tree
            dt = s.get_dep_tree()
            #print(dt)
            #self.processTreeITeratively(dt)
            self.deptrees.append(self.processTreeRecursive(dt))




    # def extract_lemma_and_sense(self,w) :
    #    lem = w.get_lemma()
    #    sens=""
    #    if len(w.get_senses())>0 :
    #        sens = w.get_senses()[0][0]
    #    return lem, sens





    #     node = dt.begin()
    #     #print(dir(node))
    #     while node != dt.end():
    #         self.processNode(child)
    #         node.incr()

    # def processNode(self, node, theend=None):

        
    #     while node != theend:
    #         ssubj=""; lsubj=""; sdobj=""; ldobj=""
    #         #print(node)
    #         for ch in range(0,node.num_children()) :
    #             child = node.nth_child(ch)
    #             l= child.get_label()
    #             n=child.get_word()
    #             print(l,n)
    #             self.processNode(child, theend)

    #         node.incr()

    # def processTreeITeratively(self, dt):

    #     result = []
    #     node = dt.begin()
    #     while node != dt.end():
    #         ssubj=""; lsubj=""; sdobj=""; ldobj=""
    #         # if it is a verb, check dependants
    #         print(node.get_label(), node.get_word().get_form())



    #         # if node.get_word().get_tag()[0]=='V' :
    #         #     for ch in range(0,node.num_children()) :
    #         #         child = node.nth_child(ch)
    #         #         if child.get_label()=="SBJ" :
    #         #            (lsubj,ssubj) = self.extract_lemma_and_sense(child.get_word())
    #         #         elif child.get_label()=="OBJ" :
    #         #            (ldobj,sdobj) = self.extract_lemma_and_sense(child.get_word())

    #         #     if lsubj!="" and ldobj!="" :
    #         #        (lpred,spred) =  self.extract_lemma_and_sense(node.get_word())
    #         #        print ("SVO : (pred:   " , lpred, "[" + spred + "]")
    #         #        print ("       subject:" , lsubj, "[" + ssubj + "]")
    #         #        print ("       dobject:" , ldobj, "[" + sdobj + "]")
    #         #        print ("      )")

    #         node.incr()


    def processTreeRecursive(self, dt):

        #print("start tree traversal")

        
        node = dt.begin()
        result = tuple(self.processTreeRec(node))

        return result

    def processTreeRec(self, node):

        childs = []

        numc = node.num_children()
        #print("after node.num_children()",numc)
        if numc == 0:
            #print(dir(node))
            result = node.get_word().get_form()
            return result

        for ch in range(0,numc):

            child = node.nth_child(ch)
            #print("after node.nth_child(ch)")
            childs.append(self.processTreeRec(child))
        
        if len(childs) == 1:
            result = (node.get_word().get_form(),childs[0] )
        else:    
            result = [node.get_word().get_form()]
            result.extend(childs)
            result = tuple(result)
        return result