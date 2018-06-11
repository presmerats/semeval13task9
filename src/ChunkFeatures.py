import re
import copy
import nltk

class MyChunker():

    def chunkElement(self, element):
        
        # get the tagged sentence
        sent = []
        for feature in element["features"]:
            sent.append((feature["word"] , feature["pos"]))
        
        # perform chunking
        if len(sent)<0:
            return
        
        chunktree = nltk.ne_chunk(sent)

        element["chunktree"] = chunktree
        
        # what to save?
        # belongs to a NE / NE-PERSON etc
        # belongs to the NE1, NE2,... / NE-PERSON1, NE-GPE2,NE-PERSON2, ..)
        # save all and then decide later

        
        # traverse all word in the tree IN ORDER and save to features
        feature_index = 0
        for t in chunktree:
            #print(t)
            #print(type(t))
            
            if isinstance(t,nltk.Tree):
                # descend this level
                
                # get label
                label = t.label()
                #print(label)
                
                # add each word with the chunk label
                for w, tr in t:
                    # l is a dict
                    l = element["features"][feature_index]
                    #print(" feature word: " + str(l[0]))
                    #print(" chunk Entity word: " + str(tr) + " "  + str(w) + " " + label)
                    if l["word"] == w:
                        l["chunk"]='NE'
                        l["chunkGroup"]='NE-' + label
                    element["features"][feature_index] = l
                    
                    feature_index = feature_index + 1
            else:
                # just a word from the sentence
                
                l = element["features"][feature_index]
                #print(" feature word: " + str(l[0]))
                #print(" chunk word: " + str(t))
                if l["word"] == t[0]:
                    l["chunk"]=''  #no named entity
                    l["chunkGroup"]='' # no named entity group
                element["features"][feature_index] = l

                feature_index = feature_index + 1
            

            
            
                            
        