import re
import copy
#import nltk
import FreelingFeatures
import pprint

class BIOTagger2():
    
    def __init__(self):
        
        self.f = FreelingFeatures.FreelingFeatures()
    
    def cleanElement(self, element):
        """
            Action:
                in the text of the element, replace any \n\r or \r\n by \n
       
            Purpose: 
                tokenized text retains the same offsets
               
            Problem: 
                when modifying the text, entity offsets must also be modified!
                use the offsetgroups structure saved inside each element for that
            
        """
        txt = element["text"]
        regex_generic = r'\W+'
        regex_specific = r'\r\n(.+$)'
        regex = regex_specific
        
        """ 
        
        # 1- finditer
        
          for each
            get offset
            get entities offsets
                if any entity offsets > offset
                    then decrement that entity offsets
                    but don't modify the string
                    
        # 2- finally perform the sub
        
        """
        result=re.search(regex,txt)
        if result:
            print("Initial text:")
            print(element)
            try:
                for e in element["entities"]:
                    co = e["charOffset"]
                    print(co)
                    print(e["text"])
                    
                    for offsetg in element["offsets"]:
                        print(element["text"][offsetg[0]:offsetg[1]])
                    
                print()
            except:
                pass
            
            # procedure
            # detect offset of the \r elimnated
            # decrement all offsets that are greater than this one
            
            element["text"] = re.sub(regex,'\n\g<1>',txt)
            try:
                for e in element["entities"]:
                    co = e["charOffset"]
                    print(co)
                    print(e["text"])
                    
                    for offsetg in element["offsets"]:
                        print(element["text"][offsetg[0]:offsetg[1]])
                    
                print(element)
                print()
                print()
            except:
                pass
            
            
    def debugOffsets(self, element):
        """Small function for debugginf offsets of drug names
           their extraction, their update when cleaning \r\n in the text
        """
        print("raw text:")
        print((element["text"],""))
        try:
            for e in element["entities"]:
                co = e["charOffset"]
                print(co)
                print(e["text"])

            for offsetg in element["offsets"]:
                print(str(offsetg) + ": " + element["text"][offsetg[0]:offsetg[1]])

            print()
        except:
            pass
        
    def decrementCharOffset(self, element, index_from, how_many):
        """
            decrements the "charOffsets" of all entities of an element
            by how_many
            if they are > index_from
        """
        for entity in element['entities']:
            # Extract all offsets
            offsetgroups = self.splitBIOOffsets(entity['charOffset'])
           
            for offsetg in offsetgroups:
                d1 = offsetg[0]
                if d1 > index_from:
                    # decrement its offset
                    offsetg[0] = offsetg[0] - 1
                    offsetg[1] = offsetg[1] - 1
                    
            
            # rebuil the string
            charoffsets = ""
            for offsetg in offsetgroups:
                if len(offsetg)==2:
                    charoffsets += str(offsetg[0]) + "-" + str(offsetg[1]) + ";"
                elif len(offsetg)==1:
                    charoffsets += str(offsetg[0]) + ";"
            # remove last ;
            entity["charOffset"] = charoffsets[:-1]  if charoffsets[-1]==';' else charoffsets
                
            

    def decrementGroupOffsets(self, element, index_from, how_many):
        """
            decrements the "Offsets" of an element
            by how_many
            if they are > index_from
        """
        for i in range(len(element['offsets'])):
            d1,d2,l,dtype = element['offsets'][i]
            if d1 > index_from:
                show = True
                # decrement its offset
                newtuple = (d1 - how_many,d2 - how_many,l,dtype)
                element['offsets'][i]=newtuple
                
        
        
    def cleanElement2(self, element):
        """
            Action:
                in the text of the element, replace any \n\r or \r\n by \n
       
            Purpose: 
                tokenized text retains the same offsets
               
            Problem: 
                when modifying the text, entity offsets must also be modified!
                use the offsetgroups structure saved inside each element for that      
        """
        txt = element["text"]
        regex_generic = r'\W+'
        regex_specific = r'\r\n(.+$)'
        p=re.compile(regex_specific)

        
        iterator = p.finditer(txt)
        for match in iterator:
            # regex match offsets 
            o1,o2 = match.span()
            
            # drug offsets decrement 
            self.decrementGroupOffsets(element, o1, 1)
                    
            # decrement in the element["entities"]["charOffsets"]
            self.decrementCharOffset(element, o1, 1)
                    
        result=p.search(txt)
        if result:
            #self.debugOffsets(element)
            element["text"] = p.sub('\n\g<1>',txt)
            #self.debugOffsets(element)
                
    
            
    def verifyOffsets(self,element):
        """ verify offsets from Freeling tokens are correct
            
            FOR EACH TOKEN 
                GET o1 and o2
                then extract element["text"][o1:o2]
                compare with feature[0]
        
        """
        element["errors"]=[]
        for feature in element["features"]:
            o1 = feature["offsetstart"]
            o2 = feature["offsetend"]
            extracted_text = element["text"][o1:o2]
            if feature["word"]!= extracted_text:
                # exception cannot separated into can and not
                if element["text"][o1:o1+6].lower()=="cannot":
                    continue
                   
                element["errors"].append(
                    (feature["word"],o1,o2,extracted_text, element["text"][o1:o1+6]))
                # element["errors"].append(
                #    (feature[0],o1,o2,extracted_text,element["text"], element["features"]))

    def verifyGlobalOffsets(self, data):
        return sum([1 for element in data if len(element["errors"])>0])
          
    def printErrorOffsets(self, data):
        for i in range(len(data)):
            element = data[i]
            if len(element["errors"])>0:
                print("element " + str(i) + " " + str(element["errors"]))
        
    def splitBIOOffsets(self, offsets_string):
        """
            -> trivial case '89'
            -> easy case '89-91'
            -> complex '112-125;146-150'
                split hiearchically first ; , then -
                if no - , then assume only one word
            -> always return a hierarchical grouping of offsets
            
            return value : [[112,115],[146,150]]
                        or for example if '89'
                        [[89]]
        """
        #offsets  = re.split(r'[\W\b\-\;]+' ,offsets_string)
        
        if ';' in offsets_string:
            groupoffsets = re.split(';',offsets_string)
            offsets = [ [int(elem) for elem in re.split('-',group)] for group in groupoffsets ]
            
        else:
            
            offsets = [ [int(elem) for elem in re.split('-',offsets_string) ]]
            
        #offsets = sorted(offsets, key=lambda x: x[0])
        
        return offsets
    
    def addTokenizedWordsToBIOOffsets(self, text, offsetgroups, drugtype):
        """
            reads the offsets and offsets groups and transforms them into
            offsetgroups is a tuple with:
                - initial offset
                [- ending offset]
                - tuple of enclosed words tokenized by freeling
                
            offsetgroups: [ [i1,i2],[i3],[i4,i5],..]
        """
        resultgroups = []
        for el in offsetgroups:
            if len(el)==2:
                # look for slice of element text, then tokenize by freeling
                #print(el)
                #print(type(el[0]))
                fragment = text[int(el[0]):int(el[1])+1]
                tokens = self.f.tokenize(fragment)
                resultgroups.append((int(el[0]),int(el[1])+1,tokens,drugtype))
                
                
            elif len(el)==1:
                # look for word in element text at offset and word boundary (use regexp
                fragment = text[int(el[0]):]
                p = re.compile(r'\W+')
                tokens = p.split(fragment)
                # we return the first offsets, the second offset as null and the list of tokens(just the first one)
                resultgroups.append((int(el[0]),None,[tokens[0]],drugtype))
        
        return resultgroups
    
    def addTokenizedWordsToDDIOffsets(self, text, offsetgroups, drugtype, eid):
        """
            reads the offsets and offsets groups and transforms them into
            offsetgroups is a tuple with:
                - initial offset
                [- ending offset]
                - tuple of enclosed words tokenized by freeling
                
            offsetgroups: [ [i1,i2],[i3],[i4,i5],..]
        """
        resultgroups = []
        for el in offsetgroups:
            if len(el)==2:
                # look for slice of element text, then tokenize by freeling
                #print(el)
                #print(type(el[0]))
                fragment = text[int(el[0]):int(el[1])+1]
                tokens = self.f.tokenize(fragment)
                resultgroups.append((int(el[0]),int(el[1])+1,tokens,drugtype, eid))
                
                
            elif len(el)==1:
                # look for word in element text at offset and word boundary (use regexp
                fragment = text[int(el[0]):]
                p = re.compile(r'\W+')
                tokens = p.split(fragment)
                # we return the first offsets, the second offset as null and the list of tokens(just the first one)
                resultgroups.append((int(el[0]),None,[tokens[0]],drugtype,eid))
        
        return resultgroups

    def prepareElementBIOOffsets(self, element):
        """
            saves offsets into a list of offsetgroups:
            
            offsetgroups is a tuple with:
                - initial offset
                [- ending offset]
                - tuple of enclosed words tokenized by freeling
        
        """
        
        # Warning with this!! maybe it is better to remove it!
        element['offsets']=[]
        element['token_entities']=[]
        
        for entity in element["entities"]:

            etext = entity["text"]
            drugtype = entity["type"]
            element["Error"]=False
           
            offsetgroups = self.splitBIOOffsets(entity['charOffset'])
            offsetgroups = self.addTokenizedWordsToBIOOffsets(element["text"], offsetgroups,drugtype)
            element['offsets'].extend(offsetgroups)
            
            #element['token_entities'].extend
            

        element['offsets'] = sorted(element['offsets'], key=lambda x:x[0])

    def prepareElementDDIOffsets(self, element):
        """
            saves offsets into a list of offsetgroups:
            
            offsetgroups is a tuple with:
                - initial offset
                [- ending offset]
                - tuple of enclosed words tokenized by freeling
        
        """
        
        # Warning with this!! maybe it is better to remove it!
        element['ddioffsets']=[]
        
        
        for entity in element["entities"]:

            etext = entity["text"]
            drugtype = entity["type"]
            eid = entity["id"]
            element["Error"]=False
           
            offsetgroups = self.splitBIOOffsets(entity['charOffset'])
            offsetgroups = self.addTokenizedWordsToDDIOffsets(element["text"], offsetgroups,drugtype,eid)
            element['ddioffsets'].extend(offsetgroups)
            
            
            

        element['offsets'] = sorted(element['offsets'], key=lambda x:x[0])
        
    def belongsToOffset(self, element, o1, o2):
        """
            finds if and offset corresponds to an entity
            returns its corresponding tag B, I or O if it is not an entity
        
           possible offsets:
               'offsets': [[29, 39], [116, 128], [237, 254]]
               'offsets': [[39], [116] ]
               'offsets': [[29, 49], [39, 100]]

        """
        found = False
        for og in element["offsets"]:
            if len(og)==3:
                if o1>=og[0] and o1<=og[1] \
                   and o2>=og[0] and o2<=og[1]:
                    found = True
                    break
            
            if len(og)==2:
                if o1==og[0]:
                    found = True
                    break
        
        return found
        
    def setBIOTag(self, element, index, tag, drugtype=''):
        
        oldlist = element["features"][index]
        oldlist["biotag"]=tag
        oldlist["drugtype"]=drugtype

        element["features"][index] = oldlist

            
    def bIOtag(self,element):
        """ 
            example:
                'features': [('[', '[', 'Fca', 0, 0, 1),
                   ('Dose-time', 'dose-time', 'NN', 1, 1, 10),
                   ('effects', 'effect', 'NNS', 2, 11, 18),
                   ('of', 'of', 'IN', 3, 19, 21),
                   
              feature tuple elements
                  word, lemma, tag, position in sentence, span start, span finish
               
            For each tag, extract offset, 
                if offset is equal to any offsets
                    aplly tagging
                    depending on offset where it fits
                        - more elements not the first -> I
                        - first element -> B
                    verify it is the same word
        """
        for i in range(len(element["features"])):
            feature = element["features"][i]
            
            o1 = feature["offsetstart"]
            o2 = feature["offsetend"]

            self.setBIOTag(element,i,"O")
            
            
            
            for og in element["offsets"]:
                #print(og)
                if len(og)==4 \
                   and o1==og[0] and o1<=og[1] \
                   and o2>og[0] and o2<=og[1]:
                       self.setBIOTag(element,i,"B",og[3])
                elif len(og)==4 \
                     and o1>og[0] and o1<=og[1] \
                     and o2>og[0] and o2<=og[1]:
                    self.setBIOTag(element,i,"I",og[3])
                
                elif len(og)==3 and o1==og[0]:
                    self.setBIOTag(element,i,"B",og[3])
                
            print
                    
        
   
        
    def verifyBIOtag(self, element):
        """
            total_element_entity_tokens
            for each entity
                number_of_; = count number of ;  + 1 in charOffset
                tokens = tokenize entity[text]
                count = number_of_; * len(tokens)
                total_element_entity_tokens += count
                
            for all features
                count number of B, I tokens
                
            both numbers should be the same
        
        """
        
        element["bioerrors"]= 0
        
        total_entities = 0
        for entity in element["entities"]:
            count1 = len(re.findall(";",entity["charOffset"]))+1
            tokens = self.f.tokenize(entity["text"])
            count = count1 * len(tokens)
            total_entities += count
            
        found_entities =len([
            feature for feature in element["features"] if feature['biotag'] in ['B','I'] ])
        #print(total_entities,found_entities)
            
        if found_entities != total_entities:
            #print(total_entities,found_entities)
            #pprint.pprint((element,""))
            #print()
            element["bioerrors"]=  total_entities - found_entities
   
    def verifyGlobalBIOtags(self, data):
        return sum([element["bioerrors"] for element in data])
  
    

class BIOTagger():

    def __init__(self):
        self.BItag = 'BiotagIni'
        self.BItagWSpace = '   ' + self.BItag + '   '
        self.BIOtag = 'BiotagEnd'
        self.BIOtagWSpace = '   ' + self.BIOtag + '  '

        self.f = FreelingFeatures.FreelingFeatures()

    
    def splitBIOOffsets(self, offsets_string):
        """
            -> trivial case '89'
            -> easy case '89-91'
            -> complex '112-125;146-150'
                split hiearchically first ; , then -
                if no - , then assume only one word
            -> always return a hierarchical grouping of offsets
            
            return value : [[112,1125],[146,150]]
                        or for example if '89'
                        [[89]]
        """
        #offsets  = re.split(r'[\W\b\-\;]+' ,offsets_string)
        
        if ';' in offsets_string:
            groupoffsets = re.split(';',offsets_string)
            offsets = [ [int(elem) for elem in re.split('-',group)] for group in groupoffsets ]
            
        else:
            
            offsets = [ [int(elem) for elem in re.split('-',offsets_string) ]]
            
        #offsets = sorted(offsets, key=lambda x: x[0])
        
        return offsets
    
    def prepareElementBIOOffsets(self, element):
        for entity in element["entities"]:

            etext = entity["text"]
            element["Error"]=False
            
            offsetgroups = self.splitBIOOffsets(entity['charOffset'])
            element['offsets'].extend(offsetgroups)

        element['offsets'] = sorted(element['offsets'], key=lambda x:x[0])
        
    def BIOtagOffset_NoOverlap(self, element, text2, i1, i2, last_offset):
        # before drug words start
        before_part = text2[last_offset:i1]
        element["text_splits"].append(before_part)

        # drug words
        element["text_splits"].append(self.BItagWSpace)
        entity_text = text2[i1:i2+1]
        element["text_splits"].append(entity_text)
        #entity_text2 = self.BIOTagWord(entity_text)
        #deviation += len(entity_text2) - len(entity_text)
        element["text_splits"].append(self.BIOtagWSpace) 

        last_offset = i2 + 1
        
        return last_offset
    
    def BIOtagOffset_i1Overlap(self, element, text2, i1,i2, last_offset):
        # crossed case: combine the next two approaches

        # i1=i1' but i2<i2'
        idiff = last_offset - i1

        # just insert the _BIO tag in between:
        # _BI_ , drug1 drug2 drug3 , _BIO_
        # _BI_ , drug1 drug2, _BI_ , drug2 , _BIO_

        # extract last 2 parts
        part_to_modify = element["text_splits"][-2]
        BIO_tag = element["text_splits"][-1]

        # remove the last 2 parts
        element["text_splits"].pop(-2)
        element["text_splits"].pop(-1)


        # modify them
        before_part = part_to_modify[:len(part_to_modify)-idiff]

        # add them to the list
        if len(before_part)>0:
            element["text_splits"].append(before_part)

        #element["text_splits"].append(self.BItagWSpace)

        if i2>=last_offset-1:
            after_part = part_to_modify[len(part_to_modify)-idiff:]
            if len(after_part)>0:
                element["text_splits"].append(after_part)

            element["text_splits"].append(BIO_tag)

            # now add the last _BIO_
            # drug words
            entity_text = text2[last_offset:i2+1]
            if len(entity_text)>0:
                element["text_splits"].append(entity_text)
                #entity_text2 = self.BIOTagWord(entity_text)
                #deviation += len(entity_text2) - len(entity_text)
                element["text_splits"].append(self.BIOtagWSpace)

        if i2> last_offset:
            last_offset = i2 + 1
            
        return last_offset

    def BIOtagOffset_i1i2Overlap(self, element, text2, i1,i2, last_offset):
        # crossed case: combine the next two approaches
        # i1=i1' but i2<i2'
        idiff = last_offset - i1

        # just insert the _BIO tag in between:
        # _BI_ , drug1 drug2 drug3 , _BIO_
        # _BI_ , drug1 drug2, _BI_ , drug2 , _BIO_

        # extract last 2 parts
        part_to_modify = element["text_splits"][-2]
        BIO_tag = element["text_splits"][-1]

        # remove the last 2 parts
        element["text_splits"].pop(-2)
        element["text_splits"].pop(-1)

        # modify them
        before_part = part_to_modify[:len(part_to_modify)-idiff]

        # add them to the list
        if len(before_part)>0:
            element["text_splits"].append(before_part)

        #element["text_splits"].append(self.BItagWSpace)

        # i2 < last_offset
        overlapping_part = part_to_modify[len(part_to_modify)-idiff:-(last_offset - i2)+1]

        if len(overlapping_part)>0:
            element["text_splits"].append(overlapping_part)
        

        #remaining part between i2 and last_offset
        after_part = part_to_modify[len(part_to_modify)-idiff+len(overlapping_part):]
        element["text_splits"].append(after_part)
        element["text_splits"].append(BIO_tag)

        if i2> last_offset:
            last_offset = i2 + 1
        
        return last_offset
        
       
    def BIOtagSingleOffset_NoOverlap(self, element, text2, i1, i2, last_offset):
        
        # before drug words start
        before_part = text2[last_offset:i1]
        element["text_splits"].append(before_part)

        # drug words
        element["text_splits"].append(self.BItagWSpace)

        # now add the next word only? or what?

        i2 = re.find(r'\b', text2[i1:])

        if i2>-1:

            entity_text = text2[i1:]
            element["text_splits"].append(entity_text)
            #entity_text2 = self.BIOTagWord(entity_text)
            #deviation += len(entity_text2) - len(entity_text)
            element["text_splits"].append(self.BIOtagWSpace)

            last_offset = i1 + len(entity_text) + 1
            
        return last_offset
    
    def BIOtagingleOffset_i1Overlap(self, element, text2, i1,i2, last_offset):
        # crossed case: combine the next two approaches

        # i1=i1' but i2<i2'
        idiff = last_offset - i1

        # just insert the _BIO tag in between:
        # _BI_ , drug1 drug2 drug3 , _BIO_
        # _BI_ , drug1 drug2, _BI_ , drug2 , _BIO_

        # extract last 2 parts
        part_to_modify = element["text_splits"][-2]
        BIO_tag = element["text_splits"][-1]

        # remove the last 2 parts
        element["text_splits"].pop(-2)
        element["text_splits"].pop(-1)

        # modify them
        before_part = part_to_modify[:len(part_to_modify)-idiff]
        after_part = part_to_modify[len(part_to_modify)-idiff:]

        
        # add them to the list
        element["text_splits"].append(before_part)
        element["text_splits"].append(self.BItagWSpace)
        element["text_splits"].append(after_part)
        element["text_splits"].append(BIO_tag)

        # now add the last _BIO_
        # skip this if last_offset contains a _BIO_
            
        return last_offset
        
    def BIOfeaturesPreparation(self, element):
        """
        for a given element:
            text
            words
            entities
                text
                charOffset
                
            -> get charOffset and words and lean of each word
            -> go to texxt and add a suffix _B _I
            -> tokenize 
            -> call BIOfeatures3
                -> add 'O' token excep for those where _B or _I suffix
                -> add 'B' for those with _B or _I except for those which
                  previous token is B also
        """        
        deviation = 0
        element['offsets']=[]
        element["text_original"] = element["text"]
        element["text_transformation"] = []
        
        element["text_splits"]=[]
        element["text_splits_history"]=[]
        last_offset = 0
        
        
        self.prepareElementBIOOffsets(element)

            
        for eoffset in element['offsets']:
            # extract all segments from offsets
            # in cases like [40,69],[40,60] go back to the previous segment
            # and split it
            
            if len(eoffset)==2:
                # start and finish offsets
                text2 = element['text']
                i1 = int(eoffset[0]) 
                i2 = int(eoffset[1]) 

                if i2 < last_offset:
                    last_offset = self.BIOtagOffset_i1i2Overlap(element, text2, i1,i2, last_offset)
                
                elif i1 < last_offset:
                    last_offset = self.BIOtagOffset_i1Overlap(element, text2, i1,i2, last_offset)

                else:
                    last_offset = self.BIOtagOffset_NoOverlap( element, text2, i1, i2, last_offset)
            
            else:
                text2 = element['text']
                i1 = int(eoffset[0]) 

                # verify no overlap exists
                if i1 < last_offset:
                    last_offset = self.BIOtagSingleOffset_i1Overlap(element, text2, i1,i2, last_offset)
                    
                else:  
                    last_offset = self.BIOtagSingleOffset_NoOverlap( element, text2, i1, i2, last_offset)

            element["text_splits_history"].append(copy.deepcopy(element["text_splits"]))
                     
        final_text = ''.join(element["text_splits"])
        element["text"] = final_text


  
    def BIOfeaturesTag(self, element):
        """
            for each features we compare the word (position 0)
            and then we append a tag in the position 1
            (we need the rebuild a new tuple)
            
        """
        
        features = element['features']
        features2 = []
        
        insideBIO = False
        

        for i in range(0,len(features)):
            biotag = 'O'
            isTag = False

            # Cleaning words 
            
            if features[i][0] == self.BItag:
                insideBIO = True
                isTag = True
                
            elif features[i][0] == self.BIOtag:
                insideBIO = False
                isTag = True
    
            elif i==0:
                pass
            
            elif insideBIO and \
                 (features[i-1][1]=='B' or \
                 features[i-1][1]=='I'):
                
                biotag = 'I'
                
            elif insideBIO and \
                 features[i-1][1]!='B' and\
                 features[i-1][1]!='I':
                
                biotag = 'B'
                
            fnew = (features[i][0],biotag , features[i][1] , features[i][2])
            features[i]=fnew
            
        
        for i in range(0,len(features)):
            if features[i][0] == self.BItag or\
               features[i][0] == self.BIOtag:
                continue
            features2.append(features[i])
                
        element['features']=features2

        #add metadata for verification
        self.BIOverification(element)
    

    def countNumDrugWords(self, element):
        
        count = 0
        element["verified_drug_words"] = []
        
        # get all sorted offsets
        # merge offsets that overlap
        overlaped_offsets = []
        for off in element["offsets"]:
            if len(overlaped_offsets)==0:
                overlaped_offsets.append(off)
            else:
                i1 = off[0]
                i2 = off[1]
                
                if i2<= overlaped_offsets[-1][1]:
                    # as the offsets are sorted by i1,
                    # we could skip this offsets as it is contained in the previous one
                    pass
                
                elif i2> overlaped_offsets[-1][1] and i1< overlaped_offsets[-1][1]:
                    # this one overlaps but includes more words after it
                    # so we should increment i2 of the previous offset to current i2
                    overlaped_offsets[-1][1] = i2
                
                else:
                    overlaped_offsets.append(off)
                
        for groupoffset in overlaped_offsets:
            if len(groupoffset)==2:
                i1 = groupoffset[0]
                i2 = groupoffset[1]
                fragment = element["text_original"][i1:i2+1]
                fwords =self.tokenize(fragment)
                element["verified_drug_words"].extend(fwords)
                count += sum([1 for word in fwords]) 
            else:
                i1 = int(groupoffset[0])
                i2 = i1 + int(re.find(r'\b',element["text_original"][i1:]))
                if i2>i1:
                    fragment = element["text_original"][i1:i2+1]
                    fwords = self.tokenize(fragment)
                    element["verified_drug_words"].extend(fwords)
                    count += sum([1 for word in fwords]) 
        
        return count
    
    def BIOverification(self, element):
        
        numBIs =sum([1 for w,biotag,l,p in element["features"] if biotag in ['B','I']])
        numDrugWords = self.countNumDrugWords(element)
        element["numBIs"]=numBIs
        element["numDrugWords"]=numDrugWords
        #element["BIOok"] = numBIs == (numDrugWords - element["BIOcrossedwords"])
        element["BIOok"] = numBIs == numDrugWords
        


    def BIOcountErrors(self, data):
        return sum([1 for element in data if 'BIOok' in element.keys() and not element['BIOok'] ])
        
    def tokenize(self, text):

        #nltk version
        # return nltk.word_tokenize(text)

        # Freeling version
        return self.f.tokenize(text)
