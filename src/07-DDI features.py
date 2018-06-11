import nltk



groucho_grammar = nltk.CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    Det -> 'an' | 'my'
    N -> 'elephant' | 'pajamas'
    V -> 'shot'
    P -> 'in'
    """)

sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
parser = nltk.ChartParser(groucho_grammar)

for tree in parser.parse(sent):
    print(tree)

selected = list(parser.parse(sent))[0]

# Contexxt Free Grammar
grammar1 = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)
sent = "Mary saw Bob".split()
sent = "John saw Mary with a cat ".split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sent):
     print(tree)
        
tree = list(rd_parser.parse(sent))[0]
        
def findword(tree, word):
    #print("tree", type(tree),tree, tree==word)
    #print(dir(tree))
    if isinstance(tree,nltk.tree.Tree):
        result=[tree.label()]
        for stree in tree:
            subresult = findword(stree, word)
            #print("sub",subresult)
            if subresult is not None:
                result.extend(subresult)
                return result 
                break
        return None
    elif isinstance(tree,str) and tree==word:
        return []
    else:
        return None
    

path1 = findword(tree,"Mary")
path2 = findword(tree,"with")
print(path1)
print(path2)

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
print("sublist1",sublist1)
if j< len(path2)-1:
    j=j+1
sublist2 = path2[j:]
print("sublist2",sublist2)
sublist2.reverse()
print("sublist2",sublist2)
shortespath = sublist2 + sublist1
print(shortespath)
