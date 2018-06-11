import nltk
from nltk.parse.malt import MaltParser

# mp = MaltParser('../data/grammars/maltparser-1.8.1/', '../data/grammars/engmalt.poly-1.7.mco')
# pt = mp.parse_one('I shot an elephant in my pajamas .'.split()).tree()

# print(pt)


# shortest path test
def findword(tree, word):
    # print("tree", type(tree), tree==word)
    # print(tree.label())


    if isinstance(tree,str) \
        and tree.lower() == word.lower():
        return [tree]

    elif isinstance(tree,nltk.tree.Tree) \
        and tree.label().lower()==word.lower():
        return [tree.label()]
    elif isinstance(tree,nltk.tree.Tree):
        result=[tree.label()]
        for stree in tree:
            subresult = findword(stree, word)
            #print("sub",subresult)
            if subresult is not None:
                result.extend(subresult)
                #print("result",result)
                return result 
                break
        return None
    
    else:
        return None
    

def shortestPath(tree, word1, word2):
    
    #print(tree)
    #print(type(tree))

    #print(tree)
    #print("calling findword",word1.lower())
    path1 = findword(tree,word1 )

    #print("","calling findword",word2.lower())
    path2 = findword(tree,word2 )
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



dparser = MaltParser('../data/grammars/maltparser-1.8.1/', 'engmalt.linear-1.7.mco')
pt = dparser.parse_one('I shot an elephant in my pajamas .'.split()).tree()

# print(pt)


# print(pt) 
# print(shortestPath(pt,'I','pajamas'))
# print(shortestPath(pt,'I','pajamas'))
# print(shortestPath(pt,'elephant','pajamas'))
# print(shortestPath(pt,'I','elephant'))


# parsing many sentences

tagged_sents= [
"The other day I went to the beach.".split(),
"It was a hot day so I swimmed in the water.".split()]

# feed all tagged_sents to Maltparser
# deptrees = list(dparser.parse_tagged_sents(
#     tagged_sents))


# feed all tagged_sents to Maltparser
deptrees = dparser.parse_sents(tagged_sents)
print(deptrees)

deptrees = [list(e) for e in deptrees ]
# e = deptrees[0]
# print(e[0])
# print(type(e[0]))
# print(dir(e[0]))
# print(e[0].root)
# print(e[0].tree())

for e in deptrees:
    try:
        print(e[0].tree())
    except:
        pass