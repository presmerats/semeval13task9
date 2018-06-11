from stat_parser import Parser
from time import time
import nltk
       
def findword(tree, word):
    #print("tree", type(tree), tree==word)
    #print(dir(tree))
    if isinstance(tree,nltk.tree.Tree):
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
    elif isinstance(tree,str) and tree==word:
        return []
    else:
        return None
    

def shortestPath(sentence, word1, word2):

    parser = Parser()
    tree = parser.parse(sentence)

    print(tree)
    #print(type(tree))

    path1 = findword(tree,word1.lower())
    path2 = findword(tree,word2.lower())
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




start = time()
sp = shortestPath("This nltk package is strange so I won't use it", "use", "package")
print(sp)
end = time()
print(end - start)



# PyStatParser

# parser = Parser()
# print(parser.parse(
#     "How can the net amount of entropy of the universe be massively decreased?"))

# start = time()
# print(parser.parse(
#     "How can the net amount of entropy of the universe be massively decreased?"))
# print(parser.parse(
#     "How can the net amount of entropy of the universe be massively decreased?"))
# print(parser.parse(
#     "How can the net amount of entropy of the universe be massively decreased?"))
# print(parser.parse(
#     "How can the net amount of entropy of the universe be massively decreased?"))
# print(parser.parse(
#     "How can the net amount of entropy of the universe be massively decreased?"))
# print(parser.parse(
#     "How can the net amount of entropy of the universe be massively decreased?"))
# print(parser.parse(
#     "How can the net amount of entropy of the universe be massively decreased?"))
# end = time()
# print(end - start)

