import json
import pprint


n=20

# d = json.load(open('../data/features/finalDDI.json','r'))
# pprint.pprint(d["data"][:n])

# #pprint.pprint(d["allfeatures"][:n])


# exit()

d = json.load(open('../data/features/preprocessBDDIstep3DrugBank-test.json','r'))



print(d.keys())



data = d["data"][:n]
datadict = d["allfeatures"][:n]
X = d["X"][:n]
Y = d["Y"][:n]

# pprint.pprint(data)
# pprint.pprint(datadict)
pprint.pprint(X)
pprint.pprint(Y)


# dict_keys(['sentenceid', 'pid', 'e1id', 'e2id', 'ddi', 'type', 'shortestpathDep', 'trigrams', 'verbLemma', 'negationLemma', 'vb_count', 'md_count', 'dt_count', 'cc_count', 'wordlist', 'lemmalist'])

json.dump(datadict,open('../data/features/inspect_ddifeatureset.json','w+'))
json.dump(X,open('../data/features/inspect_ddiX.json','w+'))
json.dump(Y,open('../data/features/inspect_ddiY.json','w+'))