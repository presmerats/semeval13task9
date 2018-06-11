import os, sys
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn import cluster


class W2VFeatures():
    """
    datastructure:
    dict
    { word : vector representation}
  
    """
    w2v_dic = []
    num_clusters = 100
    default_vector = np.zeros(200)
    default_cluster = num_clusters  # The cluseters will be in the range of: 0-(num_clusters-1), 
                                    # the default cluseter will be num_clusters.
    default_dic = (default_vector, default_cluster)
    w2v_options = ["PubMed-w2v"] #, "PubMed-and-PMC-w2v", "wikipedia-pubmed-and-PMC-w2v"]
    dic_dir_path = "../data/word2vec/"
    
    #string - dic_type: {PubMed-w2v, PubMed-and-PMC-w2v, wikipedia-pubmed-and-PMC-w2v}
    def __init__(self, dic_type="PubMed-w2v"):
        if dic_type not in self.w2v_options:
            print(dic_type + " Is not supported, currently we support only PubMed-w2v option")
            print("The model will use the default dic: PubMed-w2v")
            dic_type = "PubMed-w2v"
        
        self.files_path = self.dic_dir_path+dic_type
        
        self.load_dic()

        # if self.file_exist_not_empty():
        #     #self.load_dic_workaround() 
        #     self.load_dic()
            
        # else:
        #     self.create_and_load_dic()

    def file_exist_not_empty(self):
        if os.path.isfile(self.files_path + '.pkl'):
            if os.stat(self.files_path + '.pkl').st_size>0 :
                return True
        return False
        
    def load_dic(self):
        print("loading w2v from pickle")
        pkl_file = open(self.files_path + '.pkl', 'rb')
        self.w2v_dic = pickle.load(pkl_file)
        pkl_file.close()

    def create_and_load_dic(self):
        model = KeyedVectors.load_word2vec_format(self.files_path + '.bin', binary=True)
        kmeans = cluster.KMeans(n_clusters=self.num_clusters)
        kmeans.fit(model.wv.vectors)
        self.w2v_dic = dict(zip(model.wv.index2word, zip(model.wv.vectors, kmeans.labels_)))
        output = open(self.files_path + '.pkl', 'wb')
        pickle.dump(self.w2v_dic, output)
        output.close()

    # temp function, should be run once (not on virtual machine)
    # to run this function we need to uncomment line 32, and comment out line 33, 
    # then, run it by using "play.py" when it finish 
    # we should replace PubMed-w2v.pkl with PubMed-w2vcluster.pkl 
    def load_dic_workaround(self):
        pkl_file = open(self.files_path + '.pkl', 'rb')
        temp_w2v_dic = pickle.load(pkl_file)
        pkl_file.close()
        kmeans = cluster.KMeans(n_clusters=self.num_clusters)
        kmeans.fit(list(temp_w2v_dic.values()))
        self.w2v_dic = dict(zip(list(temp_w2v_dic.keys()), zip(list(temp_w2v_dic.values()), kmeans.labels_)))
        output = open(self.files_path + 'cluster'+ '.pkl', 'wb')
        pickle.dump(self.w2v_dic, output)
        output.close()
    
    def get(self, word):
        return self.w2v_dic.get(word, self.default_vector)
        
    def get_vector(self, word):
        return self.w2v_dic.get(word, self.default_dic)[0]
    
    def get_cluster(self, word):
        return self.w2v_dic.get(word, self.default_dic)[1]
    
    def addW2VFeature(self,element):
        #print("adding w2v features!")
        for feature in element["features"]:
            coords = self.get(feature["word"])
            
            for i in range(len(coords[:2])):
                dimension = coords[i]
                feature["w2v"+str(i)] = dimension
    
    def addW2VClusterFeature(self,element):
        for feature in element["features"]:
            feature["w2vcluster"] = self.get_cluster(feature["word"])
            

if __name__ == "__main__":

    W2V = W2VFeatures()
