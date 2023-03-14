import streamlit as st
import netlsd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
import multiprocessing as mp
from itertools import repeat
import pickle
import numpy as np
import networkx as nx
import sys

def train_with_10_folds(embeddings,labels):
    estimator  = RandomForestClassifier(criterion='gini', max_depth=None, min_weight_fraction_leaf=0.0,
                                            max_leaf_nodes=None, bootstrap=True, 
                                            oob_score=False, n_jobs=5,verbose=0, warm_start=False,
                                            class_weight=None)
    kf = StratifiedKFold(n_splits=10, random_state = 123, shuffle = True)
    st.write("running model!")
    score = cross_val_score(estimator, embeddings, labels, cv = kf, scoring="accuracy")
    st.write("accuracy:", round(np.mean(score)*100,3))
    
def train_fixed_train_test(embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=123,stratify=labels)
    estimator  = RandomForestClassifier(criterion='gini', max_depth=None, min_weight_fraction_leaf=0.0,
                                            max_leaf_nodes=None, bootstrap=True, 
                                            oob_score=False, n_jobs=5,verbose=0, warm_start=False,
                                            class_weight=None)
    estimator.fit(X_train, y_train)
    name = "models/PTC_NetLSD_trained.sav"
    pickle.dump(estimator, open(name,'wb'))
    test_obj = {"embeddings":np.array(X_test), "labels":y_test}
    pickle.dump(test_obj,open("data/PTC_NetLSD_test_set.pkl","wb"))
    pred = estimator.predict(X_test)
    acc = accuracy_score(pred, y_test)
    st.write("accuracy:", round(acc*100,3))

def train_descriptor(dataset,model, eval, metric):
    graphs,labels = dataset['graphs'],dataset['labels']
    if model=="DGSD":
        dgsd = DGSD()
        embeddings = [dgsd.get_descriptor(g,bins = 50, workers = 1) for g in graphs]
    elif model=="NetLSD":
        embeddings = [netlsd.heat(g) for g in graphs]
    if eval=="10-folds":
        train_with_10_folds(embeddings,labels)
    else:
        train_fixed_train_test(embeddings,labels)

def train_model(dataset_name,model, eval, metric):
        
    with open('data/'+dataset_name+'.pkl', 'rb') as file:
        dataset = pickle.load(file)
        file.close()
    st.write(dataset_name," loaded successfully")
    if model=="DGSD" or model=="NetLSD":
        train_descriptor(dataset,model, eval, metric)
    else:
        st.write("not implemented yet!")



def inference_with_pre_trained(dataset,model, eval, metric):
    estimator = pickle.load(open("models/"+dataset+"_"+model+"_trained.sav","rb"))
    test_data = pickle.load(open("data/"+dataset+"_"+model+"_test_set.pkl","rb"))
    pred = estimator.predict(test_data['embeddings'])
    acc = accuracy_score(pred, test_data['labels'])
    

    st.write("accuracy:", round(acc*100,3))

    

  
def classify_graph(graph):
    name = "models/shapes_NetLSD_trained.sav"
    estimator = pickle.load(open(name, "rb"))
    emb = netlsd.heat(graph).reshape(1,-1)
    pred = list(estimator.predict_proba(emb).reshape(-1,))
    results = {'complete_graph':pred[0],'cycle_graph':pred[1],
               'ladder_graph':pred[2],'path_graph':pred[3],
               'star_graph':pred[4],'wheel_graph':pred[5]
               ,'barbell_graph':pred[6],'lollipop_graph':pred[7],
               'fast_gnp_random_graph':pred[8],'turan_graph':pred[9]}
    st.write("Predicted Probabilities")
    st.write(results)

def run_model(dataset,model, eval, metric,train):
   
    if model=='GCN' or model=='GAT':
        st.write("not implemented yet!")
        sys.exit()
    if train=="train":
        train_model(dataset,model, eval, metric)
    else:
        inference_with_pre_trained(dataset,model, eval, metric)



class DGSD:
    def __init__(self):
        self.graph = None
    def get_descriptor(self, graph, bins = 50, workers=1):
        p = mp.Pool(workers)
        self.graph = nx.convert_node_labels_to_integers(graph)
        nodes = list(self.graph.nodes())
        if workers<len(nodes):
            batches = np.array_split(nodes, workers)
        else:
            workers = len(nodes)
            batches = np.array_split(nodes, workers)
        emb = p.starmap(self.Generate_Embeddings, zip(batches, repeat(bins)))
        embeddings = np.sum(np.array(emb),axis = 0)
        p.terminate()
        return embeddings

    def Generate_Embeddings(self, batch, nbins):
        total_nodes = self.get_nodes()
        d = []
        for v in batch:
            N_v = self.request_neighbors(v)
            d_v = len(N_v)
            for u in range(total_nodes):
                if u == v:
                    d.append(0)
                    continue
                N_u = self.request_neighbors(u)
                delta = 0
                if u in N_v:
                    delta = 1
                else:
                    delta = 0
                d_u = len(N_u)
                common = len(list(set(N_u) & set(N_v)))
                dist = 0
                if (((d_u + d_v) + common + delta))>0:
                    dist = (d_u + d_v) / ((d_u + d_v) + common + delta)
                delta = 0
                d.append(dist)
        hist, bin_edges = np.histogram(d, range=(0, 1), bins=nbins)
        return hist

    def request_neighbors(self, node):
        N_i = list(self.graph.neighbors(node))
        return N_i

    def get_nodes(self):
        return self.graph.number_of_nodes()
