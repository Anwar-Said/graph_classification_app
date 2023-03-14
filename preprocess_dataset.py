from torch_geometric.datasets import TUDataset,GeometricShapes
from torch_geometric.utils import to_networkx
import pickle
import networkx as nx
import torch_geometric.transforms as T
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from networkx import complete_graph,cycle_graph,ladder_graph,path_graph,star_graph,diamond_graph,wheel_graph,barbell_graph,lollipop_graph,fast_gnp_random_graph,turan_graph
import netlsd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
# dataset = TUDataset("torch_datasets/", name ="ENZYMES")
# graphs = [to_networkx(x, to_undirected =True) for x in dataset]
# labels = [x.y.item() for x in dataset]
# print(len(graphs), len(labels))
# data_dic = {"graphs":graphs,"labels":labels}
# file = "data/ENZYMES.pkl"
# pickle.dump(data_dic, open(file, "wb"))
# dataset = pickle.load(open("data/ENZYMES.pkl","rb"))
# print(len(dataset['graphs']))
# shapes = GeometricShapes("torch_datasets/",train= True,transform=T.FaceToEdge())
# for s in shapes:
#     print(s)
# sys.exit()
# graphs = [to_networkx(x, to_undirected =True) for x in shapes]
# labels = [x.y.item() for x in shapes]
# print(len(graphs), labels)
# data_dic = {"graphs":graphs,"labels":labels}
# file = "data/shapes.pkl"
# pickle.dump(data_dic, open(file, "wb"))
# dataset = pickle.load(open("data/shapes.pkl","rb"))
# print(len(dataset['graphs']))
# nx.draw_networkx(graphs[0])
# plt.show()


graph_types =['complete_graph','cycle_graph','ladder_graph','path_graph','star_graph','wheel_graph','barbell_graph','lollipop_graph','fast_gnp_random_graph','turan_graph'] 
def generate_turan_graphs(name):
    type_g = eval(name)
    graphs = []
    sizes = np.arange(10,50)
    comp = np.arange(2,10)
    for i in range(200):
        n1 = random.choice(sizes)
        n2 = random.choice(comp)
        g = type_g(n1,n2)
        node_mapping = dict(zip(g.nodes(), sorted(g.nodes(), key=lambda k: random.random())))
        permuted_g = nx.relabel_nodes(g, node_mapping)
        graphs.append(permuted_g)
    return graphs
def generate_fast_gnp_random_graphs(name):
    type_g = eval(name)
    graphs = []
    sizes = np.arange(10,50)
    for i in range(200):
        n1 = random.choice(sizes)
        n2 = random.random()
        g = type_g(n1,n2)
        node_mapping = dict(zip(g.nodes(), sorted(g.nodes(), key=lambda k: random.random())))
        permuted_g = nx.relabel_nodes(g, node_mapping)
        graphs.append(permuted_g)
        
    return graphs
def generate_lollipop_graphs(name):
    type_g = eval(name)
    graphs = []
    comp = np.arange(5,20)
    tail = np.arange(1,10)
    for i in range(200):
        n1 = random.choice(comp)
        n2 = random.choice(tail)
        g = type_g(n1,n2)
        node_mapping = dict(zip(g.nodes(), sorted(g.nodes(), key=lambda k: random.random())))
        permuted_g = nx.relabel_nodes(g, node_mapping)
        graphs.append(permuted_g)
    return graphs
def generate_barbell_graphs(name):
    type_g = eval(name)
    graphs = []
    sizes = np.arange(5,20)
    for i in range(200):
        n1 = random.choice(sizes)
        n2 = random.choice(sizes)
        g = type_g(n1,n2)
        node_mapping = dict(zip(g.nodes(), sorted(g.nodes(), key=lambda k: random.random())))
        permuted_g = nx.relabel_nodes(g, node_mapping)
        graphs.append(permuted_g)
    return graphs

def generate_graphs(name):
    type_g = eval(name)
    graphs = []
    sizes = np.arange(10,50)
    
    if name=='barbell_graph':
        graphs = generate_barbell_graphs(name)
        return graphs
    elif name =='lollipop_graph':
        graphs = generate_lollipop_graphs(name)
        return graphs
    elif name=='fast_gnp_random_graph':
        graphs = generate_fast_gnp_random_graphs(name)
        return graphs
    elif name=='turan_graph':
        graphs = generate_turan_graphs(name)
        return graphs
    else:    
        for i in range(200):
            n = random.choice(sizes)
            g = type_g(n)
            node_mapping = dict(zip(g.nodes(), sorted(g.nodes(), key=lambda k: random.random())))
            permuted_g = nx.relabel_nodes(g, node_mapping)
            graphs.append(permuted_g)
        return graphs
# dataset = {}
# index = 0

# for t in graph_types:
#     print(t)
#     data = generate_graphs(t)
#     dataset[index] = data
#     index +=1
# print(len(dataset))
file = 'data/shapes.pkl'
# pickle.dump(dataset, open(file, "wb"))
new_data = pickle.load(open(file,'rb'))
# print(len(new_data))

# for k, v in new_data.items():
#     print(k)
#     for g in v:
#         print(nx.info(g))
#         break

def get_embeddings(new_data):
    embeddings,labels = [],[]
    for y, graphs in new_data.items():
        for g in graphs:
            emb = netlsd.heat(g)
            embeddings.append(emb)
            labels.append(y)
    return embeddings,labels
def train_fixed_train_test(embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.1, random_state=123,stratify=labels)
    print(len(y_test))
    estimator  = RandomForestClassifier(criterion='gini', max_depth=None, min_weight_fraction_leaf=0.0,
                                            max_leaf_nodes=None, bootstrap=True, 
                                            oob_score=False, n_jobs=5,verbose=0, warm_start=False,
                                            class_weight=None)
    estimator.fit(X_train, y_train)
    name = "models/shapes_NetLSD_trained.sav"
    pickle.dump(estimator, open(name,'wb'))
    # test_obj = {"embeddings":np.array(X_test), "labels":y_test}
    # pickle.dump(test_obj,open("data/PTC_NetLSD_test_set.pkl","wb"))
    pred = estimator.predict(X_test)
    acc = accuracy_score(pred, y_test)
    print("accuracy:", round(acc*100,3))

# embeddings, labels = get_embeddings(new_data)

# train_fixed_train_test(embeddings, labels)

def save_test_samples(new_data):
    graphs, labels = [],[]
    for k, v in new_data.items():
        for g in v:
            graphs.append(g)
            labels.append(k)
        
    X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.1, random_state=123,stratify=labels)
    print(y_test)
    indexes = [5,6,0,8,25,1,10,4,3,14]
    for index, i in enumerate(indexes):
        g = X_test[i]
        pickle.dump(g, open("test_samples/"+str(index)+"_test.pkl","wb"))
save_test_samples(new_data)
