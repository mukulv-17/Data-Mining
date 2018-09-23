from sklearn import svm
import networkx as nx
import operator
import math
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


graphx = nx.read_weighted_edgelist("higgs-social_network.edgelist.txt")

deg = list(graphx.degree)
deg.sort(key=lambda x : x[1], reverse=True)
max_deg_list = [x[0] for x in deg][:500]

subgraph=graphx.subgraph( max_deg_list ).copy()
nodes=list(subgraph.nodes)


def common_neighbours(G,a,b):
    a_n = list(G.neighbors(a))
    b_n = list(G.neighbors(b))
    return len(set(a_n) & set(b_n))

def jacard_coefficient(G,a,b):
    a_n = list(G.neighbors(a))
    b_n = list(G.neighbors(b))
    common = len(set(a_n) & set(b_n))
    union = len(set(a_n) | set(b_n))
    if union > 0:
        return common/union
    return 0

def adamic_adar(G,a,b):
    a_n = list(G.neighbors(a))
    b_n = list(G.neighbors(b))
    common_n = list(set(a_n).intersection(b_n))
    aa_value = 0
    for node in common_n:
        aa_value += 1/math.log(len(list(G.neighbors(node))))
    return aa_value

def resource_allocation(G,a,b):
    a_n = list(G.neighbors(a))
    b_n = list(G.neighbors(b))
    common_n = list(set(a_n).intersection(b_n))
    ra_value = 0
    for node in common_n:
        ra_value += 1/len(list(G.neighbors(node)))
    return ra_value

def preferential_attachment (G,a,b):
    a_n = len(list(G.neighbors(a)))
    b_n = len(list(G.neighbors(b)))
    return a_n*b_n

# getting 1000 non-edges and 1000 edges 
def get_edges():#return edges_0, edges_1
    edges_1 = []
    edges_0 = []

    num_nodes = len(subgraph.nodes())
    num_edges = len(subgraph.edges())
    nodes = list(subgraph.nodes())
    edges = list(subgraph.edges())
    size = 1000

    i=0
    while(i<size):
        n1 = nodes[random.randint(0, num_nodes - 1)]
        n2 = nodes[random.randint(0, num_nodes - 1)]
        while((str(n1),str(n2)) in edges or (str(n2),str(n1)) in edges or (str(n1),str(n2)) in edges_0 or (str(n2),str(n1)) in edges_0):
            n1 = nodes[random.randint(0, num_nodes - 1)]
            n2 = nodes[random.randint(0, num_nodes - 1)]
        edges_0+=[(str(n1),str(n2))]
        i+=1
        
    i=0    
    while(i<size):
        e1 = edges[random.randint(0, num_edges - 1)]
        while(e1 in edges_1):
            e1 = edges[random.randint(0, num_edges - 1)]
        edges_1+=[e1]
        i+=1

    for e in edges_1:
        subgraph.remove_edge(e[0],e[1])

    # nodes_n = list(subgraph.nodes())
    # edges_n = list(subgraph.edges())
    return edges_0, edges_1

#feature calculation
def calc_feature(edges_0,edges_1):#return features
    edges_t = edges_1 + edges_0 
    pa = {}
    ra = {}
    jc = {}
    aa = {}
    cn = {}

    for node_pair in edges_t:
        cn[node_pair] = common_neighbours(subgraph, node_pair[0],node_pair[1])
        jc[node_pair] = jacard_coefficient(subgraph, node_pair[0],node_pair[1])
        aa[node_pair] = adamic_adar(subgraph, node_pair[0],node_pair[1])
        ra[node_pair] = resource_allocation(subgraph, node_pair[0],node_pair[1])
        pa[node_pair] = preferential_attachment(subgraph, node_pair[0],node_pair[1])


    features=[cn,jc,aa,ra,pa]   
    return features


def classifier(graph,edges_0,edges_1,features):
    # non_existing_edges = random_fake_edges(graph, size)
    # t1 = existing_edge_dataset(graph,size)
    # t2 = non_existing_edge_dataset(graph,size,non_existing_edges)
    t1=[]
    for i in edges_1:
        t1+=[[f[i] for f in features]]
    t2=[]
    for i in edges_0:
        t2+=[[f[i] for f in features]]
    
    total_entries = len(t1)
    dataset = list()
    for entry in t1:#adding yes edges
        dataset.append(entry)
    for entry in t2:#adding no edges
        dataset.append(entry)
    y = list()#adding corresponding 1/0 values
    i = 0
    while i < total_entries:
        y.append(1)
        i += 1
    i = 0
    while i < total_entries:
        y.append(0)
        i += 1

    shuff_arr = list()
    j = 0
    for i in dataset:
        shuff_arr.append((i,y[j]))
        j += 1

    random.shuffle(shuff_arr)
    dataset = list()
    actual_result = list()

    for entry in shuff_arr:
        dataset.append(entry[0])
        actual_result.append(entry[1])

    kf = KFold(n_splits=5)
    for train, test in kf.split(dataset):
        training_dataset = [dataset[i] for i in train]
        tr_dataset_class = [actual_result[i] for i in train]
        testing_dataset = [dataset[i] for i in test]
        testing_dataset_class = [actual_result[i] for i in test]
        print("\n")
        x = np.array(training_dataset)
        clf = svm.SVC()
        clf.fit(x,tr_dataset_class)

        ### test
        correct = 0
        j = 0
        test_result = []
        for entry in testing_dataset:
            prediction = clf.predict([entry])
            # print(prediction)
            test_result.append(prediction[0])
            if prediction[0] == testing_dataset_class[j]:
                correct += 1
            j += 1

        precision = precision_score(testing_dataset_class, test_result)
        recall = recall_score(testing_dataset_class, test_result)
        accuracy = accuracy_score(testing_dataset_class, test_result)
        fpr,trp,threshold = metrics.roc_curve(testing_dataset_class,test_result)
        auc_val = metrics.auc(fpr,trp)
        print( "Correct predictions --> " + str(correct))
        print( "Precision score --> " + str(precision))
        print("recall score --> " + str(recall))
        print("Accuracy score --> " + str(accuracy))
        print("AUC --> " + str(auc_val))

## Main execution
edges_0,edges_1=get_edges()
features = calc_feature(edges_0,edges_1)
classifier(subgraph,edges_0,edges_1,features)
