import networkx as nx
import math
import itertools
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


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
def get_edges():
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
def calc_feature(edges_0,edges_1):
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

def gaussian(x,mean,stdev):
	return math.exp(-pow((x-mean),2)/(2*pow(stdev,2)))/(math.sqrt(2*math.pi)*stdev);

def prob(x,mean,stdev,ci,c):
	ans=1
	j=0
	#print(x)
	for i in x:
		ans*=gaussian(i,mean[j],stdev[j])
		j+=1
	ans*=(ci/c)
	return ans

def prediction(test,mean,stdev,c1,c2):
	yes_p=[]
	no_p=[]
	for np in test:
		p1=prob([f[np] for f in features],mean[0],stdev[0],c1,c1+c2)
		p2=prob([f[np] for f in features],mean[1],stdev[1],c2,c1+c2)
		if p1 > p2:
			no_p+=[np]
		else:
			yes_p+=[np]
#     print(len(no_p),len(yes_p))
	return [no_p, yes_p]

#shuffling and folding into 5
def calc_results(features,edges_0,edges_1):
	edges_t = edges_1 + edges_0 
	edges_t = shuffle(edges_t)
	def chunks(l, n):
		for i in range(0, len(l), n):
			yield l[i:i+n]
	edges_t=list(chunks(edges_t,int(len(edges_t)/5)))

	precision=[]
	recall=[]
	accuracy=[]
	for i in range(5):
		test=edges_t[i]
		train=[]
		for j in range(5):
			if i!=j:
				train+=edges_t[j]
		
		split_edges = []
		split_edges+=[[x for x in train if x in edges_0 ]] #no edges
		split_edges+=[[x for x in train if x in edges_1 ]] #yes edges
		
		mean=[]
		stdev=[]
		for et in split_edges:
			m=[]
			s=[]
			for f in features:
				temp=[f[e] for e in et]
				m+=[np.average(temp)]
				s+=[np.std(temp)]
			mean+=[m]
			stdev+=[s]
			
	#     print(mean,stdev)
		predicted = prediction(test,mean,stdev,len(split_edges[0]),len(split_edges[1]))
		
		#calculate precision/recall
		true_pos=0
		actual_p=0
		for x in predicted[1]:
			if x in edges_1:
				true_pos+=1
		precision+=[true_pos/len(predicted[1])*100]
		
		for x in test:
			if x in edges_1:
				actual_p+=1
	#     print(true_pos , actual_p)
		recall+=[true_pos/actual_p*100]
		
		#calculation of accuracy
		true_neg=0
		for x in predicted[0]:
			if x in edges_0:
				true_neg += 1

		accuracy += [( (true_neg+true_pos)/len(test))]
		
		print( "Correct predictions --> " + str(true_pos+true_neg))
		print( "Precision score --> " + str(precision[i]))
		print("recall score --> " + str(recall[i]))
		print("Accuracy score --> " + str(accuracy[i]))


edges_0,edges_1 = get_edges()
features = calc_feature(edges_0,edges_1)
calc_results(features,edges_0,edges_1)
