import networkx as nx
import math

graphx = nx.read_weighted_edgelist("higgs-mention_network.edgelist.txt",create_using=nx.DiGraph())

# we take the subgraph of size 500 nodes with maximum degree
deg = list(graphx.degree)
deg.sort(key=lambda x : x[1], reverse=True)
max_deg_list = [x[0] for x in deg][:500]
subgraph=graphx.subgraph( max_deg_list )


degree_centrality={}

for x in subgraph:
    degree_centrality[x]=subgraph.out_degree(x)/(500-1)

#using own function
with open('degree_centrality.txt', 'w') as f:
    print(sorted(degree_centrality.items(), key=lambda kv: kv[1], reverse=True), file=f)  # Python 3.x

#using library function
with open('degree_centrality_lib.txt', 'w') as f:
    print (sorted(nx.out_degree_centrality(subgraph).items(), key=lambda kv: kv[1], reverse=True),file=f)
