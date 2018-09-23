import networkx as nx
import math

graphx = nx.read_weighted_edgelist("higgs-mention_network.edgelist.txt",create_using=nx.DiGraph())

# we take the subgraph of size 500 nodes with maximum degree
deg = list(graphx.degree)
deg.sort(key=lambda x : x[1], reverse=True)
max_deg_list = [x[0] for x in deg][:500]
subgraph=graphx.subgraph( max_deg_list )

shortest_dist=nx.floyd_warshall(subgraph)
closeness_centrality = {}

for key,value in shortest_dist.items():
    total_nodes=0.0
    sum_dist=0.0
    for key2,value2 in value.items():
        if( value2 != math.inf):
            sum_dist+=value2
            total_nodes+=1
    if(total_nodes == 1):
        closeness_centrality[key]=0
    else:
        closeness_centrality[key]=(total_nodes-1)/sum_dist

#using own function
with open('closeness_centrality.txt', 'w') as f:
    print(sorted(closeness_centrality.items(), key=lambda kv: kv[1], reverse=True), file=f)  # Python 3.x

#using library function
with open('closeness_centrality_lib.txt', 'w') as f:
    print (sorted(nx.closeness_centrality(subgraph,distance='weight',wf_improved =False,reverse=True).items(), key=lambda kv: kv[1], reverse=True),file=f)
