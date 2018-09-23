import networkx as nx
import math

graphx = nx.read_weighted_edgelist("higgs-mention_network.edgelist.txt",create_using=nx.DiGraph())

# we take the subgraph of size 500 nodes with maximum degree
deg = list(graphx.degree)
deg.sort(key=lambda x : x[1], reverse=True)
max_deg_list = [x[0] for x in deg][:500]
subgraph=graphx.subgraph( max_deg_list )


def eigenvector_centrality(G):
    
    from math import sqrt
    x = dict([(n,1.0/len(G)) for n in G])
    s = 1.0/sum(x.values())
    for k in x:
        x[k] *= s
    nnodes = G.number_of_nodes()
 
    for i in range(30):
        xlast = x
        x = dict.fromkeys(xlast, 0)
 
        for n in x:
            for nbr in G[n]:
                x[nbr] += xlast[n] * G[n][nbr].get('weight', 1)
 
        try:
            s = 1.0/sqrt(sum(v**2 for v in x.values()))
 
        except ZeroDivisionError:
            s = 1.0
        for n in x:
            x[n] *= s
 
        err = sum([abs(x[n]-xlast[n]) for n in x])
        if err < nnodes*1.0e-6:
            return x

#using own function
with open('eigenvector_centrality.txt', 'w') as f:
    print(sorted(eigenvector_centrality(subgraph).items(), key=lambda kv: kv[1], reverse=True), file=f)  # Python 3.x

#using library function
with open('eigenvector_centrality_lib.txt', 'w') as f:
    print (sorted(nx.eigenvector_centrality(subgraph,max_iter=30,weight='weight').items(), key=lambda kv: kv[1], reverse=True),file=f)
