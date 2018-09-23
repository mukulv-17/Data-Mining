import networkx as nx
import math

    
graphx = nx.read_weighted_edgelist("higgs-mention_network.edgelist.txt",create_using=nx.DiGraph())

# we take the subgraph of size 500 nodes with maximum degree
deg = list(graphx.degree)
deg.sort(key=lambda x : x[1], reverse=True)
max_deg_list = [x[0] for x in deg][:500]
subgraph=graphx.subgraph( max_deg_list )

# Clustering Coefficient
def clustering_coefficient(G):

    clustering_coeff = dict.fromkeys(G, 0.0)
    for u in G:
        triangles = 0
        neighb = 0
        for node1 in G.neighbors(u):
            if u!=node1 :
                neighb += 1
                for node2 in G.neighbors(u):
                    if node1 != node2 and node2 != u:
                        if G.has_edge(node1 ,node2):
                            triangles += 1
        if neighb > 1 :
            clustering_coeff[u] = ( triangles/(neighb * (neighb - 1) ))

    return clustering_coeff

und_subgraph = subgraph.to_undirected()

#using own function
with open('clustering_coeff.txt', 'w') as f:
    print(sorted(clustering_coefficient(und_subgraph).items(), key=lambda kv: kv[1], reverse=True), file=f)  # Python 3.x

#using library function
with open('clustering_coeff_lib.txt', 'w') as f:
    print (sorted(nx.clustering(und_subgraph, weight='None').items(), key=lambda kv: kv[1], reverse=True),file=f)
