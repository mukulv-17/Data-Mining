import networkx as nx
import math

graphx = nx.read_weighted_edgelist("higgs-mention_network.edgelist.txt",create_using=nx.DiGraph())

# we take the subgraph of size 500 nodes with maximum degree
deg = list(graphx.degree)
deg.sort(key=lambda x : x[1], reverse=True)
max_deg_list = [x[0] for x in deg][:500]
subgraph=graphx.subgraph( max_deg_list )

def betweenness_cent(G):
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    for s in G:
        for t in G:
            if(s==t):
                continue
            if(nx.has_path(subgraph,s,t)):
                all_shortest_paths = list(nx.all_shortest_paths(G, s, t, weight='weight'))
                sigma_st = len(all_shortest_paths)
                if(sigma_st!=0):
                    for p in all_shortest_paths:
                        for v in p[1:-1]:
                            betweenness[v]+=1/sigma_st
    return betweenness

#using own function
with open('betweenness_centrality.txt', 'w') as f:
    print(sorted(betweenness_cent(subgraph).items(), key=lambda kv: kv[1], reverse=True), file=f)  # Python 3.x

#using library function
with open('betweenness_centrality_lib.txt', 'w') as f:
    print (sorted(nx.betweenness_centrality(subgraph, normalized=False, weight='weight', endpoints=False).items(), key=lambda kv: kv[1], reverse=True),file=f)
