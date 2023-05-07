import dgl
import torch
from helper.io import *
import networkx as nx
import matplotlib.pyplot as plt

nodes = load('xxx/nodes.pkl')
edges = load('xxx/edges.pkl')

print(edges)
ts = []
for i in range(len(edges[0])):
    src = nodes[edges[0][i]]
    dst = nodes[edges[1][i]]
    ts.append((src, dst))

G = nx.DiGraph()
G.add_edges_from(ts)
nx.draw(G, with_labels=True)
plt.show()
