import networkx as nx
# check if pygraphviz is installed
import matplotlib.pyplot as plt
import copy
from itca.utils import bidict


def rename_node(node):
    level = node[0]
    if level == 0:
        name = node[1][0] + 1
    else:
        name = "c" + str(rename_node.comb_class)
        rename_node.comb_class += 1
    return name

def rename_G(G):
    rename_node.comb_class = 1
    nodes = G.nodes()
    rename_mapping = {node:  rename_node(node) for node in nodes}
    G_ = copy.copy(G)
    G_ = nx.relabel_nodes(G, rename_mapping)
    return G_
def compare_neigbor_bidict(m1, m2):
    assert(len(m1.inverse) - len(m2.inverse) == 1)
    s1 = set(tuple(t) for t in m1.inverse.values())
    s2 = set(tuple(t) for t in m2.inverse.values())
    sym_diff = s2.symmetric_difference(s1)
    i, j, ij = sorted(list(sym_diff), key=len)
    node_i = (len(i) - 1, i)
    node_j = (len(j) - 1, j)
    node_ij = (len(ij) - 1, ij)
    return node_i, node_j, node_ij


def gen_tree(path):
    G = nx.Graph()
    cur = path
    while cur.children:
        cur_m = bidict(cur.mapping)
        next_m = bidict(cur.children[0].mapping)
        node_i, node_j, node_ij = compare_neigbor_bidict(cur_m, next_m)
        G.add_node(node_i)
        G.add_node(node_j)
        G.add_node(node_ij)
        G.add_edge(node_i, node_ij)
        G.add_edge(node_j, node_ij)
        cur = cur.children[0]
    m1 = bidict(cur.mapping)
    m2 = bidict({i:0 for i in range(38)})
    node_i, node_j, node_ij = compare_neigbor_bidict(m1, m2)
    G.add_node(node_ij)
    G.add_edge(node_i, node_ij)
    G.add_edge(node_j, node_ij)
    return G


def plot_tree(strategy=None, graph=None, figsize=(6, 12), node_size=350, direction="vertical",
              return_graph=False, use_pygraphviz=False):
    """
    :param gs: instance of SearchStrategy
    :param figsize: size of the figure
    :param node_size: plot node size, input to nx.draw_networkx
    :param direction: direction of the tree, "vertical" or "horizontal"
    :param return_graph: if True, return the figure
    :return:  return the graph and figure if return_graph is True, else return figure
    """
    # Either strategy or G must be provided
    assert(strategy is not None or graph is not None)
    if graph is None:
        G = gen_tree(strategy.path)
    else:
        G = copy.copy(graph)
    G_ = rename_G(G)
    if use_pygraphviz:
        try:
            import pygraphviz
            from networkx.drawing.nx_agraph import graphviz_layout
        except ImportError:
            raise ImportError("requires pygraphviz to visualize tree")
        pos = nx.nx_agraph.graphviz_layout(G_, prog="dot")
    else:
        try:
            pos = nx.nx_pydot.pydot_layout(G_, prog='dot')
        except FileNotFoundError:
            raise FileNotFoundError("requires pydot to visualize tree; please find the troubleshooting guide at https://github.com/JSB-UCLA/ITCA")
    if direction == "horizontal":
        flipped_pos = {node: (y, -x) for (node, (x,y)) in pos.items()}
        pos = flipped_pos
    elif direction == "vertical":
        pos = {node: (x, -y) for (node, (x,y)) in pos.items()}
    fig, ax = plt.subplots(figsize=(6, 12))
    nx.draw_networkx(G_, pos, node_size=node_size, with_labels=True)
    if return_graph:
        return G_, fig
    else:
        return fig