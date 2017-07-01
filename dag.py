# some of these apply to directed graphs in general, not just DAGs
# I could consider creating a DAG class with these methods
from collections import defaultdict, deque

import networkx as nx
import numpy as np

def indegree_zero_nodes(dag):
    return [node for node in dag.nodes() if len(dag.in_edges(node)) == 0]


def ancestors(dag, node):
    return nx.ancestors(dag, node)


def descendants(dag, node):
    return nx.descendants(dag, node)


def parents(dag, node):
    return set(dag.predecessors(node))


def children(dag, node):
    return set(dag.successors(node))


def spouses(dag, node):
    children = set(dag.successors(node))
    all_spouses = set()
    for child in children:
        spouses = set(dag.predecessors(child))
        all_spouses |= spouses
    return all_spouses


def shared_children(dag, nodes):
    return set.intersection(*[set(dag.successors(node)) for node in nodes])


def parents_and_children(dag, node):
    return parents(dag, node).union(children(dag, node))


def markov_blanket(dag, node):
    mb = set()
    parents = set(dag.predecessors(node))
    mb |= parents
    children = set(dag.successors(node))
    mb |= children
    for child in children:
        spouses = set(dag.predecessors(child))
        mb |= spouses
    mb.discard(node)
    return mb


def dag_consistent_with_order(dag, order):
    ranks = {node: order.index(node) for node in dag.nodes()}
    for n1, n2 in dag.edges():
        if ranks[n2] <= ranks[n1]:
            return False
    return True


def random_topological_sort(dag):
    top_sort_list = []
    n_sorted = 0
    n_nodes = nx.number_of_nodes(dag)

    idzn = set()
    in_degrees = dag.in_degree()
    for node in dag.nodes():
        if in_degrees[node] == 0:
            idzn.add(node)

    while n_sorted < n_nodes:
        n = idzn.pop()
        top_sort_list.append(n)
        n_sorted += 1
        out_edges = dag.out_edges(n)
        for source, target in out_edges:
            in_degree = in_degrees[target]
            new_ind = in_degree - 1
            in_degrees[target] = new_ind
            if new_ind == 0:
                idzn.add(target)

    return top_sort_list


def is_valid_topological_order(dag, order):
    for n1, n2 in dag.edges():
        i1 = order.index(n1)
        i2 = order.index(n2)
        if i2 < i1:
            return False
    return True


def find_sources(dag):
    sources = set()
    in_degrees = dag.in_degree()
    #out_degrees = dag.out_degree()
    #print in_degrees
    #print out_degrees
    for node in dag.nodes():
        if in_degrees[node] == 0:
            sources.add(node)
    return sources


def distances_to_source(dag, method='min'):
    n_nodes = nx.number_of_nodes(dag)
    sources = find_sources(dag)
    all_dists = defaultdict(list)
    successors = dag.successors_iter
    for source in sources:
        #print "source: {}".format(source)
        dists = {node: n_nodes for node in dag.nodes()}
        dists[source] = 0
        queue = deque([(source, successors(source))])
        visited = set([source])
        while queue:
            parent, children = queue[0]
            child_dist = dists[parent] + 1
            try:
                child = next(children)
                if child not in visited:
                    #print "visiting {}".format(child)
                    dists[child] = child_dist
                    visited.add(child)
                    queue.append((child, successors(child)))
            except StopIteration:
                queue.popleft()
        for node in dag.nodes():
            all_dists[node].append(dists[node])
    if method == 'mean':
        return {node: np.mean(ds) for node, ds in all_dists.items()}
    elif method == 'min':
        return {node: min(ds) for node, ds in all_dists.items()}