import itertools
import math
import string
import os
import re

import networkx as nx
import numpy as np
import pandas as pd


def structural_hamming_distance(g1, g2, pdag=False):
    e1 = set(g1.edges())
    e2 = set(g2.edges())
    d1 = e1.difference(e2)
    l1 = len(d1)
    d2 = e2.difference(e1)
    l2 = len(d2)
    hamming_distance = l1 + l2
    if pdag:
        smaller_diff, graph, other_edge_set = (d1, g1, e2) if l1 < l2 else (d2, g2, e1)
        for n1, n2 in smaller_diff:
            if (n2, n1) in other_edge_set:
                if graph[n1][n2]['cpdag_edge_type'] == "reversible":
                    hamming_distance -= 2
    return hamming_distance


def pairwise_structural_hamming_distances(dags, pdag=True, add_labels=True, n_samples=None):
    if add_labels:
        for dag in dags:
            add_cpdag_edge_labels(dag, inplace=True)
    n_dags = len(dags)
    dag_idxs = range(n_dags)
    if n_samples is not None:
        sample_idxs = [np.random.choice(dag_idxs, size=2, replace=False) for n in range(n_samples)]  # repeats are possible but not worth bothering with
        combos = [(dags[idx_pair[0]], dags[idx_pair[1]]) for idx_pair in sample_idxs]
    else:
        combos = itertools.combinations(dags, 2)
    hds = [structural_hamming_distance(pd1, pd2, pdag=pdag) for pd1, pd2 in combos]
    return hds

############################# generating samples ##############################

def multi_node_query(data, nodes, states):
    query_condition_strings = ["({}=={})".format(pair[0], pair[1]) for pair in zip(nodes, states)]
    query_string = " & ".join(query_condition_strings)
    matching_samples = data.query(query_string)
    return matching_samples


def counts_over_possible_states(data, node, possible_values):
    n_possible_values = len(possible_values)
    counts = []
    cum_count = 0
    for i in range(n_possible_values - 1):
        value = possible_values[i]
        matching_samples = data[data[node] == value]
        count = matching_samples.shape[0]
        counts.append(count)
        cum_count += count
    counts.append(data.shape[0] - cum_count)
    #print node
    #print zip(possible_values, counts)
    return counts


def make_cpds(dag, unique_vals, data=None):
    """
    Todo: allow non-uniform Dirichlet parameters based on data
    since we have binary events, it might be faster to sample from uniform and subtract from it for the other...
        this could save time when we have large networks? it seems to struggle.
    """
    order = nx.topological_sort(dag)
    cpds = {node: {} for node in order}
    for node in order:
        child_uvs = unique_vals[node]
        n_child_uvs = len(child_uvs)
        ns_obs_plus_1 = [1] * n_child_uvs
        parents = dag.predecessors(node)
        if len(parents) == 0:
            if data is None:
                cpds[node][node] = np.random.dirichlet(ns_obs_plus_1, 1)[0]
            else:
                counts_plus_1 = [count + 1 for count in counts_over_possible_states(data, node, child_uvs)]
                cpds[node][node] = np.random.dirichlet(counts_plus_1, 1)[0]
        else:
            parents_uvs = [unique_vals[parent] for parent in parents]
            parent_uv_states = list(itertools.product(*parents_uvs))
            for pu in parent_uv_states:
                if data is None:
                    cpds[node][pu] = np.random.dirichlet(ns_obs_plus_1, 1)[0]
                else:
                    parent_matching_samples = multi_node_query(data, parents, pu)
                    counts_plus_1 = [count + 1 for count in counts_over_possible_states(parent_matching_samples, node, child_uvs)]
                    cpds[node][pu] = np.random.dirichlet(counts_plus_1, 1)[0]

    return cpds


def generate_samples(dag, cpds, unique_values, n_samples):
    samples = []
    order = nx.topological_sort(dag)
    for i in range(n_samples):
        node_values = {}
        for node in order:
            if node in cpds[node]:
                cpd = cpds[node][node]
            else:
                parents = dag.predecessors(node)
                parent_state = [node_values[parent] for parent in parents]
                cpd = cpds[node][tuple(parent_state)]
            node_values[node] = sample_node_value(cpd, unique_values[node])
        sample = [node_values[n] for n in order]
        samples.append(sample)
    samples_df = pd.DataFrame(samples, columns=order)
    return samples_df


def sample_node_value(cpd, sorted_possible_values):
    sample = np.random.random()
    #print sample
    i = 0
    cum_prob = 0
    for p in cpd:
        cum_prob += p
        if sample < cum_prob:
            return sorted_possible_values[i]
        i += 1


def test_bn_sample_generation():
    dag = nx.DiGraph()
    dag.add_edges_from([("B", "A"), ("E", "A"), ("A", "J"), ("A", "M")]) # Earthquake network
    unique_vals = {node: (0, 1) for node in dag.nodes()}
    cpds = make_cpds(dag, unique_vals)
    samples = generate_samples(dag, cpds, unique_vals, 10)
    print(samples)

    cpds = make_cpds(dag, unique_vals, data=samples)
    samples = generate_samples(dag, cpds, unique_vals, 10)
    print(samples)
    """
    I want a similar distribution of rates of mutation
    And edges between nodes that really are dependent
    Could use the MAP model or any from its equivalence class
    While not guaranteed to be right, it probably has similar qualities...?

    """


def generate_random_connected_dag(n_nodes, prob_edge):
    alphabet = string.ascii_uppercase
    exponent = int(math.ceil(math.log(n_nodes, 26)))
    node_names = itertools.product(alphabet, repeat=exponent)
    connected = False
    while not connected:
        #print "Trying again for a connected graph"
        dag = generate_random_dag(n_nodes, prob_edge)
        connected = nx.is_connected(dag.to_undirected())
    dag = nx.relabel_nodes(dag, {node: str("".join(node_names.next())) for node in dag.nodes()})
    return dag


def generate_random_dag(n_nodes, prob_edge):
    """
    Todo: limit number of parents
    Todo: generate with different distributions of in and out degrees
    Todo: choose a way that guarantees connectedness? repeatedly trying for it may be expensive
    """
    adj = np.random.random((n_nodes, n_nodes))
    adj[adj > prob_edge] = 0
    adj = np.tril(adj, -1)
    #print adj
    dag = nx.DiGraph(adj)
    assert(nx.is_directed_acyclic_graph(dag))
    return dag


def generate_random_connected_dags(n_dags, n_nodes, prob_edge):
    return [generate_random_dag(n_nodes, prob_edge) for _ in n_dags]


def save_dags_and_datasets(dags, datasets, out_dir):
    for i in range(len(dags)):
        dag = dags[i]
        dataset = datasets[i]
        dag_file = os.path.join(out_dir, "dag_{}.graphml".format(i))
        nx.write_graphml(dag, dag_file)
        dataset_file = os.path.join(out_dir, "dataset_{}.txt".format(i))
        dataset.to_csv(dataset.to_csv(dataset_file))


def load_dags_and_datasets(out_dir):
    files = os.listdir(out_dir)
    index_re = re.compile("_([0-9]+)\.")
    dag_files = [f for f in files if f.startswith("dag")]
    dataset_files = [f for f in files if f.startswith("dat")]
    dag_files.sort(key=lambda x: int(index_re.search(x).group(1)))
    #print dag_files
    dataset_files.sort(key=lambda x: int(index_re.search(x).group(1)))
    #print dataset_files
    dags = [nx.read_graphml(os.path.join(out_dir, f)) for f in dag_files]
    datasets = [pd.read_csv(os.path.join(out_dir, f), index_col=0) for f in dataset_files]
    return dags, datasets

###############################################################################
############################## DAGs and CPDAGs ################################

def markov_equivalent_dags(dag):
    labeled_dag = add_cpdag_edge_labels(dag)
    dags = dags_from_cpdag(labeled_dag)
    return dags


def order_edges(dag):
    """
    Chickering 1995, 2002
    """
    n_nodes = nx.number_of_nodes(dag)
    edge_order_dict = {}
    unordered_edges = set(dag.edges())
    #tsort = random_topological_sort(dag) # needn't be random
    tsort = nx.topological_sort(dag)
    i = 0
    while len(unordered_edges) > 0:
        for i in range(n_nodes):
            y = tsort[i]
            #print "y:", y
            in_edges =  dag.in_edges(y)
            unordered_incident_edges = [in_edge for in_edge in in_edges if in_edge in unordered_edges]
            if len(unordered_incident_edges) > 0:
                xs = sorted([e[0] for e in unordered_incident_edges], reverse=True)
                for x in xs:
                    edge_order_dict[(x, y)] = i
                    i += 1
                    unordered_edges.discard((x, y))
    edge_order_pairs = edge_order_dict.items()
    #print edge_order_pairs
    sorted_edges = sorted(edge_order_pairs, key=lambda x:x[1])
    return [s[0] for s in sorted_edges]


def add_cpdag_edge_labels(source_dag, inplace=False, refresh=False):
    if inplace:
        dag = source_dag
    else:
        dag = source_dag.copy()
    sorted_edges = order_edges(dag)
    n_edges = len(sorted_edges)
    some_edge = dag.edges()[0]
    if "cpdag_edge_type" not in dag[some_edge[0]][some_edge[1]] or refresh:
        unlabeled_edges = set(dag.edges())
    else:
        unlabeled_edges = set(edge for edge in dag.edges() if dag[edge[0]][edge[1]]['cpdag_edge_type'] == 'reversible')

    lowest_order_unlabeled_edge_index = 0
    while len(unlabeled_edges) > 0:
        for i in range(lowest_order_unlabeled_edge_index, n_edges):
            x, y = sorted_edges[i]
            if (x, y) in unlabeled_edges:
                lowest_order_unlabeled_edge_index = i + 1
                break
        in_edges = dag.in_edges(x)
        skip_rest = False
        for in_edge in in_edges:
            w, x_again = in_edge
            if "cpdag_edge_type" in dag[w][x]:
                if dag[w][x]["cpdag_edge_type"] == "compelled":
                    if w not in dag.predecessors(y):
                        dag[x][y]["cpdag_edge_type"] = "compelled"
                        unlabeled_edges.discard((x, y))
                        skip_rest = True
                        break
                        for y_in_edge in dag.in_edges(y):
                            q, y_again = y_in_edge
                            dag[q][y]["cpdag_edge_type"] = "compelled"
                            unlabeled_edges.discard(y_in_edge)
                    else:
                        dag[w][y]["cpdag_edge_type"] = "compelled"
                        unlabeled_edges.discard((w, y))

        if not skip_rest:
            zb = False
            for y_in_edge in dag.in_edges(y):
                z, y_again = y_in_edge
                if z != x and z not in dag.predecessors(x):
                    zb = True
                    break
            if zb:
                zlabel = "compelled"
            else:
                zlabel = "reversible"
            dag[x][y]["cpdag_edge_type"] = zlabel
            unlabeled_edges.discard((x, y))
            for y_in_edge in dag.in_edges(y):
                if y_in_edge in unlabeled_edges:
                    q, y_again = y_in_edge
                    dag[q][y]["cpdag_edge_type"] = zlabel
                    unlabeled_edges.discard(y_in_edge)
    return dag


def dags_from_cpdag(labeled_dag):
    dag_list = []
    dag_list = recurse_reversible_edge(labeled_dag, dag_list)
    return dag_list


def recurse_reversible_edge(labeled_dag, dag_list):
    rev_edges = select_reversible_edges(labeled_dag)
    if len(rev_edges) == 0:
        dag_list.append(labeled_dag)
    else:
        rev_edge = rev_edges[0]
        n1, n2 = rev_edge

        g1 = labeled_dag.copy()
        g1[n1][n2]["cpdag_edge_type"] = "compelled"

        g2 = labeled_dag.copy()
        g2.remove_edge(n1, n2)
        g2.add_edge(n2, n1)
        g2[n2][n1]["cpdag_edge_type"] = "compelled"

        for g in [g1, g2]:
            #cpdag = add_cpdag_edge_labels(g) # faster but not right?
            cpdag = complete_pdag(g)
            recurse_reversible_edge(cpdag, dag_list)
    return dag_list


def select_reversible_edges(pdag):
    return [edge for edge in pdag.edges() if pdag[edge[0]][edge[1]]['cpdag_edge_type'] == 'reversible']


def apply_rule_1(pdag):
    change_count = 0
    edge_set = set(pdag.edges())
    rev_edges = select_reversible_edges(pdag)
    for edge in rev_edges:
        for i1, i2 in [(0,1)]:
            n1, n2 = edge[i1], edge[i2]
            if len(pdag[n2]) == 0: # ?
                n1_parents = pdag.predecessors(n1)
                for n1p in n1_parents:
                    if (n1p, n2) not in edge_set and (n2, n1p) not in edge_set:
                        if (n2, n1) in edge_set:
                            pdag.remove_edge(n2, n1)
                        pdag.add_edge(n1, n2)
                        pdag[n1][n2]["cpdag_edge_type"] = "compelled"
                        change_count += 1
    return change_count

    """
    [A,B] = find(pdag==-1); % a -> b
     for i=1:length(A)
       a = A(i); b = B(i);
       undirected = abs(pdag) + abs(pdag)';
       % Adjacency test in undirected matrix:
       %   a adjacent b  <=>  undirected(a,b) ==0
       % That's easier to use than adjacency test in pdag:
       %   a adjacent b  <=>  pdag(a,b)==0 and pdag(b,a)==0

       % Find all nodes c such that  b-c  and c not adjacent a
       C = find(pdag(b,:)==1 & undirected(a,:)==0);
       if ~isempty(C)
         pdag(b,C) = -1; pdag(C,b) = 0;

       end
     end
    """


def apply_rule_2(pdag):
    change_count = 0
    edge_set = set(pdag.edges())
    rev_edges = select_reversible_edges(pdag)
    for edge in rev_edges:
        for i1, i2 in [(0,1)]:
            n1, n2 = edge[i1], edge[i2]
            for n3 in pdag.successors(n1):
                if n3 in pdag.predecessors(n2):
                    if (n2, n1) in edge_set:
                        pdag.remove_edge(n2, n1)
                    pdag.add_edge(n1, n2)
                    pdag[n1][n2]["cpdag_edge_type"] = "compelled"
                    change_count += 1
    return change_count

    """
     [A,B] = find(pdag==1); % unoriented a-b edge
     for i=1:length(A)
       a = A(i); b = B(i);
       if any( (pdag(a,:)==-1) & (pdag(:,b)==-1)' );
         pdag(a,b) = -1; pdag(b,a) = 0;

       end
     end
    """


def apply_rule_3(pdag):
    change_count = 0
    edge_set = set(pdag.edges())
    rev_edges = select_reversible_edges(pdag)
    for edge in rev_edges:
        for i1, i2 in [(0,1)]:
            n1, n2 = edge[i1], edge[i2]
            candidates = []
            for n3 in pdag.predecessors(n2):
                if (n3, n1) in edge_set and pdag[n3][n1]["cpdag_edge_type"] == "reversible":
                    candidates.append(n3)
                elif (n1, n3) in edge_set and pdag[n1][n3]["cpdag_edge_type"] == "reversible":
                    candidates.append(n3)
            for c1, c2 in itertools.combinations(candidates, 2):
                if (c1, c2) not in edge_set and (c2, c1) not in edge_set:
                    if (n2, n1) in edge_set:
                        pdag.remove_edge(n2, n1)
                    pdag.add_edge(n1, n2)
                    pdag[n1][n2]["cpdag_edge_type"] = "compelled"
                    change_count += 1
    return change_count

    """
     [A,B] = find(pdag==1); % a-b
     for i=1:length(A)
       a = A(i); b = B(i);
       C = find( (pdag(a,:)==1) & (pdag(:,b)==-1)' );
       % C contains nodes c s.t. a-c->b-a

       % Extract lines and columns corresponding only to the set of nodes C
       core = pdag(C,C);

       % Prepare adjacency test:
       unoriented = abs(core) + abs(core)';
       % Now:  a non-adjacent b <==> unoriented(a,b) == 0

       % Prepare to detect existence of non-adjacent pairs of nodes in C.
       % Set diagonal to 1, to prevent finding pairs of IDENTICAL nodes:
       unoriented = setdiag(unoriented, 1);
       if any(unoriented(:)==0) % C contains 2 different non adjacent elements
         pdag(a,b) = -1; pdag(b,a) = 0;

       end
     end
    """


def apply_rule_4(pdag):
    change_count = 0
    edge_set = set(pdag.edges())
    rev_edges = select_reversible_edges(pdag)
    for edge in rev_edges:
        for i1, i2 in [(0,1)]:
            n1, n2 = edge[i1], edge[i2]
            candidates = []
            for n3 in pdag.predecessors(n2):
                if (n3, n2) in edge_set and pdag[n3][n2]["cpdag_edge_type"] == "reversible":
                    if (n1, n3) in edge_set or (n3, n1) in edge_set:
                        candidates.append(n3)
            for c in candidates:
                ds = pdag.predecessors(c)
                for d in ds:
                    if (n1, d) in edge_set or (d, n1) in edge_set:
                        if (n2, d) not in edge_set and (d, n2) in edge_set:
                            if (n2, n1) in edge_set:
                                pdag.remove_edge(n2, n1)
                            pdag.add_edge(n1, n2)
                            pdag[n1][n2]["cpdag_edge_type"] = "compelled"
                            change_count += 1
    return change_count

    """
    [A,B] = find(pdag==1); % unoriented a-b edge
     for i=1:length(A)
       a = A(i); b = B(i);

       % Prepare adjacency test:
       % unoriented(i,j) is 0 (non-adj) or 1 (directed) or 2 (undirected)
       unoriented = abs(pdag) + abs(pdag)';

       % Find c such that c -> b and a,c are adjacent (a-c or a->c or a<-c)
       C = find( (pdag(:,b)==-1)' & (unoriented(a,:)>=1) );
       for j=1:length(C)
          c = C(j);
          % Check whether there is any node d, such that
          % d->c  AND  a-d  AND  b NOT adjacent to d
          if any( (pdag(:,c)==-1)' & (pdag(a,:)==1) & (unoriented(b,:)==0) )
	     pdag(a,b) = -1;  pdag(b,a) = 0;

          end
       end
     end
    """


def complete_pdag(source_pdag, inplace=False):
    if inplace:
        pdag = source_pdag
    else:
        pdag = source_pdag.copy()
    n_changes = 1
    while n_changes > 0:
        n_changes_1 = apply_rule_1(pdag)
        n_changes_2 = apply_rule_2(pdag)
        n_changes_3 = apply_rule_3(pdag)
        n_changes_4 = apply_rule_4(pdag)
        n_changes = sum([n_changes_1, n_changes_2, n_changes_3, n_changes_4])
    return pdag




