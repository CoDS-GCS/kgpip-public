import os
import pickle
import networkx as nx

from training.graphgen_training_utils import mkdir


def check_graph_size(
    graph, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):

    if min_num_nodes and graph.number_of_nodes() < min_num_nodes:
        return False
    if max_num_nodes and graph.number_of_nodes() > max_num_nodes:
        return False

    if min_num_edges and graph.number_of_edges() < min_num_edges:
        return False
    if max_num_edges and graph.number_of_edges() > max_num_edges:
        return False

    return True


def produce_graphs_from_raw_format(
    inputfile, output_path, num_graphs=None, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """

    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)

    index = 0
    count = 0
    graphs_ids = set()
    while index < len(lines):
        if lines[index][0][1:] not in graphs_ids:
            graph_id = lines[index][0][1:]
            G = nx.Graph(id=graph_id)

            index += 1
            vert = int(lines[index][0])
            index += 1
            for i in range(vert):
                G.add_node(i, label=lines[index][0])
                index += 1

            edges = int(lines[index][0])
            index += 1
            for i in range(edges):
                G.add_edge(int(lines[index][0]), int(
                    lines[index][1]), label=lines[index][2])
                index += 1

            index += 1

            if not check_graph_size(
                G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
            ):
                continue

            if G and nx.is_connected(G):
                with open(os.path.join(
                        output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G, f)

                graphs_ids.add(graph_id)
                count += 1

                if num_graphs and count >= num_graphs:
                    break

        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4

    return count


# Routine to create datasets
def create_graphs(args):

    #if 'graph4code' in args.graph_type:
    base_path = os.path.join(args.dataset_path, f'{args.graph_type}/')
    input_path = base_path + f'{args.graph_type}.txt'
    min_num_nodes, max_num_nodes = None, None
    min_num_edges, max_num_edges = None, None


    args.current_dataset_path = os.path.join(base_path, 'graphs/')

    args.current_processed_dataset_path = args.current_dataset_path

    if args.produce_graphs:
        mkdir(args.current_dataset_path)

        # if 'graph4code' in args.graph_type:
        count = produce_graphs_from_raw_format(input_path, args.current_dataset_path, args.num_graphs,
                                               min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                                               min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        print('Graphs produced', count)
    else:
        count = len([name for name in os.listdir(args.current_dataset_path) if name.endswith(".dat")])
        print('Graphs counted', count)


    graphs = [i for i in range(count)]
    return graphs
