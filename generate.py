import os
import shutil

from graph_generation_model.train import predict_graphs as gen_graphs_dgmg
from training.graphgen_training_utils import get_model_attribute, save_graphs


class ArgsEvaluate:
    def __init__(self, dataset_name=None,
                 graph_gen_model_path='training_artifacts/graph_generation/graph_generation_model.dat',
                 num_graphs=500, graphs_path=''):
        # Can manually select the device too
        self.device = 'cpu'
        self.model_path = graph_gen_model_path

        if dataset_name:
            self.starting_nodes = [dataset_name, 'pandas.read_csv']
            # print('Starting Nodes:', self.starting_nodes)

        self.num_epochs = get_model_attribute('epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        self.count = num_graphs
        self.batch_size = 200  # Must be a factor of count

        self.metric_eval_batch_size = 1

        # Specific to GraphRNN and DGMG
        self.max_num_node = 15

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        self.graphs_save_path = 'graphs/'
        self.current_graphs_save_path = graphs_path



def generate_pipeline_graphs(injected_dataset_name=None,
                             graph_gen_model_path='training_artifacts/graph_generation/graph_generation_model.dat',
                             num_graphs=500, graphs_path=''):
    """
    Generate graphs (networkx format) given a trained generative model
    and save them to a directory
    """
    eval_args = ArgsEvaluate(dataset_name=injected_dataset_name, graph_gen_model_path=graph_gen_model_path,
                             num_graphs=num_graphs, graphs_path=graphs_path)

    gen_graphs = gen_graphs_dgmg(eval_args)

    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)

    os.makedirs(eval_args.current_graphs_save_path)

    save_graphs(eval_args.current_graphs_save_path, gen_graphs)
        


