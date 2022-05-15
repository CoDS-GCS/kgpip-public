import random
from datetime import datetime
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from args import Args
from training.graphgen_training_utils import create_dirs
from datasets.process_dataset import create_graphs
from graph_generation_model.data import DGMG_Dataset_from_file
from graph_generation_model.model import create_model
from training.train import train


if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    create_dirs(args)

    random.seed(123)
    torch.manual_seed(123)
    np.random.seed(123)
    
    graphs = create_graphs(args)

    random.shuffle(graphs)
    graphs_train = graphs[: int(0.90 * len(graphs))]
    graphs_validate = graphs[int(0.90 * len(graphs)):]

    # show graphs statistics
    print('Device:', args.device)
    print('Graph type:', args.graph_type)
    print('Training set: {}, Validation set: {}'.format(
        len(graphs_train), len(graphs_validate)))

    # Loading the feature map
    with open(args.current_dataset_path + 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))



    dataset_train = DGMG_Dataset_from_file(args, graphs_train, feature_map)
    dataset_validate = DGMG_Dataset_from_file(args, graphs_validate, feature_map)

    
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.num_workers, collate_fn=dataset_train.collate_batch)
    dataloader_validate = DataLoader(dataset_validate, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=dataset_validate.collate_batch)


    model = create_model(args, feature_map)
    
    d0 = datetime.now()
    print('Training Started at:', d0)
    
    # with torch.autograd.detect_anomaly():
    train(args, dataloader_train, model, feature_map, dataloader_validate)
    
    print('Training Ended at:', datetime.now())
    print('Total Time:', datetime.now() - d0)
    