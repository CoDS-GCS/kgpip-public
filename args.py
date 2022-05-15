from datetime import datetime
import torch
from training.graphgen_training_utils import get_model_attribute


class Args:
    """
    Program configuration
    """

    def __init__(self):
        # Can manually select the device too
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        # Clean tensorboard
        self.clean_tensorboard = False
        # Clean temp folder
        self.clean_temp = True

        # Whether to use tensorboard for logging
        self.log_tensorboard = False

        # Algorithm Version - # Algorithm Version DGMG (Deep GMG)
        self.note = 'DGMG'

        # Check datasets/process_dataset for datasets
        # Select dataset to train the model
        self.graph_type = 'graph4code_large'
        self.num_graphs = None  # Set it None to take complete dataset

        # Whether to produce networkx format graphs for real datasets
        self.produce_graphs = True


        self.batch_size = 16  # normal: 32, and the rest should be changed accordingly

        # Specific to DGMG
        # Model parameters
        self.feat_size = 32
        self.hops = 1
        self.dropout = 0.2

        # training config
        self.num_workers = 0  # num workers to load data, default 4
        self.epochs = 400

        self.lr = 0.001  # Learning rate
        # Learning rate decay factor at each milestone (no. of epochs)
        self.gamma = 0.5
        self.milestones = [50, 100, 150, 200]  # List of milestones

        # Whether to do gradient clipping
        self.gradient_clipping = True

        # Output config
        self.dir_input = ''
        self.model_save_path = self.dir_input + 'model_save/'
        self.tensorboard_path = self.dir_input + 'tensorboard/'
        self.dataset_path = self.dir_input + 'datasets/'
        self.temp_path = self.dir_input + 'tmp/'

        # Model save and validate parameters
        self.save_model = True
        self.epochs_save = round(self.epochs / 20)
        self.epochs_validate = 1

        # Time at which code is run
        self.time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.now())

        # Filenames to save intermediate and final outputs
        self.fname = self.note + '_' + self.graph_type

        # Calcuated at run time
        self.current_model_save_path = self.model_save_path + \
            self.fname + '_' + self.time + '/'
        self.current_dataset_path = None
        self.current_processed_dataset_path = None
        self.current_temp_path = self.temp_path + self.fname + '_' + self.time + '/'

        # Model load parameters
        self.load_model = False
        self.load_model_path = ''
        self.load_device = torch.device('cuda:0')
        self.epochs_end = 10000

    def update_args(self):
        if self.load_model:
            args = get_model_attribute(
                'saved_args', self.load_model_path, self.load_device)
            args.device = self.load_device
            args.load_model = True
            args.load_model_path = self.load_model_path
            args.epochs = self.epochs_end

            args.clean_tensorboard = False
            args.clean_temp = False

            args.produce_graphs = False

            return args

        return self
