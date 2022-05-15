# KGpip - A Scalable AutoML Approach Based on Graph Neural Networks


## Quickstart
KGpip provides easy-to-use APIs similar to the scikit-learn style.

```
from kgpip import KGpip
kgpip = KGpip(num_graphs=7, hpo='autosklearn', time_budget=300)
kgpip.fit(X_train, y_train, task='regression')
kgpip.predict(X_test)
```


## Reproducing Our Results
To reproduce our results, you can use [evaluate_automl_systems.py](experiments/evaluate_automl_systems.py),
to test all systems (KGpipFLAML, KGpipAutoSklearn, FLAML, AutoSklearn, VolcanoML) on all 121 benchmark datasets.

Example usage:

```
python evaluate_automl_systems.py --time 3600 --dataset-id 39 --system KGpipFLAML
```
Which evalutes KGpipFLAML on dataset #39 with a time budget of 1 hour. Dataset IDs and info can be found in [benchmark_datasets](benchmark_datasets)

For more help on possible arguments and values:
```
python evaluate_automl_systems.py --help
```

## Installation

Create `kgpip` Conda environment (Python 3.7) and install pip requirements. Or use [init.sh](init.sh):
```
. ./init.sh
```
Note:
* The `kgpip` environment needs to be active to run the system and the provided scripts: `conda activate kgpip`
* PyTorch and DGL are installed for CUDA 11.1. Adjust [requirements.txt](requirements.txt) to match your architecture.
While the code is tested on a GPU machine, it should work fine on CPU only.

## Benchmark Datasets
We used a collection of 121 benchmark datasets. The datasets and their information can be found in [benchmark_datasets](benchmark_datasets).
The datasets need to be extracted in their respective directories before evaluation.

## Training the Graph Generator
### TODO: To be tested.

The [training](training) directory contains the needed scripts to:
1. Fetch the raw GraphGen4Code pipeline graphs from a SPARQL endpoint.
2. Clean the fetched graphs.
3. Train the graph generation model.

### Training Set Collection and Cleaning:
[fetch_and_clean_pipeline_graphs.py](training/fetch_and_clean_pipeline_graphs.py) queries an Apache Jena SPARQL
endpoint on GraphGen4Code graphs are loaded. The graphs are cleaned by removing noisy nodes and edges.
The cleaned graphs are saved to be used for training.

### Training Args:
Training Arguments can be found in [args.py](args.py).

Most important ones are:

* `graph_type`: name of the dataset. The 11K Kaggle scripts dataset has the name: `graph4code_large`
* `feat_size`: size of the node and edge embeddings.
* `epochs`: number of epochs. For now set at 400. We might need to increase it to further decrease the loss.
* `batch_size`: Batch size. 32 is a reasonable value if your GPU memory has allows it.
* `lr`: learning rate. For now set at 0.001
* `milestones`: (int list) epochs at which the learning rate will be decayed.
* `gamma`: learning rate decay factor.
* `epochs_save`: save the model checkpoint every this amount of epochs. Currently set to save the model 20 times. Models are saved in `model_save/`

### Training Start:
To start training run:

```bash
python train_graph_generation_model.py
```


## Notes
- The implementation of the graph neural network is in PyTorch and based on [GraphGen](https://github.com/idea-iitd/graphgen).
