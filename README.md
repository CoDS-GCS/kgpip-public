# KGpip - A Scalable AutoML Approach Based on Graph Neural Networks

AutoML systems build machine learning models automatically by performing a search over valid data transformations and learners, along with hyper-parameter optimization for each learner. Many AutoML systems use meta-learning to guide search for optimal pipelines.  In this work, we present a novel meta-learning system called KGpip which (1) builds a database of datasets and corresponding pipelines by mining thousands of scripts with program analysis, (2) uses dataset embeddings to find similar datasets in the database based on its content instead of metadata-based features, (3) models AutoML pipeline creation as a graph generation problem, to succinctly characterize the diverse pipelines seen for a single dataset. KGpip's meta-learning is a sub-component for AutoML systems. We demonstrate this by integrating KGpip with two AutoML systems. Our comprehensive evaluation using 121 datasets, including those used by the state-of-the-art systems, shows that KGpip significantly outperforms these systems.

## Install KGpip

Create `kgpip` Conda environment (Python 3.7) and install pip requirements. Or use [init.sh](init.sh) (for CPU machines use `init-cpu.sh`):
```
. ./init.sh
```
Note: 
* The `kgpip` environment needs to be active to run the system and the provided scripts: `conda activate kgpip`
* PyTorch and DGL are installed for CUDA 11.0. Adjust [requirements.txt](requirements.txt) to match your CUDA version.
* For CPU-only machines, `init-cpu.sh` installs `torch==1.7.0+cpu` and `dgl==0.5.3` instead of `torch==1.7.0+cu110` and `dgl-cu110==0.5.3`, respectively.

## Benchmark Datasets
We used a collection of 121 benchmark datasets. 
The datasets can be downloaded ([here](https://drive.google.com/file/d/1SzPi0l7ICUhXPPOpJGZ57f1873WmJS1D/view?usp=sharing)), 
except 6 Kaggle datasets, which should be downloaded directly from Kaggle.
The Kaggle webpages can be found ([here](https://drive.google.com/file/d/1GEj-LNx0jUqPRiGxhIYcaanzG_EkXQP5/view?usp=sharing)).

The dataset information and statistics can be found in [benchmark_datasets](benchmark_datasets).

The datasets need to be extracted in the `benchmark_datasets` directory, where each dataset is stored under its own directory.
After extracting the .zip file in `benchmark_datasets`, you can extract individual datasets using the e.g.:

```
cd benchmark_datasets   
find -name *.csv.bz2 -exec bzip2 -d {} \;
```

The final structure of `benchmark_datasets`  should look like:
```
benchmark_datasets/
├── 2dplanes
│   └── 2dplanes.csv
├── abalone
│   └── abalone.csv
├── adult
│   └── adult.csv
├ ...
...
```


## Quickstart
KGpip provides easy-to-use APIs similar to the scikit-learn style. 
The following shows an example of loading a dataset and fitting KGpip.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from kgpip import KGpip

def main():
    # load and split the dataset
    df = pd.read_csv('benchmark_datasets/volkert/volkert.csv')
    X, y = df.drop('class', axis=1), df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, 
                                                        random_state=123)
    
    # fit KGpip
    kgpip = KGpip(num_graphs=5, hpo='flaml', time_budget=900)
    kgpip.fit(X_train, y_train, task='classification')
    predictions = kgpip.predict(X_test)
    print('Score:', f1_score(y_test, predictions, average='macro'))

if __name__ == '__main__':
    main()
```


## Reproducing Our Results
To reproduce our results, you can use [evaluate_automl_systems.py](experiments/evaluate_automl_systems.py), 
to test all systems (KGpipFLAML, KGpipAutoSklearn, FLAML, AutoSklearn, VolcanoML) on all 121 benchmark datasets.
Please make sure all datasets are downloaded and extracted first.

Example usage:

```
python experiments/evaluate_automl_systems.py --time 3600 --dataset-id 39 --system KGpipFLAML
```
Dataset IDs and info can be found in [benchmark_datasets](benchmark_datasets).
The above command evaluates KGpipFLAML on dataset #39 ([volkert](benchmark_datasets/volkert)) with a time budget of 1 hour. 
You should get an F1-Score of ~0.67. The scores are saved in the `results` directory.

For more help on possible arguments and values:
```
python experiments/evaluate_automl_systems.py --help
```


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


### Note:
- The implementation of the graph neural network is in PyTorch and based on [GraphGen](https://github.com/idea-iitd/graphgen).


## Technical Report
Our technical report is available on [ArXiv](https://arxiv.org/abs/2111.00083). 

## Citing Our Work
If you find our work useful, please cite it in your research:
```
@article{kgpip,
         title={A Scalable AutoML Approach Based on Graph Neural Networks}, 
         author={Mossad Helali and Essam Mansour and Ibrahim Abdelaziz and Julian Dolby and Kavitha Srinivas},
         year={2022},
         journal={Proceedings of the VLDB Endowment},
         doi={10.14778/3551793.3551804},
         volume={15},
         number={11},
         pages={2428-2436}  
}
```

## Questions
For any questions please contact us at: mossad.helali@concordia.ca, essam.mansour@concordia.ca, ibrahim.abdelaziz1@ibm.com, dolby@us.ibm.com, kavitha.srinivas@ibm.com
