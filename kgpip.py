import warnings;warnings.filterwarnings('ignore')
import os
from glob import glob
import shutil
from tqdm import tqdm
tqdm.pandas()
from random import randint
from datetime import datetime as dt
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
import autosklearn
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

from flaml import AutoML

from generate import generate_pipeline_graphs
from dataset_embedding_model.api import embed_dataset
from utils.kgpip_utils import is_text_column
from utils.kgpip_utils import supported_autosklearn_classifiers, supported_autosklearn_regressors, supported_autosklearn_preprocessors
from utils.kgpip_utils import supported_flaml_classifiers, supported_flaml_regressors, supported_flaml_preprocessors
from utils.constants import KGPIP_PATH


class KGpip:
    
    def __init__(self, num_graphs=7, time_budget=300, hpo='autosklearn', 
                 graph_gen_model_path=None, dataset_embeddings_path=None,
                 seed=123, num_threads=None):
        """
        @param num_graphs: number of graphs/pipelines to generate by the graph generator
        @param time_budget: total time budget in seconds
        @param hpo: hyperparameter optimizer. Either 'autosklearn' or 'flaml'
        @param graph_gen_model_path: path to the trained graph generation model. If None, defaults to 'training_artifacts/graph_generation/graph_generation_model.dat'
        @param dataset_embeddings_path: path to an object containing the embeddings of training datasets. if None, defaults to 'training_artifacts/dataset_embeddings/training_set_embeddings.pickle']
        @param seed: random seed for data split and fitting.
        @param num_threads: number of threads to use while fitting.
        """
        
        self.num_graphs = num_graphs
        self.time_budget = time_budget
        assert hpo in ['autosklearn', 'flaml'], "HPO must be either 'autosklearn' or 'flaml'"
        self.hpo = hpo
        self.is_autosklearn = self.hpo == 'autosklearn'
        self.graph_gen_model_path = graph_gen_model_path or f'{KGPIP_PATH}/training_artifacts/graph_generation/graph_generation_model.dat'
        self.dataset_embeddings_path = dataset_embeddings_path or f'{KGPIP_PATH}/training_artifacts/dataset_embeddings/training_set_embeddings.pickle'
        self.seed = seed
        self.num_threads = num_threads or os.cpu_count() - 1
        self.is_regression = None
        self.target_preprocessors = None
        self.target_estimators = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.closest_dataset = None
        self.pipeline_graphs = None
        self.automl_model = None
          
    

    def __find_closest_training_dataset_by_embedding(self):
        
        X = self.X_train.sample(min(1000, len(self.X_train)), random_state=self.seed)
        embedding = embed_dataset(X)

        training_datasets_embeddings = pd.read_pickle(self.dataset_embeddings_path)
        if self.is_regression:
            task_embeddings = training_datasets_embeddings['regression']
        else:
            task_embeddings = training_datasets_embeddings['classification']

        shortest_distance = np.inf
        closest_dataset = None
        for dataset_name, properties in task_embeddings.items():
            distance = float(cosine(embedding, properties['embedding']))
            if distance < shortest_distance:
                shortest_distance = distance
                closest_dataset = dataset_name

        return closest_dataset


    def __preprocess(self, X, y, is_train):
        """
        Do pre-processing for the input dataframe before passing it to AutoSklearn.
        Pre-processing:
            1. Drop rows missing values in target variable          TODO: any way around this?
            2. Impute missing values in each column with the mean in case of numerical column and mode otherwise 
            3. Vectorize text columns by using CountVectorizer and selecting the top 500 words by frequency (i.e. columns). 
                TODO: other options include TfidfVectorizer
                TODO: the whole vocabulary could be 10s of thousands. But does this make sense?
            4. Convert String columns to categorical type
        """
        # 1. drop rows with missing values in y
        X, y = X.replace('?', np.nan), y.replace('?', np.nan)  # some datasets have NaNs as "?"
        X, y = X.apply(pd.to_numeric, errors='ignore'), pd.to_numeric(y, errors='ignore')  # re-evaluate column types
        X, y = X[~y.isna()].reset_index(drop=True), y[~y.isna()].reset_index(drop=True)


        # 2. Impute missing values with mean or mode depending on column type.
        # for test set, use mean/mode of training set
        for col in [i for i in X.columns if i in self.X_train.columns]:
            if is_train:
                X[col] = X[col].fillna(X[col].mean() if X[col].dtype.kind in 'biufc' else X[col].mode()[0])
            else:
                X[col] = X[col].fillna(self.X_train[col].mean() if self.X_train[col].dtype.kind in 'biufc' else self.X_train[col].mode()[0])

        # 3. vectorize textual columns 
        for c in X.columns:
            if is_text_column(X[c]):
                import spacy_universal_sentence_encoder
                nlp = spacy_universal_sentence_encoder.load_model('en_use_md')
                print('Vectorizing column:', c)
                vectorized = X[c].apply(lambda x: nlp(x).vector.round(3)).values.tolist()
                new_column_names = [f'embed_{c}_{i}' for i in range(len(vectorized[0]))]
                # instead of using all columns (might be too high), use the top 500 words
                vectorized_df = pd.DataFrame(vectorized, columns=new_column_names)
                # add the new columns
                X = pd.concat([X, vectorized_df], axis=1)
                # drop the text column
                X = X.drop(c, axis=1)

        # For test set: add missing columns from X_train and re-order them 
        if not is_train:
            for col in [i for i in self.X_train.columns if i not in X.columns]:
                X[col] = [self.X_train[col].mean() if self.X_train[col].dtype.kind in 'biufc' else self.X_train[col].mode()[0]] * len(X)

            # re-order columns
            X = X[self.X_train.columns]

        # 4. convert string columns to categorical.
        for c in X.columns:
            if X[c].dtype == object:
                X[c] = X[c].astype('category')

        # if y is a column of string type, convert it into categorical type
        if y.dtype == object:
            y = y.astype('category')

        return X, y


    def __filter_invalid_pipeline_graphs(self):
        """
        Graph is valid if it:
        1. Starts with the dataset name node then pandas.read_csv
        2. Has a linear structure.
        3. Has at least one estimator (from the target AutoML library) that matches the task (regression vs. classification)
        """
        valid_pipeline_graphs = []
        for graph in self.pipeline_graphs:
            # check 1
            if self.closest_dataset not in graph or 'pandas.read_csv' not in graph:
                continue
            dfs_edges = list(nx.dfs_edges(graph, source=self.closest_dataset))
            # check 2
            is_linear_graph = True
            for i in range(len(dfs_edges) - 1):
                if dfs_edges[i][1] != dfs_edges[i + 1][0]:
                    is_linear_graph = False
                    break
            if not is_linear_graph:
                continue
                
            ordered_nodes = [i[0] for i in dfs_edges] + [dfs_edges[-1][1]]
            # check 3
            for target_estimator in self.target_estimators.keys():
                if any([target_estimator in i for i in ordered_nodes]):
                    valid_pipeline_graphs.append(graph)  # condition 3 satisfied.
        self.pipeline_graphs = valid_pipeline_graphs


    def __extract_pipeline_skeletons(self):
        pipeline_skeletons = []
        explored_estimators = set()
        for idx, graph in enumerate(self.pipeline_graphs):
            #  get the preprocessors and estimators from the graphs
            extracted_preprocessors = set()
            extracted_estimators = set()
            for graph_preprocessor, automl_preprocessor in self.target_preprocessors.items():
                if any([graph_preprocessor in i for i in graph]):
                    # add the equivalent preprocessor from the target AutoML library.
                    extracted_preprocessors.add(automl_preprocessor)

            extracted_preprocessors = extracted_preprocessors or ['no_preprocessing']

            for graph_estimator, automl_estimator in self.target_estimators.items():
                if any([graph_estimator in i for i in graph]):
                    # add the equivalent estimator(s) from the target AutoML library.
                    if isinstance(automl_estimator, list):
                        extracted_estimators.update(automl_estimator)
                    else:
                        extracted_estimators.add(automl_estimator)

            extracted_preprocessors, extracted_estimators = sorted(extracted_preprocessors), sorted(extracted_estimators)
            
            # remove previously explored estimators
            # TODO: refactor (merge explored_estimators and pipeline_skeletons)
            unexplored_estimators = list(
                set([e for e in extracted_estimators if (str(extracted_preprocessors), e) not in explored_estimators]))
            if unexplored_estimators:
                explored_estimators.update([(str(extracted_preprocessors), e) for e in unexplored_estimators])
                pipeline_skeletons.extend([(extracted_preprocessors, [e]) for e in unexplored_estimators])


        ordered_pipeline_skeletons = []

        for estimator in self.target_estimators.values():
            for p_list, e_list in pipeline_skeletons:
                if [estimator] == e_list and (p_list, e_list) not in ordered_pipeline_skeletons:
                    ordered_pipeline_skeletons.append((p_list, e_list))
        
        return ordered_pipeline_skeletons[:self.num_graphs]


    def fit(self, X, y, task, verbose=True):
        """
        
        @param X: A pandas DataFrame of features
        @param y: A pandas DataFrame/Series of the target column
        @param task: Either 'classification' or 'regression' 
        @param verbose: whether to print the progress. 
        """
        assert task in ['classification', 'regression'], "Task must be either 'classification' or 'regression'"
        self.is_regression = task == 'regression'
        if self.is_regression:
            self.target_estimators = supported_autosklearn_regressors if self.is_autosklearn else supported_flaml_regressors
        else:
            self.target_estimators = supported_autosklearn_classifiers if self.is_autosklearn else supported_flaml_classifiers
        self.target_preprocessors = supported_autosklearn_preprocessors if self.is_autosklearn else supported_flaml_preprocessors
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.8, random_state=self.seed)

        if verbose:
            print('===', dt.now(), '===', 'Regression?', self.is_regression, '===', 'Time Budget:', self.time_budget,
                  '===', 'Graph Gen Model:', self.graph_gen_model_path, '===', 'Optimizer:', self.hpo, '===',
                  'Num Graphs:', self.num_graphs, '===')

        d0 = dt.now()

        # 1. finding closest dataset from the training set of KGpip
        self.closest_dataset = self.__find_closest_training_dataset_by_embedding()

        if verbose:
            print(dt.now(), f'Closest Dataset is: {self.closest_dataset}')

        d1 = dt.now()

        # 2. Pre-processing
        self.X_train, self.y_train = self.__preprocess(self.X_train, self.y_train, is_train=True)
        self.X_test, self.y_test = self.__preprocess(self.X_test, self.y_test, is_train=False)

        d2 = dt.now()

        # 3. Graph Generation
        # TODO: move this to a separate method: generate pipeline graphs
        tmp_graphs_dir = f'tmp/pipeline_graphs_{randint(1, 100000000)}/'
        generate_pipeline_graphs(self.closest_dataset, graph_gen_model_path=self.graph_gen_model_path, num_graphs=1000,
                                 graphs_path=tmp_graphs_dir)

        print(dt.now(), 'Using Graphs generated at:', tmp_graphs_dir)

        graphs = []
        for graph_file in glob(tmp_graphs_dir+'*.dat'):
            # read the graph
            g = pd.read_pickle(graph_file)
            # relabel nodes and edges (use label instead of IDs)
            node_labels = {n: g.node[n]['label'] for n in g.nodes}
            # TODO: is this needed?
            for k, v in node_labels.items():
                node_labels[k] = v.replace('http://purl.org/twc/', '')

            g = nx.relabel_nodes(g, node_labels)
            edge_labels = {e: g.get_edge_data(*e)['label'] for e in g.edges}
            # TODO: is this needed?
            for k, v in edge_labels.items():
                edge_labels[k] = v.replace('http://purl.org/twc/graph4code/', '')
            graphs.append(g)

        shutil.rmtree(tmp_graphs_dir)
        self.pipeline_graphs = graphs
        
        
        # Filter out invalid pipeline graphs
        self.__filter_invalid_pipeline_graphs()
        
        # extract pipeline skeletons from validated pipeline graphs
        pipeline_skeletons = self.__extract_pipeline_skeletons()
        
        if verbose:
            print(dt.now(), 'Considering', len(pipeline_skeletons), 'Skeletons:', pipeline_skeletons)
        d3 = dt.now()

        # time budget is calculated end to end (i.e. include graph generation time in the calculation)
        time_budget_per_graph = 0
        if pipeline_skeletons:
            time_budget_per_graph = int((self.time_budget - (d3 - d0).seconds) / len(pipeline_skeletons))
        
        if verbose:
            print(dt.now(), 'Time budget per graph:', time_budget_per_graph)

        # 6. train the automl models on X_train
        top_score = -1
        top_skeleton = None
        for preprocessors, estimators in pipeline_skeletons:

            if self.is_autosklearn:
                AutoSklearnModel = AutoSklearnRegressor if self.is_regression else AutoSklearnClassifier
                include_param = {'regressor': estimators, 'feature_preprocessor': preprocessors} if self.is_regression \
                           else {'classifier': estimators, 'feature_preprocessor': preprocessors}

                automl_model = AutoSklearnModel(include=include_param,
                                                n_jobs=self.num_threads,
                                                memory_limit=8000,
                                                time_left_for_this_task=time_budget_per_graph,
                                                resampling_strategy='cv',
                                                resampling_strategy_arguments={'folds': 5},
                                                metric=autosklearn.metrics.r2 if self.is_regression else
                                                       autosklearn.metrics.make_scorer('f1', f1_score, average='macro'),
                                                tmp_folder=f'{KGPIP_PATH}/autosklearn-{randint(1, 10000000000)}',
                                                seed=self.seed,
                                                initial_configurations_via_metalearning=0
                                                )
            else:
                # use FLAML
                automl_model = AutoML()
            try:
                if self.is_autosklearn:
                    automl_model.fit(self.X_train, self.y_train)
                else:
                    automl_model.fit(self.X_train, self.y_train,
                                     task='regression' if self.is_regression else 'classification',
                                     estimator_list=estimators,
                                     time_budget=time_budget_per_graph,
                                     metric='r2' if self.is_regression else 'macro_f1',
                                     eval_method='cv', n_splits=5,
                                     retrain_full='budget',
                                     verbose=False,
                                     mem_thres=15 * 1024 ** 3,
                                     n_jobs=self.num_threads)

                # evaluate the automl models in X_test
                if self.is_regression:
                    model_score = r2_score(self.y_test, automl_model.predict(self.X_test))
                else:
                    model_score = f1_score(self.y_test, automl_model.predict(self.X_test), average='macro')

                if model_score > top_score:
                    top_score = model_score
                    top_skeleton = (preprocessors, estimators)
                    self.automl_model = automl_model
                if verbose:
                    print(dt.now(), 'Score of:', preprocessors, ' - ', estimators, ':', model_score)

            except Exception as e:
                print(dt.now(), 'AutoML automl_model failed: Pre-processors', preprocessors, 'Estimators:', estimators)
                print(e)
        
        if verbose:
            print(dt.now(), 'Fitting Done. Best Score:', top_score, '. Best Skeleton:', top_skeleton)


    def predict(self, X):
        dummy_target = pd.Series([1] * len(X), index=X.index)
        X_test, _ = self.__preprocess(X, dummy_target, is_train=False)
        return self.automl_model.predict(X_test)
    
    
if __name__ == '__main__':
    dataset_dir = 'benchmark_datasets'
    dataset = 'titanic'
    target_column = 'Survived'
    is_regression = False
    df = pd.read_csv(f'{dataset_dir}/{dataset}/{dataset}.csv')
    X, y = df.drop(target_column, axis=1), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    kgpip = KGpip(hpo='autosklearn')
    kgpip.fit(X_train, y_train, 'regression' if is_regression else 'classification')
    score = r2_score(y_test, kgpip.predict(X_test)) if is_regression else f1_score(y_test, kgpip.predict(X_test),
                                                                                 average='macro')
    print('Score:', score)