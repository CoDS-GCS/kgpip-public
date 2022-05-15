import os
from pathlib import Path
import pandas as pd
import argparse
import traceback
from datetime import datetime as dt
from random import randint
import sys
KGPIP_PATH = Path(__file__).resolve().parent.parent
if KGPIP_PATH not in sys.path:
    sys.path.insert(0, str(KGPIP_PATH))

from flaml import AutoML
import autosklearn
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from mindware.utils.data_manager import DataManager
from mindware.estimators import Classifier, Regressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import train_test_split

from kgpip import KGpip


def main(dataset_id, system_id, time_budget=3600, results_dir='results', num_graphs=7,
         graph_gen_model_path=None, dataset_embeddings_path=None):
    benchmark_datasets = pd.read_csv(f'{KGPIP_PATH}/benchmark_datasets/benchmark_datasets_info.csv')
    ds = benchmark_datasets.iloc[dataset_id]
    base_dir, dataset_name, is_regression, target = ds['base_dir'], ds['name'], ds['is_regression'], ds['target']

    save_path = f'{results_dir}/{dataset_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    d0 = dt.now()

    # load dataset
    df = pd.read_csv(f'{KGPIP_PATH}/{base_dir}/{dataset_name}/{dataset_name}.csv', low_memory=False)
    X, y = df.drop(target, axis=1), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    d1 = dt.now()
    loading_time = (d1 - d0).seconds

    assert 0 <= system_id <= 4, f'Invalid system id: {system_id}'
    target_system = ['KGpipFLAML', 'KGpipAutoSklearn', 'FLAML', 'AutoSklearn', 'VolcanoML'][system_id]
    print(dt.now(), f'Task #{dataset_id * 5 + system_id}: Evaluating {target_system} on dataset: {dataset_name}')

    try:
        if system_id == 0 or system_id == 1:
            # KGpipFLAML or KGpipAutoSklearn
            automl_model = KGpip(num_graphs=num_graphs,
                                 time_budget=time_budget - loading_time,
                                 hpo='flaml' if system_id == 0 else 'autosklearn',
                                 graph_gen_model_path=graph_gen_model_path,
                                 dataset_embeddings_path=dataset_embeddings_path)
            automl_model.fit(X_train, y_train,
                             task='regression' if is_regression else 'classification')
            predictions = automl_model.predict(X_test)

        elif system_id == 2:
            # FLAML
            automl_model = AutoML()
            automl_model.fit(X_train, y_train, task="regression" if is_regression else "classification",
                             time_budget=time_budget - loading_time,
                             retrain_full='budget',
                             verbose=0, metric='r2' if is_regression else 'macro_f1')
            predictions = automl_model.predict(X_test)

        elif system_id == 3:
            # AutoSklearn
            for col in X_train.columns:
                if X_train[col].dtype == object:
                    X_train[col] = X_train[col].astype('category')
                    X_test[col] = X_test[col].astype('category')
            if y_train.dtype == object:
                y_train = y_train.astype('category')
                y_test = y_test.astype('category')

            AutoSklearnModel = AutoSklearnRegressor if is_regression else AutoSklearnClassifier
            automl_model = AutoSklearnModel(n_jobs=os.cpu_count() - 1,
                                            memory_limit=8000,
                                            time_left_for_this_task=time_budget - loading_time,
                                            metric=autosklearn.metrics.r2 if is_regression else
                                            autosklearn.metrics.make_scorer('f1', f1_score, average='macro'),
                                            tmp_folder=f'{KGPIP_PATH}/tmp/autosklearn-{randint(1, 1000000)}',
                                            seed=123,
                                            initial_configurations_via_metalearning=0)
            automl_model.fit(X_train, y_train)
            predictions = automl_model.predict(X_test)

        else:
            # VolcanoML
            if not is_regression:
                # encode the target column categories so it works with Volcano
                encoder = LabelEncoder()
                y = pd.concat([y_train, y_test])
                y = pd.Series(encoder.fit_transform(y))
                y_train, y_test = y.iloc[:len(y_train)], y.iloc[len(y_train):]
            dm = DataManager(X_train, y_train)
            train_data = dm.get_data_node(X_train, y_train)
            test_data = dm.get_data_node(X_test, y_test)

            VolcanoModel = Regressor if is_regression else Classifier
            automl_model = VolcanoModel(time_limit=time_budget - loading_time,
                                        metric='mse' if is_regression else 'f1',
                                        n_jobs=os.cpu_count() - 1,
                                        ensemble_method=None,
                                        # enable_meta_algorithm_selection=False,
                                        # enable_fe=False,
                                        output_dir="tmp/")
            automl_model.fit(train_data)
            predictions = automl_model.predict(test_data)
            
        score = r2_score(y_test, predictions) if is_regression else f1_score(y_test, predictions, average='macro')

    except Exception:
        print(f'{target_system} Failed with dataset: {dataset_name}')
        traceback.print_exc(file=sys.stdout)
        score = 0

    print(dt.now(), f'Score of {target_system} on dataset {dataset_name}: {score}')
    with open(f'{save_path}{target_system}.txt', 'w') as f:
        f.write(str(score))
    print('Results saved in', f'{save_path}{target_system}.txt')
    print(dt.now(), 'Done. Total Time:', dt.now() - d0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', type=int, required=False, default=None, 
                        help='Dataset ID (see benchmark_datasets/benchmark_datasets_info.csv). Range: [1, 121]')
    parser.add_argument('--system', type=str, default=None, const=None, nargs='?',
                        choices=['KGpipFLAML', 'KGpipAutoSklearn', 'FLAML', 'AutoSklearn', 'VolcanoML'],
                        help='Name of the system to evaluate. Possible values: [KGpipFLAML, KGpipAutoSklearn, FLAML, '
                             'AutoSklearn, VolcanoML]') 
    # task_id goes from 0 to (num_datasets x num_systems) - 1 = 121 x 5 - 1
    parser.add_argument('--task-id', type=int, required=False, default=None, 
                        help='Task ID for batch jobs to evaluate all systems and datasets. Range: [0, 604]')
    parser.add_argument('--time', type=int, required=False, default=3600, 
                        help='Total time budget in seconds.')
    parser.add_argument('--dir', type=str, required=False, default='results', help='Directory to save the results.')
    parser.add_argument('--graphs', type=int, required=False, default=7, 
                        help='Number of pipeline graphs to consider for KGpip.')
    parser.add_argument('--graph-gen-model-path', type=str, required=False, default=None)
    parser.add_argument('--dataset-embeddings-path', type=str, required=False, default=None)
    args = parser.parse_args()
    
    if (args.dataset_id is None or args.system is None) and args.task_id is None:
        print('Either Task ID or both Dataset ID and System name have to be provided.')
        exit()
    
    if args.dataset_id:
        dataset_id = args.dataset_id - 1
        system_id = ['KGpipFLAML', 'KGpipAutoSklearn', 'FLAML', 'AutoSklearn', 'VolcanoML'].index(args.system)
    else:
        dataset_id = args.task_id // 5
        system_id = args.task_id % 5
        
    main(dataset_id=dataset_id, system_id=system_id, time_budget=args.time, 
         results_dir=args.dir, num_graphs=args.graphs,
         graph_gen_model_path=args.graph_gen_model_path, dataset_embeddings_path=args.dataset_embeddings_path)
