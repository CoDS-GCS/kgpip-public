import os
import re

import pandas as pd
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from natsort import natsort_keygen

from utils.constants import dataset_names_11k_scripts

def fetch_and_clean_graphs():

    FORMATTED_DATASET_PATH = 'datasets/graph4code_large_p3/graph4code_large_p3.txt'
    ENDPOINT = 'http://localhost:3030/Graph4CodeLarge/sparql'
    GRAPH_TRIPLES_QUERY = """
                          select ?s ?p ?o
                          where{ 
                                graph ?g {?s ?p ?o}
                                values ?g {<GRAPH_URL>}
                          }
                          """
    ALL_GRAPHS_QUERY = """
                       select distinct ?g
                       where {
                         filter regex(str(?g), "PLACEHOLDER", "i") .
                         graph ?g {?s ?p ?o}
                       }
                       """
    """
    Definition of an ML pipeline:
        1. has an import statement of sklearn
        2. has a .fit() call (a statement ending with ".fit"). This is a heuristic to check if an estimator is used.
    Note: normalizedLabel is used instead of label. This is to work with the latest version of Graph4Code.
    """
    ML_PIPELINES_QUERY = """
                            prefix sch: <http://www.w3.org/2000/01/rdf-schema#>
                            prefix g4c: <http://purl.org/twc/graph4code/>
                            prefix syntax: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            select distinct ?g
                            where {
                              filter regex(str(?g), "PLACEHOLDER", "i") .
                              graph ?g {?s syntax:type g4c:Imported .
                                ?s sch:label ?lib .
                                filter (?lib = 'sklearn' || ?lib = 'xgboost' || ?lib = 'lgbm')
                                ?s2 <http://schema.org/about> "fit"
                                }
                            }
                         """

    DATASETS_AL_TRAINING = ['bnp-paribas-cardif-claims-management', 'crowdflower-search-relevance',
                            'flavours-of-physics', 'home-depot-product-search-relevance',
                            'liberty-mutual-group-property-inspection-prediction', 'machinery-tube-pricing',
                            'predict-west-nile-virus', 'rossmann-store-sales', 'santander-customer-satisfaction']

    print('Getting Graph URLS ...')
    graph_urls, dataset_names = get_graphs_for_datasets(dataset_names_11k_scripts, ML_PIPELINES_QUERY, ENDPOINT)

    # bad graph: 'http://github/data.kg21.unsdsn.world-happiness.manojkumarvk.eda-ensemble-learning.eda-ensemble-learning.py/eda-ensemble-learning'

    print(len(graph_urls), 'Graphs')

    if not os.path.exists(FORMATTED_DATASET_PATH[:FORMATTED_DATASET_PATH.rindex('/')]):
        os.makedirs(FORMATTED_DATASET_PATH[:FORMATTED_DATASET_PATH.rindex('/')])

    if os.path.exists(FORMATTED_DATASET_PATH):
        os.rename(FORMATTED_DATASET_PATH, FORMATTED_DATASET_PATH + '.old')

    for i, graph_url in enumerate(tqdm(graph_urls), start=1):
        dataset_name = dataset_names[i-1]
        node2id = {}
        id2node = {}
        edges = []
        lines = [f'#{10000 + i+1}']

        sparql = SPARQLWrapper(ENDPOINT, returnFormat=JSON)
        sparql.setQuery(GRAPH_TRIPLES_QUERY.replace('GRAPH_URL', graph_url))
        results = sparql.query().convert()

        triples = []
        for result in results["results"]["bindings"]:
            subj, pred, obj = result['s']["value"], result['p']["value"].strip(), result['o']["value"]

            # For RDF* triples ignore the edge label (they indicate ordinal position and "expression" name of se nodes)
            if isinstance(subj, dict):
                subj, pred, obj = subj['subject']['value'], subj['property']['value'], subj['object']['value']

            # fix literal objects that are empty strings or contain new lines
            if len(obj) == 0:
                obj = '-'
            obj = obj.replace('\n', ' ').replace(' ', '___')

            triples.append((subj, pred, obj))
        df = pd.DataFrame(triples, columns=['subject', 'predicate', 'object'])


        df = clean_graph4code_pipeline_graph(df, dataset_name)
        if len(df) == 0:
            print(f'WARNING: no flowsTo triples in graph {i} - dataset: {dataset_name}')
            continue

        triples = df.values.tolist()
        for subj, pred, obj in triples:
            if subj not in node2id:
                node2id[subj] = len(node2id)
                id2node[len(id2node)] = subj
            if obj not in node2id:
                node2id[obj] = len(node2id)
                id2node[len(id2node)] = obj

            edges.append(f'{node2id[subj]} {node2id[obj]} {pred}')

        # number of nodes + nodes
        lines.append(str(len(node2id)))
        lines.extend(list(node2id.keys()))

        # number of edges + edges
        lines.append(str(len(edges)))
        lines.extend(edges)

        with open(FORMATTED_DATASET_PATH, 'a') as f:
            f.write('\n'.join(lines) + '\n\n')

    # remove the old dataset if the generation is successful.
    if os.path.exists(FORMATTED_DATASET_PATH + '.old'):
        os.remove(FORMATTED_DATASET_PATH + '.old')
    print('Done')



def clean_graph4code_pipeline_graph(df_spo, dataset_name, use_data_flow=True):
    """
    Graph4Code Filtration: only extract Sklearn and XGBoost calls.
    Works as follows:
    1. filter out non sklearn/xgboost se nodes
    2. filter out Import se nodes
    3. filter out all BNodes
    4. filter out non-flowsTo triples (choose between data flow (default) or code flow
    5. add extra flowsTo triples between the remaining nodes to make the graph a straight line
    6. replace the remaining URI with their normalizedLabel
    7. add closest_dataset and pandas.read_csv nodes to the start of the graph
    """
    g4c_prefix = 'http://purl.org/twc/graph4code/'
    if use_data_flow:
        flowsTo_predicate = g4c_prefix + 'flowsTo'
    else:
        flowsTo_predicate = 'http://semanticscience.org/resource/SIO_000250'
    # filter out non-(sklearn.|xgboost.) label nodes and imported se nodes
    imported_se_nodes = df_spo[df_spo['object'] == g4c_prefix + 'Imported']['subject'].tolist()
    pandas_se_nodes = get_pandas_se_nodes_from_spo(df_spo[df_spo['predicate'] == g4c_prefix + 'normalizedLabel'])
    sklearn_xgb_label_spo = df_spo[(df_spo['predicate'] == g4c_prefix + 'normalizedLabel') &
                                   (df_spo['object'].str.contains(r'^(sklearn|xgboost|lgbm)\.', regex=True)) &
                                   (~df_spo['subject'].isin(imported_se_nodes + pandas_se_nodes))]
    se_to_label = {i: j['object'] for i, j in sklearn_xgb_label_spo.set_index('subject').to_dict(orient='index').items()}
    flowsTo_spo = df_spo[(df_spo['predicate'] == flowsTo_predicate) &
                         (df_spo['subject'].isin(sklearn_xgb_label_spo['subject']))]

    # skip this graph if no flowsTo triples (probably a bad graph)
    if len(flowsTo_spo) == 0:
        return pd.DataFrame()

    # natural sort by se name
    flowsTo_sorted_spo = flowsTo_spo.sort_values('subject', key=natsort_keygen())

    # replace non-sklearn/xgb se objects with ones in the subjects. i.e. connect nodes together
    last_se_node = flowsTo_sorted_spo['subject'].tolist()[-1]    # last se node that has an sklearn/xgb label
    last_se_index = int(last_se_node[last_se_node.index('/se') + 3:]) # se nodes have the format: http://purl.org/twc/graph4code/se65
    triples_to_remove = []   # remove the triples having objects in this list
    for subj, obj in zip(flowsTo_sorted_spo['subject'].tolist(), flowsTo_sorted_spo['object'].tolist()):
        # no need to change anything if the object is already in the subject list
        if obj in flowsTo_sorted_spo['subject'].values:
            continue
        # increment it till we find a node in the subject, starting from the greater index between subj and obj.
        # (sometimes subject has higher index than object)
        se_index = max(int(obj[obj.index('/se') + 3:]), int(subj[subj.index('/se') + 3:])) + 1
        while se_index <= last_se_index:
            next_se_node = f'{g4c_prefix}se{se_index}'
            if next_se_node in flowsTo_sorted_spo['subject'].values:
                break
            se_index += 1

        # if se_index is greater than the last index, remove triples having this object
        # otherwise, set the object to the se node having this index
        if se_index > last_se_index:
            triples_to_remove.append(obj)
        else:
            flowsTo_sorted_spo['object'] = flowsTo_sorted_spo.apply(lambda x: next_se_node if x['subject'] == subj and
                                                                                              x['object'] == obj
                                                                                           else x['object'], axis=1)

    flowsTo_sorted_spo = flowsTo_sorted_spo[~flowsTo_sorted_spo['object'].isin(triples_to_remove)]
    # add dataset name and pandas.read_csv nodes
    flowsTo_sorted_spo = flowsTo_sorted_spo.append({'subject': dataset_name,
                                                    'predicate': flowsTo_predicate,
                                                    'object': 'pandas.read_csv'}, ignore_index=True)
    flowsTo_sorted_spo = flowsTo_sorted_spo.append({'subject': 'pandas.read_csv',
                                                    'predicate': flowsTo_predicate,
                                                    'object': flowsTo_sorted_spo['subject'].tolist()[0]},
                                                   ignore_index=True)

    filtered_df_spo = flowsTo_sorted_spo.replace(se_to_label)
    # remove self-loops
    filtered_df_spo = filtered_df_spo[filtered_df_spo['subject'] != filtered_df_spo['object']]
    # remove duplicate edges
    filtered_df_spo = filtered_df_spo.drop_duplicates()

    return filtered_df_spo


def get_pandas_se_nodes_from_spo(df_spo_labels):
    pandas_calls = ['shape', 'null', 'expr', 'columns', 'T', 'sum', 'max', 'min', 'mean', 'astype',
                    'drop', 'value_counts', 'iloc', 'loc']
    bad_labels = set()
    for label in df_spo_labels['object'].tolist():
        if re.search(r'\.[0-9]', label):
            bad_labels.add(label)
        for pc in pandas_calls:
            if label.endswith(f'.{pc}') or f'.{pc}.' in label:
                bad_labels.add(label)

    return df_spo_labels[df_spo_labels['object'].isin(bad_labels)]['subject'].tolist()



def get_graphs_for_datasets(dataset_names, query, sparql_endpoint):
    graphs = []
    datasets = []
    for dataset_name in tqdm(dataset_names):
        sparql = SPARQLWrapper(sparql_endpoint, returnFormat=JSON)
        sparql.setQuery(query.replace('PLACEHOLDER', dataset_name))
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        graphs.extend([i['g']['value'] for i in bindings])
        datasets.extend([dataset_name] * len(bindings))

    return graphs, datasets



def main():
    fetch_and_clean_graphs()


if __name__ == '__main__':
    main()
