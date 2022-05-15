import os
import pickle
from tqdm import tqdm
import pandas as pd

from dataset_embedding_model.api import embed_dataset

"""
This script reads the csv files of the training datasets, calculates the dataset embedding and store them in a pickle
file categorized by the task type, i.e. regression vs. classification.
It reads the information of the datasets from training_set_info.csv.
"""


def main():
    df = pd.read_csv('../../training/training_set_info.csv')
    df = df[df.Regression | df.Classification]
    df = df.dropna(subset=['Target'])
    
    embeds = {'regression': {}, 'classification': {}}
    
    for _, row in tqdm(df.iterrows(), total=len(df), disable=False):
        if not os.path.exists(f'kaggle_datasets/{row.Name}/{row.Name}.csv'):
            print('Skipping dataset:', row.Name)
            continue
        # print('Dataset:', row.Name, 'Target:', row.Target)
        dataset = pd.read_csv(f'kaggle_datasets/{row.Name}/{row.Name}.csv', low_memory=False)
        # print(row['Target'])
        if row.Target in dataset.columns:
            # print(row.Target, row.Name)
            dataset = dataset.drop(row.Target, axis=1)
        
        
        # fill nans first
        for col in dataset.columns:
            dataset[col] = dataset[col].fillna(dataset[col].mean() if dataset[col].dtype.kind in 'biufc' else dataset[col].mode()[0])
        
        # sample 1000 rows
        if len(dataset) > 1000:
            dataset = dataset.sample(1000)
        
        embedding = embed_dataset(dataset, prints=False)
        
        if row.Regression:
            embeds['regression'][row.Name] = {'target': row.Target if row.Target else '', 'embedding': embedding}
        
        if row.Classification:
            embeds['classification'][row.Name] = {'target': row.Target if row.Target else '', 'embedding': embedding}
        
    with open('training_set_embeddings.pickle', 'wb') as f:
        pickle.dump(embeds, f)
        



if __name__ == '__main__':
    main()
