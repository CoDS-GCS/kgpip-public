import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
pd.set_option('display.max_rows', None)

KGPIP_PATH = Path(__file__).resolve().parent.parent.parent

def is_text_column(df_column):
    """
    Check if the dataframe column is a text (as opposed to short strings).
    Simple heuristic: check if it has 20+ unique values and 30%+ of the column contains 2+ space characters.
    TODO: this will not work with timestamp columns that contain spaces.
    """

    # to speed up the search in case of string column, check how many unique values
    if df_column.dtype != object or len(df_column.unique()) < 20:
        return False

    num_text_rows = 0
    for value in df_column.values.tolist():
        if not isinstance(value, str):
            continue
        space_count = 0
        for character in value:
            if character == ' ':
                space_count += 1
            if space_count > 1:
                num_text_rows += 1
                break
        if num_text_rows > 0.3 * len(df_column):
            return True
    return False



def main():
    benchmark_info = pd.read_csv(os.path.join(KGPIP_PATH, 'benchmark_datasets/benchmark_datasets_info.csv'))
    rows = []
    for row in tqdm(benchmark_info.itertuples(index=False), total=len(benchmark_info)):
        file = f'{KGPIP_PATH}/{row.base_dir}/{row.name}/{row.name}.csv'
        size = round(os.path.getsize(file) / 1024 ** 2, 1)
        df = pd.read_csv(file, low_memory=False)
        df, y = df.drop(row.target, axis=1), df[row.target]
        df = df.replace('?', np.nan)
        df = df.apply(pd.to_numeric, errors='ignore')

        nrows = len(df)
        ncols = len(df.columns)
        nclasses = len(y.value_counts()) if not row.is_regression else '-'

        task = row.task
        source = row.source
        papers = row.papers
        sort_priority = row.sort_priority

        nnumerical = ncategorical = ntextual = 0
        for col in df.columns:
            if df[col].dtype.kind in 'iufc':
                nnumerical += 1
            elif is_text_column(df[col]):
                ntextual += 1
            elif df[col].dtype.kind in 'bO':
                ncategorical += 1
            else:
                raise ValueError('Unexpected column type: dataset:', row.name, 'Column:', col, 'Type:', df[col].dtype)

        rows.append([row.name, nrows, ncols, nclasses, nnumerical, ncategorical, ntextual, size, task, source, papers, sort_priority])

    df = pd.DataFrame(rows, columns=['Dataset', 'Rows', 'Columns', 'Classes', 'Numerical', 'Categorical', 'Textual',
                                     'Size (MB)', 'Task', 'Source', 'Papers', 'priority'])

    df['dataset_lower'] = df.Dataset.str.lower()
    df = df.sort_values(['priority', 'Task', 'dataset_lower'])
    df.index = list(range(1, len(df) + 1))
    df = df.drop(['dataset_lower', 'priority'], axis=1)
    print(df)


if __name__ == '__main__':
    main()
