import os
import torch
torch.set_num_threads(os.cpu_count() - 2)
import swifter
import bitstring
import numpy as np
from tqdm import tqdm

from dataset_embedding_model.numerical import NumericalEmbeddingModel
from utils.constants import KGPIP_PATH

EMBEDDING_SIZE = 300
DEVICE = 'cpu'


def _load_embedding_model(path):
    model = NumericalEmbeddingModel(EMBEDDING_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def _get_column_embedding(column, embedding_model):    
    # pre-process the column
    
    def get_bin_repr(val):
        return [int(j) for j in bitstring.BitArray(float=float(val), length=32).bin]
    bin_repr = column.swifter.progress_bar(False).apply(get_bin_repr, convert_dtype=False).to_list()
    bin_tensor = torch.FloatTensor(bin_repr).to(DEVICE)
    with torch.no_grad():
        embedding_tensor = embedding_model(bin_tensor).mean(axis=0)

    return embedding_tensor.tolist()


def _is_numerical_column(column):
    """
    Checks whether the dtype of a pandas column is numerical
    :param column: a Pandas Series object representing a column of a table
    :return: whether column is numerical
    """
    numerical_dtypes = ['int', 'uint', 'float', 'bool']     # bool is ok because all will be converted to float eventually.

    if any([str(column.dtype).startswith(i) for i in numerical_dtypes]):
        return True
    return False


def embed_dataset(df, prints=True):
    numerical_model = _load_embedding_model(KGPIP_PATH + '/training_artifacts/dataset_embeddings/dataset_embedding_model.pt')
    
    col_embeddings = []
    for c in tqdm(df.columns, disable=not prints):
        # first drop the NaN values in the column
        col = df[c].dropna()
        if len(col) == 0:
            continue
        if _is_numerical_column(col):
            col_embedding = _get_column_embedding(col, numerical_model)
        else:
            # TODO: for now, use an embedding of zeros for non-numerical columns.
            col_embedding = [1e-10] * EMBEDDING_SIZE
        col_embeddings.append(col_embedding)
    
    # TODO: for now, a dataset embedding is mean of column embeddings. Maybe we need to experiment with this later.
    dataset_embedding = np.average(col_embeddings, axis=0)
    
    return dataset_embedding


    
    
    