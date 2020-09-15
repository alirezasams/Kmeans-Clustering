import numpy as np
import pandas as pd
import pickle
from sklearn import decomposition
from sklearn import preprocessing


def pca(input_data, n_components=2):
    pca_model = decomposition.PCA(n_components=n_components)
    pca_model.fit(input_data)
    output_data = pd.DataFrame(pca_model.transform(input_data),
                               columns=['component%i' % i for i in range(n_components)],
                               index=input_data.index)

    model_path = 'models/pca-' + str(n_components) + 'component'
    with open(model_path, 'wb') as model_file:
        pickle.dump(pca_model, model_file)

    return output_data


def scale(input_data):
    scale_model = preprocessing.StandardScaler()
    scale_model.fit(input_data)
    output_data = pd.DataFrame(scale_model.transform(input_data),
                               columns=input_data.columns,
                               index=input_data.index)

    model_path = 'models/scale'
    with open(model_path, 'wb') as model_file:
        pickle.dump(scale_model, model_file)

    return output_data


def rmcor(input_data, cor_threshold=.85):
    # remove high correlated columns from data
    output_data = input_data.copy()
    corr_mat = output_data.corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cor_threshold)]
    output_data.drop(to_drop, inplace=True, axis=1)

    return output_data

