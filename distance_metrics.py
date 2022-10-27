import seaborn as sns
import pandas as pd

def dist_heatmap(features_matrix:dict):
    '''Function that takes a feature matrix (N*D, where N is the number of shapes and D is the number of descriptors),
    converts it to a dataframe'''
    from scipy.spatial import distance_matrix

    d_m =  pd.DataFrame(distance_matrix(features_matrix.values, features_matrix.values),
                        index=features_matrix.index, columns=features_matrix.index)
    sns.set(rc = {'figure.figsize':(15, 10)})

    return sns.heatmap(d_m, annot=False).set(title='Heatmap of distance matrix between feature vectors.')
