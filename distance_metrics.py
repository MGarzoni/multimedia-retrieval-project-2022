import pandas as pd
import seaborn as sns

# get csvs holding features
scalar_df = pd.read_csv("./features/scalar_features.csv")
hist_df = pd.read_csv('./features/hist_features.csv')

def dist_heatmap(features_matrix, title=None):
    '''Function that takes a feature matrix (N*D, where N is the number of shapes and D is the number of descriptors),
    converts it to a dataframe'''
    from scipy.spatial import distance_matrix

    features_matrix = features_matrix.drop(['Unnamed: 0', 'filename', 'category'], axis=1)

    d_m =  pd.DataFrame(distance_matrix(features_matrix.values, features_matrix.values),
                        index=features_matrix.index, columns=features_matrix.index)
    sns.set(rc = {'figure.figsize':(15, 10)})

    return sns.heatmap(d_m, annot=False).set(title=title)

scalar_heatmap = dist_heatmap(scalar_df, title='Heatmap of distance matrix between scalar feature vectors.')
hist_heatmap = dist_heatmap(hist_df, title='Heatmap of distance matrix between histogram vectors.')
