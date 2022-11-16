import pandas as pd
import seaborn as sns
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance_matrix
from scipy.stats import wasserstein_distance

# get csvs holding features
scalar_df = pd.read_csv("./features/scalar_features.csv")
hist_df = pd.read_csv('./features/hist_features.csv')
all_features = pd.read_csv('./features/features.csv') # .drop(['Unnamed: 0', 'path', 'filename', 'category'], axis=1)

data = all_features.values
dm = distance_matrix(data, data)
dm_df = pd.DataFrame(dm)
print(dm_df.info())

def dist_heatmap(features_matrix, title=None):
    '''Function that takes a feature matrix (N*D, with N = number of shapes, D = number of descriptors),
    converts it to a dataframe and return a heatmap showing euclidean distances between feat vectors'''

    features_matrix = features_matrix.drop(['Unnamed: 0', 'path', 'filename', 'category'], axis=1)
    dm = distance_matrix(features_matrix.values, features_matrix.values)
    dm_df =  pd.DataFrame(dm)
    sns.set(rc = {'figure.figsize':(25, 20)})

    return sns.heatmap(dm_df, annot=False).set(title=title)
dist_heatmap(all_features, title='Heatmap of feature values')

""" =============================== BELOW HERE ONLY FOR SCALAR FEATURES ======================================== """

# define cosine dist function
def cosine_distance(query_vec, target_vec):
    return np.dot(query_vec, target_vec) / (norm(query_vec) * norm(target_vec))

# define euclidean distance function
def euclidean_distance(query_vec, target_vec):
    query_vec = np.array(query_vec)
    target_vec = np.array(target_vec)
    # print(query_vec)
    # print(target_vec)
    res = np.sum((query_vec - target_vec) ** 2)
    distance = np.sqrt(res)
    # print("Distance:", np.sqrt(res))
    return distance


""" =============================== BELOW HERE DOES NOT RUN IF WE IMPORT THIS PY FILE ======================================== """

# ONLY RUNS IF THIS IS MAIN
if __name__ == "__main__":
        
    # euclidean distance for scalar features
    # scalar_heatmap = dist_heatmap(scalar_df, title='Heatmap of distance matrix between scalar feature vectors.')
        
    # get a random row as query shape's features
    query_scalar_vec = scalar_df.sample().drop(['Unnamed: 0', 'filename', 'category'], axis='columns')
    qsv_idx = query_scalar_vec.index

    # get rest of feature vectors without the one selected above
    rest_of_scalar_vecs = scalar_df.drop(scalar_df.index[query_scalar_vec.index]).drop(['Unnamed: 0', 'filename', 'category'], axis='columns')


    """ =============================== BELOW HERE FOR SCALAR FEATURES ======================================== """

    # compute euclidean distances of query shape to the rest of shapes
    euclidean_distances = []
    for i in range(len(rest_of_scalar_vecs)):
    
        # check that index is not the one we dropped
        if i != qsv_idx:
            target_scalar_vec = rest_of_scalar_vecs.loc[i]
            dist = round(euclidean_distance(query_scalar_vec, target_scalar_vec), 4)
            euclidean_distances.append(dist)
        else:
            continue
    
    print("EUCLIDEAN DISTANCES BETWEEN QUERY SCALAR FEAT VEC AND REST OF DB SCALAR FEAT VECS\n")
    
    # sort distances from low to high
    sorted_euclidean_distances = sorted(euclidean_distances)
    print(f"Sorted euclidean distances:\n{sorted_euclidean_distances}\n")
    
    # get k=5 best-matching shapes
    k_best_matches = sorted_euclidean_distances[:5]
    print(f"These are the k=5 best matches:\n{k_best_matches}\n")


    """ =============================== BELOW HERE FOR HISTOGRAM FEATURES ======================================== """

    # get a random row as query shape's features
    query_hist_vec = hist_df.sample().drop(['Unnamed: 0', 'filename', 'category'], axis='columns')
    qhv_idx = query_hist_vec.index
    
    # get rest of feature vectors without the one selected above
    rest_of_hist_vecs = hist_df.drop(hist_df.index[query_hist_vec.index]).drop(['Unnamed: 0', 'filename', 'category'], axis='columns')
    
    # compute EMD distances of query shape to the rest of shapes
    hist_distances = []
    for i in range(len(rest_of_hist_vecs)):
    
        # check that index is not the one we dropped
        if i != qhv_idx:
            query_hist_vec = np.asanyarray(query_hist_vec).reshape(50)
            target_hist_vec = np.asanyarray(rest_of_hist_vecs.loc[i]).reshape(50)
            dist = round(wasserstein_distance(query_hist_vec, target_hist_vec), 3)
            hist_distances.append(dist)
        else:
            continue
    
    print("\nEMD DISTANCES BETWEEN QUERY HIST FEAT VEC AND REST OF DB HIST FEAT VECS\n")
    
    # sort distances from low to high
    sorted_hist_distances = sorted(hist_distances)
    print(f'{sorted_hist_distances}\n')
    
    # get k=5 best-matching shapes
    k_best_matches = sorted_hist_distances[:5]
    print(f"These are the k=5 best matches:\n{k_best_matches}")
