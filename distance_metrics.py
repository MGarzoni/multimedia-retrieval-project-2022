import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial import distance_matrix

# get csvs holding features
scalar_df = pd.read_csv("./features/scalar_features.csv")
hist_df = pd.read_csv('./features/hist_features.csv')

def dist_heatmap(features_matrix, title=None):
    '''Function that takes a feature matrix (N*D, where N is the number of shapes and D is the number of descriptors),
    converts it to a dataframe'''

    features_matrix = features_matrix.drop(['Unnamed: 0', 'filename', 'category'], axis=1)

    d_m =  pd.DataFrame(distance_matrix(features_matrix.values, features_matrix.values),
                        index=features_matrix.index, columns=features_matrix.index)
    sns.set(rc = {'figure.figsize':(15, 10)})

    return sns.heatmap(d_m, annot=False).set(title=title)

# euclidean distance for scalar features
# scalar_heatmap = dist_heatmap(scalar_df, title='Heatmap of distance matrix between scalar feature vectors.')

"""Implement a simple interface for querying:

    user picks a shape in the database (or alternatively can load a 3D mesh file containing a shape which is not part of the database; note that you can do this easily by simply selecting several shapes, from the benchmarks mentioned in Step 2, and removing them from the database)
    extract features of the query shape
    compute distances of query shape to all database shapes
    sort distances from low to high
    present the K best-matching shapes, plus their distances to the query, in some visual way so the user can inspect them. Alternative: set a distance threshold t and present all shapes in the database having a distance to the query lower than t"""

"""BELOW HERE ONLY FOR SCALAR FEATURES"""

# get a random row as query shape's features
query_scalar_vec = scalar_df.sample().drop(['Unnamed: 0', 'filename', 'category'], axis='columns')
qsv_idx = query_scalar_vec.index
# get rest of feature vectors without the one selected above
rest_of_scalar_vecs = scalar_df.drop(scalar_df.index[query_scalar_vec.index]).drop(['Unnamed: 0', 'filename', 'category'], axis='columns')

# define euclidean distance function
def euclidean_distance(query_vec, target_vec):

    # here could first drop the str columns from query and target
    # so we can keep only numbers for dist calculation
    # this way we avoid dropping that info initially

    query_vec = np.array(query_vec)
    target_vec = np.array(target_vec)
    res = np.sum((query_vec - target_vec) ** 2)
    distance = np.sqrt(res)

    return distance

# compute distances of query shape to the rest of shapes
scalar_distances = []
for i in range(len(rest_of_scalar_vecs)):

    # check that index is not the one we dropped
    if i != qsv_idx:
        target_scalar_vec = rest_of_scalar_vecs.loc[i]
        dist = round(euclidean_distance(query_scalar_vec, target_scalar_vec), 3)
        scalar_distances.append(dist)
    else:
        continue

print("=== EUCLIDEAN DISTANCES BETWEEN QUERY SCALAR FEAT VEC AND REST OF DB SCALAR FEAT VECS ===\n")

# sort distances from high to low
sorted_scalar_distances = sorted(scalar_distances, reverse=True)

# get k=5 best-matching shapes
k_best_matches = sorted_scalar_distances[:5]
print(f"These are the k=5 best matches:\n{k_best_matches}\n")

# get best-matching shapes based on t=0.7 distance treshold
t_best_matches = [dist for dist in scalar_distances if dist >= 0.7]
print(f"These are the t=0.7 best matches:\n{t_best_matches}\n")


"""BELOW HERE ONLY FOR HISTOGRAM FEATURES"""
# get a random row as query shape's features
query_hist_vec = hist_df.sample().drop(['Unnamed: 0', 'filename', 'category'], axis='columns')
qhv_idx = query_hist_vec.index
# print(query_hist_vec)

# get rest of feature vectors without the one selected above
rest_of_hist_vecs = hist_df.drop(hist_df.index[query_hist_vec.index]).drop(['Unnamed: 0', 'filename', 'category'], axis='columns')
# print(rest_of_hist_vecs.head(37))

# compute EMD distances of query shape to the rest of shapes
from scipy.stats import wasserstein_distance

hist_distances = []
for i in range(len(rest_of_hist_vecs)):

    # check that index is not the one we dropped
    if i != qhv_idx:
        query_hist_vec = np.asanyarray(query_hist_vec).reshape(50)
        target_hist_vec = np.asanyarray(rest_of_hist_vecs.loc[i]).reshape(50)
        # print(f"QUERY VEC: {type(query_hist_vec)}, {query_hist_vec.shape}\nTARGET VEC: {type(target_hist_vec)}, {target_hist_vec.shape}")
        dist = round(wasserstein_distance(query_hist_vec, target_hist_vec), 3)
        # print(f"DIST: {dist}\n")
        hist_distances.append(dist)
    else:
        continue

print("\n=== EMD DISTANCES BETWEEN QUERY HIST FEAT VEC AND REST OF DB HIST FEAT VECS ===\n")

# sort distances from low to high
sorted_hist_distances = sorted(hist_distances)

# get k=5 best-matching shapes
k_best_matches = sorted_hist_distances[:5]
print(f"These are the k=5 best matches:\n{k_best_matches}\n")

# get best-matching shapes based on t=0.02 distance treshold
t_best_matches = [dist for dist in hist_distances if dist >= 0.02]
print(f"These are the t=0.02 best matches:\n{t_best_matches}\n")

