import os
import trimesh
import numpy as np
import pandas as pd

import scaling
from utils import *
import copy
from features_extraction import *
from distance_metrics import *
from collections import Counter

STANDARDIZATION_CSV = "./features/standardization_parameters.csv"
FEATURES_CSV = "./features/features.csv"

"""
PIPELINE:
- open GUI
    - in GUI, user can load in a query mesh or input a string representing the mesh that wants to be retrieved
    - once mesh is loaded, user can pick a k or t value as hyperparameter defining how many similar meshs should be returned
- after that, normalization of query mesh is triggered:
    - resampling (subdivision or decimation and/or hole stitching) if needed
    - centering
    - scaling
    - alignment
    - flipping
    - store new normalized mesh
- after that, feature extraction:
    - extract scalar and hist features from query mesh
- after that, distance metrics calculation between the output of previous step and all other pre-computed feature vectors
    - euclidean for scalar values
    - EMD for hist values
- final retrieval:
    - select k or t user defined closest distance metrics
    - retrieve meshs based on that feature vector
    - display k or t user defined most similar meshs
"""



# CALL NORMALIZATION PIPELINE
def normalize_mesh_from_path(test_mesh_path):

    print("Normalizing query mesh...")

    if test_mesh_path.endswith(('.ply', '.obj', '.off')):

        # load mesh from path
        norm_mesh = trimesh.load(test_mesh_path)

        # extract attributes as a dict
        raw_mesh_attributes = extract_attributes_from_mesh(norm_mesh, test_mesh_path)
        # print("\n\nImported new mesh. Initial RAW attributes:\n\n", raw_mesh_attributes)

        ''' resampling '''

        if raw_mesh_attributes['is_out_low']:

            # while the # vertices is lower than 3500
            # use mesh.subdivide() for resampling
            while len(norm_mesh.vertices) <= IS_OUT_LOW:
                norm_mesh = norm_mesh.subdivide()

        if raw_mesh_attributes['is_out_high']:

            # while the # vertices is higher than 17500
            # use open3d for decreasing # vertices
            while len(norm_mesh.vertices) >= IS_OUT_HIGH:
                norm_mesh = norm_mesh.simplify_quadratic_decimation(17500)


        ''' normalization '''

        # translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
        norm_mesh = normalize_mesh(norm_mesh)
        
        # calculate attributes of NEW mesh.
        norm_mesh_attributes = extract_attributes_from_mesh(norm_mesh, 
                                                            mesh_path = None, 
                                                            filename = raw_mesh_attributes["filename"]+"_(normalized)") # the normalized mesh has no file path
        # print("\n\nNormalized mesh. New attributes:\n\n", norm_mesh_attributes)

    return norm_mesh, norm_mesh_attributes



# CALL FEATURE EXTRACTION
def extract_features(norm_mesh, norm_mesh_attributes, filename = None, verbose = False):
    """Extract features from a normalized mesh.
    Scalar features are standardized with respect to standardization parameters of database
    
    RETURNS: scalar features as dictionary, histogram features as one long vector"""
    
    if verbose: print("Extracting features from query mesh...")

    # # extract scalar features as dictionary
    # scalar_feats = extract_scalar_features_single_mesh(norm_mesh, 
    #                                               standardization_parameters_csv = STANDARDIZATION_CSV,
    #                                               verbose = verbose)    
    
    # # extract histogram features as ONE LONG VECTOR!!! (should be same order as hist columns in .csv file)
    # hist_feats_vector = extract_hist_features_single_mesh(norm_mesh, 
    #                                                       returntype = "vector",
    #                                                       verbose = verbose)

    #  now save these entries in the hist_bins dictionary
    features = defaultdict(list)
    features['filename'].append(norm_mesh_attributes['filename'])
    features['category'].append(norm_mesh_attributes['category'])
    if verbose: print(features)
    

    # append only the VALUES of the histogram, not the bins
    # (these are assumed to be consistent)

    scalar_features = extract_scalar_features_single_mesh(norm_mesh, standardization_parameters_csv=STANDARDIZATION_CSV) # this returns a dict of features for the mesh
    for key, value in scalar_features.items(): # add the new values to the scalar_features dictionary 
        features[key] = (value)

    # calcualte the histograms for each feature as a dictionary
    feature_hists = extract_hist_features_single_mesh(norm_mesh, filename = filename, verbose = False, returntype="dictionary")
    for feature in hist_feature_methods.keys():
        for i in range(BINS):
            features[f"{feature}_{i}"] = (feature_hists[feature][i])

    features = pd.DataFrame.from_dict(features, orient='columns')
    
    if verbose: print("\n\nFeatures of query mesh:\n\n", type(features), features)

    return features



# COMPUTE QUERY FEATURE VECTOR DISTANCES FROM ALL REST OF OTHER VECTORS
def compute_distances(query_feats, db_feats, verbose = False):

    if verbose: print("Computing distances from query to rest of DB...")

    # saving scalar and hist feat names
    scalar_labels = ['area', 'volume', 'aabb_volume', 'compactness', 'diameter', 'eccentricity']
    hist_labels = ['a3_0', 'a3_1', 'a3_2', 'a3_3', 'a3_4', 'a3_5', 'a3_6', 'a3_7', 'a3_8', 'a3_9',
                     'd1_0', 'd1_1', 'd1_2', 'd1_3', 'd1_4', 'd1_5', 'd1_6', 'd1_7', 'd1_8', 'd1_9',
                     'd2_0', 'd2_1', 'd2_2', 'd2_3', 'd2_4', 'd2_5', 'd2_6', 'd2_7', 'd2_8', 'd2_9',
                     'd3_0', 'd3_1', 'd3_2', 'd3_3', 'd3_4', 'd3_5', 'd3_6', 'd3_7', 'd3_8', 'd3_9',
                     'd4_0', 'd4_1', 'd4_2', 'd4_3', 'd4_4', 'd4_5', 'd4_6', 'd4_7', 'd4_8', 'd4_9']

    # making copies of query and target vectors
    query_scalar_copy = copy.deepcopy(query_feats[scalar_labels]).loc[0]
    query_hist_copy = copy.deepcopy(query_feats[hist_labels]).loc[0]
    db_scalar_copy = copy.deepcopy(db_feats[scalar_labels])
    db_hist_copy = copy.deepcopy(db_feats[hist_labels])
    
    if verbose: print(f"Query scalar features: {query_scalar_copy}\n\nQuery hist: {query_hist_copy}")

    # define main distances dict holding required information
    distances = {'path': [path for path in db_feats['path']], 'category':db_feats['category'], 'hist_dist': [], "scalar_dist":[]}

    # compute euclidian distances on scalar features of query shape to the rest of shapes
    for i in range(len(db_feats[scalar_labels])):
        target_scalar_vec = db_scalar_copy.loc[i]
        dist = round(euclidean_distance(query_scalar_copy, target_scalar_vec), 4)
        distances['scalar_dist'].append(dist)
    
    query_hist_copy = np.asanyarray(query_hist_copy).reshape(50) #turn histogram dataframe into vector
        
    # compute EMD distances on hist features of query shape to the rest of shapes
    from scipy.stats import wasserstein_distance
    for i in range(len(db_feats[hist_labels])):
        target_hist_vec = np.asanyarray(db_hist_copy.loc[i]).reshape(50)
        dist = round(wasserstein_distance(query_hist_copy, target_hist_vec), 4)
        distances['hist_dist'].append(dist)
        
    # sort distances from low to high before returning
    distances = pd.DataFrame.from_dict(distances)
    if verbose: print(distances)
    
    return distances


def run_query(mesh_path, k=5, scalar_weight = 0.5, verbose = False, exclude_self = False):
    
    
    norm_mesh, norm_mesh_attributes = normalize_mesh_from_path(mesh_path)
    if verbose: print("NORM QUERY ATTRIBUTES:", norm_mesh_attributes)
    query_feats = extract_features(norm_mesh, norm_mesh_attributes, filename = os.path.basename(mesh_path), verbose = False)
    if verbose: print("QUERY FEATURES", query_feats)
    
    # GET FEATURE VECTORS FROM ALL NORMALIZED DB
    db_feats = pd.read_csv(FEATURES_CSV)
    
    # get distances and STANDARDIZE
    dist_df = compute_distances(query_feats, db_feats, verbose = verbose)
    dist_df['scalar_dist_standard'] = standardize_column(dist_df['scalar_dist'])[0]
    dist_df['hist_dist_standard'] = standardize_column(dist_df['hist_dist'])[0]
    
    # calculate COMBINED DISTANCE weighted by the scalar_weight and 1-scalar_weight for the histogram distance
    dist_df['dist'] = (scalar_weight * dist_df['scalar_dist_standard'] + (1-scalar_weight)*dist_df['hist_dist_standard'])/2
    
    # sort by combined distance (note that this can be a negative value due to standardization)
    dist_df = dist_df.sort_values(by="dist", ascending=True)
    
    if exclude_self: # if want to exclude query filename from results
        dist_df = dist_df[~dist_df['path'].str.contains(os.path.basename(mesh_path)+"$")] # use regex to skip files that have same filename as query
    
    # export distances csv if verbose
    if verbose: dist_df.to_csv("distances_dataframe.csv")
    
    # SELECT K OR T USER DEFINED CLOSEST FEAT VECTORS AND RETRIEVE MESHES
    # get k=5 best-matching shapes (the 5 lowest distances)
    k_best_matches = dist_df.head(k)
    
    if verbose: print(k_best_matches)

    return k_best_matches, norm_mesh # return the k best matches dict, and the normalized mesh too

def predict_class(mesh_path, k=5, scalar_weight = 0.5, return_format = "multiple", method = "distance_query", verbose = False):
    """Given a mesh path, return the most common class in the query results (if multiple most common classes, choose arbitrarily)."""
    if method == 'distance_query':
        query_results = run_query(mesh_path, k=k, scalar_weight = scalar_weight, exclude_self = True, verbose = verbose)
        labels = query_results[0]['category']
    else:
        idx = scaling.ANNIndex(pd.read_csv(FEATURES_CSV))

        norm_mesh, norm_mesh_attributes = normalize_mesh_from_path(mesh_path)
        query_feats = extract_features(norm_mesh, norm_mesh_attributes, filename=os.path.basename(mesh_path),
                                       verbose=True)

        columns = [c for c in query_feats.columns if any(f in c for f in scaling.ANNIndex.features)]

        top_matches = idx.query(list(query_feats[columns].iloc[0]), k)
        labels = top_matches['category']
    
    if return_format == "multiple": # we want to return labels of all predictions
        return labels
    
    elif return_format == "majority":
    
        prediction = Counter(labels).most_common(1)[0][0] # most common prediction label
        
        
        if verbose:
            print(mesh_path)
            # print(query_results)
            print("Predicted class:", prediction)
        
        return prediction
    


if __name__ == "__main__":

    # CALL GUI
    # USER LOADS IN QUERY MESH (for now just get a random mesh from test db)
    test_mesh_path = os.path.join("./test-db/", np.random.choice(os.listdir("./test-db/")))
    
    run_query(test_mesh_path)
    
    
    
    # DISPLAY MESHES IN GUI


