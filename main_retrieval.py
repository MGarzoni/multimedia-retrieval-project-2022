import os
import trimesh
import numpy as np
import pandas as pd
from utils import *
import copy
from features_extraction import *
from distance_metrics import *

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

# CALL GUI
# USER LOADS IN QUERY MESH
mesh_path = input("Paste path to query mesh from disk: ")
mesh_path = str(mesh_path)

STANDARDIZATION_CSV = "./features/standardization_parameters.csv"

# CALL NORMALIZATION PIPELINE
def normalize_mesh_from_path(mesh_path):

    print("Normalizing mesh...")

    if mesh_path.endswith(('.ply', '.obj', '.off')):

        # load mesh from path
        norm_mesh = trimesh.load(mesh_path)

        # extract attributes as a dict
        raw_mesh_attributes = extract_attributes_from_mesh(norm_mesh, mesh_path)
        print("Iported new mesh. Initial RAW attributes:", raw_mesh_attributes)

        ''' resampling '''

        if raw_mesh_attributes['is_out_low']:

            # while the # vertices is lower than 3500
            # use mesh.subdivide() for resampling
            while len(norm_mesh.vertices) <= IS_OUT_LOW:
                norm_mesh = norm_mesh.subdivide()
                print("# vertices after subdivision:", len(norm_mesh.vertices))

        if raw_mesh_attributes['is_out_high']:

            # while the # vertices is higher than 17500
            # use open3d for decreasing # vertices
            while len(norm_mesh.vertices) >= IS_OUT_HIGH:
                norm_mesh = norm_mesh.simplify_quadratic_decimation(17500)


        ''' normalization '''

        # translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
        norm_mesh = normalize_mesh(norm_mesh)
        
        # calculate attributes of NEW mesh.
        norm_mesh_attributes = extract_attributes_from_mesh(norm_mesh, mesh_path = None) # the normalized mesh has no file path
        print("Normalized mesh. New attributes:", norm_mesh_attributes)

    return norm_mesh, norm_mesh_attributes

# CALL FEATURE EXTRACTION
norm_mesh, norm_mesh_attributes = normalize_mesh_from_path(mesh_path)

def extract_features(norm_mesh):

    print("Extracting features...")

    # extract scalar features as dictionary
    scalar_feats = extract_scalar_features_single_mesh(norm_mesh, 
                                                  standardization_parameters_csv=STANDARDIZATION_CSV,
                                                  verbose = True)    
    
    # extract histogram features as ONE LONG VECTOR!!! (should be same order as hist columns in .csv file)
    hist_feats_vector = extract_hist_features_single_mesh(norm_mesh, 
                                                          returntype = "vector")

    return scalar_feats, hist_feats_vector

query_scalar, hist_feats_vector = extract_features(norm_mesh)

# GET FEATURE VECTORS FROM ALL NORMALIZED DB
db_feats = pd.read_csv("norm_db_features.csv") # THIS HAS TO BE CREATED

# COMPUTE QUERY FEATURE VECTOR DISTANCES FROM ALL REST OF OTHER VECTORS
def compute_distances(query_feat_vector, db_feat_vectors):

    print("Computing distances from query to rest of DB...")

    # define main distances dict holding required information
    distances = {'filenames': [fname for fname in db_feat_vectors['filename']], 'all_distances': []}

    # compute cosine distances on scalar features of query shape to the rest of shapes
    for i in range(len(db_feat_vectors[['area', 'volume', 'aabb_volume', 'compactness', 'diameter', 'eccentricity']])):
        target_scalar_vec = db_feat_vectors.loc[i]
        dist = round(cosine_distance(query_feat_vector, target_scalar_vec), 3)
        distances['all_distances'].append(dist)
        
    query_feat_vector = np.asanyarray(query_feat_vector).reshape(50)

    # compute EMD distances on hist features of query shape to the rest of shapes
    from scipy.stats import wasserstein_distance
    for i in range(len(db_feat_vectors[['A3_0', 'A3_1', 'A3_2', 'A3_3', 'A3_4', 'A3_5', 'A3_6', 'A3_7', 'A3_8', 'A3_9',
                                        'D1_0', 'D1_1', 'D1_2', 'D1_3', 'D1_4', 'D1_5', 'D1_6', 'D1_7', 'D1_8', 'D1_9',
                                        'D2_0', 'D2_1', 'D2_2', 'D2_3', 'D2_4', 'D2_5', 'D2_6', 'D2_7', 'D2_8', 'D2_9',
                                        'D3_0', 'D3_1', 'D3_2', 'D3_3', 'D3_4', 'D3_5', 'D3_6', 'D3_7', 'D3_8', 'D3_9',
                                        'D4_0', 'D4_1', 'D4_2', 'D4_3', 'D4_4', 'D4_5', 'D4_6', 'D4_7', 'D4_8', 'D4_9']])):

        target_hist_vec = np.asanyarray(db_feat_vectors.loc[i]).reshape(50)
        dist = round(wasserstein_distance(query_feat_vector, target_hist_vec), 3)
        distances['all_distances'].append(dist)
        
    # sort distances from low to high before returning
    # HERE WE ACTUALLY ALSO WANT TO SORT THE FILENAMES (I.E. KEEP THEIR REFERENCE TO THE SHAPES)
    # OTHERWISE WE CAN'T DISPLAY THEM AFTER
    # BUT CAN'T JUST SORT ALSO THE FILENAME KEY IN THE DICT CAUSE THAT WOULD BE BASED ON ANOTHER CRITERIA (E.G. ALPHABETICAL)
    distances['all_distances'] = sorted(distances['all_distances'])
    
    return distances

# SELECT K OR T USER DEFINED CLOSEST FEAT VECTORS AND RETRIEVE MESHES
distances = compute_distances(query_feats, db_feats)

# get k=5 best-matching shapes (the 5 lowest distances)
k_best_matches = [(fname, dist) for fname, dist in zip(distances['filenames'][:5], distances['all_distances'][:5])]
print(f"These are the k=5 best matches:\n{k_best_matches}\n")

# DISPLAY MESHES IN GUI

