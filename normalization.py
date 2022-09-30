'''
1. read each shape
2. translate it so that its barycenter coincides with the coordinate-frame origin
3. scale it uniformly so that it tightly fits in a unit-sized cube
4. run stats to show that the normalization worked OK for all shapes
'''

# imports
import csv
import trimesh
import os
import numpy as np
import pandas as pd

# utils
from utils import *

# corners of image used for png export
corners = [[-1, -1, -1],
       [ 1, -1, -1],
       [ 1,  1, -1],
       [-1,  1, -1],
       [-1, -1,  1],
       [ 1, -1,  1],
       [ 1,  1,  1],
       [-1,  1,  1]]

# read shape
sample = "./psb-labeled-db/Bird/242.off"
original_mesh = trimesh.load(sample)
mesh = original_mesh.copy()
save_mesh_png(mesh, "original", corners = corners)

# translation to origin
mesh = center_at_origin(mesh)
save_mesh_png(mesh, "translated", corners = corners)
translated_mesh = mesh.copy()

# scaling so it fits in unit-sized cube
mesh = scale_to_unit(mesh)
save_mesh_png(mesh, "scaled", corners = corners)
scaled_mesh = mesh.copy()

# PCA
eigenvalues, eigenvectors = pca_eigenvalues_eigenvectors(mesh)
print(f"=> eigenvalues for (x, y, z)\n{eigenvalues}")
print(f"=> eigenvectors\n{eigenvectors}")

# print stats to demonstrate changes
print(f"Original barycenter: {original_mesh.centroid}\nOriginal bounds:\n{original_mesh.bounds}")
print(f"NEW barycenter: {mesh.centroid}\nNEW bounds:\n{mesh.bounds}")


'''
function pipeline( path, out_dir, display_meshes=False ):

    define output dictionary holding info about updated values to add to newshapes.csv
    
    read original_csv file

    check if path is outlier (possibly treat this exception):
        run java subdivider and get refined mesh (here num_vertices changes value)
        update num_vertices column

    load shape from path

    translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
    scale to cube vector (with scale_to_unit function from utils) (here bounds value changes)
    compute eigenvalues and eigenvectors
    apply pca on them (with pca function from utils) (here value changes)

    call function to extract attributes and add them to output_dictionary

    return output_dictionary
'''

def normalization_pipeline(path, csv_file, out_dir, display_mesh=False):
    
#     check if path is outlier (possibly treat this exception):
#         run java subdivider and get refined mesh (here num_vertices changes value)
#         update num_vertices column

    # load shape from path
    mesh = trimesh.load(path)

    # translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
    mesh = center_at_origin(mesh)
    translated_mesh = mesh.copy()

    # scale to cube vector (with scale_to_unit function from utils) (here bounds value changes)
    mesh = scale_to_unit(mesh)
    scaled_mesh = mesh.copy()

    # get eigenvalues and eigenvectors (with pca function from utils) (here value changes)
    eigenvalues, eigenvectors = pca_eigenvalues_eigenvectors(mesh)
    print(f"=> eigenvalues for (x, y, z)\n{eigenvalues}")
    print(f"=> eigenvectors\n{eigenvectors}")

    # print stats to demonstrate changes
    print(f"Original barycenter: {original_mesh.centroid}\nOriginal bounds:\n{original_mesh.bounds}")
    print(f"NEW barycenter: {mesh.centroid}\nNEW bounds:\n{mesh.bounds}")

    # call function to extract attributes and add them to output_dictionary
    out_dict = extract_attributes(path, outliers_range=range(3500))

    return out_dict

