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
import trimesh

# utils
from utils import *

# corners of image, array used for visually consistent png export of meshes
corners = [[-1, -1, -1],
       [ 1, -1, -1],
       [ 1,  1, -1],
       [-1,  1, -1],
       [-1, -1,  1],
       [ 1, -1,  1],
       [ 1,  1,  1],
       [-1,  1,  1]]


'''
function pipeline( path, files_dictionary, out_dir, display_meshes=False ):
    
    input requires a dictionary of attributes indexed by filename. very easy to create from csv

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
    
    save new mesh in out_dir

    return output_dictionary
'''

def normalization_pipeline(path, files_dictionary, out_dir, verbose=False):
    """Verbose includes IMAGES"""
    
    # load attributes of filename (from files_dictionary)
    attributes = files_dictionary[os.path.basename(path)]
    
    if attributes['is_out']:
        print(attributes['filename'], "is an outlier!\n")
    #     check if path is outlier (possibly treat this exception):
    #         run java subdivider and get refined mesh (here num_vertices changes value)
    #         export refined mesh to the "remeshed" directory
    #         change variable: path = refined mesh's path
    #         update num_vertices column #may be unnecessary since we re-extract attributes at the end
    
    
    mesh = trimesh.load(path) # load mesh from path -- should load REMESHED path if it was an outlier
    
    if verbose: print("Before:", attributes)
    
    original_mesh = mesh.copy()
    if verbose:
        save_mesh_png(mesh, "1.original", corners = corners)

    # translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
    mesh = center_at_origin(mesh)
    if verbose:
        save_mesh_png(mesh, "2.translated", corners = corners)
    translated_mesh = mesh.copy()

    # scale to cube vector (with scale_to_unit function from utils) (here bounds value changes)
    mesh = scale_to_unit(mesh)
    if verbose:
        save_mesh_png(mesh, "3.scaled", corners = corners)
    scaled_mesh = mesh.copy()

    # get eigenvalues and eigenvectors (with pca function from utils) (here value changes)
    eigenvalues, eigenvectors = pca_eigenvalues_eigenvectors(mesh)
    # print(f"=> eigenvalues for (x, y, z)\n{eigenvalues}")
    # print(f"=> eigenvectors\n{eigenvectors}")


    # EXPORT MODIFIED MESH AS .OFF FILE INTO "NORMALIZED" FOLDER
    output_path = os.path.join(out_dir, attributes['filename']) # where the exported mesh will go
    off_file = trimesh.exchange.off.export_off(mesh)
    with open(output_path, 'w+') as file:
        file.write(off_file)

    # call function to extract attributes and add them to output_dictionary
    out_dict = extract_attributes_from_mesh(mesh, output_path, outliers_range=range(3500))
    if verbose: print("After:", out_dict)

    return out_dict


def loop_pipeline(paths_list, csv_path):

    files_dict = attributes_csv_to_dict(csv_path)
    
    # new attributes dict, initialize
    new_files_dict = {}
    
    # loop through paths in list
    for path in paths_list:
        filename = os.path.basename(path)

        # normalize and extract attributes into new dictionary
        new_files_dict[filename] = normalization_pipeline(path, files_dict, out_dir = "./normalized", verbose=True)
    
    # export updated attributes to new csv file
    output = pd.DataFrame.from_dict(new_files_dict, orient='index')
    output.to_csv('./normalized/normalized_attributes.csv')
    
# test normalization pipeline
test_path = "./psb-labeled-db/Bird/242.off"
outlier_path = "./psb-labeled-db/Hand/185.off"

# list of paths to normalize
paths_list = [test_path, outlier_path]

# path of csv
csv_path = "./psb_analysis.csv"

loop_pipeline(paths_list, csv_path)