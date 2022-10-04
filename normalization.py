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
import subprocess

# utils
from utils import *




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
    
    if attributes['is_out']: #if it's an outlier, remesh
        print(attributes['filename'], "is an outlier!\n")

        # define path to the shape to remesh and path to where save the remeshed shape
        shape_to_remesh_path = path
        remeshed_shape_path = f"./remeshed/remeshed-{attributes['filename']}.off"
        
        #the command that will be run in command line
        subdivider_command = f"java -jar catmullclark.jar {shape_to_remesh_path} {remeshed_shape_path}"

        # call mccullark subdivider java program
        subprocess.call(subdivider_command, shell=True)

        # update variable: path should now point to refined mesh's path, not the original mesh
        path = remeshed_shape_path
        
        if verbose: print("\nRemeshed and loading updated file from", remeshed_shape_path)
    
    mesh = trimesh.load(path) # load mesh from path -- should load REMESHED path if it was an outlier
    
    if verbose: print("Initial attributes:", attributes)
    
    if verbose:
        save_mesh_png(mesh, "1.original", corners = CORNERS)

    # translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
    mesh = center_at_origin(mesh)
    if verbose: save_mesh_png(mesh, "2.translated", corners = CORNERS)

    # scale to cube vector (with scale_to_unit function from utils) (here bounds value changes)
    mesh = scale_to_unit(mesh)
    if verbose: save_mesh_png(mesh, "3.scaled", corners = CORNERS)

    # align pca: x axis is most variance, z axis is least variance
    mesh = pca_align(mesh)
    if verbose: save_mesh_png(mesh, "4.pca", corners = CORNERS)
    
    # moment test
    mesh = moment_flip(mesh)
    if verbose: save_mesh_png(mesh, "5.moment", corners = CORNERS)

    # EXPORT MODIFIED MESH AS .OFF FILE INTO "NORMALIZED" FOLDER
    output_path = os.path.join(out_dir, attributes['filename']) # where the exported mesh will go
    off_file = trimesh.exchange.off.export_off(mesh)
    with open(output_path, 'w+') as file:
        file.write(off_file)

    # call function to extract attributes and add them to output_dictionary
    out_dict = extract_attributes_from_mesh(mesh, output_path)
    if verbose: print("Final attributes:", out_dict)

    return out_dict


def loop_pipeline(paths_list, csv_path, verbose = False):
    """Run normalization pipeline on all paths in the paths_list. 
    the csv file at csv_path is used to extract attributes about the shapes in the paths_list"""
    files_dict = attributes_csv_to_dict(csv_path)
    
    # new attributes dict, initialize
    new_files_dict = {}
    
    # loop through paths in list
    for path in paths_list:
        filename = os.path.basename(path)

        # normalize and extract attributes into new dictionary
        new_files_dict[filename] = normalization_pipeline(path, files_dict, out_dir = "./normalized", verbose=verbose)
        
        print("Processed", filename)
    
    # export updated attributes to new csv file
    output = pd.DataFrame.from_dict(new_files_dict, orient='index')
    output.to_csv('./normalized/normalized_attributes.csv')
    
# test normalization pipeline
test_path = "./psb-labeled-db/Bird/242.off"
outlier_path = "./psb-labeled-db/Hand/185.off"

# list of paths to normalize
paths_list = [test_path]

# path of original csv
csv_path = "./psb_analysis.csv"

#loop_pipeline(paths_list, csv_path, verbose = True)