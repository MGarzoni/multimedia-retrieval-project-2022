import os
import trimesh
import pandas as pd

# DB PATHS
PRINCETON_PATH = "./princeton-labeled-db/"
PSB_PATH = "./psb-labeled-db/"

# step 2.1: function to analyse shapes from database
# returns a dict of categories and objects along with desired attributes
def inspect_db(db_path):

    # initialize dict where to store info about db
    db_info = {}

    # for each category (folder=dirname) create an empty list as value associated with that category
    for dirname in os.listdir(db_path):
        if dirname != ".DS_Store": # this irrelevant filename shows up on some machines
            db_info[dirname] = {}
    
            # for each object (filename)
            for filename in os.listdir(os.path.join(db_path, dirname)):
    
                # consider only 3D mesh files and use trimesh to load the mesh
                if filename.endswith(('.ply', '.obj', '.off')):
                    db_info[dirname].update({filename: {}})
                    mesh = trimesh.load(os.path.join(db_path, dirname, filename))
    
                    # save a dict containing all the desired attributes
                    db_info[dirname][filename].update({'class': dirname, 'num_faces': len(mesh.faces), 'num_vertices': len(mesh.vertices),
                                                        'faces_type': 'triangles', 'axis_aligned_bounding_box': mesh.bounding_box.extents,
                                                        'path':os.path.join(db_path, dirname, filename)})

    return db_info

out_dict = inspect_db(PSB_PATH)

# create csv with class as ATTRIBUTE rather than column
out_dict_2 = {} # this dictionary will have filename as key, and dictionary of attributes as value

for dirname, files in out_dict.items():
    for file in files:
        out_dict_2[file] = out_dict[dirname][file]
        
# output dictionary to csv
output2 = pd.DataFrame.from_dict(out_dict_2, orient='index')
output2 = output2.rename_axis('filename').reset_index()
output2.to_csv('psb_analysis.csv')

output2.head()

'''
need to create a function (universal, add to utils) that extracts useful attributes from shape and returns them as dictionary to then add to csv

function extract_attributes( path, out_bound=(0, 2500) ):
    define out_dict
    load mesh
    add to dict following attributes:
        'path', 'category', 'num_faces', 'num_vertices', 'faces_type', 'axis_aligned_bounding_box',
        'is_out', 'centroid'

    return out_dict
'''

def extract_attributes(path, outliers_range=range(3500)):
    """Given a path, loads the mesh, checks if it's an outlier,
    and then adds required attributes of mesh to the out_dict to be returned;
    can also set the outlier range (default is (0, 3500))."""

    # load mesh
    mesh = trimesh.load(path)

    # add attributes to out_dict
    out_dict = {"category" : path.split('/')[7], # this may change
                "num_faces" : len(mesh.faces),
                "num_vertices" : len(mesh.vertices),
                "faces_type" : 'triangles',
                "axis_aligned_bounding_box" : mesh.bounding_box.extents,
                "is_out" : True if len(mesh.vertices) in outliers_range else False,
                "centroid" : mesh.centroid}
    
    return out_dict
