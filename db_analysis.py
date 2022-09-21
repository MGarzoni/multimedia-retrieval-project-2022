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
        if dirname != ".DS_Store": #this irrelevant filename shows up on some machines
            db_info[dirname] = {}
    
            # for each object (filename)
            for filename in os.listdir(os.path.join(db_path, dirname)):
    
                # consider only 3D mesh files and use trimesh to load the mesh
                if filename.endswith(('.ply', '.obj', '.off')):
                    db_info[dirname].update({filename: {}})
                    mesh = trimesh.load(os.path.join(db_path, dirname, filename))
    
                    # save a dict containing all the desired attributes
                    db_info[dirname][filename].update({'class':dirname, 'num_faces': len(mesh.faces), 'num_vertices': len(mesh.vertices), 'faces_type': 'triangles', 'axis_aligned_bounding_box': mesh.bounding_box.extents})

    return db_info

out_dict = inspect_db(PSB_PATH)

# # save to csv
# output = pd.DataFrame.from_dict(out_dict, orient='columns')
# output.to_csv('psb_analysis.csv')

#create csv with class as ATTRIBUTE rather than column
out_dict_2 = {} #this dictionary will have filename as key, and dictionary of attributes as value

for dirname, files in out_dict.items():
    for file in files:
        out_dict_2[file] = out_dict[dirname][file]
        
#output dictionary to csv
output2 = pd.DataFrame.from_dict(out_dict_2, orient = 'index')
output2.to_csv('psb_analysis.csv')
