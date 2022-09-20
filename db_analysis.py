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
        db_info[dirname] = {}

        # for each object (filename)
        for filename in os.listdir(os.path.join(db_path, dirname)):

            # consider only 3D mesh files and use trimesh to load the mesh
            if filename.endswith(('.ply', '.obj', '.off')):
                db_info[dirname].update({filename: {}})
                mesh = trimesh.load(os.path.join(db_path, dirname, filename))

                # save a dict containing all the desired attributes
                db_info[dirname][filename].update({'num_faces': len(mesh.faces), 'num_vertices': len(mesh.vertices), 'faces_type': 'triangles', 'axis_aligned_bounding_box': mesh.bounding_box.extents})

    return db_info

out_dict = inspect_db(PSB_PATH)

# save to csv
output = pd.DataFrame.from_dict(out_dict, orient='columns')
output.to_csv('psb_analysis.csv')
