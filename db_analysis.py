import os
import trimesh
import pandas as pd

import utils

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
            # for each object (filename)
            for filename in os.listdir(os.path.join(db_path, dirname)):
    
                # consider only 3D mesh files and use trimesh to load the mesh
                if filename.endswith(('.ply', '.obj', '.off')):
                    
                    full_path = os.path.join(db_path, dirname, filename)
    
                    # save a dict containing all the desired attributes
                    db_info[filename] = utils.extract_attributes(full_path)

    return db_info

out_dict = inspect_db(PSB_PATH)

        
# output dictionary to csv
output2 = pd.DataFrame.from_dict(out_dict_2, orient='index')
output2 = output2.rename_axis('filename').reset_index()
output2.to_csv('psb_analysis.csv')

output2.head()

