import os
import pandas as pd
from utils import *

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
    
                # consider only 3D mesh files
                if filename.endswith(('.ply', '.obj', '.off')):
                    
                    full_path = os.path.join(db_path, dirname, filename)
    
                    # save a dict containing all the desired attributes
                    db_info[filename] = extract_attributes_from_path(full_path)
    
    print("Attributes successfully exported to csv.")

    return db_info

# output dictionary to csv
out_dict = inspect_db(PSB_PATH)
output = pd.DataFrame.from_dict(out_dict, orient='index')
output.to_csv("./attributes/original-PSB-attributes.csv")
output.head()
