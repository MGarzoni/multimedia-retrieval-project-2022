#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 08:07:07 2022

@author: eduardsaakashvili
"""

"CREATE A COPY OF THE NORMALIZED DATABASE WITH N OBJECTS PER CATEGORY, ALSO A NEW ATTRIBUTES CSV FILE"

N = 2

import pandas as pd
import os
import shutil

original_df = pd.read_csv("./attributes/normalized-PSB-attributes.csv")
paths_to_keep = []
new_paths = []

new_folder_path = "./reduced-normalized-psb-db/" # include slash at the end
new_csv = "./attributes/reduced-normalized-PSB-attributes.csv"

categories = set(original_df['category'])

# delete the current files in the folder
for filename in os.listdir(new_folder_path):
    file_path = os.path.join(new_folder_path, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

for category in categories:
    #sample 3 files from this category
    category_df = original_df[original_df["category"] == category].sample(N, random_state = 42)
    paths = list(category_df["path"])
    paths_to_keep += paths # add to list of paths to keep
    
for path in paths_to_keep:
    filename = os.path.basename(path)
    category = os.path.basename(os.path.dirname(path))

    os.makedirs(new_folder_path+category, exist_ok=True)
    
    shutil.copyfile(path, new_folder_path+category+"/"+filename)
    new_paths.append(new_folder_path+filename)
    
new_attributes_df = original_df[original_df["path"].isin(paths_to_keep)]
new_attributes_df['path'] = new_paths
new_attributes_df.to_csv(new_csv)
    

