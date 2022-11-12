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

# select files we want to keep
for category in categories:
    #sample 3 files from this category
    category_df = original_df[original_df["category"] == category].sample(N, random_state = 42)
    paths = list(category_df["path"])
    paths_to_keep += paths # add to list of paths to keep
    
    
new_df = original_df[original_df["path"].isin(paths_to_keep)]
new_paths = []
# move these files and their attributes to the new folder and csv
for index, row in new_df.iterrows():
    filename = os.path.basename(row["path"])
    category = os.path.basename(os.path.dirname(row["path"]))

    os.makedirs(new_folder_path+category, exist_ok=True)
    
    new_path = os.path.join(new_folder_path, category, filename)
    
    shutil.copyfile(row["path"], new_path)
    new_paths.append(new_folder_path+filename)
    
new_df['path'] = new_paths
new_df.to_csv(new_csv)
    

