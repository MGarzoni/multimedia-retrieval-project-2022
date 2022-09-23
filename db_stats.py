'''
Use output of filter (csv file) to find out:
    (a) what is the average shape in the database (in terms of vertex and face counts)
    (b) if there are significant outliers from this average (e.g. shapes having many, or few, vertices or cells)
    best way to do this is to show a histogram counting how many shapes are in the database for every range of the property of interest
    (e.g., number of vertices, number of faces, shape class)

Use the viewer constructed in step 1 to show an average shape and a few such outliers (if any)
'''

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random
import trimesh
import os

# import and load analysis data
analysis = './psb_analysis.csv'
analysis_in = pd.read_csv(analysis)

# inspect dataframe
print(analysis_in.describe())
print(analysis_in.info())

# plotting distribution of categories
cat_df = pd.DataFrame.from_dict(Counter(analysis_in['class']), orient='index', columns=['Total count'])
cat_df.plot.bar()

#plotting distribution of faces
analysis_in.num_faces.plot.hist()

# find an outlier with very few faces

# random filename with fewer than __ faces
few_faces_path = random.choice(list(analysis_in[analysis_in['num_faces'] < 3000].path))
print(f"Random 3D entity with fewer than 3000 faces: {few_faces_path}")

# random filename with more than __ faces
many_faces_path = random.choice(list(analysis_in[analysis_in['num_faces'] > 40000].path))
print(f"Random 3D entity with more than 4000 faces: {many_faces_path}")

def save_image_of_path(path, tag=None):
    
    # generate png
    scene = trimesh.load(path).scene()
    png = scene.save_image()
    
    # generate filename
    file_name = os.path.basename(path)
    
    if tag != None: # add tag to filename if there is one
        file_name = file_name + "_" + tag
    
    with open(file_name+".png", 'wb') as f:
        f.write(png)
        f.close()
                    
save_image_of_path(many_faces_path)
