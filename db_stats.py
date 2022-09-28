'''
Use output of filter (csv file) to find out:
    (a) what is the average shape in the database (in terms of vertex and face counts)
    (b) if there are significant outliers from this average (e.g. shapes having many, or few, vertices or cells)
    best way to do this is to show a histogram counting how many shapes are in the database for every range of the property of interest
    (e.g., number of vertices, number of faces, shape class)

Use the viewer constructed in step 1 to show an average shape and a few such outliers (if any)
'''

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random
import trimesh
import os
import seaborn as sns

#import our util functions
from utils import *

# import and load analysis data
psb_analysis_in = './psb_analysis.csv'
psb_df = pd.read_csv(psb_analysis_in)

# inspect dataframe
print(psb_df.describe())
print(psb_df.info())

# plot distribution of vertices and faces across PSB DB
sns.kdeplot(psb_df['num_vertices'], color='r', shade=True, label='num_vertices')
sns.kdeplot(psb_df['num_faces'], color='g', shade=True, label='num_faces')
plt.xlabel("Frequency")
plt.ylabel("Density")
plt.title("Distribution of vertices and faces across PSB DB")
plt.legend()
plt.show()

# check average shape in DB in terms vertex and face counts -> class: ANT
print(psb_df.groupby('num_vertices')['class'].head())
print(psb_df.groupby('num_faces')['class'].head())

# plotting distribution of categories
cat_df = pd.DataFrame.from_dict(Counter(psb_df['class']), orient='index', columns=['Total count'])
cat_df.plot.bar()

# random filename with fewer than __ faces
few_faces_path = random.choice(list(psb_df[psb_df['num_faces'] < 3000].path))
print(f"Random 3D entity with fewer than 3000 faces: {few_faces_path}")

# random filename with more than __ faces
many_faces_path = random.choice(list(psb_df[psb_df['num_faces'] > 40000].path))
print(f"Random 3D entity with more than 4000 faces: {many_faces_path}")
   
#save png                 
save_image_of_path(many_faces_path, tag="many_faces")

#save_image_of_path("./psb-labeled-db/Bird/243.off")