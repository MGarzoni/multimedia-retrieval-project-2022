'''
1. read each shape
2. translate it so that its barycenter coincides with the coordinate-frame origin
3. scale it uniformly so that it tightly fits in a unit-sized cube
4. run stats to show that the normalization worked OK for all shapes
'''

# imports
import trimesh
import os
import numpy as np

#utils
from utils import *

#corners of image for png export
corners = [[-1, -1, -1],
       [ 1, -1, -1],
       [ 1,  1, -1],
       [-1,  1, -1],
       [-1, -1,  1],
       [ 1, -1,  1],
       [ 1,  1,  1],
       [-1,  1,  1]]

# read shape
sample = "./psb-labeled-db/Bird/242.off"
original_mesh = trimesh.load(sample)
mesh = original_mesh.copy()
save_mesh_png(mesh, "1", corners = corners)

# translation to origin
mesh = center_at_origin(mesh)
save_mesh_png(mesh, "2", corners = corners)
translated_mesh = mesh.copy()

# scaling so it fits in unit-sized cube
maxsize = np.amax(np.abs(mesh.bounds))
mesh.apply_scale((1/maxsize, 1/maxsize, 1/maxsize))
scaled_mesh = mesh.copy()
save_mesh_png(mesh, "3", corners = corners)

#BEFORE AND AFTER
#before_after(original_mesh, mesh, corners)

#print stats to demonstrate changes
print("Original barrycenter: {}\nOriginal bounds:\n{}".format(original_mesh.centroid, original_mesh.bounds) )
print("\nNEW barrycenter: {}\nNEW bounds:\n{}".format(mesh.centroid, mesh.bounds) )

#This is giong to be a very small difference actually
print(translated_mesh.centroid - scaled_mesh.centroid)


# #the below loop is currently throwing an error
# ois = original_mesh.__dict__.items()
# tis = translated_mesh.__dict__.items()
# sis = scaled_mesh.__dict__.items()
# for (oa,ov), (ta,tv), (sa,sv) in zip(ois, tis, sis):

#     print(f"checking {ov} change between original and translated mesh")
#     if oa == ta:
#         print(ov == tv)

#     print(f"\nchecking {tv} change between translated and scaled mesh")
#     if ta == sa:
#         print(tv == sv)

