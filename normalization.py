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

# utils
from utils import *

# corners of image used for png export
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
save_mesh_png(mesh, "original", corners = corners)

# translation to origin
mesh = center_at_origin(mesh)
save_mesh_png(mesh, "translated", corners = corners)
translated_mesh = mesh.copy()

# scaling so it fits in unit-sized cube
maxsize = np.amax(np.abs(mesh.bounds)) #find max coordinate magnitude in any dim
mesh.apply_scale((1/maxsize, 1/maxsize, 1/maxsize))
save_mesh_png(mesh, "scaled", corners = corners)
scaled_mesh = mesh.copy()



# PCA 
eigenvalues, eigenvectors = pca_eigenvalues_eigenvectors(mesh)
print("==> eigenvalues for (x, y, z)")
print(eigenvalues)
print("\n==> eigenvectors")
print(eigenvectors)




# # before and after pictures
# before_after(original_mesh, mesh, corners)

# print stats to demonstrate changes
print("Original barrycenter: {}\nOriginal bounds:\n{}".format(original_mesh.centroid, original_mesh.bounds) )
print("\nNEW barrycenter: {}\nNEW bounds:\n{}".format(mesh.centroid, mesh.bounds) )



#more statistics

#This is giong to be a very small difference actually
#print(translated_mesh.centroid - scaled_mesh.centroid)
