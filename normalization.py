'''
1. read each shape
2. translate it so that its barycenter coincides with the coordinate-frame origin
3. scale it uniformly so that it tightly fits in a unit-sized cube
4. run stats to show that the normalization worked OK for all shapes
'''

# imports
import trimesh
import os

#utils
from utils import *

# read shape
sample = "./psb-labeled-db/Airplane/80.off"
original_mesh = trimesh.load(sample)
#original_mesh.show(viewer='gl')
save_mesh_png(original_mesh, "original")

original_camera = original_mesh.scene().camera.copy()

# translation
translated_mesh = original_mesh.copy()
translated_mesh.vertices = original_mesh.vertices + [1000, 1000, 1000]
#translated_mesh = translated_mesh.apply_translation((1000,1000,0)) # not sure if correct
#translated_mesh.show()
save_mesh_png(translated_mesh, "translated")

# scaling
scaled_mesh = translated_mesh.apply_scale((1,1,1)) # not sure if correct
scaled_mesh.show()

# check if things actually changed (nothing did - YET)
# here comparing any of the attributes of the meshes will do
print(translated_mesh.centroid == scaled_mesh.centroid)

ois = original_mesh.__dict__.items()
tis = translated_mesh.__dict__.items()
sis = scaled_mesh.__dict__.items()
for (oa,ov), (ta,tv), (sa,sv) in zip(ois, tis, sis):

    print(f"checking {ov} change between original and translated mesh")
    if oa == ta:
        print(ov == tv)

    print(f"\nchecking {tv} change between translated and scaled mesh")
    if ta == sa:
        print(tv == sv)

