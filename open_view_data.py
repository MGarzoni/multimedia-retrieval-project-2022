import os
import trimesh
import random
from utils import *

# step 1: function to load and display a random mesh from given db_path
def open_and_display_mesh(db_path):
    categories = [cat for cat in os.listdir(db_path) if len(cat)!=0]
    rand_cat = random.choice(categories)

    # consider only 3D mesh file types and pick a random entity
    entities = [ent for ent in os.listdir(db_path + rand_cat) if ent.endswith(('.ply', '.obj', '.off'))]
    rand_ent = random.choice(entities)

    # load and display mesh
    mesh = trimesh.load(os.path.join(db_path, rand_cat, rand_ent))
    return mesh.show(viewer='gl')

if __name__ == '__main__':
    open_and_display_mesh('psb-labeled-db/')