import os
import trimesh
import random

# DB PATHS
PRINCETON_PATH = "./princeton-labeled-db/"
PSB_PATH = "./psb-labeled-db/"

# step 1: function to load and display a random mesh from given db_path
def open_and_display_mesh(db_path):
    categories = os.listdir(db_path)
    rand_cat = random.choice(categories)

    # consider only 3D mesh file types and pick a random entity
    entities = [ent for ent in os.listdir(db_path + rand_cat) if ent.endswith(('.ply', '.obj', '.off'))]
    rand_ent = random.choice(entities)

    # load and display mesh
    mesh = trimesh.load(os.path.join(db_path, rand_cat, rand_ent))
    return mesh.show(viewer='gl')
    
open_and_display_mesh(PSB_PATH)
