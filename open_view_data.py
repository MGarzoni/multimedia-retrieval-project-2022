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
    entities = os.listdir(db_path + rand_cat)
    rand_ent = random.choice(entities)
    

    # consider only 3D mesh file types
    if rand_ent.endswith(('.ply', '.obj', '.off')):
        mesh = trimesh.load(os.path.join(db_path, rand_cat, rand_ent))
        return mesh.show()
    else:
        print("File is not a 3D mesh")
    
open_and_display_mesh(PSB_PATH)