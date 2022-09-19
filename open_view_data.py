import os
import trimesh
import random

# set paths to folder containing 3D shapes
princeton_path = "./princeton-labeled-db/plant/"
psb_path = "./psb-labeled-db/Octopus/"
# plants = os.listdir(princeton_path)
# octopi = os.listdir(psb_path)

# # use trimesh to load and display 3D meshes
# plant_mesh = trimesh.load(princeton_path + plants[5])
# plant_mesh.show()
# octopus_mesh = trimesh.load(psb_path + octopi[20])
# octopus_mesh.show()

def open_and_display_mesh(db_path):
    ents = os.listdir(db_path)
    fits = [True if ents[i].endswith(('.ply', '.obj', '.off')) else False for i in range(len(ents))]
    if fits:
        mesh = trimesh.load(db_path + ents[random.randint(0, len(ents))])
        print()
        return mesh.show()

    # for i in range(len(ents)):
    #     if ents[i].endswith(('.ply', '.obj', '.off')):
    #         mesh = trimesh.load(db_path + ents[random.randint(0, len(ents))])
    #         return mesh.show()
    
open_and_display_mesh(psb_path)