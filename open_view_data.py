import os
import trimesh
import random

# set paths to folder containing 3D shapes
princeton_path = "./princeton-labeled-db/plant/"
psb_path = "./psb-labeled-db/Octopus/"
plants = os.listdir(princeton_path)
octopi = os.listdir(psb_path)

# use trimesh to load and display 3D meshes
plant_mesh = trimesh.load(princeton_path + plants[5])
faces, vertices = plant_mesh.faces, plant_mesh.vertices, plant_mesh.
print(len(faces), 3 * len(vertices))
plant_mesh.show()
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
    
# open_and_display_mesh(psb_path)

'''
Start building a simple filter that checks all shapes in the database. The filter should output, for each shape

    the class of the shape
    the number of faces and vertices of the shape
    the type of faces (e.g. only triangles, only quads, mixes of triangles and quads)
    the axis-aligned 3D bounding box of the shapes 
'''

# for each shape in shapes:
#     print(class)
#     print(numfaces & numvertices)
#     print(faces types)
#     print(axis aligned 3D bounding box of the shapes)

def inspect_db(db_path):
    pass

princeton_shape_classes = {}
for dirname in os.listdir("./princeton-labeled-db/"):
    princeton_shape_classes[dirname] = {}
    for filename in os.listdir(os.path.join("./princeton-labeled-db/", dirname)):
        if filename.endswith(('.ply', '.obj', '.off')):
            princeton_shape_classes[dirname].update({filename: []})

for classes, objects in princeton_shape_classes.items():
    for obj in objects:
        mesh = trimesh.load("./princeton-labeled-db/" + obj)
        print(princeton_shape_classes[classes][obj].update({'num_faces': (mesh.faces),
                                                            'num_vertices': (mesh.vertices),
                                                            'faces_types': mesh.faces_types(),
                                                            'axis_aligned_bounding_box': mesh.axis_aligned_bounding_box()}))
        

print("Princeton DB:")
for keys, vals in princeton_shape_classes.items():
    print(f"\tClass name: {keys}, num of shapes in it: {len([v for v in vals if v.endswith(('.ply', '.obj', '.off'))])}")

psb_shape_classes = {}
for dirname in os.listdir("./psb-labeled-db/"):
    psb_shape_classes[dirname] = []
    for filename in os.listdir(os.path.join("./psb-labeled-db/", dirname)):
        if filename.endswith(('.ply', '.obj', '.off')):
            psb_shape_classes[dirname].append(filename)

print("Labeled PSB DB:")
for keys, vals in psb_shape_classes.items():
    print(f"\tClass name: {keys}, num of shapes in it: {len([v for v in vals if v.endswith(('.ply', '.obj', '.off'))])}")

