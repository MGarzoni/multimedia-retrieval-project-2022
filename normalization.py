'''
1. read each shape
2. translate it so that its barycenter coincides with the coordinate-frame origin
3. scale it uniformly so that it tightly fits in a unit-sized cube
4. run stats to show that the normalization worked OK for all shapes
'''

# imports
import trimesh

# read original mesh
sample = "./test-data-db/airplane/61.off"
original_mesh = trimesh.load(sample)
original_mesh.show(viewer='gl')

orig_scene = original_mesh.scene()
orig_png = orig_scene.save_image()

# generate filename
with open("original.png", 'wb') as f:
    f.write(orig_png)
    f.close()

# save quantities from original mesh
num_vertices_original = len(original_mesh.vertices)
bbox_size_original = (original_mesh.bounding_box.extents[2] - original_mesh.bounding_box.extents[0]) * (original_mesh.bounding_box.extents[3] - original_mesh.bounding_box.extents[1])
bbox_position_original = None # to implement
pose_original = None # to implement
original_quantities = {'num_vertices_original': num_vertices_original,
                        'bbox_size_original': bbox_size_original,
                        'bbox_position_original': bbox_position_original,
                        'pose_original': pose_original}

# translation
translated_mesh = original_mesh.apply_translation((0,0,0)) # not sure if correct
translated_mesh.show()

# save quantities from translated mesh
num_vertices_translated = len(translated_mesh.vertices)
bbox_size_translated = (translated_mesh.bounding_box.extents[2] - translated_mesh.bounding_box.extents[0]) * (translated_mesh.bounding_box.extents[3] - translated_mesh.bounding_box.extents[1])
bbox_position_translated = None # to implement
pose_translated = None # to implement
translated_quantities = {'num_vertices_translated': num_vertices_translated,
                        'bbox_size_translated': bbox_size_translated,
                        'bbox_position_translated': bbox_position_translated,
                        'pose_translated': pose_translated}

# scaling
scaled_mesh = translated_mesh.apply_scale((1,1,1)) # not sure if correct
scaled_mesh.show()

# save quantities from scaled mesh
num_vertices_scaled = len(scaled_mesh.vertices)
bbox_size_scaled = (scaled_mesh.bounding_box.extents[2] - scaled_mesh.bounding_box.extents[0]) * (scaled_mesh.bounding_box.extents[3] - scaled_mesh.bounding_box.extents[1])
bbox_position_scaled = None # to implement
pose_scaled = None # to implement
scaled_quantities = {'num_vertices_scaled': num_vertices_scaled,
                        'bbox_size_scaled': bbox_size_scaled,
                        'bbox_position_scaled': bbox_position_scaled,
                        'pose_scaled': pose_scaled}

# check variance difference quatities to see if normalizations worked:
    # number of sampling points (vertices)
    # size of bounding box (computed via its diagonal)
    # position of bounding box (distance of its center to the origin)
    # pose (absolute value of cosine of angle between major eigenvector and, say, the X axis)
