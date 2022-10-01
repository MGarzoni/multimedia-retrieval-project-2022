# imports
import trimesh
import os
import math
import numpy as np
import pandas as pd


def extract_attributes_from_path(mesh_path, outliers_range=range(3500)):
    """Given a path, loads the mesh, checks if it's an outlier,
    and then adds required attributes of mesh to the out_dict to be returned;
    can also set the outlier range (default is (0, 3500))."""

    mesh = trimesh.load(mesh_path)

    return extract_attributes_from_mesh(mesh, mesh_path, outliers_range)


def extract_attributes_from_mesh(mesh, mesh_path, outliers_range = range(3500)):
    """Extract features from a mesh that has already been loaded"""

    out_dict = {"filename" : mesh_path.split('/')[-1],
                "path" : mesh_path,
                "category" : mesh_path.split('/')[-2], # this may change
                "num_faces" : len(mesh.faces),
                "num_vertices" : len(mesh.vertices),
                "faces_type" : 'triangles',
                "axis_aligned_bounding_box" : mesh.bounding_box.extents,
                "is_out" : True if len(mesh.vertices) in outliers_range else False,
                "centroid" : mesh.centroid}
    
    return out_dict
    

def attributes_csv_to_dict(csv_path):
    """Turn csv file of attributes into dictionary of dictionaries indexed by filename, each attribute is a key in each file's dictionary"""

    files_df = pd.read_csv(csv_path)
    files_dict = {row['filename']:row.to_dict() for index, row in files_df.iterrows()}

    return files_dict


def center_at_origin(mesh):
    """Given a trimesh object,
    returns a new mesh that has been translated so barycenter is at origin"""

    translated_mesh = mesh.copy()
    translated_mesh.vertices = mesh.vertices - mesh.centroid

    return translated_mesh


def scale_to_unit(mesh):
    """Return mesh scaled to unit cube"""

    scaled_mesh = mesh.copy()
    maxsize = np.max(mesh.bounding_box.extents) #find max coordinate magnitude in any dim
    scaled_mesh.apply_scale((1/maxsize, 1/maxsize, 1/maxsize))

    return scaled_mesh


def save_mesh_png(mesh, filename, corners = None):
    """Save mesh object as png along with the x,y,z axes visualized"""

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(trimesh.creation.axis(axis_length = 1)) #add x, y, z axes to scene

    if corners is None: # add corners
        corners = scene.bounds_corners
    
    # set 45 degree view so all axes are visible
    r_e = trimesh.transformations.euler_matrix(
        math.radians(45),
        math.radians(45),
        math.radians(45),
        "ryxz")
    
    # use corners and angles to define camera's point of view
    t_r = scene.camera.look_at(corners, rotation=r_e)
    scene.camera_transform = t_r 
    
    # scene.Camera= trimesh.scene.Camera(fov=(camera_fov))
    png = scene.save_image()
    
    # save png
    with open("./pics/"+filename+".png", 'wb') as f:
        f.write(png)
        f.close()


def save_image_of_path(path, tag=None):
    """Save shape at path to .png, with tag added after underscore"""

    mesh = trimesh.load(path)
    file_name = os.path.basename(path)
    
    if tag != None: # add tag to filename if one is given
        file_name = file_name + "_" + tag
    
    save_mesh_png(mesh, file_name)
    

def before_after(mesh1, mesh2, corners = None):
    """Save "before.png" and "after.png" with two meshes;
    Camera bounding box set by SECOND image"""

    if corners is None:
        corners = mesh2.scene().bounds_corners 

    save_mesh_png(mesh1, "before", corners = corners)
    save_mesh_png(mesh2, "after", corners = corners)
    

def pca_eigenvalues_eigenvectors(mesh):
    """Matrix of points of shape (3, nr points)"""

    A = np.transpose(mesh.vertices)
    A_cov = np.cov(A)
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    return eigenvalues, eigenvectors
