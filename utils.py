import base64
from io import BytesIO

import trimesh
import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from math import sqrt
from ast import literal_eval
from collections import defaultdict
import pyrender
    
from PIL import Image, ImageTk

# corners of image, array used for visually consistent png export of meshes
CORNERS = [[-0.75, -0.75, -0.75],
       [ 0.75, -0.75, -0.75],
       [ 0.75,  0.75, -0.75],
       [-0.75,  0.75, -0.75],
       [-0.75, -0.75,  0.75],
       [ 0.75, -0.75,  0.75],
       [ 0.75,  0.75,  0.75],
       [-0.75,  0.75,  0.75]]

# db paths
PRINCETON_PATH = "./princeton-labeled-db/"
PSB_PATH = "./psb-labeled-db/"

# parameters
IS_OUT_LOW = 3500
IS_OUT_HIGH = 35000

def extract_attributes_from_path(mesh_path):
    """Given a path, loads the mesh, checks if it's an outlier,
    and then adds required attributes of mesh to the out_dict to be returned"""

    mesh = trimesh.load(mesh_path)

    return extract_attributes_from_mesh(mesh, mesh_path)

def extract_attributes_from_mesh(mesh, mesh_path=None, filename = "NA"):
    """Extract features from a mesh that has already been loaded."""
    
    if mesh_path == None:
        mesh_path = "NA/NA/"+filename
        
    
    # get moments of inertia just ONCE
    fx, fy, fz = moments_of_inertia(mesh)

    out_dict = {"filename" : mesh_path.split('/')[-1],
                "path" : mesh_path,
                "area":mesh.area,
                "category" : mesh_path.split('/')[-2], # this may change
                "num_faces" : len(mesh.faces),
                "num_vertices" : len(mesh.vertices),
                "max_extent": max(mesh.bounding_box.extents),
                "faces_type" : 'triangles',
                "axis_aligned_bounding_box" : mesh.bounding_box.extents,
                "boundingbox_diagonal": sqrt(sum([x*x for x in mesh.bounding_box.extents])), # diagonal of bounding box
                "has_holes" : True if not mesh.is_watertight else False,
                "is_out_low" : True if len(mesh.vertices) <= IS_OUT_LOW else False,
                "is_out_high" : True if len(mesh.vertices) >= IS_OUT_HIGH else False,
                "centroid" : mesh.centroid,
                "centroid_to_origin" : sqrt(sum([x*x for x in mesh.centroid])), # distance of centroid to origin
                "boundingbox_distance":sqrt(sum([x*x for x in 0.5*(mesh.bounds[1]+mesh.bounds[0])])), # boundingbox center, distance to origin
                "pca_pose": pca_pose(mesh), # abs value of cosine of major variance direction with x axis
                "fx":fx, "fy":fy, "fz":fz, # moments of inertia along each axis
                }
    
    return out_dict
    
def attributes_csv_to_dict(csv_path):
    """Turn csv file of attributes into dictionary of dictionaries indexed by filename, each attribute is a key in each file's dictionary"""

    files_df = pd.read_csv(csv_path)
    attributes_dict = {row['filename']:row.to_dict() for index, row in files_df.iterrows()}

    return attributes_dict

def center_at_origin(mesh):
    """Given a trimesh object,
    returns a new mesh that has been translated so barycenter is at origin"""

    translated_mesh = mesh.copy()
    translated_mesh.vertices = mesh.vertices - mesh.centroid

    return translated_mesh

def scale_to_unit(mesh):
    """Return mesh scaled to unit cube"""

    scaled_mesh = mesh.copy()
    maxsize = np.max(mesh.bounding_box.extents) # find max coordinate magnitude in any dim
    scaled_mesh.apply_scale((1/maxsize, 1/maxsize, 1/maxsize))

    return scaled_mesh

def display_mesh_with_axes(mesh):
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(trimesh.creation.axis(axis_length = 1)) # add x, y, z axes to scene
    scene.show(viewer='gl')

def save_mesh_png(mesh, filename, corners = None):
    """Save mesh object as png along with the x,y,z axes visualized"""

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(trimesh.creation.axis(axis_length = 1)) # add x, y, z axes to scene

    if corners is None: # add corners
        corners = scene.bounds
    
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
        
def mesh_to_PIL_img(mesh):
    """SIMPLE SOLUTION USING PYRENDER
    Returns PIL image object
    
    
    Using code from https://stackoverflow.com/questions/43724600/rendering-a-3d-mesh-into-an-image-using-python
    """
    
    
    scene = pyrender.Scene.from_trimesh_scene(mesh.scene())

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)

    c = 2**-0.5
    cam_pose = pose=[[ 1,  0,  0,  0],
                            [ 0,  c, -c, -2],
                            [ 0,  c,  c,  2],
                            [ 0,  0,  0,  1]]
    
    scene.add(camera, pose = cam_pose)
    
    light = pyrender.SpotLight(color=np.ones(3), intensity=8.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    scene.add(light, pose=cam_pose)

    # render scene
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)

    img = Image.fromarray(color)
    
    return img


def mesh_to_ImageTk(mesh, size):
    """size should be a tuple of length 2"""
    im = mesh_to_PIL_img(mesh)
    im = im.resize(size)
    return ImageTk.PhotoImage(image=im)

def mesh_to_buffer(mesh, size):
    """size should be a tuple of length 2"""
    im = mesh_to_PIL_img(mesh)
    im = im.resize(size)
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())


# def save_image_of_path(path, tag=None):
#     """Save shape at path to .png, with tag added after underscore"""

#     mesh = trimesh.load(path)
#     file_name = os.path.basename(path)
    
#     if tag != None: # add tag to filename if one is given
#         file_name = file_name + "_" + tag
    
#     save_mesh_png(mesh, file_name)
    
def before_after(mesh1, mesh2, corners = None):
    """Save "before.png" and "after.png" with two meshes;
    Camera bounding box set by SECOND image"""

    if corners is None:
        corners = mesh2.scene().bounds_corners 

    save_mesh_png(mesh1, "before", corners = corners)
    save_mesh_png(mesh2, "after", corners = corners)
    

    
def pca_eigenvectors(mesh, verbose = False):
    """"Return PCA eigenvectors (major variance first, least variance last)"""
    
    # this is a matrix of points of shape (3, nr points)
    A = np.transpose(mesh.vertices)
    A_cov = np.cov(A)
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)
    
    # we now sort eigenvalues by ascending order, saving the index of each rank position:
    ascend_order = np.argsort(eigenvalues)
    
    if verbose: print("PCA before alignment\n", eigenvalues, eigenvectors)
    
    # e1, e2, e3 based on the order of the eigenvalues magnitudes
    # NOTE: e1, e2, e3 all have magnitude 1
    e3, e2, e1 = (eigenvectors[:,index] for index in ascend_order) # the eigenvectors are the COLUMNS of the vector matrix

    return e1, e2, e3 # we return them in descending order

def pca_pose(mesh):
    """Return abs val of cosine of angle between the axis of most variance and the x axis"""

    e1, e2, e3 = pca_eigenvectors(mesh)
    return abs(e1[0]) # return the abs of the 1st element of the e1 vector (i.e. its x coord). This coordinate is the cosine we need 

def pca_align(mesh, verbose=False):
    """Largest variance will align with x axis, least variance will align with z axis. 
    Use pca eigenvectors and eigenvalues to project correctly."""
    
    # find PCA values
    e1, e2, e3 = pca_eigenvectors(mesh, verbose = verbose)
    
    # create new mesh to store pca-aligned object
    aligned_mesh = mesh.copy()

    # set all vertices to 0
    aligned_mesh.vertices = np.zeros(mesh.vertices.shape)
    
    # calculate new mesh's vertex coordinates based on PCA dot product formulas
    for index in range(mesh.vertices.shape[0]): # loop through vertices
        point = mesh.vertices[index] # take point from original mesh

        # calculate new x, y, z coordinates and put into the new mesh
        aligned_mesh.vertices[index] = np.dot(point, e1), np.dot(point, e2), np.dot(point, np.cross(e1, e2))
            
    if verbose: before_after(mesh, aligned_mesh, corners = CORNERS)
    
    if verbose:
        new_pca_vectors = pca_eigenvectors(aligned_mesh)
        print("PCA after alignment (most to least) \n", new_pca_vectors)
    
    return aligned_mesh

def moment_flip(mesh, verbose=False):
    """Flip based on moment: the greater part of the object should be on the POSITIVE side of each axis"""
    
    fx, fy, fz = moments_of_inertia(mesh)
    
    # find the corresponding signs
    sx, sy, sz = np.sign([fx, fy, fz])
    
    # multiply each vertex coordinate by the sign of the corresponding f-value
    for index in range(mesh.vertices.shape[0]):
        mesh.vertices[index] = np.multiply(mesh.vertices[index], (sx, sy, sz))
        
    return mesh

def moments_of_inertia(mesh):
    """Find moments of inertia along the x y and z axes"""

    # get centers of triangles
    triangles = mesh.triangles_center
    
    # calculate the sum of f values for each axis
    fx, fy, fz = np.sum(
        [ (np.sign(x)*x*x, np.sign(y)*y*y, np.sign(z)*z*z) for x,y,z in triangles],
                        axis = 0)
    
    return fx, fy, fz

def normalize_mesh(mesh, verbose = False):
    # translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
    mesh = center_at_origin(mesh)
    # if verbose: save_mesh_png(mesh, "2-translated", corners = CORNERS)

    # align pca: x axis is most variance, z axis is least variance
    mesh = pca_align(mesh)
    # if verbose: save_mesh_png(mesh, "4-pca", corners = CORNERS)
    
    # moment test
    mesh = moment_flip(mesh)
    # if verbose: save_mesh_png(mesh, "5-moment", corners = CORNERS)
    
    # scale to cube vector (with scale_to_unit function from utils) (here bounds value changes)
    mesh = scale_to_unit(mesh)
    # if verbose: save_mesh_png(mesh, "3-scaled", corners = CORNERS)
    
    return mesh

def test_mesh_transformation(function):
    """Takes a hand mesh and transports it so that it is highly off-center. 
    Then exports before and after PNG for the given function"""
    
    mesh = trimesh.load("./psb-labeled-db/Hand/185.off")
    mesh.vertices += (-0.5,-0.5,-0.5)
    mesh.vertices *= (1.2, 1.2, 1.2)
    newmesh = mesh.copy()
    before_after(mesh, function(newmesh), corners = CORNERS)

def before_after_hist(original_csv, norm_csv, attributes):
    """Read in attributes before and after and print their descriptive stats"""

    before = pd.read_csv(original_csv)
    after = pd.read_csv(norm_csv)
    print(f"Summary of attributes BEFORE normalization:\n{before[['axis_aligned_bounding_box', 'centroid', 'area']].describe(include='all')}")
    print(f"\nSummary of attributes AFTER normalization:\n{after[['axis_aligned_bounding_box', 'centroid', 'area']].describe(include='all')}")

    for attribute in attributes:

        # plot hist to compare distr of num_vertices before and after normalization
        sns.kdeplot(before[attribute], color='r', shade=True, label='before')
        sns.kdeplot(after[attribute], color='g', shade=True, label='after')
        plt.title(f"{attribute} before and after normalization pipeline")
        plt.legend()
        plt.show()
    
def update_csv(db_path, csv_path, flat_dir = False):
    """Given a database path, iterate through all shapes in the database and extract attributes into dictionary.
    Export dictionary as CSV to csv_path"""
    
    attributes_dict = {}
    
    # if it's a flat directory
    if flat_dir == True:
        for filename in os.listdir(db_path):
            full_path = os.path.join(db_path, filename)

            # consider only 3D mesh files
            if filename.endswith(('.ply', '.obj', '.off')):
                attributes_dict[filename] = extract_attributes_from_path(full_path)
                print("Extracted attributes of", full_path)
            
    # if it's a database with categories   
    elif flat_dir == False:
        for dirname in os.listdir(db_path):
            if dirname != ".DS_Store":
                for filename in os.listdir(os.path.join(db_path, dirname)):
                    full_path = os.path.join(db_path, dirname, filename)

                    # consider only 3D mesh files
                    if filename.endswith(('.ply', '.obj', '.off')):
                        attributes_dict[filename] = extract_attributes_from_path(full_path)
                        print("Extracted attributes of", full_path)
                        
    # write dictionary to csv
    output = pd.DataFrame.from_dict(attributes_dict, orient='index')
    output.to_csv(csv_path)

def filename_to_class(attributes_df):
    """Return dict mapping filename to class/category in a given attributes dataframe"""
    file2class = {}
    class2files= defaultdict(list)
    for index, row in attributes_df.iterrows():
        file2class[row["filename"]] = row["category"]
        class2files[row['category']].append(row["filename"])
    return file2class, class2files

def standardize_single_value(value, mean, std, verbose = False):
    """Standardize a single vanue given a mean and std
    CENTERED at 0.5 and most values will be within [0,1]"""
    standardized = (0.5 + (value-mean)/(7*std)) #distance from 0 to 1 should be n standard deviations
    
    if verbose:
        if standardized<0 or standardized > 1:
            print(f"Value: {value}, Mean: {mean}, Std: {std}, Standardized: {standardized}")
    
    return standardized

def standardize_column(column, mean=None, std=None):
    """Take an iterable and standardize it (with given mean/std if given)
    Return new set of values, also mean and std"""
    
    if mean == None or std == None: # calculate standardization parameters if not given
        mean = np.mean(column)
        std = np.std(column) 
    
    newcolumn = []
    for value in column:
        newcolumn.append(standardize_single_value(value, mean, std, verbose = False))
    
    return newcolumn, mean, std
        