
"""
Compute the following 3D elementary descriptors presented in Module 4: Feature extraction:

    surface area
    compactness (with respect to a sphere)
    axis-aligned bounding-box volume
    diameter
    eccentricity (ratio of largest to smallest eigenvalues of covariance matrix) 

All above are simple global descriptors, that is, they yield a single real value.

Besides these, compute also the following shape property descriptors:

    a3: angle between 3 random vertices
    D1: distance between barycenter and random vertex
    D2: distance between 2 random vertices
    D3: square root of area of triangle given by 3 random vertices
    D4: cube root of volume of tetrahedron formed by 4 random vertices 

These last five descriptors are distributions, not single values. Reduce them to a fixed-length descriptor using histograms.
For this, compute these descriptors for a given (large) number of random points on a given shape (you see now why you need to have finely-meshed shapes?).
Next, bin the ranges of these descriptors on a fixed number of bins B, e.g., 8..10, and compute how many values fall within each bin.
This gives you a B-dimensional descriptor.
"""

import trimesh
import random
import numpy as np
import os
import pandas as pd
from math import dist
import seaborn as sns

# load sample mesh
test_mesh = "./normalized-psb-db/24.off"
mesh = trimesh.load(test_mesh)

# below try to check for mesh to be watertight and try to fix it
mesh.show()
if not mesh.is_watertight:
    mesh.repair.fix_winding()
    mesh.repair.fix_invertion()
    mesh.repair.fix_normals()
    mesh.repair.fix_invertion()
mesh.show()

'''SIMPLE 3D GLOBAL DESCRIPTORS'''
area = mesh.area
# volume below makes sense only if:
    # if all triangles on mesh are consistently oriented
    # meshes has no holes = is watertight
volume = mesh.volume
aabb_volume = mesh.bounding_box_oriented.volume
compactness = pow(area, 3) / pow(volume, 2)

def get_diameter(mesh):
    '''given a mesh, get the furthest points on the convex haul and then try all possible combinations
    of the distances between points and return the max one'''

    convex_hull = mesh.convex_hull
    max_dist = 0

    for v1, v2 in zip(convex_hull.vertices, convex_hull.vertices):
        if (v1 != v2).any():
            dist = np.linalg.norm(v1 - v2)
            if dist > max_dist:
                max_dist = dist

    return max_dist
diameter = get_diameter(mesh)

def get_eccentricity(mesh):
    '''same as for alignment: given a mesh, get the covariance matrix of the vertices, get eigens
    and then divide the largest value over the smallest'''

    covariance = np.cov(np.transpose(mesh.vertices))
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    return np.max(eigenvalues) / np.min(eigenvalues)
eccentricity = get_eccentricity(mesh)

'''SHAPE PROPERTY DESCRIPTORS (DISTRIBUTIONS)'''
def calculate_a3(mesh):
    '''given an array of three-sized arrays (vertices),
    return the angles between every 3 vertices'''
    # taken from: https://stackoverflow.com/a/35178910

    results = []
    for vertex in mesh.vertices:

        # get a, b, c as the elements of the vertices list
        a, b, c = vertex[0], vertex[1], vertex[2]
        ba = a - b
        bc = b - c

        # calculate angle between the two lines and append it to the results
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        results.append(np.degrees(angle))

    return results
a3 = calculate_a3(mesh)

def calculate_d1(mesh):
    '''given a mesh, return the distances between barycenter and all vertices'''

    results = []
    for vertex in mesh.vertices:

        # get distance between centroid and vertex and append it to results
        result = float(np.sqrt(np.sum(np.square(mesh.centroid - vertex))))
        results.append(result)
    
    return results
d1 = calculate_d1(mesh)

def calculate_d2(mesh):
    '''given a mesh, return the distances between pairs of vertices'''

    # get distance between consecutive pairs of vertices
    dist_pairs = [[float(np.sqrt(np.sum(np.square((mesh.vertices[i], mesh.vertices[i + 1])))))]
                for i in range(len(mesh.vertices) - 1)]
    flat_dist_pairs = [item for sublist in dist_pairs for item in sublist]
    
    return flat_dist_pairs
d2 = calculate_d2(mesh)

def calculate_d3(mesh):
    '''given a mesh, return the square root of areas of triangles calculated by each 3 vertices'''

    results = []
    for vertex in mesh.vertices:

        # get a, b, c as the elements of the vertices list
        a, b, c = vertex[0], vertex[1], vertex[2]

        # calculate the semi-perimeter
        s = (a + b + c) / 2

        # calculate the area
        area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
        result = float(np.sqrt(area))
        results.append(result)

    return results
d3 = calculate_d3(mesh)

def calculate_d4(mesh):
    '''given a mesh, return the cube root of volume of tetrahedron formed by 4 random vertices'''

    mesh_vertices = mesh.vertices
    v1 = random.choice(mesh_vertices)
    v2 = random.choice(mesh_vertices)
    v3 = random.choice(mesh_vertices)
    v4 = random.choice(mesh_vertices)

    pass # to complete

def extract_features(root, to_csv=False):
    '''This function takes a DB path as input and returns a matrix where every row represents a sample (shape)
    and every column is a 3D elementary descriptor; the value in each cell refers to that feature value of that shape.'''

    # initialize dictionary holding feature values
    features = {'area': [], 'volume': [], 'aabb_volume': [], 'compactness': [], 'diameter': [], 'eccentricity': [],
                'A3': [], 'D1': [], 'D2': [], 'D3': [], 'D4': []}

    from tqdm import tqdm

    for file in tqdm(os.listdir(root)):
        mesh = trimesh.load(root + file)
        features['area'].append(mesh.area)
        features['volume'].append(mesh.volume)
        features['aabb_volume'].append(mesh.bounding_box_oriented.volume)
        features['compactness'].append(pow(mesh.area, 3) / pow(mesh.volume, 2))
        features['diameter'].append(get_diameter(mesh))
        features['eccentricity'].append(get_eccentricity(mesh))
        features['A3'].append(calculate_a3(mesh))
        features['D1'].append(calculate_d1(mesh))
        features['D2'].append(calculate_d2(mesh))
        features['D3'].append(calculate_d3(mesh))
        # features['D4'].append(calculate_d4(mesh))

        print(f"processed {file}")

    # below i am specifying orient='index' and then transposing the dataframe just cause for now we have some feats (such as D4) which are empty
    features_matrix = pd.DataFrame.from_dict(features, orient='index')
    features_matrix = features_matrix.transpose()

    if to_csv:
        features_matrix.to_csv('./features/features.csv')

    return features_matrix

features_matrix = extract_features(root='./reduced-normalized-psb-db/', to_csv=True)
features_matrix.head()

features_matrix = pd.read_csv("./features/features.csv")

def dist_heatmap(features_matrix:dict):
    '''Function that takes a feature matrix (N*D, where N is the number of shapes and D is the number of descriptors),
    converts it to a dataframe'''
    from scipy.spatial import distance_matrix

    d_m =  pd.DataFrame(distance_matrix(features_matrix.values, features_matrix.values),
                        index=features_matrix.index, columns=features_matrix.index)
    sns.set(rc = {'figure.figsize':(15, 10)})

    return sns.heatmap(d_m, annot=False).set(title='Heatmap of distance matrix between feature vectors.')

dist_heatmap(features_matrix)
