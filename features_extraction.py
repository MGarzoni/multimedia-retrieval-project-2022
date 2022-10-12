
"""
Compute the following 3D elementary descriptors presented in Module 4: Feature extraction:

    surface area
    compactness (with respect to a sphere)
    axis-aligned bounding-box volume
    diameter
    eccentricity (ratio of largest to smallest eigenvalues of covariance matrix) 

Note that the definitions given in Module 4 are for 2D shapes. You need to adapt them to 3D shapes (easy).

All above are simple global descriptors, that is, they yield a single real value. Besides these, compute also the following shape property descriptors:

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

# imports
import trimesh
import random
import numpy as np

# load sample mesh
test_mesh = "./normalized/22.off"
mesh = trimesh.load(test_mesh)

'''FEATURE EXTRACTION'''

area = mesh.area
volume = mesh.volume # this quantity only makes sense if the mesh has no holes (watertight)
aabb_volume = mesh.bounding_box_oriented.volume
compactness = pow(area, 3) / pow(volume, 2)
# calculate diameter
# calculate eccentricity

def calculate_a3(rand_mesh_vertices):
    '''angle between 3 random vertices'''
    # taken from: https://stackoverflow.com/a/35178910

    ba = rand_mesh_vertices[0] - rand_mesh_vertices[1]
    bc = rand_mesh_vertices[2] - rand_mesh_vertices[1]

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)
calculate_a3(random.choice(mesh.vertices))

def calculate_d1(mesh_vertices):
    '''distance between barycenter and random vertex'''
    return np.sqrt(np.sum(np.square(mesh.centroid - random.choice(random.choice(mesh_vertices)))))
calculate_d1(mesh.vertices)

def calculate_d2(mesh_vertices):
    '''square root of area of triangle given by 3 random vertices'''
    return np.sqrt(np.sum(np.square(random.choice(mesh_vertices), random.choice(mesh_vertices))))

def calculate_d3(mesh_vertices):
    '''square root of area of triangle given by 3 random vertices'''
    v1 = random.choice(mesh_vertices)
    v2 = random.choice(mesh_vertices)
    x1, x2, x3 = v1[0], v1[1], v1[2]
    y1, y2, y3 = v2[0], v2[1], v2[2]
    return np.sqrt(abs((0.5)*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))))
calculate_d3(mesh.vertices)

def calculate_d4(mesh_vertices):
    '''cube root of volume of tetrahedron formed by 4 random vertices'''
    v1 = random.choice(mesh_vertices)
    v2 = random.choice(mesh_vertices)
    v3 = random.choice(mesh_vertices)
    v4 = random.choice(mesh_vertices)
    pass # to complete


