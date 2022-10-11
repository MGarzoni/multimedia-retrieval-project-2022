"""
Extract the following descriptors and add them to the attributes files:
    Area A  		number of pixels inside the segmented shape
    Perimeter l  		number of pixels along the boundary of A
    Compactness c 	l2/(4pA); how close is the shape to a disk (circle)
    Circularity  		1/c; basically same as compactness, just a different scale
    Centroid  		average of (x,y) coordinates of all pixels in shape
    Rectangularity 	 	A/AOBB; how close to a rectangle the shape is (OBB=oriented bounding box)
    Diameter  		largest distance between any two contour points
    Eccentricity  		|1|/| 2|, where 1 and 2 are the eigenvalues of the shape covariance matrix (in 2D, similar for 3D)

"""

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


import trimesh
import random
import numpy as np

test_mesh = "./normalized/22.off"
mesh = trimesh.load(test_mesh)
area = mesh.area
volume = mesh.volume # this quantity only makes sense if the mesh has no holes (watertight)
aabb_volume = mesh.bounding_box_oriented.volume
compactness = pow(area, 3) / pow(volume, 2)
# diameter
# eccentricity

def calculate_a3(vertices):
    # taken from: https://stackoverflow.com/a/35178910

    ba = vertices[0] - vertices[1]
    bc = vertices[2] - vertices[1]

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)
a3 = calculate_a3(random.choice(mesh.vertices))
d1 = np.sqrt(np.sum(np.square(mesh.centroid - random.choice(random.choice(mesh.vertices)))))
d2 = np.sqrt(np.sum(np.square(random.choice(random.choice(mesh.vertices)), random.choice(random.choice(mesh.vertices)))))
print(d2)

d3 = np.sqrt()
# d3
# d4