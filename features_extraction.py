import trimesh
import random
import numpy as np
import os
import pandas as pd
from math import dist, hypot, sqrt
import math
import seaborn as sns
import random
from matplotlib import pyplot as plt
from utils import *
from collections import defaultdict

# load sample mesh
test_mesh = "./psb-labeled-db/Armadillo/284.off"
mesh = trimesh.load(test_mesh)

# define which database we are extracting features from here
NORM_PATH = "./reduced-normalized-psb-db/"
attributes_df = pd.read_csv("./attributes/reduced-normalized-PSB-attributes.csv")
file2class, class2files = filename_to_class(attributes_df) # create dicts mapping filenames to categories


# code below to check if any shape has holes
# root = "./normalized-psb-db/"
# for mesh in os.listdir(root):
#     if ".txt" not in mesh:
#         mesh = trimesh.load(root + mesh)
#         if not mesh.is_watertight:
#             print('mesh has holes:')
#             print(f"\tmesh volume before stitching: {mesh.volume}")
#             mesh.fill_holes()
#             print(f"\tmesh volume after stitching {mesh.volume}")
#         else:
#             print('mesh has NO holes')

'''SIMPLE 3D GLOBAL DESCRIPTORS'''
area = mesh.area
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
# constants for histograms
SAMPLE_N = 2000 # nr random samples taken for each distributional feature
BINS = 10
random.seed(42)

def density_histogram(values, range = None):
    """Integrates to 1, BINS nr of bins"""
    return np.histogram(values, range = range, bins = BINS, density = True)

def plot_hist(histogram):
    """Take as input the output of density_histogram"""
    hist, bins = histogram
    plt.step(bins[:-1], hist)

def calculate_a3(mesh):
    '''given an array of three-sized arrays (vertices),
    return the angles between every 3 vertices'''
    
    vertices = list(mesh.vertices)

    # generatre N trios of vertices (could be repeats)
    trios = [random.sample(vertices, 3) for i in range(SAMPLE_N)]

    results = []

    for trio in trios:

        # three points
        a, b, c = trio
        
        # create two vectors defining the triangle
        ab = b-a
        ac = c-a
        
        cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
        angle = np.arccos(cosine_angle)
                
        results.append(angle)

    return density_histogram(results, range=(0, math.pi))

def calculate_d1(mesh):
    '''given a mesh, return density histogrma of distances between barycenter and SAMPLE_N vertices
    Due to scaling ot unit cube distances will not be greater than 1.5'''

    results = []
    
    center = mesh.centroid
    all_vertices = list(mesh.vertices)
    
    vertices = random.sample(all_vertices, 1000) # repeats are possible
    
    for vertex in vertices:

        # get distance between centroid and vertex and append it to results
        result = float(np.sqrt(np.sum(np.square(vertex - center))))
        results.append(result)
    
    return density_histogram(results, range = (0, 1.5))

def calculate_d2(mesh):
    '''given a mesh, return hist of distances between SAMPLE_N pairs of vertices
     Range is set to 0, 1.5 as a greater distance is not possible due to unit cube normalization'''

    vertices = list(mesh.vertices)

    # generatre N pairs (could be repeats)
    pairs = [random.sample(vertices, 2) for i in range(SAMPLE_N)]
    
    # get distance between each pair of vertices
    distances = [float(np.sqrt(np.sum(np.square(pair[1]-pair[0]))))
               for pair in pairs]
    
    
    return density_histogram(distances, range = (0, 1.5))

def calculate_d3(mesh):
    '''given a mesh, return the square roots of areas of SAMPLE_N triangles
    chosen by random trios of three vertices
    Area of a triangle made inside a unit cube can be no more than half the max
    Cross-section area, so no more than 0.7. Square root of that is no more than 0.85'''
    
    vertices = list(mesh.vertices)

    # generatre N trios of vertices (could be repeats)
    trios = [random.sample(vertices, 3) for i in range(SAMPLE_N)]

    sqr_areas = []

    for trio in trios:

        # three points
        p1, p2, p3 = trio
        
        # create two vectors defining the triangle
        a = p2- p1
        b = p3 - p1
                
        # calculate cross product to get area
        cross_pr = np.cross(a, b)
        
        # magnitude of cross product / 2 = triangle area
        area = 0.5 * hypot(cross_pr[0], cross_pr[1], cross_pr[2])
        
        # square root of area added to results
        sqr_areas.append(sqrt(area))

    return density_histogram(sqr_areas, range=(0, 0.85))

def calculate_d4(mesh):
    '''given a mesh, return the cube roots of volume of 
    SAMPLE_N tetrahedrons formed by 4 random vertices
    Volume could not be greater than 1 due to unit cube bounding box'''

    vertices = list(mesh.vertices)
    quartets = [random.sample(vertices, 4) for i in range(SAMPLE_N)]
    
    results = []
    
    for p1, p2, p3, p4 in quartets:
        volume = (1/6) * abs(np.linalg.det((p1-p4, p2-p4, p3-p4))) # formula from Wikipedia
        results.append(np.cbrt(volume)) # add cubic root of volume to results
    
    return density_histogram(results, range = (0, 1))

'''FEATURE EXTRACTION'''

# this dict stores the feature names and corresponding calculation methods
feature_methods = {"a3":calculate_a3, 
                   "d1":calculate_d1, 
                   "d2":calculate_d2,
                   "d3":calculate_d3, 
                   "d4":calculate_d4}

def extract_scalar_features(root, to_csv=False):
    '''This function takes a DB path as input and returns a matrix where every row represents a sample (shape)
    and every column is a 3D elementary descriptor; the value in each cell refers to that feature value of that shape.'''

    # initialize dictionary holding feature values
    scalar_features = defaultdict(list)

    from tqdm import tqdm

    for file in tqdm(os.listdir(root)):
        mesh = trimesh.load(root + file)
        scalar_features['filename'].append(file)
        scalar_features['category'].append(file2class[file])
        scalar_features['area'].append(mesh.area)
        scalar_features['volume'].append(mesh.volume) # no need to check if mesh has holes cause already checked that no mesh does
        scalar_features['aabb_volume'].append(mesh.bounding_box_oriented.volume)
        scalar_features['compactness'].append(pow(mesh.area, 3) / pow(mesh.volume, 2))
        scalar_features['diameter'].append(get_diameter(mesh))
        scalar_features['eccentricity'].append(get_eccentricity(mesh))

        # print(f"processed {file}")

    # construct df holding feat values
    scalar_features_matrix = pd.DataFrame.from_dict(scalar_features)

    # export to csv
    if to_csv:
        scalar_features_matrix.to_csv('./features/scalar_features.csv')

    return scalar_features_matrix


def extract_hist_features(root, to_csv=False):
    '''This function takes a DB path as input and returns a matrix where every row represents a sample (shape)
    and every column is a 3D elementary descriptor; the value in each cell refers to that feature value of that shape.'''

    from tqdm import tqdm
    
    
    
    # this dict will hold the feature histograms, bin by bin
    hist_bins = defaultdict(list)


    for file in tqdm(os.listdir(root)):
        mesh = trimesh.load(root + file)

        # append only the VALUES of the histogram, not the bins
        # (these are assumed to be consistent)
    
        
        # calcualte the histograms for each feature using the corresponding method from the dict
        feature_hists = {feature:method_name(mesh)[0] for feature, method_name in feature_methods.items()}
        
        #  now save these entries in the hist_bins dictionary
        hist_bins['filename'].append(file)
        hist_bins['category'].append(file2class[file])
        for feature in feature_methods.keys():
            for i in range(BINS):
                hist_bins[f"{feature}_{i}"].append(feature_hists[feature][i])

        print(f"processed {file}")
        
    # construct df holding feat values
    hist_features_matrix = pd.DataFrame.from_dict(hist_bins)
    # hist_features_matrix = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hist_features.items()]))

    # export to csv
    if to_csv:
        hist_features_matrix.to_csv('./features/hist_features.csv')

    return hist_features_matrix

'''FEATURE EXTRACTION'''
def categories_visualize(hist_df):
    
    feature_names = feature_methods.keys()
    for bin_index in range(BINS):
        pass
    
    # histogram holding classes, files, and each files histograms
    # class_file_histograms = defaultdict(defaultdict(defaultdict()))
    
    # to do:
    # load histograms in
    # create another dictionary in this py file that holds the bins for all these histograms
    # use those bins to graph the histograms, one graph per class
    # then put the graphs in a nice grid
        
        
        
    


extract_scalar_features(NORM_PATH, to_csv=True)
extract_hist_features(NORM_PATH, to_csv=True)


# checking feature extraction by picking some very different samples and showing that feat values are also very different
scalar_df = pd.read_csv("./features/scalar_features.csv")
hist_df = pd.read_csv('./features/hist_features.csv')

categories_visualize(hist_df)
