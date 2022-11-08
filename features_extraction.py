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

import reporting
from utils import *
from collections import defaultdict
from tqdm import tqdm

'''SHAPE PROPERTY DESCRIPTORS (DISTRIBUTIONS)'''
# constants for histograms
SAMPLE_N = 2000 # nr random samples taken for each distributional feature
BINS = 10
random.seed(46)



def get_diameter(mesh, method="fast"):
    '''given a mesh, get the furthest points on the convex haul and then try all possible combinations
    of the distances between points and return the max one'''

    # convex_hull = mesh.convex_hull
    # max_dist = 0
    # vertices = list(convex_hull.vertices)
    #
    #
    # if method == "fast": # if fast method, REDUCE nr vertices
    #     """THIS CODE IS NOT THE RIGHT APPROACH JUST FASTER BY SAMPLING 100 POINTS"""
    #
    #     if len(vertices) > 100:
    #         vertices = random.sample(vertices, 100)
    #
    # if method == "slow":
    #     pass
    #
    # for i in range(len(vertices)):
    #     for j in range(i, len(vertices)):
    #         dist = np.linalg.norm(vertices[i] - vertices[j])
    #         if dist > max_dist:
    #             max_dist = dist
    #
    # return max_dist

    # This does basically the same as the code above but using some kind of splitting algorithm to make the lookup faster
    return trimesh.nsphere.minimum_nsphere(mesh)[1] * 2

def get_eccentricity(mesh):
    '''same as for alignment: given a mesh, get the covariance matrix of the vertices, get eigens
    and then divide the largest value over the smallest'''

    covariance = np.cov(np.transpose(mesh.vertices))
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    return np.max(eigenvalues) / np.min(eigenvalues)


def normalized_histogram_no_bins(values, range = None):
    """Sums to 1, BINS nr of bins, range given by range
    Histogram returned WITHOUT info about bins"""
    hist = np.histogram(values, range = range, bins = BINS)
    
    return hist[0]/np.sum(hist[0])

def plot_hist(histogram):
    """Take as input the output of normalized_histogram"""
    hist, bins = histogram
    plt.step(bins[:-1], hist)

def calculate_a3(mesh):
    '''given an array of three-sized arrays (vertices),
    return the angles between every 3 vertices'''

    random.seed(45)

    
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

    return normalized_histogram_no_bins(results, range=(0, math.pi))

def calculate_d1(mesh):
    '''given a mesh, return density histogrma of distances between barycenter and SAMPLE_N vertices
    Due to scaling ot unit cube distances will not be greater than 1.73 (diagonal of unit cube)'''

    results = []
    
    center = mesh.centroid
    all_vertices = list(mesh.vertices)
    
    random.seed(44)
    
    vertices = random.sample(all_vertices, 1000) # repeats are possible
    
    for vertex in vertices:

        # get distance between centroid and vertex and append it to results
        result = float(np.sqrt(np.sum(np.square(vertex - center))))
        results.append(result)
    
    return normalized_histogram_no_bins(results, range = (0, 1.73))

def calculate_d2(mesh):
    '''given a mesh, return hist of distances between SAMPLE_N pairs of vertices
     Range is set to 0, 1.73 as a greater distance is not possible due to unit cube normalization'''


    random.seed(43)
    
    vertices = list(mesh.vertices)

    # generatre N pairs (could be repeats)
    pairs = [random.sample(vertices, 2) for i in range(SAMPLE_N)]
    
    # get distance between each pair of vertices
    distances = [float(np.sqrt(np.sum(np.square(pair[1]-pair[0]))))
               for pair in pairs]
    
    
    return normalized_histogram_no_bins(distances, range = (0, 1.73))

def calculate_d3(mesh):
    '''given a mesh, return the square roots of areas of SAMPLE_N triangles
    chosen by random trios of three vertices
    Area of a triangle made inside a unit cube can be no more than half the max
    Cross-section area, so no more than 0.7. Square root of that is no more than 0.85'''
    
    random.seed(42)

    
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

    return normalized_histogram_no_bins(sqr_areas, range=(0, 0.85))

def calculate_d4(mesh):
    '''given a mesh, return the cube roots of volume of 
    SAMPLE_N tetrahedrons formed by 4 random vertices
    Volume could not be greater than 1 due to unit cube bounding box'''

    random.seed(41)


    vertices = list(mesh.vertices)
    quartets = [random.sample(vertices, 4) for i in range(SAMPLE_N)]
    
    results = []
    
    for p1, p2, p3, p4 in quartets:
        volume = (1/6) * abs(np.linalg.det((p1-p4, p2-p4, p3-p4))) # formula from Wikipedia
        results.append(np.cbrt(volume)) # add cubic root of volume to results
    
    return normalized_histogram_no_bins(results, range = (0, 1))

'''FEATURE EXTRACTION'''

# this dict stores the distributional feature names and corresponding calculation methods
hist_feature_methods = {"a3":calculate_a3, 
                   "d1":calculate_d1, 
                   "d2":calculate_d2,
                   "d3":calculate_d3, 
                   "d4":calculate_d4}

def extract_scalar_features_from_db(root, to_csv=False, standardize = False):
    '''This function takes a DB path as input and returns a matrix where every row represents a sample (shape)
    and every column is a 3D elementary descriptor; the value in each cell refers to that feature value of that shape.'''

    # initialize dictionary holding feature values
    scalar_features = defaultdict(list)

    for category in tqdm(os.listdir(root)):
        for file in os.listdir(os.path.join(root, category)):
            mesh = trimesh.load(os.path.join(root, category, file))
            scalar_features['filename'].append(file)
            # scalar_features['category'].append(file2class[file])
            scalar_features['category'].append(category)

            # calculate features and put into dictionary
            # do NOT standardize yet!!!! this is done on the column as a whole
            new_object_features = extract_scalar_features_single_mesh(mesh, standardization_parameters_csv=None) # this returns a dict of features for the mesh
            for key, value in new_object_features.items(): # append the new values to the scalar_features dictionary lists
                scalar_features[key].append(value)

        # print(f"processed {file}")
        
    print(scalar_features["diameter"])


    # construct df holding feat values
    scalar_features_matrix = pd.DataFrame.from_dict(scalar_features)
    if standardize:
        standardization_dict = {"feature":[], "mean":[], "std":[]}
        features_to_standardize = ["area", "volume", "aabb_volume", 
                        "compactness", "diameter", "eccentricity"]
        for feature in features_to_standardize:
            
            # STANDARDIZE EACH FEATURE'S COLUMN 
            scalar_features_matrix[feature], mean, std = standardize_column(scalar_features_matrix[feature])
            
            # SAVE STANDARDIZATION PARAMETERS
            standardization_dict["feature"].append(feature)
            standardization_dict["mean"].append(mean)
            standardization_dict["std"].append(std)
            

    # export to csv 
    # (STANDARDIZATION PARAMETERS EXPORTED too)
    if to_csv:
        scalar_features_matrix.to_csv('./features/scalar_features.csv')
        if standardize:
            pd.DataFrame.from_dict(standardization_dict).to_csv('./features/standardization_parameters.csv')

    return scalar_features_matrix

def extract_scalar_features_single_mesh(mesh, standardization_parameters_csv = None, verbose = False):
    """Extract scalar features from a single mesh. Return as dictionary.
        standardization_parameters_csv points to standardization
        Return a DICTIONARY
        
        If no standardization csv given, standardization will not happen"""
        
        
    scalar_features = {} #initialize dictionary
    scalar_features['area'] = mesh.area
    scalar_features['volume'] = mesh.volume # assume mesh has NO HOLES
    scalar_features['aabb_volume'] = mesh.bounding_box_oriented.volume
    scalar_features['compactness'] = pow(mesh.area, 3) / pow(mesh.volume, 2)
    scalar_features['diameter'] = get_diameter(mesh)
    scalar_features['eccentricity'] = get_eccentricity(mesh)
    
    if standardization_parameters_csv: 
        # if a standardization parameters csv is given, STANDARDIZE for each feature in parameters csv
        
        # get standardization parameters
        params_df = pd.read_csv(standardization_parameters_csv)
        params_dict = {row["feature"]:{"mean":row["mean"], 
                                       "std":row["std"]} for _, row in params_df.iterrows()}
        if verbose: 
            print("\n\nScalar features before standardization:\n", scalar_features)
            print("\n\n Loaded standardization parameters:\n", params_dict)
        
        # apply standardization parameters to respective features in scalar_features dicitonary
        for feature in scalar_features.keys():
            scalar_features[feature] = standardize_value(scalar_features[feature], 
                                                         params_dict[feature]["mean"], 
                                                         params_dict[feature]["std"])
        if verbose: print("\n\nScalar features AFTER standardization:\n", scalar_features)
    
    return scalar_features

def extract_hist_features_from_db(root, to_csv=False):
    '''This function takes a DB path as input and returns a matrix where every row represents a sample (shape)
    and every column is a 3D elementary descriptor; the value in each cell refers to that feature value of that shape.'''

    from tqdm import tqdm
    
    
    
    # this dict will hold the feature histograms, bin by bin
    hist_bins = defaultdict(list)


    for category in tqdm(os.listdir(root)):
        for file in os.listdir(os.path.join(root, category)):
            mesh = trimesh.load(os.path.join(root, category, file))

            # append only the VALUES of the histogram, not the bins
            # (these are assumed to be consistent)


            # calcualte the histograms for each feature as a dictionary
            feature_hists = extract_hist_features_single_mesh(mesh,
                                                              returntype = "dictionary")

            #  now save these entries in the hist_bins dictionary
            hist_bins['filename'].append(file)
            hist_bins['category'].append(category)
            for feature in hist_feature_methods.keys():
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

def extract_features_db(root, to_csv=False, standardize = False):
    '''This function takes a DB path as input and returns a matrix where every row represents a sample (shape)
    and every column is a 3D elementary descriptor; the value in each cell refers to that feature value of that shape.'''

    from tqdm import tqdm

    # this dict will hold the feature histograms, bin by bin
    feature_list = defaultdict(list)

    for category in tqdm(os.listdir(root)):
        for file in os.listdir(os.path.join(root, category)):
            mesh = trimesh.load(os.path.join(root, category, file))

            #  now save these entries in the hist_bins dictionary
            feature_list['filename'].append(file)
            feature_list['category'].append(category)

            # append only the VALUES of the histogram, not the bins
            # (these are assumed to be consistent)

            scalar_features = extract_scalar_features_single_mesh(mesh, standardization_parameters_csv=None) # this returns a dict of features for the mesh
            for key, value in scalar_features.items(): # append the new values to the scalar_features dictionary lists
                feature_list[key].append(value)

            # calcualte the histograms for each feature as a dictionary
            feature_hists = extract_hist_features_single_mesh(mesh,
                                                              returntype="dictionary")
            for feature in hist_feature_methods.keys():
                for i in range(BINS):
                    feature_list[f"{feature}_{i}"].append(feature_hists[feature][i])

            print(f"processed {file}")

    # construct df holding feat values
    features_matrix = pd.DataFrame.from_dict(feature_list)

    if standardize:
        standardization_dict = {"feature":[], "mean":[], "std":[]}
        features_to_standardize = ["area", "volume", "aabb_volume",
                        "compactness", "diameter", "eccentricity"]
        for feature in features_to_standardize:

            # STANDARDIZE EACH FEATURE'S COLUMN
            features_matrix[feature], mean, std = standardize_column(features_matrix[feature])

            # SAVE STANDARDIZATION PARAMETERS
            standardization_dict["feature"].append(feature)
            standardization_dict["mean"].append(mean)
            standardization_dict["std"].append(std)


    # export to csv
    # (STANDARDIZATION PARAMETERS EXPORTED too)
    if to_csv:
        features_matrix.to_csv('./features/features.csv')
        if standardize:
            pd.DataFrame.from_dict(standardization_dict).to_csv('./features/standardization_parameters.csv')

    return features_matrix

def extract_hist_features_single_mesh(mesh, 
                                      returntype = "dictionary",
                                      verbose = False):
    
    """RETURN a vector of histogram bins, total length = nfeatures*BINS
    Order of features is based on order of hist_feature_methods.keys()"""
    
    # get the histograms for each feature
    feature_hists = {feature:hist_feature_methods[feature](mesh) for feature in hist_feature_methods.keys()}
    
    # if as dictionary, return dictionary of histograms
    if returntype == "dictionary":
        return feature_hists
    
    # if as vector, return vector
    elif returntype == "vector":
        
        if verbose: print("\nExtracting histograms as single vector...\n")
        
        vector_length = BINS*len(hist_feature_methods)
        output_vector = np.empty([vector_length]) # initialize empty vector
        
        # put them in a SINGLE vector in order of hist_feature_methods.keys())
        for feature_index, feature in enumerate(hist_feature_methods.keys()):
            for bin_index in range(BINS):
                if verbose: print(f"Feature: {feature}, Bin: {bin_index}, Value: {feature_hists[feature][bin_index]}, Vector index: {feature_index*10+bin_index}")
                output_vector[feature_index*10 + bin_index] = feature_hists[feature][bin_index]
        return output_vector
            

def standardize_column(column, mean=None, std=None):
    """Take an iterable and standardize it (with given mean/std if given)
    Return new set of values, also mean and std"""
    
    if mean == None or std == None: # calculate standardization parameters if not given
        mean = np.mean(column)
        std = np.std(column) 
    
    newcolumn = []
    for value in column:
        newcolumn.append(standardize_value(value, mean, std, verbose = False))
    
    return newcolumn, mean, std
    

def standardize_value(value, mean, std, verbose = True):
    """Standardize a single vanue given a mean and std
    CENTERED at 0.5 and most values will be within [0,1]"""
    standardized = (0.5 + (value-mean)/(7*std)) #distance from 0 to 1 should be n standard deviations
    
    if verbose:
        if standardized<0 or standardized > 1:
            print(f"Value: {value}, Mean: {mean}, Std: {std}, Standardized: {standardized}")
    
    return standardized

'''FEATURE EXTRACTION'''
def categories_visualize(hist_df):
    
    feature_names = hist_feature_methods.keys()
    for bin_index in range(BINS):
        pass
    
    # histogram holding classes, files, and each files histograms
    # class_file_histograms = defaultdict(defaultdict(defaultdict()))
    
    # to do:
    # load histograms in
    # create another dictionary in this py file that holds the bins for all these histograms
    # use those bins to graph the histograms, one graph per class
    # then put the graphs in a nice grid
    pass



""" Checking feature extraction by picking some very different samples
and showing that feat values are also very different """

scalar_df = pd.read_csv("./features/scalar_features.csv")
hist_df = pd.read_csv('./features/hist_features.csv')
hist_df.columns
scalar_labels = ['area', 'volume', 'aabb_volume', 'compactness', 'diameter', 'eccentricity']
hist_labels = ['a3_0', 'a3_1', 'a3_2', 'a3_3',
       'a3_4', 'a3_5', 'a3_6', 'a3_7', 'a3_8', 'a3_9', 'd1_0', 'd1_1', 'd1_2',
       'd1_3', 'd1_4', 'd1_5', 'd1_6', 'd1_7', 'd1_8', 'd1_9', 'd2_0', 'd2_1',
       'd2_2', 'd2_3', 'd2_4', 'd2_5', 'd2_6', 'd2_7', 'd2_8', 'd2_9', 'd3_0',
       'd3_1', 'd3_2', 'd3_3', 'd3_4', 'd3_5', 'd3_6', 'd3_7', 'd3_8', 'd3_9',
       'd4_0', 'd4_1', 'd4_2', 'd4_3', 'd4_4', 'd4_5', 'd4_6', 'd4_7', 'd4_8',
       'd4_9']

eccentric_example_0 = scalar_df.loc[scalar_df['category'] == 'Airplane'].iloc[0].drop(['Unnamed: 0', 'filename', 'category'])
eccentric_example_1 = scalar_df.loc[scalar_df['category'] == 'Airplane'].iloc[1].drop(['Unnamed: 0', 'filename', 'category'])
non_eccentric_example_0 = scalar_df.loc[scalar_df['category'] == 'Cup'].iloc[0].drop(['Unnamed: 0', 'filename', 'category'])
non_eccentric_example_1 = scalar_df.loc[scalar_df['category'] == 'Cup'].iloc[1].drop(['Unnamed: 0', 'filename', 'category'])

fig, axs = plt.subplots(2, 2)
# fig.suptitle('Comparison of scalar feature values of two airplanes (right) and two cups (left)')
axs[0, 0].bar(scalar_labels, eccentric_example_0)
axs[0, 0].tick_params(labelrotation=90)
axs[0, 0].set_title('Airplane 1')
axs[0, 1].bar(scalar_labels, non_eccentric_example_0)
axs[0, 1].tick_params(labelrotation=90)
axs[0, 1].set_title('Cup 1')
axs[1, 0].bar(scalar_labels, eccentric_example_1)
axs[1, 0].tick_params(labelrotation=90)
axs[1, 0].set_title('Airplane 2')
axs[1, 1].bar(scalar_labels, non_eccentric_example_1)
axs[1, 1].tick_params(labelrotation=90)
axs[1, 1].set_title('Cup 2')
for ax in axs.flat:
    ax.label_outer()
plt.show()


# Code below will not run if we are only importing
if __name__ == "__main__":
    
    """============Loading and extracting features from sample mesh==============="""
    
    # # load SAMPLE MESH mesh
    # test_mesh = "./psb-labeled-db/Armadillo/284.off"
    # mesh = trimesh.load(test_mesh)
    #
    #
    # '''SIMPLE 3D GLOBAL DESCRIPTORS of SAMPLE MESH'''
    # area = mesh.area
    # volume = mesh.volume
    # aabb_volume = mesh.bounding_box_oriented.volume
    # compactness = pow(area, 3) / pow(volume, 2)
    #
    # # scalar feature functions
    # diameter = get_diameter(mesh)
    # eccentricity = get_eccentricity(mesh)
    #
    # # define which database we are extracting features from here
    NORM_PATH = "./normalized-psb-db/"
    attributes_df = pd.read_csv("./attributes/normalized-PSB-attributes.csv")
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


    """============EXTRACT FEATURES FORM DATABASE=========="""

    
    # EXTRACT_SCALAR_FEATURES = False
    # EXTRACT_HIST_FEATURES = True
    #
    # if EXTRACT_SCALAR_FEATURES:
    #     scalar_matrix = extract_scalar_features_from_db(NORM_PATH, to_csv=True, standardize = True)
    #
    # if EXTRACT_HIST_FEATURES:
    #     extract_hist_features_from_db(NORM_PATH, to_csv=True)

    extract_features_db(NORM_PATH, to_csv=True, standardize=True)
    
    report = reporting.FeatureReport(hist_df)
    report.save('feature_plots')
    report.save('feature_plots_grouped', True)
    
    # testing diameter, fast vs. slow
    # errors = []
    # nvert = []
    # for file in tqdm(os.listdir(NORM_PATH)):
    #     mesh = trimesh.load(NORM_PATH + file)
    #     fast_diam = get_diameter(mesh, method = "fast")
    #     slow_diam = get_diameter(mesh, method = "slow")
    #     error = (fast_diam-slow_diam)/slow_diam
    #     print("Fast dimeter:", fast_diam, "Slow diameter:", slow_diam,
    #           "Error fraction:", error, "#vertices", len(mesh.vertices))
    #     errors.append(error)
    #     nvert.append(len(mesh.vertices))
    # plt.plot(nvert, errors)

