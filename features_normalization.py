# """
# normalize the features so they have the same range. This should be done differently for
# single-value features and for histogram features
#     - Single-value features: Normalize by standardization, which is less sensitive to outliers than min-max
#     normalization
#     - Histogram features: Normalize by dividing by the area (element count)

# The above two normalizations bring, statistically speaking, both single-value and histogram features in the (absolute-
# value) range [0,1]; single-value features could still be outside of the [0,1], but we want to keep such outliers

# Single value features are normalized considering the entire set of feature-values in the process (when computing the
# standard deviation). Histogram features are normalized independently per histogram: We dont compute anything
# like the standard deviation histogram

# After the normalization (1), the range of the distances between feature-values for the different features 
# can be very different, e.g.:
#     - eccentricity values can span the whole spectrum [0,1] after normalization
#     - the D4 histogram values for a set of shapes can be very similar, so the range of distances between D4 histograms can be much smaller than 1
#     - for area-normalized histograms, a distance of 1 between two histograms is huge and would likely never be attained in practice

# If we do not address this problem, then features having a small range of distances will count little in the overall distance function.
# There are multiple ways to address this problem:
#     - Feature weighting: For every set of feature-values which, together, form a feature, add a weight wi (so that the
#     sum of all wi equals one). Then, adjust the weights wi , relative to each other, so as to boost the variations of the features having small
#     ranges. In practice, this would typically mean leaving w1 ...w5 equal to each other and making them smaller than
#     the weights w7 ...w9 used for the narrow-range histograms. The advantage of this method is that it is very simple
#     to implement. However, playing around with weights can cost quite some time until the desired results are
#     achieved.
#     - Distance weighting: Rather than weighting the feature values, we can go to the root of the problem and weigh
#     the distance values themselves. This is exactly what the standardization does for the single-value features: It
#     actually considers the spread of values (by measuring their standard deviation, which is a distance) and
#     normalizes them by this spread. We can generalize this also for multiple-value features such as histograms.
#     Consider e.g. the feature-vector elements a6 ...a15 which, together, create the D1 descriptor. We can them
#     compute all distances d(a6 ...a15 ,b6 ...b15 ) between the elements 6..15 of two feature vectors A and B over an
#     entire shape database. These give all possible values for the distances between D1 descriptors over our
#     database, computed by any desired distance function (Euclidean, EMD, cross-bin matching, etc). Then, we can
#     standardize these distances, just before combining them with the other feature distances (that is, for a1..a5 and
#     a16...a45) to yield the final distance. This way, even small variations in D1 will count similarly to e.g. large
#     variations in a1 . (Note that this histogram standardization is strongly dependent on the quality of the extracted descriptors: If,
#     for instance, we have a poor computation of D1, which yields more or less the same values for all shapes, then
#     the standardization above will artificially amplify tiny differences in D1 which likely mean nothing, leading to
#     poor matching.)
# """

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# get features csv
features_matrix = pd.read_csv("./features/features.csv")
features_matrix.head()

# divide feature types so we can apply different normalizations to each
scalar_features = features_matrix[['area', 'volume', 'aabb_volume', 'compactness', 'diameter', 'eccentricity']]
hist_features = features_matrix[['A3', 'D1', 'D2', 'D3', 'D4']]

# apply sclar standardization to scalar features
scaled_scalar_features = StandardScaler().fit_transform(scalar_features)

# normalize hist feats
weights = np.ones_like(hist_features['A3']) / float(len(hist_features['A3']))
p = plt.hist(hist_features['A3'], weights=weights)
plt.ylim(0,1)
plt.show()

# create new cvs with normalized feats

