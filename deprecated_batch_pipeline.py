import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

# importing own functions
from deprecated_normalization import *
from utils import *

# read file attributes
csv_path = "./original_TEST_DATA_attributes.csv"
# files_df = pd.read_csv(csv_path)

# # find unique categories -- in l-psb, each contains 19 items
# categories = Counter(files_df.category)

# # SAMPLE n_items from each category -- whole database is too much
# n_items = 2
# sample_df = pd.concat(
#     [files_df[files_df['category'] == category].sample(n_items) for category in categories.keys()], 
#                       axis = 0)

# loop pipeline on TEST paths
loop_normalization_pipeline(TEST_DATA_PATH, csv_path, verbose = False)

# plot hist to compare distr of centroids before and after normalization
before = pd.read_csv("./original_TEST_DATA_attributes.csv")
after = pd.read_csv("./normalized/normalized_TEST_DATA_attributes.csv")
plt.figure()
plt.subplot(211)
plt.hist([centroid[0] for centroid in before.centroid], bins = 1000)
plt.title('before normalization')
plt.subplot(212)
plt.hist([centroid[0] for centroid in after.centroid], bins = 1000)
plt.title('after normalization')
plt.show()
