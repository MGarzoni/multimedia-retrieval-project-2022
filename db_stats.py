'''
Use output of filter (csv file) to find out:
    (a) what is the average shape in the database (in terms of vertex and face counts)
    (b) if there are significant outliers from this average (e.g. shapes having many, or few, vertices or cells)
    best way to do this is to show a histogram counting how many shapes are in the database for every range of the property of interest
    (e.g., number of vertices, number of faces, shape class)

Use the viewer constructed in step 1 to show an average shape and a few such outliers (if any)
'''

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# import and load analysis data
analysis = './psb_analysis.csv'
analysis_in = pd.read_csv(analysis)

# inspect dataframe
analysis_in.describe()
analysis_in.info()

# plotting histogram of integer fields in dataframe
analysis_in.hist()

#plotting distribution of categories
cat_df = pd.DataFrame.from_dict(Counter(analysis_in['class']), orient='index', columns=['Total count'])
cat_df.plot.bar()

