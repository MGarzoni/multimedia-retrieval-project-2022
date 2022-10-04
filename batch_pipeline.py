#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 18:13:00 2022

@author: eduardsaakashvili
"""

import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

#importing own functions
from normalization import *
from utils import *

csv_path = "./psb_analysis.csv"

# read file attributes
files_df = pd.read_csv(csv_path)

# find unique categories -- in l-psb, each contains 19 items
categories = Counter(files_df.category)

# SAMPLE n_items from each category -- whole database is too much
n_items = 2
sample_df = pd.concat(
    [files_df[files_df['category'] == category].sample(n_items) for category in categories.keys()], 
                      axis = 0)

#loop pipeline on SAMPLED paths
loop_pipeline(sample_df.path, csv_path, verbose = False)

new_df = pd.read_csv('./normalized/normalized_attributes.csv')

plt.hist([centroid[0] for centroid in new_df.centroid], bins = 1000)
