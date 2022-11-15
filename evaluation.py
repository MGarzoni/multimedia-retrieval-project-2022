"""
Class IDs of the shapes in the database are key to understanding how to evaluate your CBSR system.
Simply put: When querying with each shape in the database, the ideal response should list only shapes 
of the same class. When enough items are allowed in the query result, all shapes of the same class should 
be returned as first elements of the query.

You only need to implement a single quality metric. 
Motivate your choice and describe how you implement the respective metric based on how the query system works 
(e.g., you provide a query shape and number K, you get K shapes in return). 
Compute next the metric for all shapes in the database. 
Present the metric results by aggregating it for each class individually and also by computing a grand aggregate 
(over all classes). 
Discuss which are the types of shapes where your system performs best and, respectively, worst. 
"""
import os
import random
import pandas as pd
from main_retrieval import *

# get attributes from a given query mesh
attributes = pd.read_csv("./attributes/normalized-PSB-attributes.csv")
rand_cat = random.choice(os.listdir("./normalized-psb-db/"))
random_query_mesh = random.choice(os.listdir(f"./normalized-psb-db/{rand_cat}/"))
path_random_query_mesh = f"./normalized-psb-db/{rand_cat}/{random_query_mesh}"
query_attributes = attributes.loc[attributes["path"] == path_random_query_mesh]

# run query and retrieve 5 best results
query_results = run_query(path_random_query_mesh)

# get labels of query and results
target_label = list(query_attributes["category"])
query_labels = list(query_results[0]['category'])
print(f"target shape label: {target_label}")
print(f"query shapes (k=5) labels: {query_labels}")

# binarized labels
y_true = [1 for i in range(5)]
y_pred = [1 if ql == rand_cat else 0 for ql in query_labels]
print(f"binarized target shape labels: {y_true}")
print(f"binarized query shapes (k=5) labels: {y_pred}")

# confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
cm = confusion_matrix(y_true, y_pred)
print(f"confusion matrix:\n{cm}")
cr = classification_report(y_true, y_pred, zero_division=0)
print(f"classification report:\n{cr}")

# roc auc score
roc_auc = roc_auc_score(y_true, y_pred)
print(roc_auc)
