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

import pandas
import pandas as pd
from main_retrieval import *
from tqdm import tqdm
import seaborn as sn
import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from tqdm import tqdm

CLASSIFY_ALL_OBJECTS = False
GRAPH_PRECISION_RECALL = True


from database import Database


class Evaluator:
    metrics = {
        'Precision': (lambda tp, fp, tn, fn: tp / (tp + fp)),
        'Recall': (lambda tp, fp, tn, fn: tp / (tp + fn)),
        'Specificity': (lambda tp, fp, tn, fn: tn / (tn + fp)),
        'Accuracy': (lambda tp, fp, tn, fn: (tp + tn) / (tp + tn + fp + fn)),
        'F_1': (
            lambda tp, fp, tn, fn:
                2 * Evaluator.metrics['Precision'](tp, fp, tn, fn) * Evaluator.metrics['Recall'](tp, fp, tn, fn) /
                (Evaluator.metrics['Precision'](tp, fp, tn, fn) + Evaluator.metrics['Recall'](tp, fp, tn, fn))
        )
    }

    def __init__(self, db: Database):
        print('Performing evaluation...')

        # Create a new dict for the data
        data = {
            'all': {}
        }

        # Loop over the entire database
        with tqdm(total=len(db), unit="shapes", dynamic_ncols=True) as pbar:
            for className, filename in db:
                # Get the size of the database and the size of the class
                d = len(db)
                c = len(db.shapes[className])

                # Query the shape
                results, _ = run_query(os.path.join(db.path, className, filename), c)

                # Calculate tp and fp
                tp = len(results[results['category'] == className])
                fp = len(results[results['category'] != className])

                # Calculate the true and false negatives based on the database and class size
                fn = c - tp
                tn = d - c - fp

                # Calculate and store all the metrics
                for key, metric in self.metrics.items():
                    if className not in data:
                        data[className] = {}

                    data['all'][key] = data[className][key] = metric(tp, fp, tn, fn)

                pbar.update()

        # Normalize the values
        for metric in data['all']:
            data['all'][metric] /= len(db)

        for className in db.shapes:
            for metric in data[className]:
                data[className][metric] /= len(db.shapes[className])

        self.data = pd.DataFrame.from_dict(data, orient='index')

    def save(self, output: str):
        # Create all needed dirs
        if os.path.dirname(output):
            os.makedirs(os.path.dirname(output), exist_ok=True)

        # Write the data to a csv file
        self.data.to_csv(output + '.csv', index=False, float_format='%.6f')

        # Log that it is done
        print('Output saved to', output + '.csv')



CLASSIFY_ALL_OBJECTS = False

"""============PREDICT CLASS FOR EVERY OBJECT AND PLOT CONFUSION MATRIX==========="""

EVAL_OUTPUT = "evaluation"


def classify_eval_all_objects(scalar_weight = 0.5, prediction_type = "multiple"):

    os.makedirs(EVAL_OUTPUT, exist_ok=True)

    k = 5

    features_df = pd.read_csv(FEATURES_CSV)

    predicted_classes = []
    true_classes = []

    all_paths = features_df["path"] # list of all paths to classify
    all_true_labels = features_df["category"]


    for path, true_label in tqdm(zip(all_paths, all_true_labels)):
        predicted_classes += list(predict_class(path, scalar_weight = scalar_weight, k = k, return_format = prediction_type, verbose=False))
        if prediction_type == "multiple":
            true_classes += k*[true_label]
        elif prediction_type == "majority":
            true_classes += [true_label]

    # create and plot confusion matrix as heat map
    cm = pd.crosstab(true_classes, predicted_classes, rownames = ['True'], colnames = ["Predicted"], margins = False)
    plt.figure(figsize=(15, 15))
    sn.heatmap(cm, annot=True)
    plt.savefig(os.path.join(EVAL_OUTPUT, f'confusion_matrix_scalar_weight_{scalar_weight}.pdf')) 
    

    labels = list(set(true_classes+predicted_classes))
    precisions, recalls, f1s, _ = precision_recall_fscore_support(true_classes, predicted_classes, labels = labels)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(true_classes, predicted_classes, labels = labels, average = "weighted")
    accuracy = accuracy_score(true_classes, predicted_classes)

    #add overall scores
    labels.append("OVERALL")
    precisions = np.append(precisions, [overall_precision])
    recalls = np.append(recalls, [overall_recall])
    f1s = np.append(f1s, [overall_f1])

    for label, precision_score, recall_score, f1 in zip(labels, precisions, recalls, f1s):
        print(f"{label}: P = {precision_score}, R = {recall_score}")

    pr_df = pd.DataFrame.from_dict({"True Label": labels,
                                    "Precision": precisions,
                                    "Recall": recalls,
                                    "F1": f1s})

    pr_df.to_csv(os.path.join(EVAL_OUTPUT, f"precision_recall_scalar_weight_{scalar_weight}.csv"))

    print(f"Overall accuracy: {accuracy} for scalar weight {scalar_weight}")




if __name__ == "__main__":

    scalar_weights = [0, 0.25, 0.5, 0.75, 1]
    
    if CLASSIFY_ALL_OBJECTS:
    # make prediction for EVERY object in the feature database
        
        for scalar_weight in scalar_weights:
            classify_eval_all_objects(scalar_weight = scalar_weight)
        
    if GRAPH_PRECISION_RECALL:
        paths = ["./evaluation/precision_recall_scalar_weight_0.csv",
        "./evaluation/precision_recall_scalar_weight_0.25.csv",
        "./evaluation/precision_recall_scalar_weight_0.5.csv",
        "./evaluation/precision_recall_scalar_weight_0.75.csv",
        "./evaluation/precision_recall_scalar_weight_1.csv"]
        
        performance_data = {"Scalar Weight":[], "Precision":[], "Recall":[], "F1":[]}
        
        for sw, path in zip(scalar_weights, paths):
            df = pd.read_csv(path)
            df = df[df['True Label'] == "OVERALL"] # get overall data
            
            print(df)
            
            performance_data["Scalar Weight"].append(sw)
            performance_data["Precision"].append(float(df['Precision']))
            performance_data["Recall"].append(float(df['Recall']))
            performance_data["F1"].append(float(df['F1']))
            
        performance_df = pd.DataFrame.from_dict(performance_data)
        performance_df.plot(x="Scalar Weight", fontsize = 50)
        plt.legend(prop={'size': 50})
        plt.xlabel("Scalar Weight", fontsize = 40)
        performance_df.to_csv(os.path.join("evaluation", "precision_recall_overall.csv"))
        plt.savefig(os.path.join("evaluation", "precision_recall_scalar_weight.pdf"))
            
            
            
        
    
    

    database = Database("psb-labeled-db", ['.off', '.ply', '.obj'])
    Evaluator(database).save("metrics.csv")


