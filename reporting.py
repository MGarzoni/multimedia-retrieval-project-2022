import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ShapeReport:
    def __init__(self, data: pd.DataFrame, given_ranges=None):  # histograms can be PRE defined!
        self.columns = [
            "num_faces",
            "num_vertices",
            "boundingbox_diagonal",  # diagonal of bounding box
            "centroid_to_origin",  # distance of centroid to origin
            "boundingbox_distance",  # boundingbox center, distance to origin
            "max_extent",  # maximum extent of bounding box
            "area",
            "pca_pose",  # absval of cosine of angle between major variance direction and x axis
            "fx", "fy", "fz",  # moments of inertia along each axis
        ]

        self.stats = pd.DataFrame({
            'stat': self.columns,
            'mean': data[self.columns].mean(),
            'median': data[self.columns].median(),
            'stddev': data[self.columns].std(),
            'min': data[self.columns].min(),
            'max': data[self.columns].max()
        })

        self.histograms = {}
        self.ranges = {}  # can be called externally to re-use ranges elsewhere
        self.data = data

        for column in self.columns:  # import data corresponding to self.columns and put into histograms

            # apply range if given_ranges were passed
            if given_ranges:
                given_range = list(given_ranges[column])

                # force-include 0 and 1 if they are excluded
                if given_range[0] > 0:
                    given_range[0] = -0.005
                if given_range[1] < 1:
                    given_range[1] = 1.005

                self.histograms[column] = np.histogram(data[column], bins=100, range=given_range)

            else:
                self.histograms[column] = np.histogram(data[column], bins=100)

            # save the range of this column
            self.ranges[column] = (data[column].min(), data[column].max())

    def save(self, output: str):

        # create all needed dirs
        os.makedirs(output, exist_ok=True)

        # write the stats to a csv file
        self.stats.to_csv(os.path.join(output, 'stats.csv'), index=False, float_format='%.6f')

        plt.style.use('grayscale')

        for column in self.columns:  # iterate through the columns of data in the ShapeReport object

            # save the data from this column to a PDF histogram
            counts, bins = self.histograms[column]  # will give ERROR if
            _, ax = plt.subplots()
            plt.hist(bins[:-1], bins, weights=counts)

            plt.title(column)
            plt.text(0.6, 0.8, f"Range:\n({self.data[column].min()}, \n{self.data[column].max()})",
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            plt.savefig(os.path.join(output, f'{column}.pdf'))
            plt.close()

        # log that it is done
        print('Output saved to', output)


class FeatureReport:
    config = {
        'a3': {
            'title': 'a3, $shape_class',
            'range': (0, math.pi)
        },
        'd1': {
            'title': 'd1, $shape_class',
            'range': (0, math.sqrt(3))
        },
        'd2': {
            'title': 'd2, $shape_class',
            'range': (0, math.sqrt(3))
        },
        'd3': {
            'title': 'd3, $shape_class',
            'range': (0, 0.85)
        },
        'd4': {
            'title': 'd4, $shape_class',
            'range': (0, 1.0)
        }
    }

    def __init__(self, data: pd.DataFrame):
        print('Generating features report...')

        # No data processing needed
        self.data = data

    def save(self, output: str, group: bool = False):
        # Create all needed dirs
        os.makedirs(output, exist_ok=True)

        # Loop over all the features that have to be plotted
        for key in self.config:
            # Get all the shape classes
            shape_classes = self.data['category'].unique()

            # Check if group plots must be generated
            if group:
                fig, axes = plt.subplots(nrows=math.ceil(len(shape_classes) / 4), ncols=4)

                for i, ax in enumerate(axes.flatten()):
                    # Plot all the axis object unless there are more than shape classes
                    if i >= len(shape_classes):
                        break

                    # Get the class
                    shape_class = shape_classes[i]

                    # Get the features
                    shape_features = self.data[self.data['category'] == shape_class].filter(like=key, axis=1)

                    # Recalculate the bins
                    bins = np.linspace(self.config[key]['range'][0],
                                       self.config[key]['range'][1],
                                       len(shape_features.columns) + 1)

                    # Generate all the plots
                    for _, shape_feature in shape_features.iterrows():
                        ax.plot(bins[:-1], shape_feature)

                    # Update the axes and limits
                    ax.set_title(self.config[key]['title'].replace('$shape_class', shape_class), 
                                 fontsize=8,
                                 y=1.0, pad=-14)
                    
                    ax.set_xlim(self.config[key]['range'])

                # Save the combined plots
                fig.savefig(os.path.join(output, key + '.pdf'))
            else:
                for shape_class in shape_classes:
                    # Get the features
                    shape_features = self.data[self.data['category'] == shape_class].filter(like=key, axis=1)

                    # Recalculate the bins
                    bins = np.linspace(self.config[key]['range'][0],
                                       self.config[key]['range'][1],
                                       len(shape_features.columns) + 1)

                    # Generate all the plots
                    for _, shape_feature in shape_features.iterrows():
                        plt.plot(bins[:-1], shape_feature)

                    # Update the axes and limits
                    plt.title(self.config[key]['title'].replace('$shape_class', shape_class))
                    plt.xlim(self.config[key]['range'])
                    plt.savefig(os.path.join(output, shape_class + '_' + key + '.pdf'))
                    plt.close()
