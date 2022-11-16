from typing import List

import pandas as pd
from annoy import AnnoyIndex


class ANNIndex:
    features = ['area', 'volume', 'aabb_volume', 'eccentricity', 'a3', 'd1', 'd2', 'd3', 'd4']

    def __init__(self, data: pd.DataFrame):
        # Save the dataframe to extract the records later
        self.data = data

        # Get all the column names of the columns that are feature descriptors
        columns = [c for c in data.columns if any(f in c for f in self.features)]

        # Greate the ANN index with feature vectors the size of the columns
        self.annoy = AnnoyIndex(len(columns), 'angular')

        # Add all the data
        for index, row in data[columns].iterrows():
            self.annoy.add_item(index, row)

        self.annoy.build(10)

    def query(self, features: List[float], k: int = 5):
        indices, distances = self.annoy.get_nns_by_vector(features, k, include_distances=True)

        data = self.data[self.data.index.isin(indices)][['path']]
        data['dist'] = distances

        return data
