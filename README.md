# multimedia-retrieval-project-2022
This repository holds code and resources for building a content-based 3D shape retrieval system that, given a 3D shape, finds and shows to the user the most similar shapes in a given 3D shape database. The project is part of the Multimedia Retrieval master-level course at Utrecht University.

## Preparation

Before running any of the scripts, make sure the database is present in the folder `psb-labeled-db`. 

## Scripts

Each script is runnable on its own an corresponds to a step of the pipeline. To generate a new database for querying, the scripts should be run in the following order:

1. `db-analysis.py`
2. `meshes-normalization.py`
3. `feature-extraction.py`
4. `querying.py`

This will generate a normalized database, extract and store the features and open a window for using querying the database.
