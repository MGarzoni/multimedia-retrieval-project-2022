# import libraries
import pandas as pd
import random
import trimesh

# import and load analysis data
psb_analysis_in = './psb_analysis.csv'
psb_df = pd.read_csv(psb_analysis_in)

# here want to select outliers to resample,
# i.e. data (meshes) where 'num_vertices' is OUTSIDE the range between 100 and 20000
outliers = [path for path, bool in zip(psb_df['path'], psb_df['num_vertices'].between(100, 20000)) if not bool]
print(len(outliers)) # there are 26 poorly sampled meshes

# get a random outlier and display it
outlier = random.choice(outliers)
print(f"Path of an outlier: {outlier}")
outlier_mesh = trimesh.load(outlier)
outlier_mesh.show()

# resampling here (after running java -jar off tool)
remeshed_cullark_mesh = trimesh.load("./psb-labeled-db/fourleg-398-cullark-REMESHED.off")
remeshed_doosabin_mesh = trimesh.load("./psb-labeled-db/fourleg-398-doosabin-REMESHED.off")
print(f"Num of vertices before remeshing: {len(outlier_mesh.vertices)}\nNumber of vertices after remeshing with McCullark method: {len(remeshed_cullark_mesh.vertices)}\nNumber of vertices after remeshing with Doosabin method: {len(remeshed_doosabin_mesh.vertices)}")
