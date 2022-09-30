# import libraries
import pandas as pd
import random
import trimesh

# import and load analysis data
psb_analysis_in = './psb_analysis.csv'
psb_df = pd.read_csv(psb_analysis_in)

# here want to select outliers to resample,
# i.e. data (meshes) where 'num_vertices' is OUTSIDE the range between 100 and 20000
outliers = [path for path, bool in zip(psb_df['path'], psb_df['num_vertices'].between(0, 2500)) if bool]
print(len(outliers))

# get a random outlier and display it
outlier = random.choice(outliers)
print(f"Path of an outlier: {outlier}")

outlier_mesh = trimesh.load("./psb-labeled-db/Hand/200.off")
outlier_mesh.show()

# resampling here (after running java -jar off tool)
remeshed_cullark_mesh = trimesh.load("./psb-labeled-db/Hand/200-REMESHED.off")
# remeshed_doosabin_mesh = trimesh.load("./psb-labeled-db/fourleg-398-doosabin-REMESHED.off")
print(f"Num of vertices before remeshing: {len(outlier_mesh.vertices)}\nNumber of vertices after remeshing with McCullark method: {len(remeshed_cullark_mesh.vertices)}\nNumber of vertices after remeshing with Doosabin method: {len(remeshed_doosabin_mesh.vertices)}")
