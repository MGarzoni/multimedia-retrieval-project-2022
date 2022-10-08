# imports
import trimesh
import os
import pandas as pd
import seaborn as sns
import trimesh
import subprocess

# utils
from utils import *

def normalization_pipeline(path, files_dictionary, out_dir="./normalized", verbose=False):
    """Verbose includes IMAGES"""

    # load attributes of filename (from files_dictionary)
    # print('in norm f(x) --> ', path)
    filename = os.path.basename(path)
    # print('in norm f(x) --> ', filename)
    attributes = files_dictionary[filename]
    # print(attributes)
    
    # if it's an outlier, remesh and save remeshed to out_dir
    if attributes['is_out']:
        print(attributes['filename'], "is an outlier!")

        # define path to the shape to remesh and set subdivider command so that it overwrites the file
        shape_to_remesh_path = attributes['path']
        print(f"shape_to_remesh_path --> {shape_to_remesh_path}")
        remeshed_shape_path = f"{out_dir}/{os.path.split(os.path.split(attributes['path'])[0])[1]}"
        print(f"remeshed_shape_path --> {remeshed_shape_path}")
        subdivider_command = f"java -jar catmullclark.jar {shape_to_remesh_path} {remeshed_shape_path}"

        # call mccullark subdivider java program
        subprocess.call(subdivider_command, shell=True)
        # update num_vertices column #may be unnecessary since we re-extract attributes at the end
    
    # load mesh from path
    mesh = trimesh.load(path)
    
    # print initial attributes csv
    if verbose: print("Initial attributes:", attributes)
    if verbose:
        # save original as png (this is already AFTER remeshing it if it is an outlier)
        save_mesh_png(mesh, "1-original", corners = CORNERS)

    '''From here actual normalization begins'''

    # translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
    mesh = center_at_origin(mesh)
    if verbose: save_mesh_png(mesh, "2-translated", corners = CORNERS)

    # scale to cube vector (with scale_to_unit function from utils) (here bounds value changes)
    mesh = scale_to_unit(mesh)
    if verbose: save_mesh_png(mesh, "3-scaled", corners = CORNERS)

    # align pca: x axis is most variance, z axis is least variance
    mesh = pca_align(mesh)
    if verbose: save_mesh_png(mesh, "4-pca", corners = CORNERS)
    
    # moment test
    mesh = moment_flip(mesh)
    if verbose: save_mesh_png(mesh, "5-moment", corners = CORNERS)

    '''From here we export normalized mesh as new .off file to normalized folder'''
    off_file = trimesh.exchange.off.export_off(mesh)
    # print(f"out_dir --> {out_dir}")
    # print(f"filename --> {filename}")
    # print(f"os.path.join(out_dir, filename) --> {os.path.join(out_dir, filename)}")
    with open(os.path.join(out_dir, filename), 'w+') as fp:
        fp.write(off_file)

    # call function to extract attributes and add them to output_dictionary
    out_dict = extract_attributes_from_mesh(mesh, out_dir)
    # print(out_dict)
    if verbose: print("Final attributes:", out_dict)

    return out_dict

def loop_normalization_pipeline(database, csv_path, out_dir="./normalized", verbose = False):
    """Run normalization pipeline on all paths in the paths_list;
    the csv file at csv_path is used to extract attributes about the shapes in the paths_list"""
    
    attributes_dict = attributes_csv_to_dict(csv_path)
    # print(attributes_dict)
    
    # initialize new attributes dict
    new_files_dict = {}
    
    # loop through paths in list
    for category in os.listdir(database):
        if ".DS_Store" not in category:
            for shape in os.listdir(os.path.join(database, category)):
                if ".DS_Store" not in shape:
                    filename = os.path.basename(shape)
                    # print('in loop norm f(x) --> ', shape)
                    path = os.path.join(database, category, shape)
                    # print('in loop norm f(x) --> ', path)

                    # normalize and extract attributes into new dictionary
                    new_files_dict[filename] = normalization_pipeline(path, attributes_dict)
                    print("Processed", filename)
    
    # export updated attributes to new csv file
    output = pd.DataFrame.from_dict(new_files_dict, orient='index')
    output.to_csv(f"{out_dir}/normalized_TEST_DATA_attributes.csv")
    print("SAVED TO CSV")


# loop pipeline on TEST paths
csv_path = "./original_TEST_DATA_attributes.csv"
loop_normalization_pipeline(TEST_DATA_PATH, csv_path)

# plot hist to compare distr of centroids before and after normalization
before = pd.read_csv("./original_TEST_DATA_attributes.csv")
after = pd.read_csv("./normalized/normalized_TEST_DATA_attributes.csv")

sns.kdeplot(before['num_vertices'], color='r', shade=True, label='before')
sns.kdeplot(after['num_vertices'], color='g', shade=True, label='after')
plt.title("Number of vertices before and after normalization pipeline")
plt.legend()
plt.show()

sns.kdeplot([centroid[0] for centroid in before['centroid']], color='r', shade=True, label='before')
sns.kdeplot([centroid[0] for centroid in before['centroid']], color='g', shade=True, label='after')
plt.title("Number of vertices before and after normalization pipeline")
plt.legend()
plt.show()

