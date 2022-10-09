# imports
from utils import *
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt

def normalize_db(database, original_csv, out_dir, verbose=False):
    """This function takes a database, iteratively gets a shape from a category and
    applies normalization steps to the mesh; if the shape is an outlier, the remeshing via 
    catmullclark subvider is triggered; the normalized mesh is then exported as a new .off file and 
    then we call the extract_attributes_from_mesh util function and export the new csv holding the 
    normalized attributes."""

    # call utils function to turn csv file of attributes into dict of dicts indexed by filename
    attributes_dict = attributes_csv_to_dict(original_csv)
    
    # initialize new attributes dict
    new_files_dict = {}

    # walk thru database to apply normalization pipeline to each shape
    for root, categories, shapes in os.walk(database):
        for category in categories:
            for shape in os.listdir(os.path.join(root, category)):

                full_path = os.path.join(root, category, shape)
                # print('\n', full_path)
                filename = os.path.basename(full_path)
                # print(filename)

                shape_attributes = attributes_dict[filename]
                # print(shape_attributes)
                
                # if it's an outlier, remesh and save remeshed to out_dir
                if shape_attributes['is_out']:
                    # print(f"\n{shape_attributes['filename']} is an outlier!")

                    # define path to the shape to remesh and set subdivider command so that it overwrites the file
                    shape_to_remesh_path = os.path.join(os.getcwd(), shape_attributes['path'][1:])
                    # print(f"shape_to_remesh_path --> {shape_to_remesh_path}")
                    remeshed_shape_path = os.path.join(os.getcwd(), out_dir, os.path.split(os.path.split(shape_attributes['path'][1:])[0])[1])
                    # print(f"remeshed_shape_path --> {remeshed_shape_path}")
                    subdivider_command = f"java -jar catmullclark.jar {shape_to_remesh_path} {remeshed_shape_path}"
                    # print(f"subdivider_command --> {subdivider_command}")
                    
                    # call mccullark subdivider java program
                    subprocess.call(subdivider_command, shell=True)
                    # update num_vertices column #may be unnecessary since we re-extract attributes at the end
                
                # load mesh from path
                mesh = trimesh.load(full_path)

                # print initial attributes csv
                # if verbose: print("Initial attributes:", shape_attributes)

                # save original as png (this is already AFTER remeshing it if it is an outlier)
                # if verbose: save_mesh_png(mesh, "1-original", corners = CORNERS)

                '''From here actual normalization begins'''

                # translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
                mesh = center_at_origin(mesh)
                # if verbose: save_mesh_png(mesh, "2-translated", corners = CORNERS)

                # scale to cube vector (with scale_to_unit function from utils) (here bounds value changes)
                mesh = scale_to_unit(mesh)
                # if verbose: save_mesh_png(mesh, "3-scaled", corners = CORNERS)

                # align pca: x axis is most variance, z axis is least variance
                mesh = pca_align(mesh)
                # if verbose: save_mesh_png(mesh, "4-pca", corners = CORNERS)
                
                # moment test
                mesh = moment_flip(mesh)
                # if verbose: save_mesh_png(mesh, "5-moment", corners = CORNERS)

                '''From here we export normalized mesh as new .off file to normalized folder'''

                off_file = trimesh.exchange.off.export_off(mesh)
                # print(f"\nout_dir --> {out_dir}")
                # print(f"filename --> {filename}")
                # print(f"os.path.join(out_dir, filename) --> {os.path.join(out_dir, filename)}")

                # this one below should be tweaked to create a folder for each mesh before saving, and then save it an the correct category folder
                with open(os.path.join(out_dir, filename), 'w+') as fp:
                    fp.write(off_file)

                # call function to extract attributes and add them to output_dictionary
                out_dict = extract_attributes_from_mesh(mesh, full_path)
                # if verbose: print("Final attributes:", out_dict)

                # extract attributes into new dictionary
                new_files_dict[filename] = out_dict
                # print("Processed", filename)
    
    # export updated attributes to new csv file
    output = pd.DataFrame.from_dict(new_files_dict, orient='index')
    output.to_csv(f"./attributes/normalized-PSB-attributes.csv")
    print("Normalized attributes successfully exported to csv.")

original_psb_csv = "./attributes/original-PSB-attributes.csv"
out_dir = "./normalized"
normalize_db(database=PSB_PATH, original_csv=original_psb_csv, out_dir=out_dir)

# plot hist to compare distr of num_vertices before and after normalization
before = pd.read_csv(original_psb_csv)
after = pd.read_csv("./attributes/normalized-PSB-attributes.csv")
sns.kdeplot(before['num_vertices'], color='r', shade=True, label='before')
sns.kdeplot(after['num_vertices'], color='g', shade=True, label='after')
plt.title("Number of vertices before and after normalization pipeline")
plt.legend()
plt.show()

# plot hist to compare distr of centroids before and after normalization
sns.kdeplot([len(centroid) for centroid in before['centroid']], color='r', shade=True, label='before')
sns.kdeplot([len(centroid) for centroid in after['centroid']], color='g', shade=True, label='after')
plt.title("Distribution of centroids before and after normalization pipeline")
plt.legend()
plt.show()

# save visualization of meshes before and after normalization
mesh_before = trimesh.load("./psb-labeled-db/FourLeg/397.off")
mesh_after = trimesh.load("./normalized/397.off")
before_after(mesh_before, mesh_after)

# display before and after side by side
img_before = "./pics/before.png"
img_after = "./pics/after.png"
f, axs = plt.subplots(1,2)
axs[0].imshow(img_before.astype('float64'))
axs[1].imshow(img_after.astype('float64'))

import cv2
fig = plt.figure(figsize=(10, 8))
rows, cols = 1, 2
img1 = cv2.imread(img_before)
img2 = cv2.imread(img_after)
fig.add_subplot(rows, cols, 1)
plt.imshow(img1)
plt.axis('off')
plt.title("Before normalization")
fig.add_subplot(rows, cols, 2)
plt.imshow(img2)
plt.axis('off')
plt.title("After normalization")
