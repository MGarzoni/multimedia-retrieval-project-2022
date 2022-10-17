# imports
from utils import *
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt

from reporting import ShapeReport as report

def normalize_db(database, original_csv, out_dir, out_csv, verbose=False):
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
                
                if shape.endswith(('.ply', '.obj', '.off')):

                    full_path = os.path.join(root, category, shape)
                    # print('\n', full_path)
                    filename = os.path.basename(full_path)
                    # print(filename)
    
                    shape_attributes = attributes_dict[filename]
                    # print(shape_attributes)
                    
                    '''Below remeshing thru command line is throwing errors for files not in .off format;
                    below I will try to use mesh.subdivide()'''
    
                    # load mesh from path
                    mesh = trimesh.load(full_path)
    
                    # if it's an OUTLIER, remesh until we reach REMESH_THRESHOLD (parameter set in utils)
                    if shape_attributes['is_out']:
                        print(f"\n{shape_attributes['filename']} is an outlier!")
    
                        # trying to use mesh.subdivide() for resampling
                        print("# vertices:", len(mesh.vertices))
                        
                        while len(mesh.vertices) < REMESH_THRESHOLD:
                            mesh = mesh.subdivide()
                            print("# vertices after subdivide:", len(mesh.vertices))
                    
                    # print initial attributes csv
                    # if verbose: print("Initial attributes:", shape_attributes)
    
                    # save original as png (this is already AFTER remeshing it if it is an outlier)
                    # if verbose: save_mesh_png(mesh, "1-original", corners = CORNERS)
    
                    '''From here actual normalization begins'''
    
                    # translate mesh to origin (with center_at_origin function from utils) (here bounds value changes)
                    mesh = center_at_origin(mesh)
                    # if verbose: save_mesh_png(mesh, "2-translated", corners = CORNERS)
    
    
                    # align pca: x axis is most variance, z axis is least variance
                    mesh = pca_align(mesh)
                    # if verbose: save_mesh_png(mesh, "4-pca", corners = CORNERS)
                    
                    # moment test
                    mesh = moment_flip(mesh)
                    # if verbose: save_mesh_png(mesh, "5-moment", corners = CORNERS)
                    
                    # scale to cube vector (with scale_to_unit function from utils) (here bounds value changes)
                    mesh = scale_to_unit(mesh)
                    # if verbose: save_mesh_png(mesh, "3-scaled", corners = CORNERS)
    
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
                    print("Processed", filename)
    
    # export updated attributes to new csv file
    output = pd.DataFrame.from_dict(new_files_dict, orient='index')
    output.to_csv(out_csv)
    print("Normalized attributes successfully exported to csv.")

original_psb_csv = "./attributes/original-PSB-attributes.csv"
out_dir = "./normalized"
out_csv = "./attributes/normalized-PSB-attributes.csv"

# normalize_db(database=PSB_PATH, original_csv=original_psb_csv, out_dir = out_dir, out_csv = out_csv)


'''
To check if normalization procedure was carried out correctly, check:
    - number of sampling points (vertices)
    - size of bounding box (computed via its diagonal)
    - position of bounding box (distance of its center to the origin)
    - pose (absolute value of cosine of angle between major eigenvector and, say, the X axis)
'''

# read in attributes before and after and print their descriptive stats
before = pd.read_csv(original_psb_csv)
after = pd.read_csv(out_csv)
# print(f"Summary of attributes BEFORE normalization:\n{before[['axis_aligned_bounding_box', 'centroid', 'area']].describe(include='all')}")
# print(f"\nSummary of attributes AFTER normalization:\n{after[['axis_aligned_bounding_box', 'centroid', 'area']].describe(include='all')}")


# update csv's if necessary
update_csv(PSB_PATH, original_psb_csv, flat_dir=False)
update_csv(out_dir, out_csv, flat_dir = True)

#histograms, before and after
# before_after_hist("./attributes/original-PSB-attributes.csv", "./attributes/normalized-PSB-attributes.csv",
#                   attributes = ["area", "num_vertices", "boundingbox_distance", "centroid_to_origin", "boundingbox_diagonal"])


# read in attributes before and after and print their descriptive stats
before = pd.read_csv(original_psb_csv)
after = pd.read_csv(out_csv)
# print(f"Summary of attributes BEFORE normalization:\n{before[['axis_aligned_bounding_box', 'centroid', 'area']].describe(include='all')}")
# print(f"\nSummary of attributes AFTER normalization:\n{after[['axis_aligned_bounding_box', 'centroid', 'area']].describe(include='all')}")



before_rep = report(before)
after_rep = report(after, given_ranges=before_rep.ranges)

before_rep.save("before")
after_rep.save("after")








# deprecated code:

# plot hist to compare size of bounding box before and after normalization


# plot hist to compare position of bounding box (distance of its center to the origin) before and after normalization


# # save visualization of meshes before and after normalization
# mesh_before = trimesh.load("./psb-labeled-db/FourLeg/397.off")
# mesh_after = trimesh.load("./normalized/397.off")
# before_after(mesh_before, mesh_after)

# # display before and after side by side
# import cv2
# img_before = "./pics/before.png"
# img_after = "./pics/after.png"
# fig = plt.figure(figsize=(10, 8))
# img1 = cv2.imread(img_before)
# img2 = cv2.imread(img_after)
# fig.add_subplot(1, 2, 1)
# plt.imshow(img1)
# plt.axis('off')
# plt.title("Before normalization")
# fig.add_subplot(1, 2, 2)
# plt.imshow(img2)
# plt.axis('off')
# plt.title("After normalization")

'''
What does a good normalization do? It reduces the variance of Q.
Hence, to check if your normalization works OK, you can compute an indication of Q's variance before and then after the normalization, and compare them.
If variance drops significantly, the normalization worked properly.
You can estimate this variance in several ways. From more aggregated (less informative) to less aggregated (more informative):
    compute the average and standard deviation of Q: The standard deviation should drop after normalization. The average indicates the value around which normalization brings the shapes.
    compute a histogram of Q over a fixed number of bins: Normalization should make the histogram more pointy, that is, having (ideally) a single large peak and being nearly zero away from that peak. 
'''
