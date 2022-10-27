from utils import *
from reporting import ShapeReport as report
import open3d

def normalize_db(database, original_csv, out_dir, out_csv, verbose=False):
    """This function takes a database, iteratively gets a shape from a category and
    applies normalization steps to the mesh; if the shape is an outlier, the remeshing via 
    mesh.subdivide() is triggered; the normalized mesh is then exported as a new .off file and 
    then we call the extract_attributes_from_mesh util function and export the new csv holding the 
    normalized attributes.
    NOTE: The original csv may not be up to date about attributes and features. This can be updated separately with update_csv()"""

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
    
                    # load mesh from path
                    mesh = trimesh.load(full_path)
    
                    # if mesh has less than 3500 vertices (IS_OUT_LOW), then subdivide and remesh
                    if shape_attributes['is_out_low']:
                        print(f"\n{shape_attributes['filename']} is an outlier because it has less than 3500 vertices")
                        print("# vertices before refinement:", len(mesh.vertices))

                        # while the # vertices is lower than 3500
                        # use mesh.subdivide() for resampling
                        while len(mesh.vertices) <= IS_OUT_LOW:
                            mesh = mesh.subdivide()
                            print("# vertices after subdivide:", len(mesh.vertices))

                    # if mesh has more than 17500 vertices (IS_OUT_HIGH), then remove vertices and remesh
                    if shape_attributes['is_out_high']:
                        print(f"\n{shape_attributes['filename']} is an outlier because it has more than 17500 vertices")
                        print("# vertices before refinement:", len(mesh.vertices))

                        # while the # vertices is higher than 17500
                        # use open3d for decreasing # vertices
                        while len(mesh.vertices) >= IS_OUT_HIGH:
                            # do decimation here
                            print("# vertices after open3d decimation:", len(mesh.vertices))
                    
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

# set paths
original_psb_csv = "./attributes/original-PSB-attributes.csv"
out_dir = "./normalized-psb-db"
out_csv = "./attributes/normalized-PSB-attributes.csv"

NORMALIZE = True # set to true to re-do normalization
UPDATE_CSV = True # set to true to re-update attributes CSV files for both before and after normalization

if NORMALIZE:
    normalize_db(database=PSB_PATH, original_csv=original_psb_csv, out_dir=out_dir, out_csv=out_csv)

if UPDATE_CSV:
    # update csv's if necessary
    update_csv(PSB_PATH, original_psb_csv, flat_dir=False)
    update_csv(out_dir, out_csv, flat_dir = True)

# read in attributes before and after normalization
before = pd.read_csv(original_psb_csv)
after = pd.read_csv(out_csv)

# generate histogram report objects
before_rep = report(before)
after_rep = report(after, given_ranges=before_rep.ranges)

# export histograms into folders
before_rep.save("before_hists")
after_rep.save("after_hists")
