'''
1. read each shape
2. translate it so that its barycenter coincides with the coordinate-frame origin
3. scale it uniformly so that it tightly fits in a unit-sized cube
4. run stats to show that the normalization worked OK for all shapes
'''

# imports
import trimesh

# read shape
sample = "./test-data-db/airplane/61.off"
original_mesh = trimesh.load(sample)
original_mesh.show()

# translation
translated_mesh = original_mesh.apply_translation((0,0,0)) # not sure if correct
translated_mesh.show()

# scaling
scaled_mesh = translated_mesh.apply_scale((1,1,1)) # not sure if correct
scaled_mesh.show()

# check if things actually changed (nothing did - YET)
# here comparing any of the attributes of the meshes will do
print(translated_mesh.centroid == scaled_mesh.centroid)

ois = original_mesh.__dict__.items()
tis = translated_mesh.__dict__.items()
sis = scaled_mesh.__dict__.items()
for (oa,ov), (ta,tv), (sa,sv) in zip(ois, tis, sis):

    print(f"checking {ov} change between original and translated mesh")
    if oa == ta:
        print(ov == tv)

    print(f"\nchecking {tv} change between translated and scaled mesh")
    if ta == sa:
        print(tv == sv)

