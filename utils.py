#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:55:41 2022

@author: eduardsaakashvili
"""

import trimesh
import os
import math

#save mesh object as png (file name should not include .png)
def save_mesh_png(mesh, filename, corners = None):
    
    #code following save image example from trimesh documentation
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    if corners is None: #if no corners given
        corners = scene.bounds_corners
    print("Corners", filename, corners)
    r_e = trimesh.transformations.euler_matrix(
        math.radians(45),
        math.radians(45),
        math.radians(45),
        "ryxz",
        )
    
    t_r = scene.camera.look_at(corners, rotation=r_e)
    
    scene.camera_transform = t_r
    
    
    #scene.Camera= trimesh.scene.Camera(fov=(camera_fov))
    png = scene.save_image()
    
    with open(filename+".png", 'wb') as f:
        f.write(png)
        f.close()
    

# save shape at path to .png, with tag added after underscore
def save_image_of_path(path, tag=None):
    
    mesh = trimesh.load(path)
    
    # generate filename
    file_name = os.path.basename(path)
    
    if tag != None: # add tag to filename if there is one
        file_name = file_name + "_" + tag
    
    save_mesh_png(mesh, file_name)