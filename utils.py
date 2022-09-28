#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:55:41 2022

@author: eduardsaakashvili
"""

import trimesh
import os

#save mesh object as png (file name should not include .png)
def save_mesh_png(mesh, filename, camera_fov = None):
    scene = mesh.scene()
    scene.Camera= trimesh.scene.Camera(fov=(camera_fov))
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