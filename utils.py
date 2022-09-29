#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:55:41 2022

@author: eduardsaakashvili
"""

import trimesh
import os
import math


def center_at_origin(mesh):
    """takes a trimesh object.
    returns a new mesh that has been translated so barrycenter is at origin"""
    translated_mesh = mesh.copy()
    translated_mesh.vertices = mesh.vertices - mesh.centroid
    return translated_mesh
    
    

#save mesh object as png (file name should not include .png)
def save_mesh_png(mesh, filename, corners = None):
    
    #code following save image example from trimesh documentation
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(trimesh.creation.axis(axis_length = 1)) #add axis!
    if corners is None: #if no corners given
        corners = scene.bounds_corners
    #print("Corners", filename, corners)
    
    #set 45 degree view so all axes are visible
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
    
    with open("./pics/"+filename+".png", 'wb') as f:
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
    
    
#save "before.png" and "after.png" with two meshes. camera bounding box set by SECOND image
def before_after(mesh1, mesh2, corners = None):
    if corners is None:
        corners = mesh2.scene().bounds_corners 
    save_mesh_png(mesh1, "before", corners = corners)
    save_mesh_png(mesh2, "after", corners = corners)