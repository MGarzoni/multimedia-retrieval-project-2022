import PySimpleGUI as sg
import os
import trimesh
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageTk
from main_retrieval import *
import os

# set theme
sg.theme('SystemDefault')

image_loaded = False

query_image = None

top_matches = None

result_images = []


def generate_results_window():
    results_layout = []
    for index, result in enumerate(top_matches):
        
        # add image into layout
        mesh = trimesh.load(result[0])
        query_image = mesh_to_ImageTk(mesh, (100,100))
        
        img = ImageTk.getimage( query_image )
        img.save(f"./pics/result_{index}.png")
        
        result_images.append(query_image)
        
        results_layout.append([sg.Image(filename = f"./pics/result_{index}.png")])
        
        # add file path and distance into layout
        filename = " / ".join((result[0].split("/")[-2:]))
        results_layout.append([sg.Text(filename + 
                                       "\nDistance: "
                                       +str(result[1])+"\n")])

    return sg.Window("Results", results_layout)


def open_results_window():
    # layout of second window
    window2 = generate_results_window()    
    
    while True:
        event, values = window2.read()
        if event == sg.WIN_CLOSED:
            break
        
    window2.close()

# layout of initial window
layout1 = [  
        [sg.Text('Upload query mesh from disk')], 
          [sg.FileBrowse('FileBrowse', file_types=(('Mesh files', '.off .ply .obj'),), target='-file-')],
          [sg.InputText(key='-file-')],
            [sg.Button('Load query'), sg.Button('Cancel (close window)')],
            [sg.Image(key = "-IMAGE-", size = (300,300))],
            [sg.Button('Open 3D viewer'), sg.Button('Find matches')]
            
        ]

# create the first window
window = sg.Window('MultiPlayer', layout1)

# this is the event loop to process "events"
# and get the "values" of the inputs
while True:
    event, values = window.read()
    if (event == sg.WIN_CLOSED or event == 'Cancel (close window)'): # if user closes window or clicks cancel
        break
    
    if event == 'Load query':
        if len(values["-file-"]):
            mesh = trimesh.load(values['-file-'])
            query_image = mesh_to_ImageTk(mesh, (300,300))
            window['-IMAGE-'].update(data=query_image)
            image_loaded = True
          
    if event == "Find matches" and image_loaded:
        # run the query and get matches
        top_matches, _ = run_query(values["-file-"], "./features/features.csv")
                    
        # open second window to display the results
        open_results_window()

    if event == "Open 3D viewer" and image_loaded:
        try:
            trimesh.load(values['-file-']).show(viewer="gl")
        except Exception as e:
            print("Failed to load viewer", e)

window.close()

#####

