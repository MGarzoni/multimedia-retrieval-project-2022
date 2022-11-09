import PySimpleGUI as sg
import os
import trimesh
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageTk
from main_retrieval import *

# set theme
sg.theme('Default')

query_image = None

# place here whatever we want in the window
layout = [  
    [sg.Text('Upload query mesh from disk'), sg.FileBrowse('FileBrowse', file_types=['.ply', '.obj', '.off'], target = '-file-')],
    [sg.InputText(key = '-file-'), sg.Button('Load query')],
    [sg.Text('Number of similar shapes to retrieve:'), sg.Button(1), sg.Button(2), sg.Button(3), sg.Button(4), sg.Button(5)],
    [sg.Button('Cancel (close window)')],
    [sg.Image(key = "-IMAGE-", size = (400,300))]
]

# create the GUI window
window = sg.Window('3D Shape Retrieval System', layout)

# this is the event loop to process "events"
# and get the "values" of the inputs
while True:
    print("Duck")
    event, values = window.read()
    print(event)
    if event == sg.WIN_CLOSED or event == 'Cancel (close window)': # if user closes window or clicks cancel
        break
    
    if event == 'Load query':
        if len(values["-file-"]):
            mesh = trimesh.load(values['-file-'])
            im = mesh_to_PIL_img(mesh)
            im = im.resize((300,300))
            query_image = ImageTk.PhotoImage(image=im)
            window['-IMAGE-'].update(data=query_image)
            
            # if event == 1 or event == 2 or event == 3 or event == 4 or event == 5:
            # run the query and print text output
            print(run_query(values["-file-"], "./features/features.csv")[0]) # here add k argument to add as number of similar shapes to retrieve
            
    # print to console whatever is input
    print('You loaded ', values["-file-"])

window.close()
