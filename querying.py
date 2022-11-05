import PySimpleGUI as sg
import os
import trimesh

# set theme
sg.theme('Default')

# place here whatever we want in the window
layout = [  [sg.Text('Upload query mesh from disk')], [sg.FileBrowse('FileBrowse', file_types=['.ply', '.obj', '.off'], key='-FILE-')],
            [sg.Button('Cancel (close window)')] ]

# create the GUI window
window = sg.Window('Multimedia Retrieval System', layout)

# this is the event loop to process "events"
# and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel (close window)': # if user closes window or clicks cancel
        break
    
    if event == 'Upload query mesh from disk':
        filename = values["-FILE-"]
        if os.path.exists(filename):
            mesh = trimesh.load(values["-FILE-"])
            mesh.show(viewer='gl')
            
    window.close()

    # print to console whatever is input
    print('You entered ', values[0])

window.close()
