import PySimpleGUI as sg

# set theme
sg.theme('Default')

# place here whatever we want in the window
layout = [  [sg.Text('Enter the name of a shape category you would like to retrieve from:'), sg.InputText()],
            [sg.Button('Upload query mesh from disk'), sg.Button('Cancel (close window)')] ]

# create the GUI window
window = sg.Window('Multimedia Retrieval System', layout)

# this is the event loop to process "events"
# and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel (close window)': # if user closes window or clicks cancel
        break

    if event == 'Upload query mesh from disk':
        # load and display mesh
        pass

    # print to console whatever is input
    print('You entered ', values[0])

window.close()
