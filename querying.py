import PySimpleGUI as sg

# set theme
sg.theme('DarkAmber')

# place here whatever we want in the window
layout = [  [sg.Text('Enter the name of a shape category you would liek to retrieve from:'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Cancel (close window)')] ]

# create the GUI window
window = sg.Window('Multimedia Retrieval System', layout)

# this is the event loop to process "events"
# and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel (close window)': # if user closes window or clicks cancel
        break

    # print to console whatever is input
    print('You entered ', values[0])

window.close()
