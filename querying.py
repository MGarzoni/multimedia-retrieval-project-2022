import PySimpleGUI as sg
from main_retrieval import *

# set theme
sg.theme('SystemDefault')

query_image = None

top_matches = None

last_image = None


def generate_results_window():
    columns = []
    for index, result in enumerate(top_matches):
        # add image into layout
        mesh = trimesh.load(result[0])
        query_image = mesh_to_buffer(mesh, (100, 100))
        filename = " / ".join((result[0].split("/")[-2:]))

        columns.append(
            sg.Column([
                [sg.Button(key=f'preview_{index}', image_data=query_image)],
                [sg.Text(filename + "\nScalar distance: " + str(result[1]) + 
                         "\nHistogram distance: " + str(result[2]) +"\n")]]))
    return sg.Window("Results", [columns])


def open_results_window():
    # layout of second window
    window2 = generate_results_window()

    while True:
        event_, _ = window2.read()
        if event_ == sg.WIN_CLOSED:
            break

        if event_.startswith("preview_"):
            index = event_.replace('preview_', '')

            if index.isdigit():
                index = int(index)
                mesh_ = trimesh.load(top_matches[index][0])
                try:
                    trimesh.load(mesh_).show(viewer="gl")
                except Exception as e:
                    print("Failed to load viewer", e)

    window2.close()


# layout of initial window
layout1 = [
    [sg.Text('Upload query mesh from disk')],
    [sg.Text('File', size=(15, 1)), sg.InputText(key='-file-', enable_events=True),
     sg.FileBrowse('Select', file_types=(('Mesh files', '.off .ply .obj'),), target='-file-')],
    [sg.Text('Preview', size=(15, 1), visible=False, key="Preview"), sg.Image(key='-preview-', visible=False, )],
    [sg.Text('Result count', size=(15, 1)), sg.InputText('5', key='-k-', enable_events=True)],
    [sg.Button('3D viewer', disabled=True), sg.Button('Query', disabled=True)],
]

# create the first window
window = sg.Window('Query image', layout1)

# this is the event loop to process "events"
# and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    # if event == 'Query':
    if len(values["-file-"]):
        if last_image is None or last_image != values['-file-']:
            mesh = trimesh.load(values['-file-'])
            query_image = mesh_to_buffer(mesh, (300, 300))
            window['Preview'].update(visible=True)
            window['-preview-'].update(visible=True)
            window['-preview-'].update(query_image)
            last_image = values['-file-']
            window['3D viewer'].update(disabled=False)
    else:
        window['3D viewer'].update(disabled=True)
            
            
    if values['-k-'].isdigit() and int(values['-k-']) > 0 and last_image is not None:
        window['Query'].update(disabled=False) # we have both a file and a k value -- we can query now
    else:
        window['Query'].update(disabled=True)
        
    if event == "3D viewer":
        mesh_ = trimesh.load(values['-file-'])
        try:
            trimesh.load(mesh_).show(viewer="gl")
        except Exception as e:
            print("Failed to load viewer", e)

    if event == "Query":
        # run the query and get matches
        top_matches, _ = run_query(values["-file-"], "./features/features.csv", int(values['-k-']))

        # open second window to display the results
        open_results_window()

window.close()

#####
