import PySimpleGUI as sg
from main_retrieval import *
from scaling import ANNIndex
from utils import *

# set theme
sg.theme('SystemDefault')

query_image = None

top_matches = None

last_image = None

windows = list()


def generate_results_window():
    columns = []
    for index, row in top_matches.iterrows():
        # add image into layout
        mesh = trimesh.load(row['path'])
        query_image = mesh_to_buffer(mesh, (220, 220))
        filename = " / ".join((row['path'].split("/")[-2:]))

        columns.append(
            sg.Column([
                [sg.Button(key=f'preview_{index}', image_data=query_image)],
                [sg.Text(filename + "\nDistance: " + str(row["dist"]) + "\n")]]))
    return sg.Window("Results", [columns])


# layout of initial window
layout1 = [
    [sg.Text('Upload query mesh from disk')],
    [sg.Text('File', size=(15, 1)), sg.InputText(key='-file-', enable_events=True),
     sg.FileBrowse('Select', file_types=(('Mesh files', '.off .ply .obj'),), target='-file-')],
    [sg.Text('Preview', size=(15, 1), visible=False, key="Preview"), sg.Image(key='-preview-', visible=False, )],
    [sg.Text('Result count', size=(15, 1)), sg.InputText('5', key='-k-', enable_events=True)],
    [sg.Text('Scalar weight', size=(15, 1)), sg.InputText('0.5', key='-sw-', enable_events=True)],
    [sg.Checkbox('Use ANN (ignores scalar weight)', key='-ann-', default=False)],
    [sg.Button('3D viewer', disabled=True), sg.Button('Query', disabled=True)],
]

# create the first window
main_window = sg.Window('Query image', layout1)

# this is the event loop to process "events"
# and get the "values" of the inputs
while True:
    event, values = main_window.read()
    if event == sg.WIN_CLOSED:
        break

    # if event == 'Query':
    if len(values["-file-"]):
        if last_image is None or last_image != values['-file-']:
            mesh = trimesh.load(values['-file-'])
            query_image = mesh_to_buffer(mesh, (300, 300))
            main_window['Preview'].update(visible=True)
            main_window['-preview-'].update(visible=True)
            main_window['-preview-'].update(query_image)
            last_image = values['-file-']
            main_window['3D viewer'].update(disabled=False)
    else:
        main_window['3D viewer'].update(disabled=True)

    if values['-k-'].isdigit() and int(values['-k-']) > 0 and last_image is not None:
        main_window['Query'].update(disabled=False)  # we have both a file and a k value -- we can query now
    else:
        main_window['Query'].update(disabled=True)

    if event == "3D viewer":
        mesh_ = trimesh.load(values['-file-'])
        try:
            trimesh.load(mesh_).show(viewer="gl")
        except Exception as e:
            print("Failed to load viewer", e)

    if event == "Query":
        if values['-ann-']:
            print("ann")
            idx = ANNIndex(pd.read_csv(FEATURES_CSV))

            norm_mesh, norm_mesh_attributes = normalize_mesh_from_path(values["-file-"])
            query_feats = extract_features(norm_mesh, norm_mesh_attributes, filename=os.path.basename(values["-file-"]),
                                           verbose=True)

            columns = [c for c in query_feats.columns if any(f in c for f in ANNIndex.features)]

            top_matches = idx.query(list(query_feats[columns].iloc[0]), k=int(values['-k-']))
        else:
            # run the query and get matches
            top_matches, _ = run_query(values["-file-"], k=int(values['-k-']),
                                       scalar_weight = float(values['-sw-']),
                                       verbose=True)

        # open second window to display the results
        windows.append(generate_results_window())

    for window in windows:
        event_, _ = window.read()

        if event_ == sg.WIN_CLOSED:
            window.close()
            continue

        if event_.startswith("preview_"):
            index = event_.replace('preview_', '')

            if index.isdigit():
                index = int(index)
                mesh_ = trimesh.load(top_matches[index][0])
                try:
                    trimesh.load(mesh_).show(viewer="gl")
                except Exception as e:
                    print("Failed to load viewer", e)


main_window.close()

for window in windows:
    window.close()