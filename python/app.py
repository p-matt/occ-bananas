import os
import sys

import pickle

sys.path.append(os.path.dirname(__file__))

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from utils import cwd, compute, get_fig, headers
from database import get_data
from random import choice

asset = os.path.join(str(cwd.parent.absolute()), "asset", "web")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ], assets_folder=asset)
app.title = 'One Class Classification for bananas'
server = app.server
port = 8080

df = None

index = 0


def get_single_output(header, fig):
    return html.Div(
        [
            html.Div(header, className="card-header"),
            html.Div(
                [
                    dcc.Graph(figure=fig)
                ], className="card-body"),
        ], className="card border-warning mb-3")


def get_multiple_output():
    global df
    df = get_data()
    outputs = []
    for i, row in df.iterrows():
        header = choice(headers[str(row["labels"])])
        is_correct = row["correct"]
        img = pickle.loads(row["imgs"])
        fig = get_fig(img)
        output = get_single_output(header, fig)
        outputs.append(output)
    return outputs


def get_app_layout():
    return html.Div(
        [
            dcc.Location(id='url', refresh=False),
            html.Nav(
                [
                    html.H1("Bananas Classifier", id="main-title")
                ]),
            html.Div(
                [dcc.Loading(id="loading-1", type="default", color="orange", children=html.Div(id="loading-output-1"))],
                id="container"),

            dcc.Upload(id='upload-image',
                       children=html.Div(
                           [
                               'Drag and Drop or ',
                               html.A('Select Files')
                           ]),
                       multiple=False
                       ),

            html.Div([
                html.Div(get_multiple_output(), id="page-content-online")
            ], id='page-content')
        ])


@app.callback([Output('page-content-online', 'children'), Output('loading-output-1', 'children')],
              Input('upload-image', 'contents'),
              State('page-content-online', 'children'))
def update_output(file, current_output: list):
    global index
    if file is None:
        return current_output, ""

    result = compute(file)
    if isinstance(result, str):
        return current_output, ""
    else:
        X, pred = result
        header = choice(headers[str(pred)])
        fig = get_fig(X)
        output = get_single_output(header, fig)

    current_output.insert(0, output)
    index += 1
    app.layout = get_app_layout() if index % 4 == 0 else app.layout
    return current_output, ""


app.layout = get_app_layout()
if __name__ == "__main__":
    app.run_server(port=port, host="0.0.0.0", debug=False)
