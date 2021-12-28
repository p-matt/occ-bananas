import os
import sys

import pickle

sys.path.append(os.path.dirname(__file__))

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from utils import cwd, compute, get_fig, headers
from database import get_data, cursor
from random import choice
from numpy import asarray, reshape, uint8

asset = os.path.join(str(cwd.parent.absolute()), "asset", "web")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ], assets_folder=asset)
app.title = 'One Class Classification for bananas'
server = app.server
port = 8080

df = None


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
    print("a")
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


app.layout = html.Div(
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
            html.Div(get_multiple_output(), id="page-content-container")
        ], id='page-content')
    ])


@app.callback(dash.dependencies.Output('page-content-container', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    return get_multiple_output()


@app.callback([Output('page-content-container', 'children'), Output('loading-output-1', 'children')],
              Input('upload-image', 'contents'),
              State('page-content-container', 'children'))
def update_output(file, current_output: list):
    if file is None:
        return current_output, ""

    result = compute(file)
    if isinstance(result, str):
        return current_output, ""
    else:
        X, *res = result
        header = "I think your image contains a banana" if res[0] == 1 else "I don't see any delicious banana here"
        fig = get_fig(X)
        output = get_single_output(header, fig)

    current_output.insert(0, output)
    return current_output, ""


if __name__ == "__main__":
    app.run_server(port=port, host="0.0.0.0", debug=False)
