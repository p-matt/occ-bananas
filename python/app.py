import os
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from utils import compute, cwd
import plotly.express as px

asset = os.path.join(str(cwd.parent.absolute()), "asset", "web")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ], assets_folder=asset)
app.title = 'One Class Classification for bananas'
server = app.server
port = 8080

app.layout = html.Div(
    [
        html.Nav(
            [
                # dcc.Link([html.Img(src=app.get_asset_url('/img/home.png'))], href="/"),
                html.H1("Bananas Classifier", id="main-title")
            ]),
        html.H2(),
        dcc.Upload(id='upload-image',
                   children=html.Div(
                       [
                           'Drag and Drop or ',
                           html.A('Select Files')
                       ]),
                   multiple=False
                   ),
        html.Div(id='page-content')
    ])


@app.callback(Output('page-content', 'children'),
              Input('upload-image', 'contents'),
              State('page-content', 'children'))
def update_output(file, current_output):
    output = ""
    if file is not None:
        result = compute(file)
        if isinstance(result, str):
            return result
        else:
            X, *res = result
            header = "I think your image contains a banana" if res[0] == 1 else "I don't see any delicious banana here"
            fig = px.imshow(X).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)", "margin": {"l":0, "r":0, "b":0, "t":0}})
            output = [html.Div(
                [
                    html.Div(header, className="card-header"),
                    html.Div(
                        [
                            dcc.Graph(figure=fig)
                        ], className="card-body"),
                ], className="card border-warning mb-3")]
    if current_output:
        output += current_output
    return output


if __name__ == "__main__":
    app.run_server(port=port, host="0.0.0.0", debug=False)
