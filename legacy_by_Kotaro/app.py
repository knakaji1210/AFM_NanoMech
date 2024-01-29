import plotly.express as px
import json
from dash_extensions.enrich import Output, Input, html, ALL,State,DashProxy,dash,dcc,MATCH,NoOutputTransform
import numpy as np
import plotly.graph_objs as go
import dash_colorscales
from dash.exceptions import PreventUpdate
import re
import dash_bootstrap_components as dbc

app = DashProxy(__name__,use_pages = True,external_stylesheets=[dbc.themes.DARKLY],transforms=[NoOutputTransform()],compress = True)
app.layout = html.Div(
    [
                html.Div(
                    [
                        html.Div(
                            dcc.Link(
                                f"{page['name']} - {page['path']}", href=page["relative_path"]
                            )
                        )
                        for page in dash.page_registry.values()

                    ],

                ),

	dash.page_container
                ])


if __name__ == "__main__":
    app.run_server(debug=None,threaded=True)