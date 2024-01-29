# AFM Nanomechanics
# programmed by Ken Nakajima
# 2022-09-08

ver = 0.1

import base64
import io

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px

dash_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=dash_stylesheets)
app.config.suppress_callback_exceptions = True

tab_selected_style = {
    "backgroundColor": "black",
    "color": "white",
    "fontWeight": "bold",
}

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2(f"AFM Nanomecanics v.{ver}")
            ]
        ),
        # Uploadコンポーネントを配置
        dcc.Upload(
            id="upload-data",
            children=html.Button('Upload AFM Data File'),
            style={
                "margin": "2% auto",
            },
        ),
        # 表示を切り替えるタブ
        dcc.Tabs(
            id="top_tab",
            value="one",   # デフォルトタブの選択
            children=[
                # ヘッダーを表示するタブ
                dcc.Tab(
                    label="Header",
                    value="one",
                    selected_style=tab_selected_style,
                    children=[html.Div(id="upload-header")],
                ),
                # ドロップダウンとグラフを表示するタブ
                dcc.Tab(
                    label="Main",
                    value="two",
                    selected_style=tab_selected_style,
                    children=[
                        html.Div(
                            [
                                dcc.Dropdown(id="table-dropdown", multi=True),
                                html.Div(id="update_graph")
                            ]
                        )
                    ],                      
                ),
                # ドロップダウンとグラフを表示するタブ
                dcc.Tab(
                    label="Supplement",
                    value="three",
                    selected_style=tab_selected_style,
                    children=[
                        html.Div(
                            [
                                dcc.Dropdown(id="table-dropdown2", multi=True),
                                html.Div(id="update_graph2")
                            ]
                        )
                    ],                      
                ),
            ],
        ),
    ]
)

# テーブルを作成するためのコールバック
@app.callback(
    Output("upload-header", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_contents(contents, filename):
    if contents:
        print(filename)
        with open(filename, "rb") as f:
            print(contents[:50])
            content_type, content_string = contents.split(",") # ","で分ける
            print(content_type)
            decoded = base64.b64decode(content_string) # base64でデコード
            print(decoded[:500])

#        print(contents[:50000])
#        content_type, content_string = contents.split(",") # ","で分ける
#        print(content_type) # 「data:text/csv;base64」が得られる
#        decoded = base64.b64decode(content_string) # base64でデコード
#        print(decoded[:50000])
#        print(decoded.decode("shift-jis")[:300]) # さらにそれをutf-8と見なしてデコード
        # csvファイルとエクセルファイルのデータ読み込みの処理
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            elif filename.endswith(".xls") or filename.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div(["ファイルの処理中にエラーが発生しました"])
#        print(df.columns)
        # ファイル名をタイトル、読み込んだデータのテーブルを返り値とする
        return (
            html.H2(filename),
            dash_table.DataTable(
                id="header",
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
            ),
        )

# グラフタブのドロップダウン作成用コールバック
@app.callback(
    Output("table-dropdown", "options"),
    Output("table-dropdown", "value"),
    Input("header", "columns"),
    Input("header", "derived_virtual_data"),
)
def update_dropdown(columns, rows):
    # テーブルのデータをデータフレームとする
    dff = pd.DataFrame(rows, columns=[c["name"] for c in columns])
    # 国名列にデータがあればドロップダウンの選択肢を作成し、初期値とともに返り値とする
    try:
        options = [{"value": i, "label": i} for i in dff["国名"].unique()]
        return options, [dff["国名"].unique()[0]]
    # 上記の処理ができない場合、コールバックを更新しない
    except:
        raise dash.exceptions.PreventUpdate

# ドロップダウンの選択を受けてグラフを作成するコールバック
@app.callback(
    Output("update_graph", "children"),
    Input("header", "columns"),
    Input("header", "derived_virtual_data"),
    Input("table-dropdown", "value"),
)
def update_graph(columns, rows, selected_countries):
    # テーブルのデータをデータフレームとする
    dff = pd.DataFrame(rows, columns=[c["name"] for c in columns])
    # ドロップダウンで国名が選択されていればグラフを作成する
    if not selected_countries:
        raise dash.exceptions.PreventUpdate
    
    dff["date"] = pd.to_datetime(dff["date"])
    dff = dff[dff["国名"].isin(selected_countries)]
    return dcc.Graph(figure=px.line(dff, x="date", y="value", color="国名"))

if __name__=="__main__":
    app.run_server(debug=True)