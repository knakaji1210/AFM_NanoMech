import plotly.express as px
import json
from dash_extensions.enrich import Output, Input, html, ALL,State,dash,dcc,MATCH,ServersideOutput,callback
import numpy as np
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
import re
from numba import jit, prange
import dash_bootstrap_components as dbc
import numba1
import time
import global_value

dash.register_page(__name__, path='/')
# with open(r"C:\Users\osako\Desktop\TKB3_300nmThickum.028.pfc", "rb") as f:
with open ("filename","rb") as f:
    headerdata=f.read (40960)
    #2byte=1pixelのため、16bitずつ読み込む。
    rectype1 = np.dtype(np.int16)
    # ヘッダーforceデータ　dataoffcet,datalength=count
    seek = f.seek(172032)
    data = np.fromfile(f, dtype = rectype1, count = 16777216)
    # ヘッダーimageデータ（originalheight)
    seek2 = f.seek(40960)
    data2 = np.fromfile(f, dtype = rectype1, count = 65536)
header_decode = headerdata.decode(encoding='cp932')
pattern1= r'Sens. Zsens: V (.*).n'
pattern2=r'@2:Z scale: V \[Sens. Zsens\] \((.*).V/LSB\) (.*).V'
pattern3=r'Peak Force Amplitude: (.*)'
pattern4=r'@Sens. DeflSens: V (.*).nm/V'
pattern5=r'@2:TMDeflectionLimitLockIn3LSADC1: V \((.*).V/LSB\) (.*).V'
pattern6=r'Scan Size: (.*).nm'
pattern7=r'Number of lines: (.*).'
pattern8=r'Spring Constant: (.*).'
pattern9=r'Sync Distance New: (.*).'
pattern10=r'Sample Points: (.*).'
pattern11=r'force/line: (.*).0'
pattern12=r'PFT Freq: (.*).KHz'
zSens_nm_V=float(re.findall(pattern1, header_decode)[0])
zRange_V=float(re.findall(pattern2, header_decode)[0][1])
amplitude=float(re.findall(pattern3,header_decode)[0])*(10**-9)
deflSens_nm_V=float(re.findall(pattern4,header_decode)[0])
deflRange_V=float(re.findall(pattern5,header_decode)[0][1])
sync_distance=float(re.findall(pattern9,header_decode)[0])
scanpoint=int(re.findall(pattern7,header_decode)[0])
samplepoint=int(re.findall(pattern10,header_decode)[0])
forcepoint=int(re.findall(pattern11,header_decode)[0])
pft_freq=float(re.findall(pattern12,header_decode)[0])*(10**3)
amplifier = (deflRange_V*deflSens_nm_V*(10**(-9)))/65536
springconstant=float(re.findall(pattern8,header_decode)[0])


originalheight=np.array(data2)
originalheight = (((originalheight/65536)*zRange_V*zSens_nm_V)/(10**9))
originalheight = np.reshape(originalheight,(256,256))
originalheight_max=np.nanmax(originalheight)
originalheight_min=np.nanmin(originalheight)
defl = np.array_split(data, 131072)
defl=np.fliplr(np.array(defl)*amplifier)
#欠損値検出
defl=np.where (defl<=-32767*amplifier,np.nan,defl)
defl_128=defl[::2]
defl_256=np.fliplr(defl[1::2])
concate_defl=np.reshape(np.concatenate([defl_128,defl_256],axis = 1),(scanpoint,scanpoint,forcepoint))
sync_distance=sync_distance*(forcepoint/samplepoint)
#igor　AFM4_plot.ipfに記載あり
ramp=amplitude*(np.cos(2*np.pi*((np.arange(forcepoint)-sync_distance)/forcepoint))+1)
#igor AFM Nanoscope.ipfに記載あり　SetScale x 0, (1 / pftFreq), "sec", inw4_plot
# time=np.linspace(0,1/pft_freq,forcepoint,endpoint = False)
sync_distance_plus1=int(sync_distance)+1
# extime=time[:sync_distance_plus1]
# rttime=time[sync_distance_plus1:]
global_value.exramp=ramp[:sync_distance_plus1]
global_value.rtramp=ramp[sync_distance_plus1:]
global_value.exdefl=concate_defl[:,:,:sync_distance_plus1]
global_value.rtdefl=concate_defl[:,:,sync_distance_plus1:]
exdeflection=global_value.exdefl.reshape(65536,-1)
rtdeflection=global_value.rtdefl.reshape(65536,-1)
exdeflection=exdeflection.astype(np.float32)
rtdeflection=rtdeflection.astype(np.float32)
def deflmin(start,end,k):
    selectvalue=exdeflection[:,start:end]
    baseline=np.nanmean (selectvalue,axis = 1)
    baseline=baseline[:,np.newaxis]
    exbase=exdeflection-baseline
    rtbase=rtdeflection-baseline
    global_value.rtbase=rtbase.reshape(256,256,-1)
    minimum=np.nanmin (exbase,axis = 1)
    minimum=minimum[:,np.newaxis]
    minimumindex=np.nanargmin (exbase,axis = 1)
    rampmini=global_value.exramp[minimumindex]
    rampmini=rampmini[:,np.newaxis]
    global exdelta_graph
    global rtdelta_graph
    global exforce_graph
    global rtforce_graph
    exdelta=(global_value.exramp-rampmini)-(exbase-minimum)
    rtdelta=np.fliplr((global_value.rtramp-rampmini)-(rtbase-minimum))
    exforce=np.array (springconstant*exbase)
    rtforce=np.fliplr(np.array (k*rtbase))
    exdelta_graph=np.array(exdelta).reshape(256,256,-1)
    rtdelta_graph=np.array(rtdelta).reshape (256,256,-1)
    exforce_graph=np.array(exforce).reshape (256,256,-1)
    rtforce_graph=np.array(rtforce).reshape (256,256,-1)
    return exdelta,exforce,rtdelta,rtforce

def jkr(R,DA,FA,DB,p):
    global modulus_graph
    global adhesive_graph
    K=(-1)*(((1+(16**(1/3)))/3)**(3/2))*FA/((R*(DB-DA)**3)**(1/2))
    # K=(-1)*(((1+(16**(1/3)))/3)**(3/2))*FA*((DB-DA)**(-1.5))/(R**(1/2))
    modulus=(3/4)*(1-(p**2))*K
    adhesive=-(2*FA/(3*np.pi*R))
    # np.save('modulus',module)
    # np.save('adhesive',adhesive)
    global_value.modulus=modulus.reshape(256,256)
    global_value.adhesive=adhesive.reshape(256,256)
    return modulus,adhesive


def fitting(Maugis_delta,rangedelta,poisson,w_list,R):
    modulus_list=[]
    for i in range (65536):
        rangedelta_i=rangedelta[i]
        Maugis_delta_i=Maugis_delta[i]
        w_list_i=w_list[i]
        a,b=np.polyfit(rangedelta_i,Maugis_delta_i,1)
        K=((a**3)*(np.pi**2)*(w_list_i**2)*R)**(1/2)
        modulus=(3/4)*(1-(poisson**2))*K
        modulus_list.append(modulus)
    modulus_list=np.array(modulus_list).reshape(256,256)
    w_list=np.array(w_list).reshape(256,256)
    global modulus_graph
    global adhesive_graph
    global_value.modulus=modulus_list.copy()
    global_value.adhesive=w_list.copy()
    return modulus,w_list






layout1=go.Layout (xaxis = dict (title = 'piezoscanner displacement[nm]',
                                showexponent = 'none'),
                   yaxis = dict (title = 'deflection[nm]',
                                  showexponent = 'none'),
                    template = "plotly_dark",plot_bgcolor = 'rgba(0, 0, 0, 0)',
                    paper_bgcolor = 'rgba(0, 0, 0, 0)',uirevision = 'DropDown Live Charts'
                    )
layout2=go.Layout (xaxis = dict (title = 'deformation[nm]',
                                showexponent = 'none'),
                   yaxis = dict (title = 'force[nN]',
                                  showexponent = 'none'),
                    template = "plotly_dark",plot_bgcolor = 'rgba(0, 0, 0, 0)',
                    paper_bgcolor = 'rgba(0, 0, 0, 0)',uirevision = 'DropDown Live Charts'
                    )


colorscales = px.colors.named_colorscales()


heatmap1=np.arange(65536).reshape(256,256)
heatmap2=np.random.rand(65536).reshape(256,256)
height=go.Figure()
height.add_trace(go.Heatmapgl(z=originalheight))

# fig2.add_trace(go.Heatmapgl(z=heatmap2))
height.update_layout (yaxis = dict (scaleanchor = 'x'),
                   plot_bgcolor='rgba(0,0,0,0)',
                   margin = {'l':30,'t':15,'r':10,'b':30},
                   font=dict(size=10),template="plotly_dark")

# fig2.update_layout (yaxis = dict (scaleanchor = 'x'),
#                    width=280,
#                    height=220,
#                    plot_bgcolor='rgba(0,0,0,0)',
#                    margin = {'l':0,'t':0,'r':0,'b':0},
#                    font=dict(size=10))



def create_figure():
    return height

card2 = dbc.CardBody(
            [
                html.H4("Method", className="card-title"),
                dbc.RadioItems(
                            options=[
                                {"label": "twopoint", "value": 1},
                                {"label": "linear", "value": 2},
                            ],
                            value=1,
                            id="method-input",
                        ),
                html.Div([
                    dbc.Input(type="number",id="fit_start",placeholder = "start",value = 0.2,
                              style={"width": 120, "height": 30, "display":"inline-block"}),
                    dbc.Input(type="number",id="fit_end",placeholder = "end",value = 0.7,
                              style={"width": 120, "height": 30, "display":"inline-block"})]),
                dbc.RadioItems (
                    options = [
                        {"label":"JKR","value":1},
                        {"label":"DMT","value":2},
                    ],
                    value = 1,
                    id = "model-input",
                ),


                ])



card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("parameter", className="card-title"),
                html.Div([
                    html.Div([
                        html.P("tip radius",style = {"marginBottom":0}),
                        dbc.Input(type="number",id="tip radius",placeholder = "tip radius",value = '{:g}'.format(10),
                                  style={"width": 120, "height": 30, "display":"inline-block"})],style = {"display":"inline-block"}),
                    html.Div([
                        html.P("spring constant",style = {"marginBottom":0}),
                        dbc.Input(type="number",id="spring constant",placeholder = "spring constant",value = 1,
                                  style={"width": 120, "height": 30, "display":"inline-block"})],style = {"display":"inline-block"}),
                    html.Div ([
                        html.P ("start",style = {"marginBottom":0}),
                        dbc.Input(type="number",id="start",placeholder = "start",value = 1,
                                  style={"width": 120, "height": 30, "display":"inline-block"})],style = {"display":"inline-block"}),
                    html.Div ([
                        html.P ("end",style = {"marginBottom":0}),
                        dbc.Input(type="number",id="end",placeholder = "end",value = 3,
                                  style={"width": 120, "height": 30, "display":"inline-block"})],style = {"display":"inline-block"}),
                    html.Div ([
                        html.P ("poisson's ratio",style = {"marginBottom":0}),
                        dbc.Input(type="number",id="poisson",placeholder = "poisson's ratio ",value = 0.5,
                                  style={"width": 120, "height": 30, "display":"inline-block"})],style = {"display":"inline-block"})]),
                card2,
                dbc.Button("Go somewhere",id="analyze", color="primary"),
            ]
        ),
    ],
    style={"width": "30rem"},
)



sidebar = html.Div(
    [
        # dbc.Row(
        #     [
        #         html.P('Settings'),
        #
        #         ],
        #     style={"height": "30vh"},
        #     ),
        dbc.Row(
            [
                card,
                ],
            style={"height": "100vh"}
            )
        ]
    )


content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Checkbox(
                            id="standalone-checkbox",
                            label="This is a checkbox",
                            value=0
                        ),
                        html.Div(style={"width": "60rem","height": "48rem"},
                                 children=dcc.Graph(id='curve',style = {"width": "40rem","height": "28rem"}),
                                 ),
                        ],width=10),
                dbc.Col([],width=8)

            ],style={"height": "30rem"}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P('Correlation Matrix Heatmap'),
                        html.Div ([
                            html.Div (
                                # style = {
                                #     "width":300,
                                #     "height":330,
                                #     "display":"inline-block",
                                #     "outline":"thin lightgrey solid",
                                #     "padding":10,
                                # }
                            ),
                            dbc.Button (
                                "Add Chart",
                                id = "add-chart",
                                n_clicks = 0,
                                style = {"display":"inline-block"}),
                            dcc.Store (id = 'save_data'),
                            html.Div (id = "container",className = "container",children = [])])

            ])
            ],
            style={"height": "70vh"}
            )
        ]
    )

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3),
                dbc.Col(content, width=9)
                ]
            ),
        ],
    fluid=True
    )


@callback(
    Output("container", "children"),
    Input("add-chart", "n_clicks"),
    Input({"type": "dynamic-delete", "index": ALL}, "n_clicks"),
    State("container", "children")
)
def display_dropdowns(n_clicks, _, children):
    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if "index" in input_id:
        delete_chart = json.loads(input_id)["index"]
        children = [
            chart
            for chart in children
            if "'index': " + str(delete_chart) not in str(chart)
        ]
    else:
        new_element=html.Div( className="dash-bootstrap",
            style = {
                "width":"20rem",
                "height":"20rem",
                "display":"inline-block",
                "outline":"thin lightgrey solid",
                # "padding":"3rem",
            },
            children = [
                html.Div([
                    dbc.Button(
                        "X",
                        id={"type": "dynamic-delete", "index": n_clicks},
                        n_clicks=0,
                        size = "sm",
                        style={"display": "block"},
                    ),
                    dbc.Input(id={"type":"min","index":n_clicks},
                              type = 'number',
                              placeholder = 'min',
                              value='{:.2e}'.format(originalheight_min),
                              style = {"width":"7rem","height":"1.5rem","display":"inline-block"}),
                    dbc.Input (id = {"type":"max","index":n_clicks},
                               type = 'number',
                               placeholder = 'max',
                               value = '{:.2e}'.format(originalheight_max),
                               style = {"width":"7em","height":"1.5rem","display":"inline-block"}),
                    dbc.Select (
                        id = {"type":"colorscale","index":n_clicks},
                        options = [{"value":x,"label":x}
                                   for x in colorscales],
                        value = 'viridis',
                        style = {"display":"inline-block",
                                 "width":70,
                                 "height":30},
                        size = "sm"
                    ),
                ]),
                dcc.Graph(
                    id={"type": "dynamic-output", "index": n_clicks}, config={'scrollZoom': True},
                    style={"height": "13rem","width":"19rem","display":"block"},
                    figure=create_figure(),

                ),
                dbc.Select(
                    style = {'display':'inlineblock'},
                    id = {"type":"dynamic-dropdown-x","index":n_clicks},
                    options = [{'value':'originalheight','label':'originalheight'},
                               {'value':'modulus','label':'modulus'},
                               {'value':'adhesive','label':'adhesive'},
                               ],
                    value = 'originalheight',size = "sm")])

        children.append(new_element)
    return children


@callback(
    Output("curve", "figure"),
    Input("standalone-checkbox","value"),
    Input({"type": "dynamic-output", "index": ALL}, "hoverData"),
    Input("save_data","data"),
    blocking=True,prevent_initial_call=True)
def curve(value,hoverData,data):
    if dash.callback_context.triggered_id=="save_data":
        return dash.no_update
    if dash.callback_context.triggered_id=="standalone-checkbox":
        return dash.no_update
    if hoverData==[None]:
        return dash.no_update
    else:
        #使用するインデックス0から
        if value==0:
            triggered_id=dash.callback_context.triggered_id['index']
            inputs=dash.callback_context.inputs_list[1]
            all_num=len(inputs)
            num=np.arange(all_num)
            ids=[i["id"]["index"] for i in inputs]
            index=ids.index(triggered_id)
            num=num[index]
            # selectx=hoverData[triggered_id]['points'][0]['x']
            # selecty=hoverData[triggered_id]['points'][0]['y']
            selectx=hoverData[num]['points'][0]['x']
            selecty=hoverData[num]['points'][0]['y']
            exdeflection=global_value.exdefl[selecty,selectx,:]
            rtdeflection=global_value.rtdefl[selecty,selectx,:]
            fig2=go.Figure (layout = layout1,data=[go.Scattergl (
                x = global_value.exramp,
                y = exdeflection,
                mode = 'lines',
                showlegend = False),
             go.Scattergl (
                x = global_value.rtramp,
                y = rtdeflection,
                mode = 'markers',
                showlegend = False)
            ])
            return fig2
        if data=='calculate':
            if value==1:
                triggered_id=dash.callback_context.triggered_id['index']
                inputs=dash.callback_context.inputs_list[1]
                all_num=len(inputs)
                num=np.arange(all_num)
                ids=[i["id"]["index"] for i in inputs]
                index=ids.index(triggered_id)
                num=num[index]
                # selectx=hoverData[triggered_id]['points'][0]['x']
                # selecty=hoverData[triggered_id]['points'][0]['y']
                selectx=hoverData[num]['points'][0]['x']
                selecty=hoverData[num]['points'][0]['y']
                exdelta=exdelta_graph[selecty,selectx,:]
                rtdelta=rtdelta_graph[selecty,selectx,:]
                exforce=exforce_graph[selecty,selectx,:]
                rtforce=rtforce_graph[selecty,selectx,:]
                fig2=go.Figure (layout = layout2,data=[go.Scattergl (
                    x = exdelta,
                    y = exforce,
                    mode = 'lines',
                    showlegend = False),
                 go.Scattergl (
                    x = rtdelta,
                    y = rtforce,
                    mode = 'markers',
                    showlegend = False)
                ])
                return fig2


@callback(
    Output("save_data","data"),
    Input("analyze","n_clicks"),
    State("spring constant","value"),
    State("start","value"),
    State("end","value"),
    State("poisson","value"),
    State("tip radius","value"),
    State("method-input","value"),
    State("model-input","value"),
    State("fit_start","value"),
    State("fit_end","value"),
    prevent_initial_call=True
)
def analyze(n_clicks,springconstant,start,end,poisson,radius,method,model,fit_start,fit_end):
    if n_clicks==None:
        return dash.no_update
    if method==1:
        all_deflmin=deflmin (start,end,springconstant)
        exdelta=np.array(all_deflmin[0]).astype(np.float32)
        exforce=np.array(all_deflmin[1]).astype(np.float32)
        rtdelta=np.array(all_deflmin[2]).astype (np.float32)
        rtforce=np.array(all_deflmin[3]).astype (np.float32)
        all_calculate=numba1.new_calculate2(rtdelta,rtforce)
        DA=np.array (all_calculate[0])
        FA=np.array (all_calculate[1])
        DB=np.array (all_calculate[2])
        radius=float(radius)*(10**-9)
        poisson=float(poisson)
        all_jkr=jkr (radius,DA,FA,DB,poisson)
        return 'calculate'
    # if method==2:
    #     all_deflmin=deflmin (start,end,springconstant)
    #     exdelta=np.array(all_deflmin[0]).astype(np.float32)
    #     exforce=np.array(all_deflmin[1]).astype(np.float32)
    #     rtdelta=np.array(all_deflmin[2]).astype (np.float32)
    #     rtforce=np.array(all_deflmin[3]).astype (np.float32)
    #     radius=np.float32(radius)*(10**-9)
    #     poisson=np.float32(poisson)
    #     all_jkr=numba1.linearfit(rtdelta,rtforce,radius,fit_start,fit_end)
    #     result=fitting(all_jkr[0],all_jkr[1],poisson,all_jkr[2],radius)
    #     return 'calculate'
    if method==2:
        all_deflmin=deflmin (start,end,springconstant)
        exdelta=np.array(all_deflmin[0]).astype(np.float64)
        exforce=np.array(all_deflmin[1]).astype(np.float64)
        rtdelta=np.array(all_deflmin[2]).astype (np.float64)
        rtforce=np.array(all_deflmin[3]).astype (np.float64)
        radius=np.float32(radius)*(10**-9)
        poisson=np.float32(poisson)
        all_jkr=numba1.linearfit2(rtdelta,rtforce,poisson,radius,fit_start,fit_end)
        global_value.modulus=np.array(all_jkr[0]).reshape(256,256)
        global_value.adhesive=np.array(all_jkr[1]).reshape(256,256)
        return 'calculate'

@callback(
        Output({"type": "dynamic-output", "index": MATCH}, "figure"),
        Input ({"type":"dynamic-dropdown-x","index":MATCH},"value"),
        Input ({"type":"colorscale","index":MATCH},"value"),
        Input ({"type":"min","index":MATCH},"value"),
        Input ({"type":"max","index":MATCH},"value"),
        Input("save_data","data"),
        prevent_initiall_call=True)
def display_output(channel,color,min,max,data):
    print(channel)
    if color is None:
        raise PreventUpdate
    if channel=='originalheight':
        height.data[0]['colorscale']=color
        height.data[0]['zmax']=float(max)
        height.data[0]['zmin']=float(min)
        return height
    if data is None:
        return dash.no_update
    if channel=='modulus':
        fig=go.Figure ()
        fig.add_trace (go.Heatmapgl (z = global_value.modulus,zmin=float(min),zmax=float(max),colorscale = color))
        # fig2.add_trace(go.Heatmapgl(z=heatmap2))
        fig.update_layout (yaxis = dict (scaleanchor = 'x'),
                              plot_bgcolor = 'rgba(0,0,0,0)',
                              margin = {'l':30,'t':15,'r':10,'b':30},
                              font = dict (size = 10),template = "plotly_dark")
        return fig
    if channel=='adhesive':
        fig=go.Figure ()
        fig.add_trace (go.Heatmapgl (z = global_value.adhesive,zmin=float(min),zmax=float(max),colorscale = color))
        # fig2.add_trace(go.Heatmapgl(z=heatmap2))
        fig.update_layout (yaxis = dict (scaleanchor = 'x'),
                              plot_bgcolor = 'rgba(0,0,0,0)',
                              margin = {'l':30,'t':15,'r':10,'b':30},
                              font = dict (size = 10),template = "plotly_dark")
        return fig

@callback(
    Output({"type":"min","index":MATCH},"value"),
    Output({"type":"max","index":MATCH},"value"),
    Input ({"type":"min","index":MATCH},"value"),
    Input ({"type":"max","index":MATCH},"value"),
    Input({"type":"dynamic-dropdown-x","index":MATCH},"value"))
def map_scale(scalemin,scalemax,channel):
    if dash.callback_context.triggered_id==None:
        return dash.no_update
    if dash.callback_context.triggered_id['type']=="dynamic-dropdown-x":
        if channel=='modulus':
            scalemin='{:.2e}'.format(np.nanmin(global_value.modulus))
            scalemax='{:.2e}'.format(np.nanmax(global_value.modulus))
            return scalemin,scalemax
        if channel=='originalheight':
            scalemin='{:.2e}'.format(np.nanmin(originalheight_min))
            scalemax='{:.2e}'.format(np.nanmax(originalheight_max))
            return scalemin,scalemax
        if channel=='adhesive':
            scalemin='{:.2e}'.format(np.nanmin(global_value.adhesive))
            scalemax='{:.2e}'.format(np.nanmax(global_value.adhesive))
            return scalemin,scalemax

        else:
            return dash.no_update
    else:
        return scalemin,scalemax








    # elif channel=='modulus':
    #     modulus



# if __name__ == "__main__":
#     app.run_server(debug=None, port=1234,threaded=True, dev_tools_silence_routes_logging=None)