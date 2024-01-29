import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash_extensions.enrich import Output, Input, html, ALL,State,DashProxy,dash,dcc,MATCH,ClientsideFunction,BlockingCallbackTransform,ServersideOutputTransform,ServersideOutput,callback_context,NoOutputTransform,callback
import numpy as np
import plotly.express as px
from numba import jit, prange
from sklearn.preprocessing import MinMaxScaler
from sklearn.semi_supervised import LabelSpreading
import global_value
import numba1

dash.register_page(__name__)

colorscales = px.colors.named_colorscales()
#
# heatmap=go.Figure([go.Heatmapgl(z=global_value.modulus,colorscale ='Jet',colorbar=dict(title='log modulus[Pa]',showexponent='none'))])
# heatmap.update_layout (yaxis = dict (scaleanchor = 'x',title='pixels'),
#                        xaxis=dict(title='pixels'),
#                        plot_bgcolor='rgba(0,0,0,0)',
#                        margin = {'l':60,'t':10,'r':10,'b':60},
#                        template="plotly_dark",
#                        uirevision=True)
#
#
#

fig_black=go.Figure()
fig_black.update_layout (template="plotly_dark")

card = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Button("mapping!!!!",id='mapping', color="primary",size = 'sm'),
                html.Div ([
                           html.P("slice forcecurve",style = {"marginBottom":0}),
                           dbc.Input (id = 'slice input',type = "number",size = 'sm'),
                           dbc.Button("Go somewhere",id='slice button', color="primary",size = 'sm'),
                           ]),
                html.H4("parameter", className="card-title"),
                html.Div([
                    html.Div([
                        html.P("alpha",style = {"marginBottom":0}),
                        dbc.Input(type="number",id="alpha",placeholder = "alpha",
                                  style={"width": 120, "height": 30, "display":"inline-block"})],style ={"display":"inline-block"}),
                    html.Div ([
                        html.P ("max_iter",style = {"marginBottom":0}),
                        dbc.Input(type="number",id="max_iter",placeholder = "max_iter",
                                  style={"width": 120, "height": 30, "display":"inline-block"})],style ={"display":"inline-block"}),
                      html.Div ([
                        html.P ("n_neighbor",style = {"marginBottom":0}),
                        dbc.Input(type="number",id="n_neighbors",placeholder = "n_neighbors",
                                  style={"width": 120, "height": 30, "display":"inline-block"})],style ={"display":"inline-block"}),
                ]),
                dbc.Button("Go somewhere",id='start', color="primary"),
            ]
        ),
    ],
    style={"width": "30rem"},
)



sidebar = html.Div(
    [
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
                        html.Div([
                            html.Div([
                                dbc.Input (type = "number",id = "labelnumber",placeholder = "labelnumber",
                                           style = {"width":120,"height":30,"display":"inline-block"}),
                                dbc.Button ("OK",id='label button',color = "primary",size="sm",style = {"display":"inline-block"})]),
                            html.Div([
                                html.P('min',style = {"marginBottom":0}),
                                dbc.Input (id = 'labelmap min',
                                           type = 'number',
                                           placeholder = 'min',
                                           style = {"width":"7rem","height":"1.5rem","display":"inline-block"})],style = {"display":"inline-block"}),
                            html.Div([
                                html.P('max',style = {"marginBottom":0}),
                                dbc.Input (id = 'labelmap max',
                                           type = 'number',
                                           placeholder = 'max',
                                           style = {"width":"7em","height":"1.5rem","display":"inline-block"})],style = {"display":"inline-block"}),
                                dbc.Select (
                                    id = 'labelmap color',
                                    options = [{"value":x,"label":x}
                                               for x in colorscales],
                                    value = 'viridis',
                                    style = {"display":"inline-block",
                                             "width":100,
                                             "height":30},
                                    size = "sm"
                            ),
                            dbc.Select (
                                style = {'display':'inlineblock'},
                                id = "map_dropdown",
                                options = [
                                           {'value':'modulus','label':'modulus'},
                                           {'value':'adhesive','label':'adhesive'},
                                           ],
                                value = 'modulus',size = "sm")
                        ]),
                        dcc.Graph (
                            id = 'labelmap',
                            style = {"height":"21rem","width":"30rem","display":"block"}),
                        dcc.Store (id = 'label_data'),
                        dcc.Store (id = 'all_label')
                    ],
                        width=6),
                dbc.Col([html.Div([dcc.Graph(id='labelFC')])],width=6,className='bg-white')]),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P('Correlation Matrix Heatmap'),
                        html.Div ([dcc.Graph(id='result',
                                             figure=fig_black,
                                             style={"height":"21rem","width":"30rem","display":"block"})]),
            ],
            style={"height": "70vh"}
            )
        ]
    )])

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=2),
                dbc.Col(content, width=10)
                ]
            ),
        ],
    fluid=True
    )




@callback(
    Input("slice button","n_clicks"),
    State("slice input","value")
)
def slice(n_clicks,input):
    global_value.rtbase_slice=global_value.rtbase[:,:,:input]



@callback(
    Output("labelFC","figure"),
    Input ("labelmap","clickData"),
    Input("labelmap","hoverData"),
    State("labelFC","figure")
)
def label_hover(clickData,hoverData,fig):
    if hoverData is None:
        return dash.no_update
    triggered_id=callback_context.triggered[0]['prop_id']
    if 'labelmap.clickData'==triggered_id:
        selectx_click=clickData['points'][0]['x']
        selecty_click=clickData['points'][0]['y']
        rtdefl_click=global_value.rtbase_slice[selecty_click,selectx_click,:]
        fig=go.Figure(fig)
        fig.add_trace(go.Scattergl(
            x = global_value.rtramp,
            y = rtdefl_click,
            mode = 'lines',
            name = 'retract',
            hoverinfo = None
        ))
        return fig
    elif 'labelmap.hoverData'==triggered_id:
        if fig==None:
            selectx_click=hoverData['points'][0]['x']
            selecty_click=hoverData['points'][0]['y']
            rtdefl_click=global_value.rtbase_slice[selecty_click,selectx_click,:]
            fig=go.Figure (fig)
            fig.add_trace (go.Scattergl (
                x = global_value.rtramp,
                y = rtdefl_click,
                mode = 'lines',
                name = 'retract',
                hoverinfo = None
            ))
            fig.update_layout (xaxis = dict (title = 'piezoscanner displacement[nm]',
                                              showexponent = 'none'),
                                yaxis = dict (title = 'deflection[nm]',
                                              showexponent = 'none'),
                               template = "plotly_dark",
                               showlegend=False)
            return fig
        else:
            selectx_hover=hoverData['points'][0]['x']
            selecty_hover=hoverData['points'][0]['y']
            rtdefl_hover=global_value.rtbase_slice[selecty_hover,selectx_hover,:]
            fig['data'].pop (-1)
            fig=go.Figure(fig)
            fig.add_trace(go.Scattergl(
                x = global_value.rtramp,
                y = rtdefl_hover,
                mode = 'lines',
                name = 'retract',
                hoverinfo = None
            ))
            return fig
#
@callback(
    Output("labelmap max","value"),
    Output("labelmap min","value"),
    Input ("labelmap max","value"),
    Input ("labelmap min","value"),
    Input("map_dropdown","value"),
    Input("mapping","n_clicks"))
def map_scale(scalemax,scalemin,channel,nclicks):
    if dash.callback_context.triggered_id==None:
        return dash.no_update
    if dash.callback_context.triggered_id=="mapping":
            scalemin='{:.2e}'.format(np.nanmin(global_value.modulus))
            scalemax='{:.2e}'.format(np.nanmax(global_value.modulus))
            return scalemax,scalemin
    if dash.callback_context.triggered_id=="map_dropdown":
        if channel=='modulus':
            scalemin='{:.2e}'.format(np.nanmin(global_value.modulus))
            scalemax='{:.2e}'.format(np.nanmax(global_value.modulus))
            return scalemax,scalemin
        if channel=='adhesive':
            scalemin='{:.2e}'.format (np.nanmin (global_value.adhesive))
            scalemax='{:.2e}'.format (np.nanmax (global_value.adhesive))
            return scalemax,scalemin
        else:
            return dash.no_update
    else:
        return scalemax,scalemin





@callback(
    Output("labelmap", "figure"),
    Input("labelmap", "clickData"),
    Input ("mapping","n_clicks"),
    Input ("labelmap max","value"),
    Input ("labelmap min","value"),
    Input("map_dropdown","value"),
    Input("labelmap color","value"),
    State("label_data",'data'),
    State("labelmap","figure")
)
def click_label(clickData,n_clicks,max,min,channel,color,label,my_graph):
    triggered_id=callback_context.triggered[0]['prop_id']
    if 'mapping.n_clicks'==triggered_id:
        heatmap=go.Figure ([go.Heatmapgl (z = global_value.modulus,colorscale = color,
                                          colorbar = dict (title = 'log modulus[Pa]',showexponent = 'none'))])
        heatmap.update_layout (yaxis = dict (scaleanchor = 'x',title = 'pixels'),
                               xaxis = dict (title = 'pixels'),
                               plot_bgcolor = 'rgba(0,0,0,0)',
                               margin = {'l':60,'t':10,'r':10,'b':30},
                               template = "plotly_dark",
                               uirevision = True)
        return heatmap
    if 'map_dropdown.value'==triggered_id:
        if channel=='modulus':
            heatmap=go.Figure ([go.Heatmapgl (z = global_value.modulus,zmin=float(min),zmax=float(max),colorscale = color,
                                              colorbar = dict (title = 'log modulus[Pa]',showexponent = 'none'))])
            heatmap.update_layout (yaxis = dict (scaleanchor = 'x',title = 'pixels'),
                                   xaxis = dict (title = 'pixels'),
                                   plot_bgcolor = 'rgba(0,0,0,0)',
                                   margin = {'l':60,'t':10,'r':10,'b':30},
                                   template = "plotly_dark",
                                   uirevision = True)
            return heatmap
        if channel=='adhesive':
            heatmap=go.Figure ([go.Heatmapgl (z = global_value.adhesive,zmin=float(min),zmax=float(max),colorscale = color,
                                              colorbar = dict (title = 'adhesive',showexponent = 'none'))])
            heatmap.update_layout (yaxis = dict (scaleanchor = 'x',title = 'pixels'),
                                   xaxis = dict (title = 'pixels'),
                                   plot_bgcolor = 'rgba(0,0,0,0)',
                                   margin = {'l':60,'t':10,'r':10,'b':30},
                                   template = "plotly_dark",
                                   uirevision = True)
            return heatmap
    # if clickData is None:
    #     heatmap=go.Figure ([go.Heatmapgl (z = global_value.modulus,zmin=float(min),zmax=float(max),
    #                                       colorscale = 'Jet',
    #                                       colorbar = dict (title = 'adhesive',showexponent = 'none'))])
    #     heatmap.update_layout (yaxis = dict (scaleanchor = 'x',title = 'pixels'),
    #                            xaxis = dict (title = 'pixels'),
    #                            plot_bgcolor = 'rgba(0,0,0,0)',
    #                            margin = {'l':60,'t':10,'r':10,'b':30},
    #                            template = "plotly_dark",
    #                            uirevision = True)
    #     return heatmap
    if clickData is None:
        return dash.no_update
    selectx_click=clickData['points'][0]['x']
    selecty_click=clickData['points'][0]['y']
    z=np.array(my_graph['data'][0]['z'])
    z[selecty_click,selectx_click]=label
    heatmap=go.Figure ([go.Heatmapgl(z = z,colorscale = color,zmin=float(min),zmax=float(max),
                                     colorbar = dict (title = 'log modulus[Pa]',showexponent = 'none'))])
    heatmap.update_layout (yaxis = dict (scaleanchor = 'x',title = 'pixels'),
                           xaxis = dict (title = 'pixels'),
                           plot_bgcolor = 'rgba(0,0,0,0)',
                           margin={'l':60,'t':10,'r':10,'b':30},
                           template = "plotly_dark",
                           uirevision = True)
    return heatmap


@callback(
    Output("all_label","data"),
    Output("label_data","data"),
    Input("label button","n_clicks"),
    State("labelnumber","value"),
    State("all_label","data"))
def value(n_clicks,value,all_value):
    if n_clicks==None:
        return dash.no_update
    if n_clicks==1:
        history=[value]
        return history,value
    else:
        all_value.append(value)
        return all_value,value

@callback(
    Output("result","figure"),
    Input("start","n_clicks"),
    State("labelmap","figure"),
    State("all_label","data"),
    State("n_neighbors","value"),
    State("alpha","value"),
    State("max_iter","value"))
def labelspreading(n_clicks,heatmapping,all_label,n_neighbors,alpha,n_init):
    if n_clicks==None:
        return dash.no_update
    rtbase_1d=global_value.rtbase_slice.reshape(-1,1)
    scaler=MinMaxScaler()
    Min_Max_rtbase=scaler.fit_transform(rtbase_1d)
    Min_Max_rtbase=Min_Max_rtbase.reshape(65536,-1)
    heatmapping_data=np.array(heatmapping['data'][0]['z'])
    learning_data=np.full((256,256),-1)
    for i,k in enumerate(all_label):
        index=np.where(heatmapping_data==k)
        learning_data[index]=i
    learning_data=learning_data.reshape(65536)
    label_spread=LabelSpreading (kernel = "knn",alpha = alpha,
                                 max_iter = n_init,n_neighbors = n_neighbors)
    label_spread.fit (Min_Max_rtbase,learning_data)
    output_labels=label_spread.transduction_
    label_result=output_labels.reshape(256,256)
    heatmap_result=go.Figure()
    heatmap_result.add_trace(go.Heatmapgl(z=label_result))
    heatmap_result.update_layout (yaxis = dict (scaleanchor = 'x',title = 'pixels'),
                           xaxis = dict (title = 'pixels'),
                           plot_bgcolor = 'rgba(0,0,0,0)',
                           margin={'l':60,'t':10,'r':10,'b':30},
                           template = "plotly_dark",
                           uirevision = True)
    return heatmap_result
