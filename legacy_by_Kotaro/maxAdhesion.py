import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from dash.exceptions import PreventUpdate
#ヘッダーからのパラメータ
zRange_V = 1.586909
zSens_nm_V = 8.500600
deflRange_V = 24.57600
deflSens_nm_V = 53.01765
amplifier = (deflRange_V*deflSens_nm_V*(10**(-9)))/65536
rampsize_V = 9.411100
rampsize = zSens_nm_V*rampsize_V
# print(rampsize)
#ファイルオープン
with open(r'heikou.000', "rb") as f:
    #2byte=1pixelのため、16bitずつ読み込む。
    rectype = np.dtype(np.int16)
    # ヘッダーforceデータ　dataoffcet,datalength=count
    seek = f.seek(73728)
    data = np.fromfile(f, dtype = rectype, count = 16777216)
    # ヘッダーimageデータ（originalheight)
    seek2 = f.seek(40960)
    data2 = np.fromfile(f, dtype = rectype, count = 16384)
    # ヘッダーimageデータ (module)使わない　
    seek3 = f.seek(33660928)
    data3 = np.fromfile(f, dtype = rectype, count = 16384)

# forceデータ取り出し,512点ずつ取り出す＝32768個に分離。
force_data = np.array_split(data, 32768)
#行方向で反転
force_data = np.fliplr(np.array(force_data,dtype=float)*amplifier)
#行きのforceデータに分ける
exdefl = force_data[::2]
#16bitで表せる最小値は-32768であり、最小値で出されているものは、NaNとして扱う
exdefl = np.where(exdefl < -32767*amplifier, np.nan, exdefl)
#帰りのforceデータに分ける
rtdefl = force_data[1::2]
#(128,128,512)の行列に変形する
exdefl1 = np.reshape(exdefl, (128, 128, 512))
rtdefl1 = np.reshape(rtdefl, (128, 128, 512))
#image像をarrayに変換
originallsbwave = np.array(data2)
#image像の計算処理
originallsbWave = (((originallsbwave/65536)*zRange_V*zSens_nm_V)/(10**9))
#image像の配列を(128,128）に変換
originalheight = np.reshape(originallsbWave, (128, 128))
FVmodule = np.reshape(data3, (128, 128))
# rampデータ所得
ramp = []
for i in range(512):
    x = i*(rampsize/512)
    ramp.append(x)
ramp = np.array(ramp)
# 傾き補正する箇所をスライス
rampselect = ramp[:400]
rampselect = np.array(rampselect)
# print(rampselect)
# ベースライン、原点合わせる
exbase = []
rtbase = []
delta3 = []
delta4 = []
p3 = []
p4 = []
rtadhesive_x=[]
rtadhesive_y=[]
#行きの一部分のdeflection force curveを用いて、線形回帰を行い、ベースラインを作る。
for k, i in enumerate(exdefl):
#線形回帰
    model = LinearRegression(fit_intercept = True)
#線形回帰を用いる範囲をスライス
    selectvalue = i[:400]
#NaNのあるインデックスを取得(線形回帰がNaNがあるとできないため)
    nanind = np.where(np.isnan(selectvalue))
#forceデータのNaNのインデックスに合わせて、rampデータを削除
    selectramp = np.delete(rampselect, nanind)
#NaNのあるデータを削除
    selectvalue = selectvalue[np.isfinite(selectvalue)]
    X = selectramp[:, np.newaxis]
    model.fit(X, selectvalue)
    xfit = np.linspace(0, 50)
    Xfit = xfit[:, np.newaxis]
    yfit = model.predict(Xfit)
#傾き
    coef = model.coef_
    # print(yfit)
#切片
    intercept = model.intercept_
#スライス部分(傾き補正、切片補正)
    baseline1 = coef*ramp+intercept
#スライスしていない部分(切片補正のみ）傾きも補正したほうが良い？
    #baseline2 = np.zeros(len(ramp)-len(rampselect))+intercept
    # print(baseline1)
#スライス部分とスライスしていない部分を合わせる
    #baseline3 = np.concatenate([baseline1, baseline2])
#ベースライン補正
    exbase1 = i-baseline1
    rtbase1 = rtdefl[k]-baseline1
#ベースライン補正後の行きのカーブの最小値を引く
    minimum = np.nanmin(exbase1)
#最小値のインデックスを取得
    minimumindex = np.nanargmin(exbase1)
    # print(minimumindex)
#カーブ最小値部分のランプ値を取得
    rampmini = ramp[minimumindex]
#原点決め
    delta1 = (ramp-rampmini)*(10**-9)-(exbase1-minimum)
    delta2 = (ramp-rampmini)*(10**-9)-(rtbase1-minimum)
    p1 = 0.999*exbase1
    p2 = 0.999*rtbase1
    rtminimum_p=np.nanmin (p2)
    #print(rtminimum_p)
    rtminimum_pindex = np.nanargmin(p2)
    #print(rtminimum_pindex)
    delta2minimum=delta2[rtminimum_pindex]
    #print(delta2minimum)
    rtadhesive_x.append(delta2minimum)
    rtadhesive_y.append(rtminimum_p)
    p3.append(p1)
    p4.append(p2)
    delta3.append(delta1)
    delta4.append(delta2)
# print(delta3[1])
p5 = np.array(p3)
p6 = np.array(p4)
delta5 = np.array(delta3)
delta6 = np.array(delta4)
rtadhesive_x = np.reshape(rtadhesive_x,(128,128))
rtadhesive_y = np.reshape(rtadhesive_y,(128,128))
delta7 = np.reshape(delta5, (128, 128, 512))
delta8 = np.reshape(delta6, (128, 128, 512))
p7 = np.reshape(p5, (128, 128, 512))
p8 = np.reshape(p6, (128, 128, 512))

app = dash.Dash()
app.layout = html.Div([
    html.Div(
        html.H1('AFM')),
    dcc.Dropdown(
        id = 'select',
        options = [{'value': 'originalheight', 'label': 'originalheight'},
                   {'value': 'adhesivepoint_x', 'label': 'adhesivepoint_x'},
                   {'value': 'adhesivepoint_y', 'label': 'adhesivepoint_y'}],
        value = 'originalheight'),
    html.Div([
        dcc.Graph(
            id = 'image',
            style = {'width': 600,
                     'display': "inline-block",
                     'verticalAlign': 'top'},
            figure={}),
        dcc.Graph(
            id = 'force_deformation',
            style = {'width': 600,
                     'display': 'inline-block',
                     'verticalAlign': 'top'},
            figure={}),
        dcc.Graph(
            id = 'deflection',
            style = {'width': 600,
                     'display': 'inline-block',
                     'verticalAlign': 'top'},
            figure={}
        )])])


@app.callback(
    Output('force_deformation', 'figure'),
    [Input('image', 'hoverData')]
    )
def update(hoverData):
    if hoverData is None:
        return dash.no_update
    else:
        selectx = hoverData['points'][0]['x']
        selecty = hoverData['points'][0]['y']
        deltaex = delta7[selecty,selectx,:]
        print(selectx,selecty)
        #print(deltaex)
        deltart = delta8[selecty,selectx,:]
        p_ex = p7[selecty,selectx,:]
        p_rt = p8[selecty,selectx,:]
        #print(deltaex)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x = deltaex,
            y = p_ex,
            mode = 'markers')
        )
        fig1.add_trace(go.Scatter(
            x = deltart,
            y = p_rt,
            mode = 'markers')
        )
        fig1.update_layout (xaxis = dict(title='example'))
        return fig1



@app.callback(
    Output('deflection', 'figure'),
    [Input('image', 'hoverData')]
    )
def updatedeflection(hoverData):
    if hoverData is None:
        return dash.no_update
    else:
        selectx = hoverData['points'][0]['x']
        selecty = hoverData['points'][0]['y']
        exdeflection = exdefl1[selecty,selectx,:]
        rtdeflection = rtdefl1[selecty,selectx,:]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x = ramp,
            y = exdeflection,
            mode = 'lines')
        )
        fig2.add_trace(go.Scatter(
            x = ramp,
            y = rtdeflection,
            mode = 'lines')
        )
        fig2.update_layout(xaxis = dict(title = 'example'))
        return fig2


@app.callback(
    Output('image', 'figure'),
    [Input('select', 'value')]
    )
def mapping(factor):
    fig3 = go.Figure()
    if factor == 'originalheight':
        fig3.add_trace(go.Heatmap(z = originalheight))
        fig3.update_layout(yaxis = dict(scaleanchor = 'x'))
        return fig3
    elif factor=='adhesivepoint_x':
        fig3.add_trace(go.Heatmap(z = rtadhesive_x))
        fig3.update_layout(yaxis = dict(scaleanchor = 'x'))
        return fig3
    elif factor=='adhesivepoint_y':
        fig3.add_trace(go.Heatmap(z = rtadhesive_y))
        fig3.update_layout(yaxis = dict(scaleanchor = 'x'))
        return fig3

app.run_server(debug=True)


