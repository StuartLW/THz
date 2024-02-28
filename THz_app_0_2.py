from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from THz import THzSpec, THzSpecSet, THzData
import base64
import io

temp_df=pd.DataFrame()
THz_data= THzData()
data=None

def parse_contents(contents, filename, timestamp, type):
    ''' Takes an uploaded file and turns it into a THz spectral object'''
    global temp_df
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    spec = THzSpec(io.StringIO(decoded.decode('utf-8')), type, name=filename)
    temp_df=pd.concat([temp_df, pd.DataFrame({'object':spec,'filename':filename, 'type':type},index=[0])],ignore_index=True)
    return True
def window_fig(start, stop):
    ''' generates a figure with the raw data with window superimposed'''
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(title='Raw Data and Window')
    fig.update_xaxes(title='Time (ps)')
    fig.update_yaxes(title='Amplitude (V)', secondary_y=False)
    fig.add_traces(data.window_curves(), secondary_ys=[True,True])
    fig.add_traces(data.raw_curves())
    fig.add_vline(x=stop/data.sample.sampling_rate)
    fig.add_vline(x=start/data.sample.sampling_rate)
    return fig

def processed_fig():
    '''generates the windowed and zero padded data plot'''
    fig = go.Figure()
    fig.update_layout(title='Windowed Signal (ZP)')
    fig.update_xaxes(title='Time (ps)')
    fig.update_yaxes(title='Amplitude (V)')
    fig.add_traces(data.processed_signal_curves())
    return fig
def spectral_fig():
    ''' generates the post fourier transformed spectrum plot'''
    fig = go.Figure()
    fig.update_layout(title='Amplitude of FFT (ZP)')
    fig.update_xaxes(title='Frequency (THz)')
    fig.update_yaxes(title='Amplitude (V)', type='log')
    fig.add_traces(data.spectral_curves())
    return fig
def phase_fig():
    '''generates the phase plot'''
    fig = go.Figure()
    fig.update_layout(title='Phase of FFT (ZP)')
    fig.update_xaxes(title='Frequency (THz)')
    fig.update_yaxes(title='Phase (radians)')
    fig.add_traces(data.unwrapped_phase_curves())
    fig.add_traces(data.wrapped_phase_curves())
    return fig

# read Bootstrap external theme
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)
# style the uploader
uploader_style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin-left': '25%',
            'margin-top':'5%',
            'margin-bottom':'10%',
        }

# app layout using bootstrap
app.layout = html.Div([
     dbc.Row([
        dbc.Col([
             dbc.Row(
                  dbc.Col(html.Div((html.H3('Data Selection', style={'textAlign': 'center'}))))
                  
             ),
             dbc.Row([
                  dbc.Col([
                       html.Div([
                            html.Label("Upload Sample Data", htmlFor="upload_sample"),
                            dcc.Upload(id='upload-sample', children=html.Div([
                            'Drag and Drop or ',html.A('Select Files')]),
                            style=uploader_style,
                            # Allow multiple files to be uploaded
                            multiple=True)]),
                        html.Div(dash_table.DataTable(id='sample_table'), style={'display': 'inline-block', 'margin':10})
                        
                        ],                     
                    width=6,className='text-center'),

                    dbc.Col([
                       html.Div([
                            html.Label("Upload Reference Data", htmlFor="upload_reference"),
                            dcc.Upload(id='upload-reference', children=html.Div([
                            'Drag and Drop or ',html.A('Select Files')]),
                            style=uploader_style,
                            # Allow multiple files to be uploaded
                            multiple=True)]),
                        html.Div(dash_table.DataTable(id='reference_table'), style={'display': 'inline-block', 'margin':10})
                       
                        ], 
                     width=6,className='text-center'),
                ]),
                dbc.Row(
                    dbc.Col(html.Button('Link', id='link_data', n_clicks=0),className='text-center')
                    ),
                dbc.Row(
                     dbc.Col(
                        html.Div([
                            html.Div(dash_table.DataTable(id='data_set_table') )]), style={'display': 'inline-block', 'margin':10},className='text-center'
                        )
                    )              
        ],
        width=3),
        dbc.Col([
             dbc.Row(
                  dbc.Col(html.Div(html.H1('THz Spectral Processor', style={'textAlign': 'center'})))
             ),
             dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Label("Window Start and End", htmlFor="window_range"),
                        # dcc.RangeSlider(0, len(data.sample.amp),1,marks=None,
                        # value=[0,int(len(data.sample.amp)/10)],
                        dcc.RangeSlider(0, 30000,1,marks=None,
                        value=[0,(30000/10)],
                        id='window_range')
                        ]), width = 6
                        ),
                dbc.Col(
                    html.Div([
                        html.Label("Window Curve", htmlFor="window_curve"),
                        dcc.Slider(0, 1,0.01, marks={(i/10): '{}'.format(i/10) for i in range(11)},
                        value=0.1,
                        id='window_curve')
                        ]) ,width = 6

             ),
             ]),
             dbc.Row([
                 dbc.Col(
                     html.Div([
                        html.Label("Zero Padding", htmlFor="zero_fill"),
                        dcc.Slider(2,20,1,marks={(i): '{}'.format(2 ** i) for i in range(2,21)},
                        value=0.1,
                        id='zero_fill')
                     ])
                 )
             ]),
             dbc.Row([
                  dbc.Col(html.Div(dcc.Graph(id='graph1')), width=6),
                  dbc.Col(html.Div(dcc.Graph(id='graph2')), width=6)
             ]),
             dbc.Row([
                  dbc.Col(html.Div(dcc.Graph(id='graph3')), width=6),
                  dbc.Col(html.Div(dcc.Graph(id='graph4')), width=6)
             ])


        ], width=9)

     
    ])
])

# callbacks 
# upload a sample file

@callback(Output('sample_table', 'data'),
              Input('upload-sample', 'contents'),
              State('upload-sample', 'filename'),
              State('upload-sample', 'last_modified'),
              prevent_initial_call=True)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
            parsed=[parse_contents(c, n,d, 'sample') for c, n,d in zip(list_of_contents, list_of_names, list_of_dates)]
            
    return temp_df[temp_df['type']=='sample'][['filename','type']].to_dict('records')

# upload a reference file
@callback(Output('reference_table', 'data'),
              Input('upload-reference', 'contents'),
              State('upload-reference', 'filename'),
              State('upload-reference', 'last_modified'),
              prevent_initial_call=True)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
            parsed=[parse_contents(c, n,d, 'ref') for c, n,d in zip(list_of_contents, list_of_names, list_of_dates)]
            
    return temp_df[temp_df['type']=='ref'][['filename','type']].to_dict('records')

# link a sample and reference together
@callback(Output('data_set_table','data'),
          Output('sample_table','selected_cells'),
          Output('reference_table','selected_cells'),
          Output('sample_table','active_cell'),
          Output('reference_table','active_cell'),
          Input('link_data', 'n_clicks'),
          State('sample_table', 'active_cell'),
          State('reference_table', 'active_cell'),
          prevent_initial_call=True)
def update_links(clicks,sample_cell,ref_cell):
    if not [x for x in (sample_cell, ref_cell) if x is None]:
        temp_df[temp_df['type']=='sample'].iloc[sample_cell['row']]
        THz_data.add_data(temp_df[temp_df['type']=='sample'].iloc[sample_cell['row']]['object'],
                          temp_df[temp_df['type']=='ref'].iloc[ref_cell['row']]['object'],
                          'sample_' + str(clicks))
    if len(THz_data.df)>0:
        return  THz_data.df[['name','sample_file','reference_file']].to_dict('records'),[],[],None,None
    else:
        return [{'error':'Must Select File From Both Tables to Link'}],[],[],None,None
     
# callbacks from the slider updates
@callback(
    Output('graph1', 'figure',allow_duplicate=True),
    Output('graph2', 'figure',allow_duplicate=True),
    Output('graph3', 'figure',allow_duplicate=True),
    Output('graph4', 'figure',allow_duplicate=True),
    Input('window_range', 'value'),
    Input('window_curve', 'value'),
    Input('zero_fill', 'value'),
    prevent_initial_call=True
    )
def update_figure(rng, curve, zp):
    data.process_signals(rng[0],rng[1],curve,zp)
    fig=window_fig(rng[0],rng[1])
    fig2= processed_fig()
    fig3=spectral_fig()
    fig4=phase_fig()
    return fig,fig2,fig3,fig4

#callback for data selection

@callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Output('graph3', 'figure'),
    Output('graph4', 'figure'),
    Input('data_set_table', 'active_cell'),
    prevent_initial_call=True)
def update_data(cell):
    global data
    data=THz_data.df.iloc[cell['row']]['data']
    proc_param=data.sample.last_processing
    if proc_param is not None:
        start,end,curve,zp = proc_param['start'],proc_param['end'],proc_param['curve'],proc_param['zero_padding']
    else:
        start,end,curve,zp = 0,int(len(data.sample.amp)/10),3,2
    data.process_signals(start,end,curve,zp)
    fig=window_fig(start,end)
    fig2= processed_fig()
    fig3=spectral_fig()
    fig4=phase_fig()
    return fig,fig2,fig3,fig4
        



if __name__ == '__main__':
    app.run(debug=True)

