import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import dill
import pandas as pd
import requests

a = dill.load(open('ICD9_flat.pkd','rb'))
Xy = dill.load(open('Xy_CA_2015_loc.pkd','rb'))
op_lib = dill.load(open('op_lib_norm.pkd','rb'))
classifier_model = dill.load(open('CA_classifier.sav','rb'))

############### Beginning of ML classes ###############
from math import sin, cos, sqrt, atan2, radians

def distcalc(coord, lat, long):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(coord[0])
    lon1 = radians(coord[1])
    lat2 = radians(lat)
    lon2 = radians(long)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance
    
    

############### End of ML classes ###############

geocode = pd.read_csv('us-zip-code-latitude-and-longitude.csv',sep=';')

cities_list = [{'label': x, 'value': x} for x in geocode.City.unique()]
states_list = [{'label': x, 'value': x} for x in geocode.State.unique()]

############## Stuff for making CMS database query
# My SoQL Fetcher, for all my tabulated data from CMS

class SoQLFetcher():
    
    def __init__(self):
        self.ROOT = 'https://data.cms.gov/resource/'
        
        self.ROOT_open = 'https://openpaymentsdata.cms.gov/resource/'

        self.rx_identifier = {
            2016:'xbte-dn4t',
            2015:'x77v-hecv',
            2014:'uggq-gnqc',
            2013:'hffa-2yrd'
        }
        # for years 2016, 2015, 2014 and 2013
    
        self.dx_identifier = {
            2016:'haqy-eqp7',
            2015:'4hzz-sw77',
            2014:'cng4-92f3',
            2013:'5fnr-qp4c',
            2012:'j688-dtru'            
        }
        
        self.payments_identifier = {
            2013: 'tvyk-kca8',
            2014: 'gysc-m9qm',
            2015: 'a482-xr32',
            2016: 'daa6-m7ef'
        }
    
    def retrieve(self,
                year = 2015,
                dataset = 'Rx',
                limit = 25000000,
                select = '*',
                where = None,
                group = None,
                having = None,
                order = None):
        
        if dataset == 'Dx':
            key = '.'.join([self.dx_identifier[year], 'json'])
            addr = ''.join([self.ROOT,key])
        elif dataset == 'Rx':
            key = '.'.join([self.rx_identifier[year], 'json'])
            addr = ''.join([self.ROOT,key])
        else:
            key = '.'.join([self.payments_identifier[year], 'json'])
            addr = ''.join([self.ROOT_open,key])

        if limit:
            limit_query = '='.join(['$limit',str(limit)])
            addr = '?'.join([addr,limit_query])
        
        if select:
            select_query = '='.join(['$select',select])
            addr = '&'.join([addr,select_query])

        if where:
            where_query = '='.join(['$where',where])
            addr = '&'.join([addr,where_query])

        if group:
            group_query = '='.join(['$group',group])
            addr = '&'.join([addr,group_query])
            
        if having:
            having_query = '='.join(['$group',having])
            addr = '&'.join([addr,having_query])
        
        if order:
            order_query = '='.join(['$order',order])
            addr = '&'.join([addr,order_query])
            
        data = requests.get(addr)
        
        return pd.DataFrame(data.json())

############## End of CMS database query helper



############# some code for table

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

############# end code for table



####### some helper code for dataframe apply
def extract_lat(x):
    try:
        return geocode[geocode['Zip']==x].Latitude.values[0]
    except IndexError:
        return 0

def extract_lon(x):
    try:
        return geocode[geocode['Zip']==x].Longitude.values[0]
    except IndexError:
        return 0


######## end helper code



############### some code for radar chart

data2 = [dict(
    type = 'scatterpolar',
  r = [39, 28, 8, 7, 28, 39],
  theta = ['Services provided','Rx provided', 'Rx levels', 'Patient volume', 'Vitals Rating', 'Services provided'],
  fill = 'toself'
)]

layout2 = dict(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 100]
    )
  ),
  showlegend = False
)

fig2 = dict(data=data2, layout=layout2)

############### end code for radar chart




ICD9_dx = dill.load(open('op_lib.pkd', 'rb'))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)


app.config.supress_callback_exceptions=True

app.layout = html.Div([
    html.H1(children='National Medicare Provider Lookup'),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    
    html.Div([
        html.Label('Who can help me with: '),
        dcc.Dropdown(
            options=a,
            value='V70',
            multi=False,
            id='input'
        ),
        html.Div(id='output')
    ]),

    html.Label('I live in: '),
    html.Div([
        html.Label('City: '),
        dcc.Dropdown(
            options=cities_list,
            value='Oakland',
            multi=False,
            id='input_city'
        ),
        html.Label('State: '),
        dcc.Dropdown(
            options=states_list,
            value='CA',
            multi=False,
            id='input_state'
        )
    ], style = {'columnCount': 2}),
    
    
    html.Div(id='output_loc'),
    

    #### tab insert here
    
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Your Providers',
               children=[
                   html.Div([
                   ## add list of providers here
                   dcc.Graph(
                   id='radar_chart',
                   figure=fig2
                   )
                   ],style = {'columnCount': 2})
               ]),
        
        dcc.Tab(label='Regional View',
                children=[dcc.Graph(
                            id='geodata_map_region'
                )]),
        dcc.Tab(label='Local View',
                children=[dcc.Graph(
                            id='geodata_map_local'
                )])
    
    ]),
    
    
    #### tab end here
                
                
    html.H4(children='List of most common procedures by provider'),
    html.Div(id='table_loc')
    
], style = {'columnCount': 2})

@app.callback(
    Output('table_loc','children'),
    [Input('input_state', 'value'),
    Input('input', 'value')]
)
def display_table(state, icd):
    try:
        specialty = classifier_model.predict([(op_lib[icd],{})])[0]
    except KeyError:
        specialty = 'Internal Medicine'
        
    df =  SoQLFetcher().retrieve(dataset = 'Dx', year=2015,
                                 select='hcpcs_code,hcpcs_description,sum(bene_day_srvc_cnt) as volume',
                                 where='nppes_provider_state=\'{}\'&provider_type=\'{}\''.format(state,specialty),
                                 group='hcpcs_code,hcpcs_description',
                                 order='volume DESC')
    return generate_table(df.head(10))




@app.callback(
    Output('output', 'children'),
    [Input('input', 'value')]
)      
def display_output(value):
    try:
        specialty = classifier_model.predict([(op_lib[value],{})])
    except KeyError:
        specialty = ['Not in db']
    return 'Seeking expertise in "{}"'.format(specialty[0])


@app.callback(
    Output('output_loc', 'children'),
    [Input('input_city', 'value'),
    Input('input_state', 'value')]
)        
def display_loc(city, state):
    return ', '.join([city, state])



@app.callback(
    Output('geodata_map_region', 'figure'),
    [Input('input_city', 'value'),
    Input('input_state', 'value'),
    Input('input', 'value')]
)
def update_regional(city, state, icd):
    
    ############ Some code for map
    
    try:
        specialty = classifier_model.predict([(op_lib[icd],{})])[0]
    except KeyError:
        specialty = 'Internal Medicine'
    
    df =  SoQLFetcher().retrieve(dataset = 'Dx', year=2015,
                                 select='nppes_provider_zip,sum(bene_day_srvc_cnt) as volume',
                                 where='nppes_provider_state=\'{}\'&provider_type=\'{}\''.format(state,specialty),
                                 group='nppes_provider_zip')

    df['Zip'] = df['nppes_provider_zip'].apply(lambda x: int(x[:5]))
    df['volume'] = df['volume'].apply(lambda x: int(x))
    df = df.drop(columns=['nppes_provider_zip']).groupby('Zip').sum()

    max_volume = max(df['volume'])
    df['Zip']=df.index

    df['lat'] = df['Zip'].apply(extract_lat)
    df['long'] = df['Zip'].apply(extract_lon)

    scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
        [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

    data = [ dict(
            type = 'scattergeo',
            locationmode = 'USA-California',
            lon = df['long'],
            lat = df['lat'],
            text = df['Zip'],
            mode = 'markers',
            marker = dict(
                size = 8,
                opacity = 0.8,
                reversescale = True,
                autocolorscale = False,
                symbol = 'circle',
                line = dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
                colorscale = scl,
                cmin = 0,
                color = df['volume'],
                cmax = df['volume'].max(),
                colorbar=dict(
                    title="Beneficiary Volume"
                )
            ))]

    layout = dict(
            title = 'Physicians matching your query',
            colorbar = True,
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 0.5,
                subunitwidth = 0.5
            ),
        )

    fig = dict( data=data, layout=layout )  
    return fig
############### end code for map


@app.callback(
    Output('geodata_map_local', 'figure'),
    [Input('input_city', 'value'),
    Input('input_state', 'value'),
    Input('input', 'value')]
)
def update_local(city, state, icd):
    try:
        specialty = classifier_model.predict([(op_lib[icd],{})])[0]
    except KeyError:
        specialty = 'Internal Medicine'
    
    coord = [34.137557, -118.207650]
    
    df =  SoQLFetcher().retrieve(dataset = 'Dx', year=2015,
                                 select='nppes_provider_zip',
                                 where='nppes_provider_state=\'{}\'&provider_type=\'{}\''.format(state,specialty),
                                 group='npi,nppes_provider_zip')

    df['Zip'] = df['nppes_provider_zip'].apply(lambda x: int(x[:5]))
    df['count'] = 1
    
    df = df.drop(columns=['nppes_provider_zip']).groupby('Zip').sum()
    df['Zip']=df.index

    df['lat'] = df['Zip'].apply(extract_lat)
    df['long'] = df['Zip'].apply(extract_lon)
    
    df['dist'] = df.apply(lambda row: distcalc(coord, row['lat'],row['long']), axis=1)
    df = df[df['dist']<50]
    

    
    figure={
            'data': [
                {'x': df['dist'], 'y': df['count'], 'type': 'bar', 'name': 'Providers'}
            ],
            'layout': {
                'title': 'Proximity of providers with same specialty'
            }
        }
    
    return figure





if __name__ == '__main__':
    app.run_server(debug=True)