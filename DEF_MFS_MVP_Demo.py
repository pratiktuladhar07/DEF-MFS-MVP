#to work with dataframe
import pandas as pd
#to perform mathematical and statistical calculations
import numpy as np

#for interactive graphs/visualization
#plotly express
import plotly.express as px
#plotly graphical object
import plotly.graph_objects as go
#to work with dashboards
# from jupyter_dash import JupyterDash
from dash import Dash, dcc, html

#import dash core components
from dash import dcc
from dash.dependencies import Input,Output,State
#to create html objects
from dash import html
import datetime

#for database
#!python -m pip install "pymongo[srv]"
from pymongo import MongoClient
import json

#import storage module to main module
import DEF_MFS_MVP_Storage as storage_module
import DEF_MFS_MVP_Timeseries_Forcasting as forecast

#define class object for mongoDB storage module class
mongoDB=storage_module.MongoDbAtlas()

#connect to mongodb database
#retrive username, password and host_address from JSON
username,password,host_address=mongoDB.fetch_creds("DEF-MFS-MVP-Configuration.JSON")
#use the retrived credentials to connect to mongoDB database
mongoDB.connect_database(username,password,host_address)
#connect to tesla database and collection
#for Tesla database
Tesla_collection=mongoDB.create_database("DEF-MFS-MVP-Stocks","DEF-MFS-MVP-Tesla")

dic_from_db_tesla,df_from_db_tesla=mongoDB.fetch_dbdata(Tesla_collection)
# df_from_db_tesla

#for Ford database
Ford_collection=mongoDB.create_database("DEF-MFS-MVP-Stocks","DEF-MFS-MVP-Ford")

dic_from_db_ford,df_from_db_ford=mongoDB.fetch_dbdata(Ford_collection)
# df_from_db_ford


# class to generate predictions and used them to genrate our Plotly Dash application showcasing the demo of our product
class Interactive_Dash:

    # class constructor
    def __init__(self, df_tesla, df_ford):
        self.df_tesla = df_tesla
        self.df_ford = df_ford

    def get_predictions(self):
        # we will be using the module DEF_MFS_MVP_TimeseriesForecasting for forecasting
        # we will use this class instance method to get stock predictions using the DEF_MFS_MVP_TimeseriesForecasting Module
        # one fuction to forecast stocks of any company
        def predict_stocks(data):
            # first create  a class object
            pred = forecast.Prophet_forecast(data)
            # select features as close
            pred.feature_selection("Close")
            # change dataframe format into suitable format of columns ds and y
            pred.prophet_dataframe()
            # split data into training an testing
            pred.Train_Test_split()
            # get forecasted value
            pred.forecast_values()
            # we will be returning actual data points, predicted data points and the prophet model
            return pred.dff, pred.forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], pred.fbp

        # for tesla stock
        self.df_tesla_actual, self.df_tesla_pred, self.tesla_model = predict_stocks(self.df_tesla)
        # for ford stock
        self.df_ford_actual, self.df_ford_pred, self.ford_model = predict_stocks(self.df_ford)
        # display(self.df_tesla_actual)
        ##display(self.df_ford_actual)
        # display(self.df_tesla_pred)
        # display(self.df_ford_pred)

    def create_merge_table(self):
        # in this class instance we will merge actual stock prices with the predicted stock prices into one table
        # we will use merge function
        self.tesla_stocks = pd.merge(self.df_tesla_actual, self.df_tesla_pred, on="ds", how="inner")
        self.ford_stocks = pd.merge(self.df_ford_actual, self.df_ford_pred, on="ds", how="inner")
        # display(self.tesla_stocks)
        # display(self.ford_stocks)
        # the merged dataframe will be later used to create graphs is dash application

    def Dash_Stocks(self):

        # create dash application
        app = Dash(__name__)

        # to describe what app looks like we setup its layout
        # html.Div-our main container
        app.layout = html.Div([
            # this div is to display the heading of our app- Time Series Analysis - Demo'
            html.Div([
                # html header
                html.H1('Time Series Analysis - Demo', style={'text-align': 'center'}),
                html.Br(),
                html.Br(),
            ]),
            # this div is to integrate and show dropdowns in our add- Select Ticker and Dropdown
            html.Div([
                # html paragraph
                html.P("Select Ticker:"),
                # dash core components- dcc allows us to create interactive components like graphs, dropdown menus, or date ranges
                # to enable dropdowns in dash
                dcc.Dropdown(
                    id='graph-type',
                    # title
                    placeholder='Select Stock',
                    # options to choose from
                    options=[
                        {'label': 'Tesla', 'value': 'Tesla'},
                        {'label': 'Ford', 'value': 'Ford'}
                    ],
                    # default value when app starts
                    value="Tesla",
                    style={"width": "30%", "color": "black"}
                ),
                # line break
                html.Br(),
            ]),
            # this div is used to plot candlestick plots based on the tickers selected on dropdown menu
            html.Div([
                html.H2("Candlestick Plot"),
                # to render candlestick data visualization
                dcc.Graph(
                    id='candle-graph',
                ),
                html.Br(),
                html.Br(),

            ]),

            # this div is used to plot graph for actual stock data and predicted stock data along with checklist to select from
            html.Div([
                html.H2("Actual Vs Predictions"),
                # to render any plotly-powered data visualization
                dcc.Graph(
                    id='graph'
                ),
                html.Br(),
                # P object for paragraph-"Select Values" which gets rendered by dash as html element
                html.P("Select Feature Values to Display:"),
                dcc.Checklist(
                    # we define a list of options to choose from
                    options=[
                        {"label": "HISTORIC    ", "value": "y"},
                        {"label": "PREDICTED    ", "value": "yhat"},
                        {"label": "PREDICTED UPPER    ", "value": "yhat_upper"},
                        {"label": "PREDICTED LOWER    ", "value": "yhat_lower"},

                    ],
                    value=["y"],
                    id="checklist",
                    # labelStyle={'display': 'block'},
                ),
                html.Br(),
                html.Br(),
            ]),

            # this div is used to show prediction in table format or predictions in graph format for 3 years by click of a button
            html.Div([
                # html header
                html.H2('Show Predictions for 3 Years', style={'text-align': 'left'}),
                # add button to generate table
                html.Button('Display Table ', id='table-button', n_clicks=0,
                            style={'height': '100px', 'width': '100px'}),
                # add button to generate output
                html.Button('Display Graph', id='graph-button', n_clicks=0,
                            style={'height': '100px', 'width': '100px'}),
                # the output generated when a button is pressed
                html.Div(id='output'),
                html.Br(),
                html.Br(),
            ]),
            # this div is used to display a prediction made for a certain day.
            html.Div([
                html.H2("Forecasting For Any Date"),
                # to enter date when the prediction is to be made
                dcc.Input(id='input-date', type='text',
                          value=datetime.datetime.today().strftime('%Y-%m-%d'),
                          style={'height': '60px', 'width': '100px'},
                          ),
                # button to generate prediction of that date
                html.Button('Submit', id='submit-button', n_clicks=0,
                            style={'height': '60px', 'width': '100px'},
                            ),
                html.Br(),
                html.Br(),
                # prediction made when button is pressed
                html.Div(id='output-div', children='', style={"font-size": "20px"}),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),

            ])

        ],
            # overall page background color and text color
            style={'backgroundColor': '#111111', 'color': "white"}
        )

        # to plot candlestick plot
        @app.callback(
            # what is to be displayed/altered
            Output('candle-graph', 'figure'),
            # what inputs affects the output
            [Input('graph-type', 'value')]
        )
        def choose_graph_type(graph_type):
            # logic is we first check which stock to work with, then plot candlestick using the actual values of that stock retrieved from database
            if graph_type is None:
                raise Dash.exceptions.PreventUpdate()
            if graph_type == 'Tesla':
                # will plot candlestick graph of tesla
                fig = go.Figure(data=[go.Candlestick(x=self.df_tesla['Date'],
                                                     open=self.df_tesla['Open'],
                                                     high=self.df_tesla['High'],
                                                     low=self.df_tesla['Low'],
                                                     close=self.df_tesla['Close'])])
                fig.update_layout(xaxis_title=" Date", yaxis_title="Stock Market Prices")
                fig.update_layout(template="plotly_dark")
                return fig
            elif graph_type == 'Ford':
                # will plot candlestick graph of ford
                fig = go.Figure(data=[go.Candlestick(x=self.df_ford['Date'],
                                                     open=self.df_ford['Open'],
                                                     high=self.df_ford['High'],
                                                     low=self.df_ford['Low'],
                                                     close=self.df_ford['Close'])])
                fig.update_layout(xaxis_title=" Date", yaxis_title="Stock Market Prices")
                fig.update_layout(template="plotly_dark")
                return fig
            return None

        # to plot actual vs predictions
        @app.callback(
            # what is to be displayed
            Output('graph', 'figure'),
            # what inputs affects the output
            [Input('graph-type', 'value'), Input('checklist', 'value')]
        )
        def choose_graph_type(graph_type, checklist):
            # logic is we first see which stock is to work with, and then based on that plot the actual datapoints and the predictions
            # generated by facebook prophet whist training with that actual points
            if graph_type is None:
                raise Dash.exceptions.PreventUpdate()
            if graph_type == 'Tesla':
                fig = px.line(self.tesla_stocks, x='ds', y=checklist)
                fig.update_layout(template="plotly_dark")
                return fig
            elif graph_type == 'Ford':
                fig = px.line(self.ford_stocks, x='ds', y=checklist)
                fig.update_layout(template="plotly_dark")
                return fig
            return None

        # to display table and figure in click of a button
        @app.callback(
            # what is to be displayed/altered
            Output('output', 'children'),
            # what inputs affects the output
            [Input('table-button', 'n_clicks'), Input('graph-button', 'n_clicks'), Input('graph-type', 'value'), ]
        )
        def display_output(table_clicks, graph_clicks, graph_type):
            # logic is to check stock and generate either prediction table or prediction graph based on which  buttons are clicked
            def stock_selection(graph_type):
                if graph_type is None:
                    raise Dash.exceptions.PreventUpdate()
                if graph_type == 'Tesla':
                    return self.df_tesla_pred
                elif graph_type == 'Ford':
                    return self.df_ford_pred
                return None

            df_stock = stock_selection(graph_type)

            if table_clicks == 0 and graph_clicks == 0:
                return ''
            elif table_clicks > graph_clicks:
                return dcc.Graph(
                    id='table',
                    figure={
                        'data': [go.Table(
                            header=dict(values=list(df_stock.columns)),
                            cells=dict(values=[df_stock[col] for col in df_stock.columns])),
                        ],

                    }
                )
            else:
                return dcc.Graph(

                    id='graph-predict',
                    figure={
                        'data': [go.Scatter(x=df_stock['ds'], y=df_stock['yhat'])],
                        'layout': go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Predicted stock Market Prices'},
                            template="plotly_dark",
                        )
                    }
                )
            # elif graph_clicks is not None and table_clicks is not None and graph_clicks>table_clicks

        # to generate predictions based on date entered
        @app.callback(
            # what is to be displayed/altered
            Output('output-div', 'children'),
            # what inputs affects the output
            [Input('submit-button', 'n_clicks')],
            # latest value
            [State('input-date', 'value'), State('graph-type', 'value'), ]
        )
        def update_output_div(n_clicks, date_value, graph_type):
            # logic is that will check ticker, then fetch the date from user when prediction is to be made , then uses prophet to forecast
            if n_clicks:
                if graph_type is None:
                    raise Dash.exceptions.PreventUpdate()
                if graph_type == 'Tesla':
                    forecast_value = float(self.tesla_model.predict(pd.DataFrame({'ds': [date_value]}))["yhat"])
                    return f"The forecasted value for {date_value} is {forecast_value}"

                elif graph_type == 'Ford':
                    forecast_value = float(self.ford_model.predict(pd.DataFrame({'ds': [date_value]}))["yhat"])
                    return f"The forecasted value for {date_value} is {forecast_value}"

                return graph_type

            else:
                return ''

        app.run_server(debug=True)

#create class object to work with class instance method
INT=Interactive_Dash(df_from_db_tesla,df_from_db_ford)