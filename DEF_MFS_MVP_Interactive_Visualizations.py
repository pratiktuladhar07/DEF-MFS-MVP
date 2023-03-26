#### This module will will be used to develop an interactive dashboard in which users can interact with the dashboard at least in one way. When the user interacts with the dashboard a visible change will occur on the dashboard.

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
#import dash core components
# from dash import dcc
from dash import Dash, dcc, html
from dash.dependencies import Input,Output
#to create html objects
# from dash import html

### Class to display interactive plots with plotly

#### This class is defined to plot interactive graphs that can best describe our stock market prices over time
from DEF_MFS_MVP import df_from_db_tesla, df_from_db_ford


class InteractivePlotly():

    # class constructor
    def __init__(self, dataframe):
        self.df = dataframe
        # self.feat_columns=target_features

    # Line Plot
    # our data is a time series data where stock market changes according to time
    # line plots are best used to describe a time series data
    # this instance method will plot line plots for all target_features selected ie Open, Close, High and Low
    def interactive_line_all(self):
        fig = px.line(self.df, x="Date", y=["Open", "High", "Low", "Close"],
                      title='A Plotly Express for Time Series Plot of Different Stock Market Prices')
        fig.update_layout(xaxis_title=" Date", yaxis_title="Stock Market Prices")
        # A range slider is a small subplot-like area below a plot which allows users to pan and zoom the X-axis
        # while maintaining an overview of the chart.
        fig.update_xaxes(rangeslider_visible=True)
        fig.show()
        # save plotly graph as html file
        fig.write_html("interactive_line_all.html")

    # subplots using facetplots to dislay graph of individual values
    def interactive_line_indv(self):
        # With the.melt() function, we may pivot a DataFrame from wide to long format.
        # It manipulates a DataFrame into a structure where one or more columns—Date Column—are identifier variables
        # while all other columns—Open, Close, High, and Low—which are thought of as measured variables—are unpivoted
        # to the row axis, leaving just two non-identifier columns, variable and value.
        dfm = self.df[["Date", "Open", "Close", "High", "Low"]].melt('Date', var_name='Feature_Target',
                                                                     value_name='Values')
        # use line plot and facet_col to make subplots
        fig = px.line(dfm, x="Date", y="Values",
                      facet_col="Feature_Target", facet_col_wrap=1,
                      title='A Plotly Express for Time Series Plot of Different Stock Market Prices')
        # A range slider is a small subplot-like area below a plot which allows users to pan and zoom the X-axis
        # while maintaining an overview of the chart.
        # fig.update_xaxes(rangeslider_visible=True)
        fig.show()
        # save plotly graph as html file
        fig.write_html("interactive_line_indv.html")

    # Candlestick plot
    # we will use candlestick chart to describe open, high, low and close for a given dates.
    # The boxes represent the spread between the open and close values and the lines represent the spread between the low and high values.
    # Sample points where the close value is higher (lower) then the open value are called increasing (decreasing).
    # By default, increasing candles are drawn in green whereas decreasing are drawn in red.
    def interactive_candlestick(self):
        fig = go.Figure(data=[go.Candlestick(x=self.df['Date'],
                                             open=self.df['Open'],
                                             high=self.df['High'],
                                             low=self.df['Low'],
                                             close=self.df['Close'])])
        fig.update_layout(xaxis_title=" Date", yaxis_title="Stock Market Prices",
                          title="A Plotly go Candlestick Plot to describe Open, Close, High and Low Stock Market Values for Different Dates")

        fig.show()
        # save plotly graph as html file
        fig.write_html("interactive_candlestick.html")

    # OHLC plot
    # The OHLC chart (for open, high, low and close) describes open, high, low and close values for a given x coordinate-Date.
    # The tip of the lines represent the low and high values and the horizontal segments represent the open and close values.
    # Sample points where the close value is higher (lower) then the open value are called increasing (decreasing).
    # By default, increasing items are drawn in green whereas decreasing are drawn in red.

    def interactive_OHLC(self):
        fig = go.Figure(data=[go.Ohlc(x=self.df['Date'],
                                      open=self.df['Open'],
                                      high=self.df['High'],
                                      low=self.df['Low'],
                                      close=self.df['Close'])])
        fig.update_layout(
            title="A Plotly go OHLC Plot to describe Open, Close, High and Low Stock Market Values for Different Dates")

        fig.show()
        # save plotly graph as html file
        fig.write_html("interactive_OHLC.html")


### Class to Visualize Graphs in Dash Dashboard

#### This class will be used create interactive analytic web applications with dropdown menus  and option buttons in Dash dashboard.

class Dash_Dashboard():
    # we want ti inherit the wide_to_long() function from the wideformat_to_longformat class

    # class constructor
    def __init__(self, dataframe):
        self.df = dataframe
        # self.feat_columns=target_features

    # this instance method is used to create simple lineplots on dash dashboard
    def Dash_lineplot(self):
        # constructor for creating the dash application
        app = Dash(__name__)
        # set app title
        app.title = "Stock market Values according to Time"
        fig = px.line(self.df, x='Date', y=["Open", "High", "Low", "Close"])
        fig.update_layout(template="plotly_dark")
        # layout of the dash app describes what the app looks like
        # layout is hierarchial tree of components
        # The dash HTML library provides classes for all of the HTML tags and the keyword arguments describe the HTML attributes
        # like style, class name, and ID.
        # html.Div-our main container
        app.layout = html.Div(
            id="app-container",
            children=[
                # html header
                html.H1("Stock market Values according to Time", style={'text-align': 'center'}),
                # P object for paragraph-"Select Values" which gets rendered by dash as html element
                html.P("Unit is in USD"),
                # dcc allows us to create interactive components like graphs, dropdown menus, or date ranges
                dcc.Graph(figure=fig)
            ])

        app.run_server(debug=True)

        # this class instance function will let us choose and plot line graphs for different stock values- Open, High, Low and Close

    # using dropdowns
    def Dash_dropdown_OHLC(self):
        # constructor for creating the dash application
        app = Dash(__name__)
        # set app title
        app.title = "Stock market Values according to Time"
        # layout of the dash app describes what the app looks like
        # layout is hierarchial tree of components
        # The dash HTML library provides classes for all of the HTML tags and the keyword arguments describe the HTML attributes
        # like style, class name, and ID.
        # html.Div-our main container
        app.layout = html.Div([
            # html header
            html.H4('Open High Low and Close Stock Market Values in Dash', style={'text-align': 'center'}),
            # dcc allows us to create interactive components like graphs, dropdown menus, or date ranges
            dcc.Graph(id="time-series-chart"),
            # P object for paragraph-"Select Values" which gets rendered by dash as html element
            html.P("Select Values:"),
            dcc.Dropdown(
                id="Stock_value",
                options=["Open", "High", "Low", "Close"],
                value="Open",
                clearable=False,
            ),
        ])

        # to allow interactivity in dash, we use callback
        # connects dash components with graphs
        # callback function are automatically called by Dash whenever an input component's property changes in order to update
        # output component.
        @app.callback(
            # output element
            Output("time-series-chart", "figure"),
            # input element
            # Whenever an input property changes, the callback function will get called automatically.
            # Dash passes  the new value of the input to the callback function and updates the property of the output component
            # with whatever gets returned by the function.
            Input("Stock_value", "value"))
        def display_time_series(Stock_value):
            fig = px.line(self.df, x='Date', y=Stock_value)
            fig.update_layout(
                template="plotly_dark")
            return fig

        # to run dash server
        # debug=True for debugging features
        app.run_server(debug=True)

    # this instance class will let us pick stock market values to be displayed on our figures using Dash's Checklist
    def DASH_checklist_OHLC(self):
        # constructor for creating the dash application
        app = Dash(__name__)
        # set app title
        app.title = "Stock market Values according to Time"
        # layout of the dash app describes what the app looks like
        # layout is hierarchial tree of components
        # The dash HTML library provides classes for all of the HTML tags and the keyword arguments describe the HTML attributes
        # like style, class name, and ID.
        # html.Div-our main container
        app.layout = html.Div([
            # html header
            html.H4('Open High Low and Close Stock Market Values in Dash', style={'text-align': 'center'}),
            # dcc allows us to create interactive components like graphs, dropdown menus, or date ranges
            dcc.Graph(id="time-series-chart"),
            # P object for paragraph-"Select Values" which gets rendered by dash as html element
            html.P("Select Values:"),
            dcc.Checklist(
                # we define a list of options to choose from
                options=[
                    {"label": "Open", "value": "Open"},
                    {"label": "High", "value": "High"},
                    {"label": "Low", "value": "Low"},
                    {"label": "Close", "value": "Close"},
                ],
                value=["Open"],
                id="checklist",
            ),
        ])

        @app.callback(
            Output("time-series-chart", "figure"),
            Input("checklist", "value"),
        )
        def update(checklist):
            fig = px.line(self.df, x="Date", y=checklist)
            return fig

        app.run_server(debug=True)

    # this function is used to plot interactive candlestick plot where you will have the option to use a range slider
    def DASH_candlestick(self):
        app = Dash(__name__)

        app.layout = html.Div([
            html.H4('Candlestick chart', style={'text-align': 'center'}),
            dcc.Checklist(
                id='toggle-rangeslider',
                options=[{'label': 'Include Rangeslider',
                          'value': 'slider'}],
                value=['slider']
            ),
            dcc.Graph(id="graph"),
        ])

        @app.callback(
            Output("graph", "figure"),
            Input("toggle-rangeslider", "value"))
        def display_candlestick(value):
            fig = go.Figure(go.Candlestick(
                x=self.df['Date'],
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close']
            ))

            fig.update_layout(
                xaxis_rangeslider_visible='slider' in value
            )

            return fig

        app.run_server(debug=True)

### Class to Integrate Dropdowns and Checklist into one Interactive Graph

#### This class will be used create interactive analytic web application where different graphs will be plotted based on dropdown and graph content will be changed based on checklist options.

class Interactive_StockPrices:

    # class constructor
    def __init__(self, df_tesla, df_ford):
        self.df_tesla = df_tesla
        self.df_ford = df_ford

    def Dash_Stocks(self):
        # create dash application
        app = Dash(__name__)
        # to describe what app looks like we setup its layout
        # html.Div-our main container
        app.layout = html.Div([
            # html header
            html.H1('Open High Low and Close Stock Market Values in Dash', style={'text-align': 'center'}),
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
                value="Tesla"
            ),
            # to render any plotly-powered data visualization
            dcc.Graph(
                id='graph'
            ),
            # to render set of checkbox
            dcc.Checklist(
                id='toggle-rangeslider',
                options=[{'label': 'Include Rangeslider',
                          'value': 'slider'}],
                value=['slider'],
            ),
            # P object for paragraph-"Select Values" which gets rendered by dash as html element
            html.P("Select Values:"),
            dcc.Checklist(
                # we define a list of options to choose from
                options=[
                    {"label": "Open", "value": "Open"},
                    {"label": "High", "value": "High"},
                    {"label": "Low", "value": "Low"},
                    {"label": "Close", "value": "Close"},
                ],
                value=["Open"],
                id="checklist",
            ),
        ])

        @app.callback(
            Output('graph', 'figure'),
            [Input('graph-type', 'value'), Input('checklist', 'value'), Input('toggle-rangeslider', 'value')]
        )
        def choose_graph_type(graph_type, checklist, rangeslider):
            if graph_type is None:
                raise Dash.exceptions.PreventUpdate()
            if graph_type == 'Tesla':
                fig = px.line(df_from_db_tesla, x='Date', y=checklist)
                fig.update_layout(template="presentation", xaxis_rangeslider_visible='slider' in rangeslider)
                return fig
            elif graph_type == 'Ford':
                fig = px.line(df_from_db_ford, x='Date', y=checklist)
                fig.update_layout(template="presentation", xaxis_rangeslider_visible='slider' in rangeslider)
                return fig
            return None

        app.run_server(debug=True)