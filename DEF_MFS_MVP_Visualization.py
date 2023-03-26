#### This module will visualize the data as time series and plot them with the x-axes as time and the y-axis as the target feature.

## Libraries

#to work with dataframe
import pandas as pd
#to perform mathematical and statistical calculations
import numpy as np
#for visualization to create plots and graphs
import matplotlib.pyplot as plt
#for  high-level interface for drawing attractive and informative statistical graphics.
import seaborn as sns
#for interactive graphs
import plotly.express as px

## Define class to plot boxplots to show outliers graphically

class Boxplot_Outliers:
    # class constructor
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_boxplots(self):
        # to control the general style of the plots.
        sns.set(style='whitegrid')
        # set figure size
        sns.set(rc={'figure.figsize': (20, 15)})

        outliers_plot = sns.boxplot(data=self.df[["Open", "Close", "High", "Low"]])
        # set_title('Boxplots to display outliers', fontdict = { 'fontsize': 30})
        # This will add title to plot
        outliers_plot.set_title("Boxplots to Display Outliers", fontdict={'fontsize': 30})
        # set x labels with font size 20
        outliers_plot.set_xlabel("Stock Market Feature", fontdict={'fontsize': 20})
        # set y labels with font size 20
        outliers_plot.set_ylabel("Feature Target Values ", fontdict={'fontsize': 20})
        # set ticker size
        outliers_plot.tick_params(axis='both', which='major', labelsize=14)

## Define class to display line plots of out stock market data

class timeseries_plot:
    def __init__(self, dataframe):
        self.df = dataframe

    # instance class method to plot time series graph for all  target features
    def static_lineplots_all(self):
        # we first set the style
        sns.set(style='whitegrid')
        # set figure size
        sns.set(rc={'figure.figsize': (15, 10)})
        # convert to long (tidy) form using .melt() method
        self.dfm = self.df[["Date", "Open", "Close", "High", "Low"]].melt('Date', var_name='Feature_Target',
                                                                          value_name='vals')
        # use line plot to plot all datapoints as time series plot
        all_plots = sns.lineplot(x="Date", y="vals", hue='Feature_Target', data=self.dfm)
        # set title of the figure
        all_plots.set_title("Static Lineplot to Visualize Time Series Data", fontdict={'fontsize': 30})
        # set x labels with font size 20
        all_plots.set_xlabel("Date", fontdict={'fontsize': 20})
        # set y labels with font size 20
        all_plots.set_ylabel("Feature Target ", fontdict={'fontsize': 20})
        # set ticker size
        all_plots.tick_params(axis='both', which='major', labelsize=14)

    # insstance class method to display time series graph for individual target feature
    def static_lineplot_indv(self):
        # we use subplotting to plot multiple graph in one figure background
        # setup figure and axes
        fig, axs = plt.subplots(5, 1, figsize=(15, 15))
        sns.set(style='darkgrid')
        # for "Open" stock market values
        OPEN = sns.lineplot(x="Date", y="Open", data=self.df, ax=axs[0])
        OPEN.set_title("Static Lineplot to Visualize Open Stock prices Data", fontdict={'fontsize': 20})
        # for "Close" stock market values
        CLOSE = sns.lineplot(x="Date", y="Close", data=self.df, ax=axs[1])
        CLOSE.set_title("Static Lineplot to Visualize Close Stock prices Data", fontdict={'fontsize': 20})
        # for "High" stock market values
        HIGH = sns.lineplot(x="Date", y="High", data=self.df, ax=axs[2])
        HIGH.set_title("Static Lineplot to Visualize High Stock prices Data", fontdict={'fontsize': 20})
        # for "Low" stock market values
        LOW = sns.lineplot(x="Date", y="Low", data=self.df, ax=axs[3])
        LOW.set_title("Static Lineplot to Visualize Low Stock prices Data", fontdict={'fontsize': 20})
        # for "Volume" of stocks
        VOLUME = sns.lineplot(x="Date", y="Volume", data=self.df, ax=axs[4])
        VOLUME.set_title("Static Lineplot to Visualize Volume of our Stocks", fontdict={'fontsize': 20})
        plt.tight_layout()

    # instance method to create interactive plots for all target features
    def interactive_lineplots_all(self):
        # we use plotly's line function to create interactive plots
        fig = px.line(self.dfm, x="Date", y="vals",
                      color="Feature_Target",
                      title='A Plotly Express for Time Series Plot of Different Stock Market Prices')
        fig.show()

    # instance method to create interactive plot for "Close" stock market prices
    def interactive_lineplots_close(self):
        # we use plotly's line function to create interactive plots
        fig = px.line(self.df, x="Date", y="Close",
                      title='A Plotly Express for Time Series Plot of "Close" Stock Market Prices')
        fig.show()


