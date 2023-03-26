#### In this module we load the data from the persistent storage and visualize data in a meaningful way considering that our data is a  timeseries data, hence the x-axis is set as time. Then we apply the selected Timeseries Analysis Method to the data.

### Import Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#plotly graphical object
import plotly.graph_objects as go
from matplotlib.pyplot import figure
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split


### Visualize Data as Time Series Plot

#We define a class and use plotly express to plot our stock market data as a time series plot.

class TimeSeries_Visualization:

    # class constructor
    def __init__(self, dataframe):
        self.df = dataframe
        # self.feat_columns=target_features

    # Line Plot
    # our data is a time series data where stock market changes according to time
    # line plots are best used to describe a time series data
    # this instance method will plot line plots for all target_features selected ie Open, Close, High and Low
    def line_plot(self):
        fig = px.line(self.df, x="Date", y=["Open", "High", "Low", "Close"],
                      title='A Plotly Express for Time Series Plot of Different Stock Market Prices')
        fig.update_layout(xaxis_title=" Date", yaxis_title="Stock Market Prices")
        # A range slider is a small subplot-like area below a plot which allows users to pan and zoom the X-axis
        # while maintaining an overview of the chart.
        fig.update_xaxes(rangeslider_visible=True)
        fig.show()
        # save plotly graph as html file
        fig.write_html("interactive_line_all.html")

    # Candlestick plot
    # we will use candlestick chart to describe open, high, low and close for a given dates.
    # The boxes represent the spread between the open and close values and the lines represent the spread between the low and high values.
    # Sample points where the close value is higher (lower) then the open value are called increasing (decreasing).
    # By default, increasing candles are drawn in green whereas decreasing are drawn in red.
    def candlestick_plot(self):
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

    def OHLC_plot(self):
        fig = go.Figure(data=[go.Ohlc(x=self.df['Date'],
                                      open=self.df['Open'],
                                      high=self.df['High'],
                                      low=self.df['Low'],
                                      close=self.df['Close'])])
        fig.update_layout(xaxis_title=" Date", yaxis_title="Stock Market Prices",
                          title="A Plotly go OHLC Plot to describe Open, Close, High and Low Stock Market Values for Different Dates")

        fig.show()
        # save plotly graph as html file
        fig.write_html("interactive_OHLC.html")

# Our Target feature in Close Value, so we will be working on this

#df_tesla=df_tesla=df_from_db_tesla[["Date","Close"]]
#then we set date as our indext
#df_tesla.set_index("Date",inplace=True)


### Analyse Time Series Data

# We define a class to perform different types of analysis on our time series data

class TimeSeries_Analysis:
    # class constructor
    def __init__(self, dataframe):
        self.df = dataframe

    def additive_decomposition(self):
        # this class method is use to model/decompose our time series data as an additive of base level(residue),trend and seaonality
        # value=base level+trend+seasonality
        # we perform seasonal decomposition by using the statasmodel library's seasonal decomposition module
        # Time series decomposition helps us disentangle the time series into components that may be easier to understand and  to forecas
        # used in understanding the data, forecasting, outlier detection

        seasonal_decomp = seasonal_decompose(self.df, model="additive", period=50)
        # save it to a dataframe
        additive_df = pd.concat(
            [seasonal_decomp.seasonal, seasonal_decomp.trend, seasonal_decomp.resid, seasonal_decomp.observed], axis=1)
        additive_df.columns = ['Seasonal', 'Trend', 'Residual', 'Actual_values']
        # display(additive_df.head())
        # plot these data into one figure to illustrate seasonality, trend and residue individually
        plt.figure(figsize=(15, 10))
        seasonal_decomp.seasonal.plot(label='Seasonal')
        seasonal_decomp.trend.plot(label='Trend')
        seasonal_decomp.resid.plot(label='Residual')
        seasonal_decomp.observed.plot(label='Actual Data')
        plt.legend()
        plt.title("Decomposition of Time Series Data in Seasonal, Trend and Residue", fontsize=20)
        plt.xlabel("Date", fontsize=15)
        plt.ylabel("CLOSE Stock Market Data", fontsize=15)

    def autocorellationship(self):
        # this class method is used to show the cross coreelation in our data
        # autocorrelation is the similarity between observations as a function of the time lag between them.
        plt.figure(figsize=(15, 10))
        pd.plotting.lag_plot(self.df["Close"], lag=3)
        plt.title("Auto correlation plot for Close Stock prices with lag=3")
        plt.show()

### Arima Model to Analyse our Time Series Data

# We define a class to perform analysis on our tock market data using arima model

class ARIMA_Analysis:
    # Auto-Regressive Integrated Moving Averag Model
    # based on the assumption that previous values carry inherent information and can be used to predict future values.
    # 3 parameters
    # p: the autoregressive order
    # d: the order of differencing to make the time series stationary
    # q: the moving average order

    # class constructor
    def __init__(self, dataframe):
        self.df = dataframe

        # Time series forecasting works only for a stationary time series since only the behavior of a stationary time series is predictable.

    # so we define a class function to check stationary characteristic
    def stationary_test(self):
        # stationary- time series data whose mean and variance doesnt change when time changes
        # covariance is independent of time
        # Time series forecasting works only for a stationary since only the behavior of a stationary time series is predictable.
        # to check if our time series data is stationary we use Augmentd Dicky-fuller(ADF) test
        # The default null hypothesis of the ADF test is that the time series is non-stationary.
        # p>0.05
        # if p-value of the ADF testis less than the significance level of 0.05,
        # we reject the null hypothesis and conclude that the time series is stationary
        result = adfuller(self.df["Close"])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        if result[1] > 0.05:
            print("The Time Series Data is Non-Stationary and We will Need to Find Order of Differencing")
        else:
            print("The Time Series Data is Stationary")

    def Differencing_d(self):
        # because our time series data is non stationary we need to maker it stationary by diffrencing them
        # we will need to substract previous value by present value(differencing)
        # might need to do diffrencing multiple time ie multiple order of differencing
        # to find order of diffrencing we use acf plot
        # acf plot shows serial correlation in time series data
        # tells how many turn are need to remove any auto correlation in the series
        # to find "d"

        def AutoCorrelationFunction(data, diff):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            ax1.plot(data)
            ax1.set_title(f"Order of Differencing = {diff}")
            plot_acf(data, ax=ax2)

        # print("Original Time Series Plot")
        AutoCorrelationFunction(self.df["Close"], 0)
        # 1st differencing
        diff1 = self.df["Close"].diff().dropna()
        # print("First Differencing")
        AutoCorrelationFunction(diff1, 1)
        # 2nd diff
        diff2 = self.df["Close"].diff().diff().dropna()
        # print("Second Differencing")
        AutoCorrelationFunction(diff2, 2)
        # 3rd diff
        diff3 = self.df["Close"].diff().diff().diff().dropna()
        # print("Third Differencing")
        AutoCorrelationFunction(diff3, 3)
        # we can also use ndifss from pmdarima library to get order of diffentiation
        return ndiffs(self.df["Close"], test="adf")

    def AutoRegressive_p(self):
        # p is the order of Auto Regressive term
        # refers to number of lags to be used as predictors
        # we can find out the required number of AR - p terms by inspecting Partial Autocorrelation plot
        # PACF represents correlation between the series and its lags
        def PartialAutoCorrelation(data, diff):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            ax1.plot(data)
            ax1.set_title(f"Order of Differencing = {diff}")
            ax2.set_ylim(0, 1)
            plot_pacf(data, ax=ax2)

        # print("Original Time Series Plot")
        PartialAutoCorrelation(self.df["Close"], 0)
        # since we chose diffferencing d= 1
        diff1 = self.df["Close"].diff().dropna()
        # print("First Differencing")
        PartialAutoCorrelation(diff1, 1)

    def MovingAverage_q(self):
        # q is the order of Moving Average term
        # it refers to the number of lagged forexcast errors that should go to the Arima model
        # we use the ACF plot for the number of MA terms
        def AutoCorrelation(data, diff):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            ax1.plot(data)
            ax1.set_title(f"Order of Differencing = {diff}")
            ax2.set_ylim(0, 1)
            plot_acf(data, ax=ax2)

        # print("Original Time Series Plot")
        AutoCorrelation(self.df["Close"], 0)
        # since we chose diffferencing d= 1
        diff1 = self.df["Close"].diff().dropna()
        # print("First Differencing")
        AutoCorrelation(diff1, 1)

    def BuildModel(self, p, q, d):
        # well build an ARIMA model to forcast new datapoints
        # split stock market data into training and testing set
        # self.train, self.test = train_test_split(self.df["Close"], train_size=100)
        self.model = ARIMA(self.df["Close"], order=(7, 1, 4))
        self.result = self.model.fit()
        print(self.result.summary())
        # self.model.plot_diagnostics()
        # plt.show()

    def ResidualErrors(self):
        # The “residuals” in a time series model are what is left over after fitting a model.
        # For many time series models, they are equal to the difference between the observations and the corresponding fitted values
        #: error = yt − ˆyt
        residuals = pd.DataFrame(self.result.resid)
        print(residuals)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        ax1.plot(residuals)
        ax2.hist(residuals)