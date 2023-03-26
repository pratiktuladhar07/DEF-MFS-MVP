#### In this module we load the data from the persistent storage and then feeds it to the PROPHET module to carry out a Timeseries Forecasting.


#Libraries
#to work with dataframe
import pandas as pd
#to perform mathematical and statistical calculations
import numpy as np
#DEAL WITH WARNINGS
import warnings
warnings.filterwarnings("ignore")
#to work with facebook prophet
from prophet import Prophet
#represent data figuratively
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot,plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
#compute evaluation metrices
from sklearn.metrics import max_error,mean_absolute_error,mean_squared_error,r2_score
import math


# We define a class to use facebook prophet to forecast predictions of future values of stock market prices

class Prophet_forecast:
    # class constructor
    def __init__(self, dataframe):
        self.df = dataframe

    # preprocessing step 1
    # Firstly we choose 1 feature out of many to fit our facebook prophet model
    def feature_selection(self, feature_name):
        self.dff = self.df[["Date", feature_name]]
        print(self.dff)

    # preprocessing step 2
    # facebook prophet has a special dataframe requirement inorder to train the model.
    # we use a class instance to convert our dataframe into format suitable for fitting and predicting
    def prophet_dataframe(self):
        # The input to Prophet is always a dataframe with two columns: ds and y.
        # The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date.
        # The y column must be numeric, and represents the measurement we wish to forecast.
        # thus renaming our column names
        self.dff = self.dff.rename(columns={"Date": "ds", "Close": "y"})
        print(self.dff)

    def Train_Test_split(self):
        # Split to training and Testing
        # to forecast any values we need to split the data into training and testing set so that it maybe later used for validation
        # we get the last 20 rows of data and store them into a new variable which will be later used for validation
        self.dff_test = self.dff[len(self.dff) - 20:]
        print("Testing Set")
        print(self.dff_test)
        # we get the remaining data excluding the last 20 to train the model
        self.dff_train = self.dff[:-20]
        print("Training Set")
        print(self.dff_train)

    # class instance method to predict future values
    def forecast_values(self):
        # Train Prophet Model
        # we create the facebook prophet class object to use the prophet module
        # Any forecasting procedures are passed into this constructor
        # Prophet will by default fit weekly and yearly seasonalities, we will enable daily seasonalities as well
        self.fbp = Prophet(daily_seasonality=True)
        # fit or train the model
        self.fbp.fit(self.dff_train)

        # Make future  date
        # A data frame with future dates can be obtained by make_future_dataframe method, where periods is the number of days.
        future_dates = self.fbp.make_future_dataframe(periods=50)
        print("Future Dates")
        print(future_dates)

        # Prediction
        self.forecast = self.fbp.predict(future_dates)
        forecast_100 = self.forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(100)
        # last 100 days predictions
        print("Predictions for last 100 days")
        print(forecast_100)

    def plot_prophet(self):
        # we use this class instance method to employ prophets's built-in plot function to graphs of predicted values vs actual values
        # plot actual vs predicted with upper and lower prediction limit
        figure1 = self.fbp.plot(self.forecast, figsize=(15, 10))
        # set 1st axes
        ax1 = figure1.gca()
        # plot testing data points to distinguish from training set
        ax1.plot(self.dff_test["ds"], self.dff_test["y"], 'y.', label="Test Points")
        ax1.legend()
        ax1.set_title("Prophet's in-built plot function ", size=20)
        ax1.set_xlabel("Date", size=15)
        ax1.set_ylabel("Values", size=15)
        # Get layers to overlay significant changepoints on prophet forecast plot.
        a = add_changepoints_to_plot(ax1, self.fbp, self.forecast)
        # plot components like trend and seasonality of time series close price data using prophet's plot
        figure2 = self.fbp.plot_components(self.forecast, figsize=(15, 10))

    def prediction_visualization(self):
        # we use this class instance method to create our own plot of actual and predicted values
        # new_df=pd.merge(self.dff, self.forecast[["ds","yhat"]], on = "ds", how = "inner")
        # print(new_df)
        plt.figure(figsize=(16, 8))
        axes = plt.gca()
        # to plot training data
        self.dff_train.plot(kind='line', x='ds', y='y', label='Actual Training Value', ax=axes)
        # to plot testing data
        self.dff_test.plot(kind='line', x='ds', y='y', label='Actual Testing Value', ax=axes)
        # to plot predicted data
        self.forecast.plot(kind='line', x='ds', y='yhat', label='Predicted Value', ax=axes)
        axes.set_title("Predicted Values using Faebook Prophet VS Actual Values", size=25)
        axes.set_xlabel("Date", size=15)
        axes.set_ylabel("Close Stock Values", size=15)
        plt.legend()
        plt.show()

    def prophet_evaluation(self):
        # we use this class instance to compute cross-validation and performance metrices of the model using Facebook prophet built in function
        # Cross validation
        # prophet's cross_validation perform cross-validation in different sections of the dataset instead of the last section
        # initial – training period length (training set size for the model)
        # initial=400 days means it uses 400 days worth of data to train
        # period – spacing between cutoff dates,time between each fold
        # period=60 means 60 days till next cut off
        # Cut off points are used to cut the historical data and for each cross-validation fit the model using data only up to cutoff point.
        # we treat this as the shift size of training period
        # horizon – forecasting period length
        # horizon=30 means will predict for 60days
        df_cv = cross_validation(self.fbp, initial="400 days", period='30 days', horizon='60 days',
                                 parallel="processes")
        print("Cross Validation using Facebook Prophet")
        print(df_cv)

        # Model performance metrics
        # Compute a series of performance metrics on the output of cross-validation
        df_p = performance_metrics(df_cv)
        print("Performance Metrices using Facebook Prophet")
        print(df_p)
        fig = plot_cross_validation_metric(df_cv, metric='rmse')
        # fig.set_title("Rmse Plot")
        # return df_cv,df_p

    def evaluation_metrices(self):
        # we use this class instance method to compute evaluation metrices using sklearn for 20 days worth of data and for whole dataset
        # to evaluate our model we take 20 days worth of testing data
        # we make a dataframe of actual and predicted values
        df_20days = pd.merge(self.dff_test, self.forecast[["ds", "yhat"]], on="ds", how="inner")
        print("Predicted Values VS Actual Values for last 20 days of Fetched Stock Market Data")
        print(df_20days)
        plt.figure(figsize=(16, 8))
        axes = plt.gca()
        # to plot testing data
        df_20days.plot(kind='line', x='ds', y='y', label='Actual Testing Value', ax=axes)
        # to plot predicted data
        df_20days.plot(kind='line', x='ds', y='yhat', label='Predicted Testing Value', ax=axes)
        axes.set_title("Predicted Values VS Actual Values for last 20 days of Fetched Stock Market Data", size=25)
        axes.set_xlabel("Date", size=15)
        axes.set_ylabel("Close Stock Values", size=15)
        plt.legend()
        plt.show()

        # calculate evaluation metrices

        # define a function to compute evaluation metrices
        def compute_metrices(data):
            # mean squared error
            mse = mean_squared_error(data["y"], data["yhat"])
            # Root Mean Squared Error
            rmse = math.sqrt(mse)
            # Mean Absolute Error
            mae = mean_absolute_error(data["y"], data["yhat"])
            # R2 score
            R2_Score = r2_score(data["y"], data["yhat"])
            df_metrices = pd.DataFrame(
                data={'Evaluation Metric': ["MSE", "RMSE", "MAE", "R2_Score"], 'Values': [mse, rmse, mae, R2_Score]})
            return df_metrices

        # print("Evaluation Metrics for last 20 days of Fetched Stock Market Data")
        df_20days_eval = compute_metrices(df_20days)
        # print(df_20days_eval)
        # for overall data
        # we make dataframe for predicted and actual values
        df_all = pd.merge(self.dff, self.forecast[["ds", "yhat"]], on="ds", how="inner")
        print("Predicted Values VS Actual Values of Fetched Stock Market Data")
        print(df_all)
        # evaluation metrices for overall
        df_all_eval = compute_metrices(df_all)
        # print(df_all_eval)
        return df_20days_eval, df_all_eval
